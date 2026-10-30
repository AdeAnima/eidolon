//! Byte-counted range origin for real Fetcher integration tests.

use std::io::{BufRead as _, BufReader, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

pub struct VideoServer {
    address: SocketAddr,
    bytes: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<anyhow::Result<()>>>,
}

impl VideoServer {
    pub fn new(video: Vec<u8>) -> anyhow::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let bytes = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let transferred = Arc::clone(&bytes);
        let stopping = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::Acquire) {
                let (mut stream, _) = listener.accept()?;
                if stopping.load(Ordering::Acquire) {
                    break;
                }
                stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                let count = serve_video(&mut stream, &video)?;
                transferred.fetch_add(count, Ordering::Relaxed);
            }
            Ok(())
        });
        Ok(Self {
            address,
            bytes,
            stop,
            thread: Some(thread),
        })
    }

    pub fn url(&self) -> String {
        format!("http://{}/video.mp4", self.address)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }
}

impl Drop for VideoServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect(self.address);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve_video(stream: &mut TcpStream, video: &[u8]) -> anyhow::Result<u64> {
    let mut reader = BufReader::new(&*stream);
    let mut request = String::new();
    reader.read_line(&mut request)?;
    let head = request.starts_with("HEAD ");
    let mut range = None;
    let mut header = String::new();
    loop {
        header.clear();
        if reader.read_line(&mut header)? == 0 || header == "\r\n" {
            break;
        }
        if let Some((name, value)) = header.split_once(':')
            && name.eq_ignore_ascii_case("range")
        {
            let (start, end) = value
                .trim()
                .strip_prefix("bytes=")
                .and_then(|value| value.split_once('-'))
                .ok_or_else(|| anyhow::anyhow!("invalid test HTTP range"))?;
            let start = start.parse::<usize>()?;
            let end = if end.is_empty() {
                video.len() - 1
            } else {
                end.parse::<usize>()?
            };
            range = Some((start, end.min(video.len() - 1)));
        }
    }
    drop(reader);
    let (start, end) = range.unwrap_or((0, video.len() - 1));
    anyhow::ensure!(
        start <= end && end < video.len(),
        "invalid test range bounds"
    );
    let body = &video[start..=end];
    let status = if range.is_some() {
        "206 Partial Content"
    } else {
        "200 OK"
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: video/mp4\r\nAccept-Ranges: bytes\r\nETag: \"fixture-video\"\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    )?;
    if range.is_some() {
        write!(
            stream,
            "Content-Range: bytes {start}-{end}/{}\r\n",
            video.len()
        )?;
    }
    write!(stream, "\r\n")?;
    if !head {
        stream.write_all(body)?;
    }
    stream.flush()?;
    Ok(if head { 0 } else { body.len() as u64 })
}
