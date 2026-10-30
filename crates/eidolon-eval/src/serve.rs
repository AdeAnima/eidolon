//! Loopback-only HTTP media serving with byte-range support and body-byte accounting.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead as _, BufReader, Read as _, Seek as _, SeekFrom, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context as _, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::model::Truth;

/// Select one fifth per media kind, covering each non-original variant when the budget permits.
#[must_use]
pub fn url_sample(truth: &Truth) -> HashSet<PathBuf> {
    let mut result = HashSet::new();
    for kind in ["image", "video"] {
        let mut eligible: Vec<_> = truth
            .files
            .iter()
            .filter(|file| file.kind == kind && file.variant != "original")
            .map(|file| {
                let mut hash = Sha256::new();
                hash.update(truth.seed.to_le_bytes());
                hash.update(file.path.as_os_str().as_encoded_bytes());
                (hash.finalize(), &file.path, &file.variant)
            })
            .collect();
        eligible.sort_unstable();
        let budget = eligible.len() / 5;
        let mut representatives = BTreeMap::new();
        for (_, path, variant) in &eligible {
            representatives.entry(*variant).or_insert(*path);
        }
        let mut selected: HashSet<_> = if representatives.len() <= budget {
            representatives.values().copied().collect()
        } else {
            HashSet::new()
        };
        for (_, path, _) in &eligible {
            if selected.len() == budget {
                break;
            }
            selected.insert(*path);
        }
        result.extend(selected.into_iter().cloned());
    }
    result
}

/// Recover a corpus-relative media path from the original URL, not a cache artifact path.
///
/// # Errors
/// Fails for malformed URLs or paths that are not valid UTF-8.
pub fn media_path(url: &str) -> anyhow::Result<PathBuf> {
    let url = url::Url::parse(url)?;
    Ok(PathBuf::from(
        percent_encoding::percent_decode_str(url.path())
            .decode_utf8()?
            .trim_start_matches('/'),
    ))
}

/// Measured HTTP response-body traffic for one corpus media file.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FileTraffic {
    /// Full local/remote object size, counted once in the baseline.
    pub size: u64,
    /// Measured body bytes, including repeated/overlapping reads.
    pub bytes_served: u64,
    /// Observed GET/HEAD count; null for CLI aggregates without request counts.
    pub requests: Option<u64>,
}

/// Counted media-body traffic; headers and the statistics endpoint are excluded.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Traffic {
    /// Counters keyed by relative media pathname.
    pub files: BTreeMap<String, FileTraffic>,
    /// Total response-body bytes written for the selected objects.
    pub bytes_served: u64,
    /// Sum of complete object sizes, each selected object counted once.
    pub full_remote_bytes: u64,
    /// Actual bytes divided by complete sizes; may exceed one on repeated reads.
    pub bytes_fetched_ratio: f64,
}

impl Traffic {
    /// Restrict counters and baseline to the exact URL sample, including cache hits.
    #[must_use]
    pub fn for_paths(&self, paths: &HashSet<PathBuf>) -> Self {
        let mut result = Self {
            files: self
                .files
                .iter()
                .filter(|(path, _)| paths.contains(Path::new(path.as_str())))
                .map(|(path, counts)| (path.clone(), counts.clone()))
                .collect(),
            ..Self::default()
        };
        result.finish(true);
        result
    }

    pub(crate) fn finish(&mut self, include_unrequested: bool) {
        self.bytes_served = self.files.values().map(|file| file.bytes_served).sum();
        self.full_remote_bytes = self
            .files
            .values()
            .filter(|file| include_unrequested || file.requests.unwrap_or(0) > 0)
            .map(|file| file.size)
            .sum();
        self.bytes_fetched_ratio = if self.full_remote_bytes == 0 {
            0.0
        } else {
            self.bytes_served as f64 / self.full_remote_bytes as f64
        };
    }
}

struct Asset {
    path: PathBuf,
    size: u64,
    etag: String,
    modified: String,
}

/// A scoped loopback server; dropping it closes the listener and persists counters.
pub struct CorpusServer {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    traffic: Arc<Mutex<Traffic>>,
    worker: Option<JoinHandle<anyhow::Result<()>>>,
}

impl CorpusServer {
    /// Bind `127.0.0.1:port` and serve only manifest-listed media; zero selects a free port.
    ///
    /// # Errors
    /// Fails for invalid paths, inaccessible files, or unavailable ports.
    pub fn start(truth: &Truth, port: u16) -> anyhow::Result<Self> {
        let root = truth.root.canonicalize()?;
        let mut assets = BTreeMap::new();
        let mut traffic = Traffic::default();
        for file in &truth.files {
            let path = root.join(&file.path).canonicalize()?;
            ensure!(path.starts_with(&root), "media escapes corpus root");
            let metadata = fs::metadata(&path)?;
            let name = file
                .path
                .to_str()
                .context("media path is not UTF-8")?
                .replace('\\', "/");
            assets.insert(
                format!("/{name}"),
                Asset {
                    path,
                    size: metadata.len(),
                    etag: format!("\"{}\"", file.sha256),
                    modified: httpdate::fmt_http_date(metadata.modified()?),
                },
            );
            traffic.files.insert(
                name,
                FileTraffic {
                    size: metadata.len(),
                    requests: Some(0),
                    ..FileTraffic::default()
                },
            );
        }
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let traffic = Arc::new(Mutex::new(traffic));
        let worker_stop = Arc::clone(&stop);
        let worker_traffic = Arc::clone(&traffic);
        let worker = thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut connection, _)) => {
                        // macOS accepted sockets inherit the listener's nonblocking mode.
                        connection.set_nonblocking(false)?;
                        connection.set_read_timeout(Some(Duration::from_secs(2)))?;
                        connection.set_write_timeout(Some(Duration::from_secs(5)))?;
                        // A disconnected client must not terminate the entire corpus server.
                        if let Err(error) = handle(&mut connection, &assets, &worker_traffic) {
                            eprintln!("HTTP request failed: {error:#}");
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            save(&root, &worker_traffic)
        });
        Ok(Self {
            address,
            stop,
            traffic,
            worker: Some(worker),
        })
    }

    /// Actual bound loopback address.
    #[must_use]
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Encode a manifest-relative file path as an absolute HTTP URL.
    ///
    /// # Errors
    /// Fails for non-UTF-8 or non-relative path components.
    pub fn url_for(&self, path: &Path) -> anyhow::Result<String> {
        let mut url = url::Url::parse(&format!("http://{}/", self.address))?;
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|()| anyhow::anyhow!("URL cannot hold path segments"))?;
            for component in path.components() {
                let std::path::Component::Normal(component) = component else {
                    anyhow::bail!("URL media path must be relative");
                };
                segments.push(component.to_str().context("media path is not UTF-8")?);
            }
        }
        Ok(url.into())
    }

    /// Read a consistent snapshot without stopping service.
    ///
    /// # Errors
    /// Fails if the statistics lock was poisoned.
    pub fn snapshot(&self) -> anyhow::Result<Traffic> {
        let mut result = self
            .traffic
            .lock()
            .map_err(|_| anyhow::anyhow!("traffic lock poisoned"))?
            .clone();
        result.finish(false);
        Ok(result)
    }

    /// Stop service, wait for the current response, and return final counters.
    ///
    /// # Errors
    /// Fails for worker errors or inaccessible statistics output.
    pub fn stop(mut self) -> anyhow::Result<Traffic> {
        self.shutdown()?;
        self.snapshot()
    }

    fn shutdown(&mut self) -> anyhow::Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("HTTP worker panicked"))??;
        }
        Ok(())
    }
}

impl Drop for CorpusServer {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn handle(
    stream: &mut TcpStream,
    assets: &BTreeMap<String, Asset>,
    counters: &Mutex<Traffic>,
) -> anyhow::Result<()> {
    let (method, target, range) = read_request(stream)?;
    if !matches!(method.as_str(), "GET" | "HEAD") {
        return empty(stream, "405 Method Not Allowed", "Allow: GET, HEAD\r\n");
    }
    if target == "/__eidolon__/stats" {
        let bytes = {
            let mut snapshot = counters
                .lock()
                .map_err(|_| anyhow::anyhow!("traffic lock poisoned"))?;
            snapshot.finish(false);
            serde_json::to_vec(&*snapshot)?
        };
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        )?;
        if method == "GET" {
            stream.write_all(&bytes)?;
        }
        return Ok(());
    }
    let decoded = percent_encoding::percent_decode_str(&target).decode_utf8()?;
    let Some(asset) = assets.get(decoded.as_ref()) else {
        return empty(stream, "404 Not Found", "");
    };
    let name = decoded.trim_start_matches('/');
    *counters
        .lock()
        .map_err(|_| anyhow::anyhow!("traffic lock poisoned"))?
        .files
        .get_mut(name)
        .context("missing traffic file")?
        .requests
        .get_or_insert(0) += 1;
    let (start, end, partial) = if let Some(range) = range {
        let Some((start, end)) = parse_range(&range, asset.size) else {
            return empty(
                stream,
                "416 Range Not Satisfiable",
                &format!("Content-Range: bytes */{}\r\n", asset.size),
            );
        };
        (start, end, true)
    } else {
        (0, asset.size.saturating_sub(1), false)
    };
    let count = if asset.size == 0 { 0 } else { end - start + 1 };
    let status = if partial {
        "206 Partial Content"
    } else {
        "200 OK"
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {count}\r\nAccept-Ranges: bytes\r\nETag: {}\r\nLast-Modified: {}\r\nConnection: close\r\n",
        asset.etag, asset.modified
    )?;
    if partial {
        write!(
            stream,
            "Content-Range: bytes {start}-{end}/{}\r\n",
            asset.size
        )?;
    }
    write!(stream, "\r\n")?;
    if method == "HEAD" {
        return Ok(());
    }
    let mut file = File::open(&asset.path)?;
    file.seek(SeekFrom::Start(start))?;
    let written = write_body(stream, file.take(count));
    counters
        .lock()
        .map_err(|_| anyhow::anyhow!("traffic lock poisoned"))?
        .files
        .get_mut(name)
        .context("missing traffic file")?
        .bytes_served += written;
    Ok(())
}

fn read_request(stream: &TcpStream) -> anyhow::Result<(String, String, Option<String>)> {
    let mut reader = BufReader::new(stream.try_clone()?.take(16 * 1024 + 1));
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut fields = line.split_whitespace();
    let method = fields.next().unwrap_or("").to_owned();
    let target = fields
        .next()
        .unwrap_or("")
        .split('?')
        .next()
        .unwrap_or("")
        .to_owned();
    let mut range = None;
    let mut length = line.len();
    loop {
        line.clear();
        ensure!(reader.read_line(&mut line)? > 0, "incomplete HTTP request");
        length += line.len();
        ensure!(length <= 16 * 1024, "HTTP headers exceed16 KiB");
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("range")
        {
            range = Some(value.trim().to_owned());
        }
    }
    Ok((method, target, range))
}

fn write_body(stream: &mut TcpStream, mut file: impl std::io::Read) -> u64 {
    let mut written = 0;
    let mut buffer = [0_u8; 16 * 1024];
    while let Ok(count) = file.read(&mut buffer) {
        if count == 0 {
            break;
        }
        let mut offset = 0;
        while offset < count {
            match stream.write(&buffer[offset..count]) {
                Ok(0) | Err(_) => return written,
                Ok(count) => {
                    offset += count;
                    written += count as u64;
                }
            }
        }
    }
    written
}

fn parse_range(header: &str, size: u64) -> Option<(u64, u64)> {
    let (start, end) = header.strip_prefix("bytes=")?.split_once('-')?;
    if size == 0 {
        return None;
    }
    if start.is_empty() {
        let count: u64 = end.parse().ok()?;
        return (count > 0).then(|| (size.saturating_sub(count), size - 1));
    }
    let start: u64 = start.parse().ok()?;
    let end = if end.is_empty() {
        size - 1
    } else {
        end.parse::<u64>().ok()?.min(size - 1)
    };
    (start <= end && start < size).then_some((start, end))
}

fn empty(stream: &mut TcpStream, status: &str, extra: &str) -> anyhow::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n"
    )?;
    Ok(())
}

fn save(root: &Path, counters: &Mutex<Traffic>) -> anyhow::Result<()> {
    let mut snapshot = counters
        .lock()
        .map_err(|_| anyhow::anyhow!("traffic lock poisoned"))?;
    snapshot.finish(false);
    let temporary = root.join(".url-traffic.json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&*snapshot)?)?;
    fs::rename(temporary, root.join("url-traffic.json"))?;
    Ok(())
}
