//! Real loopback HTTP tests for explicit URL inputs and offline guarantees.

use std::io::{BufRead as _, BufReader, Cursor, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

struct Server {
    address: SocketAddr,
    requests: Arc<AtomicUsize>,
    heads: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<anyhow::Result<()>>>,
}

impl Server {
    fn new(png: Vec<u8>) -> anyhow::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let requests = Arc::new(AtomicUsize::new(0));
        let heads = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let request_counter = Arc::clone(&requests);
        let head_counter = Arc::clone(&heads);
        let stopping = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::Acquire) {
                let (mut stream, _) = listener.accept()?;
                if stopping.load(Ordering::Acquire) {
                    break;
                }
                request_counter.fetch_add(1, Ordering::Relaxed);
                stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                let mut reader = BufReader::new(&stream);
                let mut request = String::new();
                reader.read_line(&mut request)?;
                let mut header = String::new();
                let mut unchanged = false;
                loop {
                    header.clear();
                    if reader.read_line(&mut header)? == 0 || header == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = header.split_once(':')
                        && name.eq_ignore_ascii_case("if-none-match")
                        && value.trim() == "\"fixture-image\""
                    {
                        unchanged = true;
                    }
                }
                drop(reader);
                let mut parts = request.split_whitespace();
                let head = parts.next() == Some("HEAD");
                if head {
                    head_counter.fetch_add(1, Ordering::Relaxed);
                }
                let target = parts.next().unwrap_or("/");
                let page = b"<!doctype html><html><head><meta property=\"og:image\" content=\"/image.png\"></head><body></body></html>";
                let (status, content_type, body): (&str, &str, &[u8]) = match target {
                    "/image.png" => ("200 OK", "image/png", &png),
                    "/page" => ("200 OK", "text/html; charset=utf-8", page),
                    _ => ("404 Not Found", "text/plain", b"missing"),
                };
                let (status, body) = if unchanged && target == "/image.png" {
                    ("304 Not Modified", &[][..])
                } else {
                    (status, body)
                };
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nETag: \"fixture-image\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )?;
                if !head {
                    stream.write_all(body)?;
                }
                stream.flush()?;
            }
            Ok(())
        });
        Ok(Self {
            address,
            requests,
            heads,
            stop,
            thread: Some(thread),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.address)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect(self.address);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn png() -> anyhow::Result<Vec<u8>> {
    let pixels = image::RgbImage::from_fn(48, 32, |x, y| {
        image::Rgb([(x * 5) as u8, (y * 7) as u8, ((x + y) * 3) as u8])
    });
    let mut encoded = Cursor::new(Vec::new());
    pixels.write_to(&mut encoded, image::ImageFormat::Png)?;
    Ok(encoded.into_inner())
}

fn run(home: &Path, args: &[&str], input: Option<&[u8]>) -> anyhow::Result<std::process::Output> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_eidolon"))
        .args(["--no-cache", "--no-embeddings", "--no-keypoints"])
        .args(args)
        .env("HOME", home)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(input) = input {
        child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing stdin"))?
            .write_all(input)?;
    }
    Ok(child.wait_with_output()?)
}

fn report(output: &std::process::Output) -> anyhow::Result<serde_json::Value> {
    anyhow::ensure!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
fn fetch_fresh_flag_skips_cached_head_but_default_revalidates() -> anyhow::Result<()> {
    let home = tempfile::Builder::new()
        .prefix("eidolon-url-fresh-")
        .tempdir_in("/tmp")?;
    let server = Server::new(png()?)?;
    let url = server.url("/image.png");
    let initial = report(&run(home.path(), &["--json", "--fetch-fresh", &url], None)?)?;
    assert_eq!(server.heads.load(Ordering::Relaxed), 1);
    assert_eq!(server.requests.load(Ordering::Relaxed), 2);
    let fresh = report(&run(home.path(), &["--json", "--fetch-fresh", &url], None)?)?;
    assert_eq!(fresh["settings"]["fetch_fresh"], true);
    assert_eq!(server.heads.load(Ordering::Relaxed), 1);
    assert_eq!(server.requests.load(Ordering::Relaxed), 2);
    let default = report(&run(home.path(), &["--json", &url], None)?)?;
    assert_eq!(default["settings"]["fetch_fresh"], false);
    assert_eq!(server.heads.load(Ordering::Relaxed), 2);
    assert_eq!(server.requests.load(Ordering::Relaxed), 3);
    for cached in [&fresh, &default] {
        assert_eq!(cached["files"][0]["path"], initial["files"][0]["path"]);
        assert_eq!(cached["files"][0]["url"]["bytes_fetched"], 0);
    }
    Ok(())
}

#[test]
fn offline_rejects_positional_and_stdin_urls_without_a_request() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-offline-")
        .tempdir_in("/tmp")?;
    let server = Server::new(png()?)?;
    let url = server.url("/image.png");
    let positional = run(dir.path(), &["--json", "--offline", &url], None)?;
    let lines = format!("{url}\n");
    let newline = run(
        dir.path(),
        &["--json", "--offline", "--stdin"],
        Some(lines.as_bytes()),
    )?;
    let nuls = format!("{url}\0");
    let nul = run(
        dir.path(),
        &["--json", "--offline", "--stdin", "-0"],
        Some(nuls.as_bytes()),
    )?;
    for output in [positional, newline, nul] {
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("URL inputs are disabled by --offline")
        );
    }
    assert_eq!(server.requests.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn url_failures_warn_and_continue_but_all_failed_inputs_exit_nonzero() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-url-errors-")
        .tempdir_in("/tmp")?;
    let local = dir.path().join("local.txt");
    std::fs::write(
        &local,
        "A local input remains reportable when a remote URL fails.",
    )?;
    let local = local
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("UTF-8 local path"))?;
    let server = Server::new(png()?)?;
    let missing = server.url("/missing");
    let mixed = run(dir.path(), &["--json", local, &missing], None)?;
    let json = report(&mixed)?;
    assert_eq!(json["files"].as_array().map(Vec::len), Some(1));
    let stderr = String::from_utf8(mixed.stderr)?;
    assert!(stderr.contains("warning: failed URL"));
    assert!(stderr.contains("1 failed URLs"));
    let all_failed = run(dir.path(), &["--json", &missing], None)?;
    assert!(!all_failed.status.success());
    let stderr = String::from_utf8(all_failed.stderr)?;
    assert!(stderr.contains("all 1 URL inputs failed"));
    assert!(stderr.contains("1 failed URLs"));
    assert!(server.requests.load(Ordering::Relaxed) >= 2);
    Ok(())
}

#[test]
fn mixed_paths_and_urls_classify_images_and_preserve_provenance() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-url-image-")
        .tempdir_in("/tmp")?;
    let bytes = png()?;
    let size = bytes.len() as u64;
    let local = dir.path().join("local.png");
    std::fs::write(&local, &bytes)?;
    let local = local
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("UTF-8 local path"))?;
    let server = Server::new(bytes)?;
    let url = server.url("/image.png");
    let output = run(dir.path(), &["--json", local, &url], None)?;
    let json = report(&output)?;
    assert_eq!(json["settings"]["thumbnails_requested"], "auto");
    assert_eq!(json["settings"]["thumbnails"], true);
    assert_eq!(json["files"].as_array().map(Vec::len), Some(2));
    let remote = json["files"]
        .as_array()
        .and_then(|files| files.iter().find(|file| file.get("url").is_some()))
        .ok_or_else(|| anyhow::anyhow!("missing remote file"))?;
    assert_eq!(remote["kind"], "image");
    assert_eq!(remote["url"]["url"], url);
    assert_eq!(remote["url"]["total_size"], size);
    assert!(
        remote["url"]["bytes_fetched"]
            .as_u64()
            .is_some_and(|count| count > 0)
    );
    assert_eq!(json["matches"][0]["relation"], "identical");
    let human = run(dir.path(), &[local, &url], None)?;
    anyhow::ensure!(
        human.status.success(),
        "{}",
        String::from_utf8_lossy(&human.stderr)
    );
    let text = String::from_utf8(human.stdout)?;
    assert!(text.contains(&url));
    assert!(!text.contains("Library/Caches"));
    assert!(text.contains("Remote transfer:"));
    assert!(text.contains("0 failed URLs"));
    Ok(())
}

#[test]
fn webpage_thumbnail_keeps_actual_url_page_url_and_og_role() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-url-page-")
        .tempdir_in("/tmp")?;
    let bytes = png()?;
    let local = dir.path().join("local.png");
    std::fs::write(&local, &bytes)?;
    let local = local
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("UTF-8 local path"))?;
    let server = Server::new(bytes)?;
    let page = server.url("/page");
    let image = server.url("/image.png");
    let output = run(dir.path(), &["--json", local, &page], None)?;
    let json = report(&output)?;
    let origin = json["files"]
        .as_array()
        .and_then(|files| {
            files.iter().find_map(|file| {
                let origin = file.get("url")?;
                (origin["url"] == image).then_some(origin)
            })
        })
        .ok_or_else(|| anyhow::anyhow!("missing discovered thumbnail provenance"))?;
    assert_eq!(origin["page_url"], page);
    assert_eq!(origin["role"], "og:image");
    let human = run(dir.path(), &[local, &page], None)?;
    anyhow::ensure!(
        human.status.success(),
        "{}",
        String::from_utf8_lossy(&human.stderr)
    );
    let text = String::from_utf8(human.stdout)?;
    assert!(text.contains(&format!("{image} [og:image]")));
    assert!(!text.contains("Library/Caches"));
    Ok(())
}
