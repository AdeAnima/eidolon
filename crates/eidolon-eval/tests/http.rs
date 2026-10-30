//! Real local HTTP requests exercise range semantics and traffic accounting.

use std::io::{Read as _, Write as _};
use std::net::TcpStream;

use eidolon_eval::model::Truth;
use eidolon_eval::serve::CorpusServer;
use serde_json::json;

#[test]
fn serves_ranges_head_suffix_and_rejects_traversal_without_counting_headers() -> anyhow::Result<()>
{
    let directory = tempfile::Builder::new()
        .prefix("eidolon-http-")
        .tempdir_in("/tmp")?;
    std::fs::create_dir_all(directory.path().join("media/image"))?;
    let bytes: Vec<_> = (0..1_000).map(|value| (value % 251) as u8).collect();
    std::fs::write(directory.path().join("media/image/a.bin"), &bytes)?;
    let truth: Truth = serde_json::from_value(json!({
        "version":1,"seed":1,"root":directory.path(),"scale":"test","sources":[],"pairs":[],"hard_negatives":[],
        "files":[{"path":"media/image/a.bin","kind":"image","variant":"exact","family":"one","sha256":""}]
    }))?;
    let server = CorpusServer::start(&truth, 0)?;
    let request =
        |method: &str, path: &str, range: Option<&str>| -> anyhow::Result<(String, Vec<u8>)> {
            let mut stream = TcpStream::connect(server.address())?;
            // Accepted sockets must wait for request bytes, not inherit nonblocking reads.
            std::thread::sleep(std::time::Duration::from_millis(30));
            write!(
                stream,
                "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
            )?;
            if let Some(range) = range {
                write!(stream, "Range: {range}\r\n")?;
            }
            write!(stream, "\r\n")?;
            let mut response = Vec::new();
            stream.read_to_end(&mut response)?;
            let boundary = response
                .windows(4)
                .position(|bytes| bytes == b"\r\n\r\n")
                .ok_or_else(|| anyhow::anyhow!("HTTP header boundary"))?;
            Ok((
                String::from_utf8(response[..boundary].to_vec())?,
                response[boundary + 4..].to_vec(),
            ))
        };
    let (headers, body) = request("HEAD", "/media/image/a.bin", None)?;
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert!(headers.contains("Content-Length: 1000"));
    assert_eq!(body, [0_u8; 0]);
    let (headers, body) = request("GET", "/media/image/a.bin", Some("bytes=10-29"))?;
    assert!(headers.starts_with("HTTP/1.1 206"));
    assert!(headers.contains("Content-Range: bytes 10-29/1000"));
    assert_eq!(body, bytes[10..30]);
    let (_, body) = request("GET", "/media/image/a.bin", Some("bytes=-12"))?;
    assert_eq!(body, bytes[988..]);
    let (headers, body) = request("GET", "/media/image/a.bin", Some("bytes=1000-"))?;
    assert!(headers.starts_with("HTTP/1.1 416"));
    assert!(headers.contains("Content-Range: bytes */1000"));
    assert_eq!(body, [0_u8; 0]);
    let (headers, _) = request("GET", "/%2e%2e/truth.json", None)?;
    assert!(headers.starts_with("HTTP/1.1 404"));
    let traffic = server.stop()?;
    assert_eq!(traffic.files["media/image/a.bin"].bytes_served, 32);
    assert_eq!(traffic.bytes_fetched_ratio, 0.032);
    Ok(())
}

#[test]
fn samples_exactly_twenty_percent_per_media_kind_and_keeps_originals_local() -> anyhow::Result<()> {
    let mut files = Vec::new();
    for kind in ["image", "video", "text"] {
        for index in 0..10 {
            files.push(json!({"path":format!("media/{kind}/{index}.bin"),"kind":kind,"variant":"exact","family":format!("{kind}-{index}"),"sha256":""}));
        }
        files.push(json!({"path":format!("media/{kind}/original.bin"),"kind":kind,"variant":"original","family":"original","sha256":""}));
    }
    let truth: Truth = serde_json::from_value(json!({
        "version":1,"seed":42,"root":"/corpus","scale":"test","sources":[],"pairs":[],"hard_negatives":[],"files":files
    }))?;
    let sample = eidolon_eval::serve::url_sample(&truth);
    assert_eq!(sample.len(), 4);
    assert_eq!(
        sample
            .iter()
            .filter(|path| path.starts_with("media/image"))
            .count(),
        2
    );
    assert_eq!(
        sample
            .iter()
            .filter(|path| path.starts_with("media/video"))
            .count(),
        2
    );
    assert!(!sample.iter().any(|path| path.ends_with("original.bin")));
    assert_eq!(sample, eidolon_eval::serve::url_sample(&truth));
    Ok(())
}

#[test]
fn remote_sample_covers_rare_clip_variants_when_budget_permits() -> anyhow::Result<()> {
    let mut files: Vec<_> = (0..40)
        .map(|index| {
            json!({
                "path":format!("media/video/{index}_exact.mp4"),"kind":"video",
                "variant":"exact","family":format!("video-{index}"),"sha256":""
            })
        })
        .collect();
    for variant in ["clip_10s", "clip_25pct", "watermark"] {
        files.push(
            json!({"path":format!("media/video/{variant}.mp4"),"kind":"video",
            "variant":variant,"family":"video-0","sha256":""}),
        );
    }
    let truth: Truth = serde_json::from_value(json!({
        "version":1,"seed":42,"root":"/corpus","scale":"test","sources":[],
        "pairs":[],"hard_negatives":[],"files":files
    }))?;
    let sample = eidolon_eval::serve::url_sample(&truth);
    assert_eq!(sample.len(), 8);
    for variant in ["clip_10s", "clip_25pct", "watermark"] {
        assert!(
            sample.contains(&std::path::PathBuf::from(format!(
                "media/video/{variant}.mp4"
            ))),
            "fixed-size URL sample omitted {variant}"
        );
    }
    Ok(())
}
