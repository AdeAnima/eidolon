//! Real range-served sparse video refinement and final transfer accounting.

use std::io::{BufRead as _, BufReader, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

struct VideoServer {
    address: SocketAddr,
    bytes: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<anyhow::Result<()>>>,
}

impl VideoServer {
    fn new(video: Vec<u8>) -> anyhow::Result<Self> {
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

    fn url(&self) -> String {
        format!("http://{}/video.mp4", self.address)
    }

    fn bytes(&self) -> u64 {
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

fn ffmpeg(arguments: &[&str], output: &Path) -> anyhow::Result<()> {
    let result = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error"])
        .args(arguments)
        .arg(output)
        .output()?;
    anyhow::ensure!(
        result.status.success(),
        "ffmpeg: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

fn run(
    home: &Path,
    input: &str,
    poster: Option<&Path>,
    max_fetch_fraction: &str,
) -> anyhow::Result<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eidolon"));
    command
        .args([
            "--json",
            "--no-cache",
            "--no-embeddings",
            "--no-keypoints",
            "--thumbnails",
            "on",
            "--max-fetch-fraction",
            max_fetch_fraction,
            "--jobs",
            "2",
            input,
        ])
        .env("HOME", home);
    if let Some(poster) = poster {
        command.arg(poster);
    }
    let output = command.output()?;
    anyhow::ensure!(
        output.status.success(),
        "CLI: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn summary_bytes(output: &std::process::Output) -> anyhow::Result<u64> {
    let stderr = std::str::from_utf8(&output.stderr)?;
    let value = stderr
        .lines()
        .find_map(|line| line.strip_prefix("Remote transfer: "))
        .and_then(|summary| summary.split_whitespace().next())
        .ok_or_else(|| anyhow::anyhow!("missing transfer summary: {stderr}"))?;
    Ok(value.parse()?)
}

fn fixture(directory: &Path, pulse: &str) -> anyhow::Result<(PathBuf, PathBuf)> {
    let source = directory.join("source.mp4");
    let poster = directory.join("poster.jpg");
    // Stable layout nominates the video without embeddings; the brief brightness
    // pulse changes every pixel and lies inside the first hash-only native window.
    let filter = format!(
        "testsrc2=size=960x540:rate=8,lutrgb=r='51+0.6*val':g='51+0.6*val':b='51+0.6*val',eq=brightness='0.2*max(0,1-abs(t-{pulse})/0.25)':eval=frame"
    );
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            &filter,
            "-t",
            "24",
            "-c:v",
            "libx264",
            "-g",
            "64",
            "-keyint_min",
            "64",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
        &source,
    )?;
    ffmpeg(
        &[
            "-protocol_whitelist",
            "file,pipe",
            "-ss",
            pulse,
            "-i",
            source
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("UTF-8 fixture path"))?,
            "-frames:v",
            "1",
            "-vf",
            "scale=320:180",
            "-q:v",
            "2",
        ],
        &poster,
    )?;
    Ok((source, poster))
}

fn retain_output(
    directory: &Path,
    name: &str,
    output: &std::process::Output,
) -> anyhow::Result<()> {
    std::fs::write(directory.join(format!("{name}.json")), &output.stdout)?;
    std::fs::write(directory.join(format!("{name}.stderr")), &output.stderr)?;
    Ok(())
}

fn check_refinement(directory: &Path) -> anyhow::Result<()> {
    let (source, poster) = fixture(directory, "0.25")?;
    let server = VideoServer::new(std::fs::read(&source)?)?;
    let url = server.url();
    let baseline = run(&directory.join("baseline-home"), &url, None, "1")?;
    retain_output(directory, "baseline", &baseline)?;
    let baseline_bytes = server.bytes();
    anyhow::ensure!(baseline_bytes > 0, "sparse discovery must fetch bytes");
    anyhow::ensure!(
        summary_bytes(&baseline)? == baseline_bytes,
        "initial transfer count"
    );
    let baseline_report: serde_json::Value = serde_json::from_slice(&baseline.stdout)?;
    let initial_artifact_bytes = baseline_report["files"][0]["url"]["bytes_fetched"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("missing initial artifact byte count"))?;
    let sparse_path = baseline_report["files"][0]["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing sparse artifact path"))?;
    let without = run(
        &directory.join("no-provider-home"),
        sparse_path,
        Some(&poster),
        "1",
    )?;
    retain_output(directory, "without-provider", &without)?;
    let unrefined: serde_json::Value = serde_json::from_slice(&without.stdout)?;
    anyhow::ensure!(
        unrefined["matches"].as_array().is_some_and(Vec::is_empty),
        "fixture must require native pixels, not already verify a sparse sample: {unrefined}"
    );
    anyhow::ensure!(
        server.bytes() == baseline_bytes,
        "local-only scan must not fetch"
    );
    let output = run(&directory.join("matching-home"), &url, Some(&poster), "1")?;
    retain_output(directory, "with-provider", &output)?;
    let refined_bytes = server.bytes() - baseline_bytes;
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let matched = report["matches"].as_array()
        .and_then(|matches| matches.iter().find(|matched| matched["relation"] == "frame_of"))
        .ok_or_else(|| anyhow::anyhow!(
            "missing refined FrameOf; initial={baseline_bytes}, with-refinement={refined_bytes}: {report}"
        ))?;
    anyhow::ensure!(
        matched["evidence"].as_array().is_some_and(|evidence| {
            evidence
                .iter()
                .any(|item| item["signal"] == "frame_refinement")
        }),
        "native provider must improve the sparse seed: {matched}"
    );
    anyhow::ensure!(
        refined_bytes > baseline_bytes,
        "refinement must fetch additional bytes"
    );
    anyhow::ensure!(
        summary_bytes(&output)? == refined_bytes,
        "final summary must include all native-window transfer bytes"
    );
    let remote = report["files"]
        .as_array()
        .and_then(|files| files.iter().find(|file| file.get("url").is_some()))
        .ok_or_else(|| anyhow::anyhow!("missing remote video provenance"))?;
    anyhow::ensure!(
        remote["url"]["bytes_fetched"] == initial_artifact_bytes + refined_bytes - baseline_bytes,
        "per-artifact provenance must include refinement bytes"
    );
    let extra = refined_bytes - baseline_bytes;
    anyhow::ensure!(
        remote["url"]["mode"] == "sparse"
            && remote["url"]["initial_bytes_fetched"] == initial_artifact_bytes
            && remote["url"]["refinement_bytes_fetched"] == extra
            && remote["url"]["mode_reason"]
                .as_str()
                .is_some_and(|reason| reason.contains("native_prefix_refinement")),
        "transport provenance must distinguish sparse acquisition and native prefix refinement: {remote}"
    );
    let stderr = std::str::from_utf8(&output.stderr)?;
    anyhow::ensure!(
        stderr.contains("Remote video: ")
            && stderr.contains("mode=sparse")
            && stderr.contains(&format!(
                "initial={initial_artifact_bytes} bytes; refinement={extra} bytes; total={} bytes",
                initial_artifact_bytes + extra,
            )),
        "missing per-video transport note: {stderr}"
    );
    Ok(())
}

#[test]
fn sparse_remote_frame_refinement_is_wired_and_included_in_final_transfer_total()
-> anyhow::Result<()> {
    let directory = tempfile::Builder::new()
        .prefix("eidolon-cli-refinement-")
        .tempdir_in("/tmp")?;
    if let Err(error) = check_refinement(directory.path()) {
        let retained = directory.keep();
        return Err(error.context(format!(
            "refinement fixture retained at {}",
            retained.display()
        )));
    }
    Ok(())
}

#[test]
fn refinement_timing_diagnostic_requires_timings_flag() -> anyhow::Result<()> {
    let directory = tempfile::Builder::new()
        .prefix("eidolon-cli-timings-")
        .tempdir_in("/tmp")?;
    let (source, poster) = fixture(directory.path(), "0.25")?;
    let server = VideoServer::new(std::fs::read(source)?)?;
    for timings in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_eidolon"));
        command
            .args([
                "--json",
                "--no-cache",
                "--no-embeddings",
                "--no-keypoints",
                "--thumbnails",
                "on",
                "--max-fetch-fraction",
                "1",
                "--jobs",
                "2",
            ])
            .arg(server.url())
            .arg(&poster)
            .env(
                "HOME",
                directory.path().join(if timings { "on" } else { "off" }),
            );
        if timings {
            command.arg("--timings");
        }
        let output = command.output()?;
        anyhow::ensure!(
            output.status.success(),
            "CLI: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let remote = report["files"]
            .as_array()
            .and_then(|files| {
                files
                    .iter()
                    .find(|file| file["kind"] == "video" && file.get("url").is_some())
            })
            .ok_or_else(|| anyhow::anyhow!("missing remote video: {report}"))?;
        let video_id = remote["id"].as_u64().expect("remote file id");
        let refinement_bytes = remote["url"]["refinement_bytes_fetched"]
            .as_u64()
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| {
                anyhow::anyhow!("native refinement must occur with timings={timings}: {report}")
            })?;
        assert!(
            remote["url"]["mode_reason"]
                .as_str()
                .is_some_and(|reason| reason.contains("native_prefix_refinement"))
        );
        let stderr = std::str::from_utf8(&output.stderr)?;
        assert!(stderr.contains(&format!("refinement={refinement_bytes} bytes;")));
        let lines: Vec<_> = stderr
            .lines()
            .filter_map(|line| line.strip_prefix("Video refinement timing: "))
            .collect();
        assert_eq!(lines.len(), usize::from(timings), "{stderr}");
        if let Some(line) = lines.first() {
            let timing: serde_json::Value = serde_json::from_str(line)?;
            assert_eq!(timing["file_id"], video_id);
            for counter in ["provider_calls", "decode_calls", "proof_calls"] {
                assert!(
                    timing[counter].as_u64().is_some_and(|calls| calls > 0),
                    "{timing}"
                );
            }
            for clock in ["provider_seconds", "decode_seconds", "proof_seconds"] {
                assert!(
                    timing[clock].as_f64().is_some_and(|seconds| seconds > 0.0),
                    "{timing}"
                );
            }
            assert!(timing["cache_hits"].is_u64());
        }
    }
    Ok(())
}

#[test]
fn fractional_native_budget_is_forwarded_and_reports_limited_recall() -> anyhow::Result<()> {
    let directory = tempfile::Builder::new()
        .prefix("eidolon-cli-budget-")
        .tempdir_in("/tmp")?;
    // Early proof intentionally avoids exhaustion; this late first-GOP target
    // needs more native bytes than the quarter-source budget permits.
    let (source, poster) = fixture(directory.path(), "7.25")?;
    let size = std::fs::metadata(&source)?.len();
    let server = VideoServer::new(std::fs::read(&source)?)?;
    let output = run(
        &directory.path().join("home"),
        &server.url(),
        Some(&poster),
        "0.25",
    )?;
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(report["settings"]["max_fetch_fraction"], 0.25);
    assert_eq!(report["settings"]["thumbnails"], true);
    let remote = report["files"]
        .as_array()
        .and_then(|files| files.iter().find_map(|file| file.get("url")))
        .ok_or_else(|| anyhow::anyhow!("missing remote provenance"))?;
    assert_eq!(remote["budget_exhausted"], true, "{report}");
    assert_eq!(summary_bytes(&output)?, server.bytes());
    assert!(
        server.bytes() <= size / 4,
        "cold native transfer exceeds the quarter-source cap"
    );
    let stderr = std::str::from_utf8(&output.stderr)?;
    assert_eq!(
        stderr
            .matches("remote video refinement budget exhausted")
            .count(),
        1
    );
    assert!(stderr.contains("recall may be limited"));
    Ok(())
}

fn captioned_crop_fixture(directory: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
    let background = directory.join("background.png");
    image::RgbImage::from_fn(640, 360, |x, y| {
        image::Rgb([
            (30 + (x / 28 * 17 + y / 24 * 29) % 190) as u8,
            (35 + (x / 36 * 31 + y / 32 * 13) % 180) as u8,
            (40 + (x / 24 * 7 + y / 28 * 37) % 170) as u8,
        ])
    })
    .save(&background)?;
    let source = directory.join("caption-source.mp4");
    ffmpeg(
        &[
            "-loop",
            "1",
            "-i",
            background
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("UTF-8 background"))?,
            "-t",
            "8",
            "-r",
            "8",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-g",
            "64",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
        &source,
    )?;
    let query = directory.join("caption-crop.png");
    ffmpeg(
        &[
            "-i",
            source
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("UTF-8 source"))?,
            "-frames:v",
            "1",
            "-vf",
            "crop=400:270:160:45,scale=320:216",
        ],
        &query,
    )?;
    let mut pixels = image::open(&query)?.to_rgb8();
    let top = pixels.height() - 36;
    for y in top..pixels.height() {
        for x in 0..pixels.width() {
            pixels.put_pixel(x, y, image::Rgb([18, 22, 30]));
        }
    }
    for glyph in 0..14 {
        for y in 0..7 {
            for x in 0..5 {
                if x == 0 || y == 0 || y == 3 || (x == 4 && y < 3) {
                    for dy in 0..3 {
                        for dx in 0..3 {
                            pixels.put_pixel(
                                7 + glyph * 21 + x * 3 + dx,
                                top + 7 + y * 3 + dy,
                                image::Rgb([242, 238, 225]),
                            );
                        }
                    }
                }
            }
        }
    }
    pixels.save(&query)?;
    Ok((source, query))
}

#[test]
fn remote_overlay_defaults_on_and_can_disable_caption_crop_retrieval() -> anyhow::Result<()> {
    let directory = tempfile::Builder::new()
        .prefix("eidolon-cli-caption-option-")
        .tempdir_in("/tmp")?;
    let (source, query) = captioned_crop_fixture(directory.path())?;
    let server = VideoServer::new(std::fs::read(source)?)?;
    for mode in [None, Some("off"), Some("on")] {
        let enabled = mode != Some("off");
        let mut command = Command::new(env!("CARGO_BIN_EXE_eidolon"));
        command
            .args([
                "--json",
                "--no-cache",
                "--no-embeddings",
                "--no-keypoints",
                "--thumbnails",
                "on",
                "--max-fetch-fraction",
                "1",
                "--jobs",
                "1",
            ])
            .arg(server.url())
            .arg(&query)
            .env("HOME", directory.path().join(mode.unwrap_or("default")));
        if let Some(mode) = mode {
            command.args(["--remote-overlay", mode]);
        }
        let output = command.output()?;
        anyhow::ensure!(
            output.status.success(),
            "mode={mode:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let matches = report["matches"].as_array().expect("matches array");
        assert_eq!(
            matches
                .iter()
                .filter(|matched| matched["relation"] == "frame_of")
                .count(),
            usize::from(enabled),
            "mode={mode:?}: {report}"
        );
        assert_eq!(report["settings"]["remote_overlay"], enabled);
        let stderr = std::str::from_utf8(&output.stderr)?;
        let settings: serde_json::Value = serde_json::from_str(
            stderr
                .lines()
                .find_map(|line| line.strip_prefix("Effective settings: "))
                .expect("effective settings"),
        )?;
        assert_eq!(settings, report["settings"]);
    }
    Ok(())
}
