use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;

use anyhow::{Context as _, ensure};
use serde_json::Value;
use url::Url;
use wait_timeout::ChildExt as _;

use crate::{FetchOptions, VideoMode, digest, http, save};

pub(crate) fn is_platform(url: &Url) -> bool {
    let host = url.host_str().unwrap_or_default();
    [
        "youtube.com",
        "youtu.be",
        "youtube-nocookie.com",
        "vimeo.com",
        "dailymotion.com",
        "twitch.tv",
        "tiktok.com",
        "bilibili.com",
    ]
    .iter()
    .any(|domain| {
        host == *domain
            || host
                .strip_suffix(domain)
                .is_some_and(|prefix| prefix.ends_with('.'))
    })
}

fn format_selector(mode: VideoMode) -> &'static str {
    match mode {
        VideoMode::Keyframes => {
            "bestvideo[vcodec~='^(avc1|h264|vp0?9)'][height<=360][protocol=https]/best[vcodec~='^(avc1|h264|vp0?9)'][height<=360][protocol=https]/bestvideo[vcodec~='^(avc1|h264|vp0?9)'][height<=360][protocol=http]/best[vcodec~='^(avc1|h264|vp0?9)'][height<=360][protocol=http]/bestvideo[vcodec~='^(avc1|h264|vp0?9)'][protocol=https]/best[vcodec~='^(avc1|h264|vp0?9)'][protocol=https]/bestvideo[vcodec~='^(avc1|h264|vp0?9)'][protocol=http]/best[vcodec~='^(avc1|h264|vp0?9)'][protocol=http]"
        }
        VideoMode::FullCadence => {
            "bestvideo[ext=mp4][protocol=https]/best[ext=mp4][protocol=https]/bestvideo[protocol=https]/best[protocol=https]/bestvideo[protocol=http]/best[protocol=http]"
        }
    }
}

pub(crate) fn discover(
    page: &Url,
    options: &FetchOptions,
) -> anyhow::Result<Vec<(Url, &'static str)>> {
    let mut child = Command::new("yt-dlp")
        .args([
            "--ignore-config",
            "--no-plugin-dirs",
            "--no-remote-components",
            "--no-update",
            "--no-sponsorblock",
            "--no-cache-dir",
            "--no-playlist",
            "--no-mark-watched",
            "--skip-download",
            "--dump-single-json",
            "--no-warnings",
            "--no-progress",
            "--retries",
            "0",
            "--extractor-retries",
            "0",
            "--socket-timeout",
        ])
        .arg(options.timeout.as_secs_f64().to_string())
        .args([
            "--user-agent",
            &format!("eidolon/{}", env!("CARGO_PKG_VERSION")),
            "--format",
            format_selector(options.video_mode),
            "--",
        ])
        .arg(page.as_str())
        .env("DO_NOT_TRACK", "1")
        .env("DNT", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context(
            "launching yt-dlp (install with brew install yt-dlp; no cookies or login are needed)",
        )?;
    let stdout = child.stdout.take().context("yt-dlp stdout unavailable")?;
    let stderr = child.stderr.take().context("yt-dlp stderr unavailable")?;
    let limit = options.max_page_bytes;
    let output = thread::spawn(move || bounded_output(stdout, limit));
    let errors = thread::spawn(move || bounded_output(stderr, 64 * 1024));
    let status = child.wait_timeout(options.resolve_timeout)?;
    if status.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let stdout = output
        .join()
        .map_err(|_| anyhow::anyhow!("yt-dlp stdout reader panicked"))??;
    let stderr = errors
        .join()
        .map_err(|_| anyhow::anyhow!("yt-dlp stderr reader panicked"))??;
    // Keep the anonymous discovery source before interpreting it; no browser/cookie files are opened.
    let key = digest(page.as_str());
    save(
        &options.cache_dir.join(format!("{key}.yt-dlp.json")),
        &stdout,
    )?;
    save(
        &options.cache_dir.join(format!("{key}.yt-dlp.stderr")),
        &stderr,
    )?;
    let status = status.context("yt-dlp resolution timed out")?;
    ensure!(
        status.success(),
        "yt-dlp failed: {}",
        String::from_utf8_lossy(&stderr).trim()
    );
    ensure!(
        stdout.len() as u64 <= limit,
        "yt-dlp metadata exceeds page size cap"
    );
    let metadata: Value = serde_json::from_slice(&stdout).context("parsing yt-dlp metadata")?;
    ensure!(
        !metadata
            .get("is_live")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "live videos are not bounded file inputs"
    );
    let mut media = Vec::new();
    if let Some(url) = metadata.get("url").and_then(Value::as_str) {
        media.push((http::validate(url)?, "video"));
    }
    if let Some(url) = metadata.get("thumbnail").and_then(Value::as_str) {
        media.push((http::validate(url)?, "video thumbnail"));
    }
    ensure!(
        !media.is_empty(),
        "yt-dlp returned no HTTP media or thumbnail URLs"
    );
    Ok(media)
}

fn bounded_output(reader: impl Read, limit: u64) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VideoMode;

    #[test]
    fn sparse_platform_selection_prefers_low_resolution_decodable_video_and_never_av1() {
        let directory = tempfile::tempdir().expect("format metadata");
        let path = directory.path().join("video.json");
        for (formats, expected) in [
            (
                vec![
                    ("av1", "av01", 1080),
                    ("h264-hd", "avc1.4d401f", 720),
                    ("h264-low", "avc1.4d401e", 360),
                    ("vp9-low", "vp9", 240),
                ],
                Some("h264-low"),
            ),
            (
                vec![("av1", "av01", 360), ("vp9-low", "vp9", 240)],
                Some("vp9-low"),
            ),
            (vec![("av1", "av01", 144)], None),
        ] {
            let formats: Vec<_> = formats
                .into_iter()
                .map(|(id, codec, height)| {
                    serde_json::json!({
                        "format_id": id, "vcodec": codec, "acodec": "none",
                        "height": height, "width": height * 16 / 9,
                        "url": format!("https://example.invalid/{id}"),
                        "ext": "mp4", "protocol": "https",
                    })
                })
                .collect();
            std::fs::write(
                &path,
                serde_json::to_vec(&serde_json::json!({
                    "id": "fixture", "title": "Offline selector fixture", "formats": formats,
                    "webpage_url": "https://example.invalid/fixture", "extractor": "generic",
                }))
                .expect("metadata"),
            )
            .expect("fixture");
            let output = Command::new("yt-dlp")
                .args([
                    "--ignore-config",
                    "--no-plugin-dirs",
                    "--no-remote-components",
                    "--no-update",
                    "--no-cache-dir",
                    "--skip-download",
                    "--dump-single-json",
                    "--format",
                    format_selector(VideoMode::Keyframes),
                    "--load-info-json",
                ])
                .arg(&path)
                .env("DO_NOT_TRACK", "1")
                .env("DNT", "1")
                .output()
                .expect("installed yt-dlp");
            if let Some(expected) = expected {
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let metadata: Value =
                    serde_json::from_slice(&output.stdout).expect("selected metadata");
                assert_eq!(metadata["format_id"], expected);
            } else {
                assert!(
                    !output.status.success(),
                    "AV1-only sparse input must be rejected"
                );
            }
        }
    }
}
