//! End-to-end image-to-video frame matching through the executable.

use std::path::{Path, PathBuf};
use std::process::Command;

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

fn fixture(directory: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
    let video = directory.join("source.mp4");
    let poster = directory.join("poster.png");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=6",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ],
        &video,
    )?;
    let video_path = video
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("UTF-8 video path"))?;
    ffmpeg(
        &[
            "-protocol_whitelist",
            "file,pipe",
            "-ss",
            "1",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-vf",
            "scale=128:96",
        ],
        &poster,
    )?;
    Ok((video, poster))
}

fn scan(
    poster: &Path,
    video: &Path,
    thumbnails: Option<&str>,
) -> anyhow::Result<serde_json::Value> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eidolon"));
    command.args([
        "--json",
        "--offline",
        "--no-embeddings",
        "--no-keypoints",
        "--no-cache",
        "--jobs",
        "2",
    ]);
    if let Some(mode) = thumbnails {
        command.args(["--thumbnails", mode]);
    }
    let output = command.arg(poster).arg(video).output()?;
    anyhow::ensure!(
        output.status.success(),
        "CLI: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
fn resized_video_thumbnail_is_dispatched_as_a_directional_frame_match() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-cli-frame-")
        .tempdir_in("/tmp")?;
    let (video, poster) = fixture(dir.path())?;
    let report = scan(&poster, &video, Some("on"))?;
    let matched = report["matches"]
        .as_array()
        .and_then(|matches| {
            matches
                .iter()
                .find(|matched| matched["relation"] == "frame_of")
        })
        .ok_or_else(|| anyhow::anyhow!("missing frame_of match: {report}"))?;
    let image = matched["a"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("image endpoint missing"))? as usize;
    let video = matched["b"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("video endpoint missing"))? as usize;
    assert_eq!(report["files"][image]["kind"], "image");
    assert_eq!(report["files"][video]["kind"], "video");
    assert_eq!(report["groups"][0]["keep_hint"], video);
    assert_eq!(
        report["groups"][0]["files"].as_array().map(Vec::len),
        Some(2)
    );
    Ok(())
}

#[test]
fn disabled_and_automatic_local_thumbnail_policies_produce_no_frame_matches() -> anyhow::Result<()>
{
    let dir = tempfile::Builder::new()
        .prefix("eidolon-cli-frame-off-")
        .tempdir_in("/tmp")?;
    let (video, poster) = fixture(dir.path())?;
    for policy in [Some("off"), None] {
        let report = scan(&poster, &video, policy)?;
        assert!(report["matches"].as_array().is_some_and(|matches| {
            matches
                .iter()
                .all(|matched| matched["relation"] != "frame_of")
        }));
        assert_eq!(report["settings"]["thumbnails"], false);
        assert_eq!(
            report["settings"]["thumbnails_requested"],
            policy.unwrap_or("auto")
        );
        assert_eq!(report["groups"].as_array().map(Vec::len), Some(0));
    }
    Ok(())
}
