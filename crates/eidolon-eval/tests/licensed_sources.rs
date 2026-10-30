//! Offline source manifests preserve native containers, attribution and frame containment.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use eidolon_eval::generate::licensed;
use serde_json::json;
use sha2::{Digest as _, Sha256};

fn sha(path: &Path) -> anyhow::Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn fixture() -> anyhow::Result<(tempfile::TempDir, PathBuf, PathBuf)> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-licensed-")
        .tempdir_in("/tmp")?;
    let image = temp.path().join("photo.png");
    image::RgbImage::from_fn(160, 120, |x, y| {
        let stripe = if (x / 13 + y / 11).is_multiple_of(3) {
            160
        } else {
            24
        };
        image::Rgb([((x * 3 + y) % 256) as u8, stripe, ((x + y * 5) % 256) as u8])
    })
    .save(&image)?;
    let video = temp.path().join("native.webm");
    let encoded = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=8",
            "-t",
            "16",
            "-c:v",
            "libvpx-vp9",
            "-threads",
            "1",
            "-g",
            "32",
        ])
        .arg(&video)
        .output()?;
    assert!(
        encoded.status.success(),
        "{}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    let text = temp.path().join("chapter.txt");
    let mut words = String::new();
    for index in 0..1200 {
        write!(words, "word{index} ")?;
    }
    fs::write(&text, words)?;
    let sources = [(&image, "image"), (&video, "video"), (&text, "text")]
        .into_iter()
        .map(|(path, kind)| {
            Ok(json!({"kind":kind,"path":path,"sha256":sha(path)?,
            "title":format!("Fixture {kind}"),"author":"generated test fixture",
            "licence":"CC0","licence_url":"https://creativecommons.org/publicdomain/zero/1.0/",
            "source_url":format!("https://example.invalid/fixture/{kind}"),"raw_capture":path}))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let manifest = temp.path().join("sources.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&json!({"version":1,"sources":sources}))?,
    )?;
    Ok((temp, manifest, video))
}

#[test]
fn generates_from_archived_sources_without_losing_native_video_or_clip_timing() -> anyhow::Result<()>
{
    let (temp, manifest, video) = fixture()?;
    let out = temp.path().join("corpus");
    let (truth, validation) = licensed::generate(&manifest, &out, 123)?;
    assert_eq!(validation.files, 38);
    let native = truth
        .files
        .iter()
        .find(|file| file.kind == "video" && file.variant == "original")
        .expect("native video");
    assert_eq!(
        native
            .path
            .extension()
            .and_then(|extension| extension.to_str()),
        Some("webm")
    );
    assert_eq!(sha(&truth.root.join(&native.path))?, sha(&video)?);
    assert_ne!(
        native.design, truth.files[0].design,
        "native videos have no implied source-image ancestry"
    );
    for variant in ["clip_10s", "clip_25pct"] {
        let clip = truth
            .files
            .iter()
            .find(|file| file.kind == "video" && file.variant == variant)
            .expect("clip");
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
            ])
            .arg(truth.root.join(&clip.path))
            .output()?;
        assert!(probe.status.success());
        let duration: f64 = std::str::from_utf8(&probe.stdout)?.trim().parse()?;
        let expected = if variant == "clip_10s" { 10.0 } else { 4.0 };
        assert!((duration - expected).abs() < 0.3, "{variant}: {duration}");
    }
    assert!(out.join("source-manifest.json").is_file());
    assert!(out.join("truth.json").is_file());
    assert!(out.join("registration.json").is_file());
    assert!(
        licensed::generate(&manifest, &out, 123).is_err(),
        "occupied destinations must not be overwritten"
    );
    Ok(())
}

#[test]
fn rejects_bad_hash_before_writing_and_resolves_relative_manifest_paths() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-source-validation-")
        .tempdir_in("/tmp")?;
    let text = temp.path().join("chapter.txt");
    fs::write(
        &text,
        "An independently generated local chapter with enough words for excerpt variants.\n",
    )?;
    let manifest = temp.path().join("sources.json");
    let out = temp.path().join("corpus");
    let mut value = json!({"version":1,"sources":[{
        "kind":"text","path":"chapter.txt","raw_capture":"chapter.txt","sha256":"invalid",
        "title":"Local fixture","author":"Fixture author","licence":"CC0",
        "licence_url":"https://creativecommons.org/publicdomain/zero/1.0/",
        "source_url":"https://example.invalid/fixture/chapter"
    }]});
    fs::write(&manifest, serde_json::to_vec(&value)?)?;
    let error = licensed::generate(&manifest, &out, 1).expect_err("bad source hash");
    assert!(error.to_string().contains("source hash mismatch"));
    assert!(
        !out.exists(),
        "bad source inputs must not touch the destination"
    );
    value["sources"][0]["sha256"] = sha(&text)?.into();
    fs::write(&manifest, serde_json::to_vec(&value)?)?;
    let (truth, validation) = licensed::generate(&manifest, &out, 1)?;
    assert_eq!(validation.files, 8);
    assert_eq!(
        fs::read(truth.root.join(&truth.files[0].path))?,
        fs::read(text)?
    );
    Ok(())
}
