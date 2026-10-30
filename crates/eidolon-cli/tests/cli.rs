//! End-to-end tests of the executable's input and report contract.

use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

fn scan(root: &Path, extra: &[&str], stdin: Option<&[u8]>) -> anyhow::Result<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eidolon"));
    command
        .args(["--json", "--no-cache", "--model-dir"])
        .arg(root.join("missing-models"))
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;
    if let Some(bytes) = stdin {
        child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing child stdin"))?
            .write_all(bytes)?;
    }
    Ok(child.wait_with_output()?)
}

fn successful_report(output: &std::process::Output) -> anyhow::Result<serde_json::Value> {
    anyhow::ensure!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn file_paths(output: &std::process::Output) -> anyhow::Result<Vec<String>> {
    let report = successful_report(output)?;
    Ok(report["files"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing files array"))?
        .iter()
        .filter_map(|file| file["path"].as_str().map(str::to_owned))
        .collect())
}

#[test]
fn folder_and_both_stdin_delimiters_report_the_same_files() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new().prefix("eidolon-cli-").tempdir()?;
    std::fs::create_dir(dir.path().join("nested"))?;
    let first = dir.path().join("first.txt");
    let second = dir.path().join("nested/second.txt");
    std::fs::write(&first, "The first document about a quiet forest.")?;
    std::fs::write(&second, "A separate recipe for freshly baked bread.")?;
    std::fs::write(dir.path().join(".hidden"), "Not selected by default.")?;
    let root = dir
        .path()
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("UTF-8 temporary path"))?;
    let walked = scan(dir.path(), &[root, "--no-embeddings"], None)?;
    let lines = format!("{}\n{}\n", first.display(), second.display());
    let from_lines = scan(
        dir.path(),
        &["--stdin", "--no-embeddings"],
        Some(lines.as_bytes()),
    )?;
    let nuls = format!("{}\0{}\0", first.display(), second.display());
    let from_nuls = scan(
        dir.path(),
        &["--stdin", "-0", "--no-embeddings"],
        Some(nuls.as_bytes()),
    )?;
    assert_eq!(file_paths(&walked)?.len(), 2);
    assert_eq!(file_paths(&walked)?, file_paths(&from_lines)?);
    assert_eq!(file_paths(&walked)?, file_paths(&from_nuls)?);
    Ok(())
}

#[test]
fn duplicate_scan_succeeds_without_modifying_inputs_or_drawing_progress() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new().prefix("eidolon-cli-").tempdir()?;
    let a = dir.path().join("a.txt");
    let b = dir.path().join("renamed.data");
    let contents = b"A short document that has a byte-identical second copy.";
    std::fs::write(&a, contents)?;
    std::fs::copy(&a, &b)?;
    let root = dir
        .path()
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("UTF-8 temporary path"))?;
    let output = scan(dir.path(), &[root], None)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(report["version"], 1);
    assert_eq!(report["matches"].as_array().map(Vec::len), Some(1));
    assert_eq!(report["matches"][0]["relation"], "identical");
    assert_eq!(report["groups"][0]["files"], serde_json::json!([0, 1]));
    let stderr = String::from_utf8(output.stderr)?;
    assert_eq!(stderr.matches("eidolon fetch-models").count(), 1);
    assert!(!stderr.contains('\u{1b}'));
    assert!(!stderr.contains("Comparing"));
    assert_eq!(std::fs::read(&a)?, contents);
    assert_eq!(std::fs::read(&b)?, contents);
    Ok(())
}

#[test]
fn reports_effective_performance_settings_in_json_and_one_stderr_line() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-cli-settings-")
        .tempdir_in("/tmp")?;
    let path = dir.path().join("document.txt");
    std::fs::write(
        &path,
        "A local document needs neither video decoding nor inference.",
    )?;
    let input = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("UTF-8 fixture path"))?;
    let output = scan(
        dir.path(),
        &[
            "--fast",
            "--video-fps",
            "1.5",
            "--embed-device",
            "cpu",
            "--thumbnails",
            "off",
            "--remote-overlay",
            "on",
            "--image-source-bank",
            "off",
            "--max-fetch-fraction",
            "0.4",
            "--fetch-fresh",
            "--timings",
            "--jobs",
            "2",
            "--no-embeddings",
            "--no-keypoints",
            input,
        ],
        None,
    )?;
    anyhow::ensure!(
        output.status.success(),
        "CLI: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(report["version"], 1);
    assert_eq!(
        report["settings"],
        serde_json::json!({
            "fast": true,
            "video_fps": 1.5,
            "embed_device_requested": "cpu",
            "embed_device": null,
            "thumbnails_requested": "off",
            "thumbnails": false,
            "remote_overlay": true,
            "image_source_bank": false,
            "max_fetch_fraction": 0.4,
            "fetch_fresh": true,
            "timings": true,
            "min_score": 0.75,
            "jobs": 2,
            "no_embeddings": true,
            "no_keypoints": true,
            "remote_video_mode": "keyframes"
        })
    );
    let stderr = std::str::from_utf8(&output.stderr)?;
    let lines: Vec<_> = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("Effective settings: "))
        .collect();
    assert_eq!(lines.len(), 1);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(lines[0])?,
        report["settings"]
    );
    Ok(())
}

#[test]
fn offline_model_fetch_never_creates_model_or_backend_caches() -> anyhow::Result<()> {
    let directory = tempfile::Builder::new()
        .prefix("eidolon-cli-offline-models-")
        .tempdir_in("/tmp")?;
    let model_dir = directory.path().join("absent-models");
    let output = Command::new(env!("CARGO_BIN_EXE_eidolon"))
        .args(["--offline", "fetch-models", "--model-dir"])
        .arg(&model_dir)
        .output()?;
    assert!(!output.status.success());
    assert!(!model_dir.exists());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("fetch-models cannot be used with --offline"));
    assert!(!stderr.contains("Prepared automatic embedding backend:"));
    Ok(())
}

#[test]
fn undecodable_images_are_skipped_but_keep_exact_matches() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-cli-skipped-")
        .tempdir_in("/tmp")?;
    let a = dir.path().join("a.jpg");
    image::RgbImage::from_pixel(32, 32, image::Rgb([70, 120, 190])).save(&a)?;
    std::fs::copy(&a, dir.path().join("b.jpg"))?;
    let truncated = &std::fs::read(&a)?[..20];
    std::fs::write(dir.path().join("truncated.jpg"), truncated)?;
    std::fs::write(dir.path().join("icon.ico"), [0, 0, 1, 0, 1, 0])?;
    std::fs::write(dir.path().join("fake.psd"), b"8BPS\0\x01\0\0\0\0\0\0")?;
    let root = dir.path().to_str().expect("UTF-8 fixture path");
    let output = scan(dir.path(), &[root, "--no-embeddings"], None)?;
    let report = successful_report(&output)?;
    assert_eq!(report["skipped"].as_array().map(Vec::len), Some(3));
    assert_eq!(report["matches"].as_array().map(Vec::len), Some(1));
    assert_eq!(report["matches"][0]["relation"], "identical");
    let stderr = std::str::from_utf8(&output.stderr)?;
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.starts_with("eidolon: skipped "))
            .count(),
        3
    );
    for skipped in report["skipped"].as_array().expect("skipped array") {
        assert_ne!(skipped["reason"].as_str().expect("reason"), "");
        assert!(stderr.contains(&format!(
            "eidolon: skipped {}: {}",
            skipped["path"].as_str().expect("path"),
            skipped["reason"].as_str().expect("reason")
        )));
    }
    std::fs::copy(
        dir.path().join("truncated.jpg"),
        dir.path().join("truncated-copy.jpg"),
    )?;
    let output = scan(dir.path(), &[root, "--no-embeddings"], None)?;
    let report = successful_report(&output)?;
    assert_eq!(report["skipped"].as_array().map(Vec::len), Some(4));
    let matches = report["matches"].as_array().expect("matches array");
    assert_eq!(matches.len(), 2);
    assert!(
        matches
            .iter()
            .all(|matched| matched["relation"] == "identical")
    );
    let files = report["files"].as_array().expect("files array");
    assert!(matches.iter().any(|matched| {
        ["a", "b"].iter().all(|side| {
            let id = matched[side].as_u64().expect("file id") as usize;
            files[id]["path"]
                .as_str()
                .expect("path")
                .contains("truncated")
        })
    }));
    std::fs::write(
        dir.path().join("invalid.mp4"),
        b"\0\0\0\x18ftypisom\0\0\0\0isom",
    )?;
    let output = scan(
        dir.path(),
        &[root, "--no-embeddings", "--thumbnails", "on"],
        None,
    )?;
    let report = successful_report(&output)?;
    assert_eq!(report["skipped"].as_array().map(Vec::len), Some(5));
    assert_eq!(report["matches"].as_array().map(Vec::len), Some(2));
    let stderr = std::str::from_utf8(&output.stderr)?;
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.starts_with("eidolon: skipped "))
            .count(),
        5
    );
    assert!(
        report["skipped"]
            .as_array()
            .expect("skipped array")
            .iter()
            .any(|file| file["path"]
                .as_str()
                .expect("path")
                .ends_with("invalid.mp4"))
    );
    let human = Command::new(env!("CARGO_BIN_EXE_eidolon"))
        .args(["--no-cache", "--no-embeddings", "--model-dir"])
        .arg(dir.path().join("missing-models"))
        .arg(dir.path())
        .output()?;
    assert!(human.status.success());
    assert!(
        std::str::from_utf8(&human.stdout)?.contains("5 skipped files (near-duplicate analysis)")
    );
    Ok(())
}

#[test]
fn pdfs_are_exact_only_and_never_reported_as_skipped_media() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-cli-documents-")
        .tempdir_in("/tmp")?;
    let pdf = b"%PDF-1.4\nA text-only PDF header\n%%EOF\n";
    std::fs::write(dir.path().join("document.jpg"), pdf)?;
    std::fs::write(dir.path().join("identical.pdf"), pdf)?;
    std::fs::write(
        dir.path().join("rewrapped.pdf"),
        b"%PDF-1.4\nA  text-only PDF header\n%%EOF\n",
    )?;
    std::fs::write(
        dir.path().join("binary.pdf"),
        b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n%%EOF\n",
    )?;
    let root = dir.path().to_str().expect("UTF-8 fixture path");
    let output = scan(dir.path(), &[root, "--no-embeddings"], None)?;
    let report = successful_report(&output)?;
    assert_eq!(report["skipped"], serde_json::json!([]));
    let files = report["files"].as_array().expect("files array");
    assert_eq!(files.len(), 4);
    assert!(files.iter().all(|file| file["kind"] == "other"));
    let matches = report["matches"].as_array().expect("matches array");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0]["relation"], "identical");
    assert!(!std::str::from_utf8(&output.stderr)?.contains("eidolon: skipped "));
    Ok(())
}

#[test]
fn template_data_variants_never_join_duplicate_groups_or_keep_hints() -> anyhow::Result<()> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-cli-text-templates-")
        .tempdir_in("/tmp")?;
    let receipt = |amount, date, transaction| {
        format!(
            "<html><body>\n\
             <h1>Example payment receipt</h1>\n\
             <p>Thank you for your payment. This document confirms that the payment \
             has been completed successfully and the merchant has received a notification. \
             Keep this receipt for your records and review the details below carefully. \
             Contact the merchant directly if you have questions about your purchase. \
             You can find the complete payment history in your account activity. \
             This message was generated automatically and replies are not monitored.</p>\n\
             <p>Amount: {amount} EUR</p><p>Date: {date}</p>\
             <p>Transaction: {transaction}</p>\n\
             </body></html>"
        )
    };
    let first = receipt("12.50", "14 July 2026", "AB1234CD5678");
    std::fs::write(dir.path().join("first.html"), &first)?;
    std::fs::write(
        dir.path().join("second.html"),
        receipt("18.75", "21 August 2026", "XY9876ZW5432"),
    )?;
    let root = dir.path().to_str().expect("UTF-8 fixture path");
    let output = scan(dir.path(), &[root, "--offline", "--no-embeddings"], None)?;
    let report = successful_report(&output)?;
    assert_eq!(report["matches"].as_array().map(Vec::len), Some(1));
    assert_eq!(report["matches"][0]["relation"], "variant");
    assert_eq!(report["variants"].as_array().map(Vec::len), Some(1));
    assert_eq!(report["groups"], serde_json::json!([]));
    assert!(!std::str::from_utf8(&output.stdout)?.contains("keep_hint"));
    let human = Command::new(env!("CARGO_BIN_EXE_eidolon"))
        .args(["--offline", "--no-cache", "--no-embeddings"])
        .arg(dir.path())
        .output()?;
    assert!(human.status.success());
    let text = std::str::from_utf8(&human.stdout)?;
    assert!(text.contains("Variants (related, not duplicates)"));
    assert!(text.contains("same template, different data"));
    assert!(!text.contains("keep_hint"));

    std::fs::write(dir.path().join("resaved.html"), first.replace('\n', "\r\n"))?;
    let output = scan(dir.path(), &[root, "--offline", "--no-embeddings"], None)?;
    let report = successful_report(&output)?;
    let groups = report["groups"].as_array().expect("groups array");
    assert_eq!(groups.len(), 1);
    let members = groups[0]["files"].as_array().expect("group files");
    assert_eq!(members.len(), 2);
    for id in members {
        let path = report["files"][id.as_u64().expect("file id") as usize]["path"]
            .as_str()
            .expect("file path");
        assert!(path.ends_with("/first.html") || path.ends_with("/resaved.html"));
    }
    assert_eq!(report["variants"].as_array().map(Vec::len), Some(2));
    Ok(())
}
