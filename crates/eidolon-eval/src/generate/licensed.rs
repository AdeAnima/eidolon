//! Generate the existing transformation matrix from explicitly archived local source files.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, ensure};
use serde::Deserialize;

use super::{
    Random, Validation, VideoTiming, annotate_hard_negatives, command, hash_file, image_family,
    image_path, record, save_original_window, scene_path, text_family, validate, video_derivatives,
    write_new_json,
};
use crate::model::{Source, Truth};

#[derive(Deserialize)]
struct Manifest {
    version: u32,
    sources: Vec<LocalSource>,
}

#[derive(Deserialize)]
struct LocalSource {
    kind: String,
    path: PathBuf,
    sha256: String,
    title: String,
    author: String,
    licence: String,
    licence_url: String,
    source_url: String,
    raw_capture: PathBuf,
    raw_sha256: Option<String>,
}

/// Generate a corpus from a version-one JSON source manifest, without network access.
///
/// Sources must name local files and archived raw captures, include SHA-256 and attribution,
/// and use Public domain, CC0, or CC BY 1.0/2.0/2.5/3.0/4.0. The full input manifest is retained.
/// Native video originals and exact copies retain their bytes and containers; H.264 derivatives
/// use the actual source duration for ten-second/quarter-length clips and frame containment.
/// All outputs are decoded/probed, hashed, and audited before `truth.json` is published.
///
/// # Errors
/// Fails for invalid source metadata/hashes, occupied output directories, unsupported media,
/// videos shorter than twelve seconds, conversion errors or distinctness audit conflicts.
/// A partial output directory is retained on generation failure.
#[allow(clippy::too_many_lines)] // One ordered orchestration path defines reproducible source/variant ordering.
pub fn generate(
    manifest_path: &Path,
    out: &Path,
    seed: u64,
) -> anyhow::Result<(Truth, Validation)> {
    let bytes = fs::read(manifest_path)?;
    let mut manifest: Manifest = serde_json::from_slice(&bytes)?;
    ensure!(manifest.version == 1, "unsupported source manifest version");
    ensure!(!manifest.sources.is_empty(), "source manifest is empty");
    // Resolve relative source names against the manifest, not the caller's working directory.
    let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    for source in &mut manifest.sources {
        if source.path.is_relative() {
            source.path = base.join(&source.path);
        }
        if source.raw_capture.is_relative() {
            source.raw_capture = base.join(&source.raw_capture);
        }
        validate_source(source)?;
    }
    fs::create_dir_all(out)?;
    ensure!(
        fs::read_dir(out)?.next().is_none(),
        "destination must be empty: {}",
        out.display()
    );
    let root = out.canonicalize()?;
    fs::create_dir(root.join("sources"))?;
    for kind in ["image", "text", "video"] {
        fs::create_dir_all(root.join("media").join(kind))?;
    }
    fs::write(root.join("source-manifest.json"), &bytes)?;
    let mut truth = Truth {
        version: 1,
        seed,
        root,
        scale: "archived-sources".into(),
        sources: Vec::new(),
        files: Vec::new(),
        pairs: Vec::new(),
        hard_negatives: Vec::new(),
    };
    let mut random = Random(seed);
    for kind in ["image", "text", "video"] {
        for (index, source) in manifest
            .sources
            .iter()
            .filter(|source| source.kind == kind)
            .enumerate()
        {
            let extension = source
                .path
                .extension()
                .and_then(|extension| extension.to_str())
                .context("source file needs an extension")?;
            let retained = PathBuf::from(format!("sources/{kind}-{index:03}.{extension}"));
            fs::copy(&source.path, truth.root.join(&retained))?;
            truth.sources.push(Source {
                origin: source.source_url.clone(),
                retained: retained.clone(),
                sha256: source.sha256.clone(),
            });
            match kind {
                "image" => image_source(&mut truth, index, source, &retained, &mut random)?,
                "text" => {
                    let text = fs::read_to_string(truth.root.join(&retained))?;
                    text_family(&mut truth, index, &text, &mut random)?;
                    truth.sources.push(Source {
                        origin: source.source_url.clone(),
                        retained: PathBuf::from(format!("media/text/{index:03}_original.txt")),
                        sha256: source.sha256.clone(),
                    });
                }
                "video" => video_source(&mut truth, index, source, &retained, &mut random)?,
                _ => unreachable!("source kinds validated before output creation"),
            }
        }
    }
    let registration = crate::relations::annotate(&mut truth)?;
    annotate_hard_negatives(&mut truth);
    let frames = crate::frame_windows::annotate(&mut truth)?;
    let validation = validate(&truth)?;
    write_new_json(&truth.root.join("registration.json"), &registration)?;
    write_new_json(&truth.root.join("frame-windows.json"), &frames)?;
    write_new_json(
        &truth.root.join("distinctness.json"),
        &validation.distinctness,
    )?;
    write_new_json(&truth.root.join("truth.json"), &truth)?;
    Ok((truth, validation))
}

fn validate_source(source: &LocalSource) -> anyhow::Result<()> {
    ensure!(
        matches!(source.kind.as_str(), "image" | "video" | "text"),
        "invalid source kind {}",
        source.kind
    );
    ensure!(
        !source.title.trim().is_empty()
            && !source.author.trim().is_empty()
            && !source.source_url.trim().is_empty(),
        "source attribution is incomplete"
    );
    ensure!(
        matches!(
            source.licence.as_str(),
            "Public domain"
                | "CC0"
                | "CC BY 1.0"
                | "CC BY 2.0"
                | "CC BY 2.5"
                | "CC BY 3.0"
                | "CC BY 4.0"
        ),
        "unsupported source licence {}",
        source.licence
    );
    ensure!(
        source.licence == "Public domain" || !source.licence_url.trim().is_empty(),
        "source licence URL is missing"
    );
    ensure!(
        hash_file(&source.path)? == source.sha256,
        "source hash mismatch: {}",
        source.path.display()
    );
    ensure!(
        source.raw_capture.is_file(),
        "raw source capture is missing: {}",
        source.raw_capture.display()
    );
    if let Some(expected) = &source.raw_sha256 {
        ensure!(
            hash_file(&source.raw_capture)? == *expected,
            "raw capture hash mismatch: {}",
            source.raw_capture.display()
        );
    }
    Ok(())
}

fn image_source(
    truth: &mut Truth,
    index: usize,
    source: &LocalSource,
    retained: &Path,
    random: &mut Random,
) -> anyhow::Result<()> {
    let scene = image::open(truth.root.join(retained))?.resize(
        960,
        960,
        image::imageops::FilterType::Lanczos3,
    );
    ensure!(
        scene.width() >= 4 && scene.height() >= 4,
        "image source is too small"
    );
    save_original_window(truth, index, &scene)?;
    let original = image_path(index, "original", "jpg");
    for path in [original.clone(), scene_path(index)] {
        truth.sources.push(Source {
            origin: source.source_url.clone(),
            sha256: hash_file(&truth.root.join(&path))?,
            retained: path,
        });
    }
    image_family(truth, index, &original, random)
}

fn video_source(
    truth: &mut Truth,
    index: usize,
    source: &LocalSource,
    retained: &Path,
    random: &mut Random,
) -> anyhow::Result<()> {
    let extension = retained
        .extension()
        .and_then(|extension| extension.to_str())
        .context("video extension")?;
    let original = PathBuf::from(format!("media/video/{index:03}_original.{extension}"));
    fs::copy(truth.root.join(retained), truth.root.join(&original))?;
    let output = command(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
            ])
            .arg(truth.root.join(&original)),
    )?;
    let duration: f64 = std::str::from_utf8(&output)?.trim().parse()?;
    ensure!(
        duration.is_finite() && duration >= 12.0,
        "video source must be at least twelve seconds: {}",
        source.path.display()
    );
    record(truth, "video", index, "original", original.clone(), None)?;
    truth.sources.push(Source {
        origin: source.source_url.clone(),
        retained: original.clone(),
        sha256: source.sha256.clone(),
    });
    video_derivatives(
        truth,
        index,
        &original,
        VideoTiming {
            duration,
            clip_10_start: (duration - 10.0) * 0.35,
            clip_25_start: duration * 0.5,
        },
        random,
    )
}
