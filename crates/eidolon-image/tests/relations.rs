//! Real-photo acceptance checks cover both geometric and neural detection.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{Context as _, ensure};
use eidolon_core::{Config, Ctx, Embedder, FileEntry, FileId, Kind, Relation, cache::FeatureCache};
use image::imageops::FilterType;

fn photo(temp: &Path, name: &str) -> anyhow::Result<image::RgbImage> {
    let source =
        Path::new("/System/Library/Desktop Pictures/.thumbnails").join(format!("{name}.heic"));
    let target = temp.join(format!("{name}.png"));
    let output = Command::new("sips")
        .args(["-s", "format", "png"])
        .arg(source)
        .arg("--out")
        .arg(&target)
        .output()?;
    ensure!(
        output.status.success(),
        "sips failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(image::open(target)?.to_rgb8())
}

fn entry(id: usize, path: PathBuf) -> anyhow::Result<FileEntry> {
    let metadata = path.metadata()?;
    Ok(FileEntry {
        id: FileId(id),
        path,
        size: metadata.len(),
        mtime_ns: i128::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
        kind: Kind::Image,
    })
}

fn config() -> Config {
    Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.80,
        model_dir: model_dir(),
        no_embeddings: false,
        no_keypoints: false,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    }
}

fn model_dir() -> PathBuf {
    Path::new(&std::env::var_os("HOME").unwrap_or_default()).join("Library/Caches/eidolon/models")
}

fn variants(temp: &Path) -> anyhow::Result<Vec<FileEntry>> {
    let original = photo(temp, "Big Sur Coastline")?;
    let other = photo(temp, "Big Sur Mountains")?;
    let path = temp.join("original.png");
    original.save(&path)?;
    let mut files = vec![entry(8, path)?];
    let small = image::imageops::resize(
        &original,
        original.width() * 2 / 3,
        original.height() * 2 / 3,
        FilterType::Lanczos3,
    );
    let path = temp.join("quality30.jpg");
    image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&path)?, 30)
        .encode_image(&small)?;
    files.push(entry(2, path)?);
    // Both sides sqrt(0.6): 60% of original area, offset away from the center.
    let side = (f64::from(original.width()) * 0.6_f64.sqrt()).round() as u32;
    let crop = image::imageops::crop_imm(&original, 13, 41, side, side).to_image();
    let path = temp.join("crop.png");
    crop.save(&path)?;
    files.push(entry(12, path)?);
    let trim = image::imageops::crop_imm(
        &original,
        13,
        8,
        original.width() - 26,
        original.height() - 16,
    )
    .to_image();
    let path = temp.join("trim.png");
    trim.save(&path)?;
    files.push(entry(16, path)?);
    let mut watermarked = original.clone();
    let width = original.width();
    let height = original.height();
    for y in height * 3 / 4..height * 9 / 10 {
        for x in width / 2..width * 9 / 10 {
            if (x / 9 + y / 7) % 3 != 0 {
                let pixel = watermarked.get_pixel_mut(x, y);
                for channel in &mut pixel.0 {
                    *channel = u16::midpoint(u16::from(*channel), 255) as u8;
                }
            }
        }
    }
    let path = temp.join("watermark.png");
    watermarked.save(&path)?;
    files.push(entry(4, path)?);
    for (id, name, mut picture) in [
        (18, "opaque_logo", original.clone()),
        (20, "cropped_logo", crop.clone()),
    ] {
        let width = picture.width();
        let height = picture.height();
        for y in height * 7 / 10..height * 9 / 10 {
            for x in width / 10..width * 9 / 10 {
                let ink = if (x / 13 + y / 11) % 2 == 0 {
                    image::Rgb([255, 20, 160])
                } else {
                    image::Rgb([255, 255, 255])
                };
                picture.put_pixel(x, y, ink);
            }
        }
        let path = temp.join(format!("{name}.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&path)?, 25)
            .encode_image(&picture)?;
        files.push(entry(id, path)?);
    }
    let path = temp.join("unrelated.png");
    other.save(&path)?;
    files.push(entry(6, path)?);
    Ok(files)
}

fn assert_relations(embedder: Option<&dyn Embedder>) -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-relations-")
        .tempdir_in("/tmp")?;
    let files = variants(temp.path())?;
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let context = Ctx {
        refiner: None,
        config: &settings,
        cache: &cache,
        embedder,
    };
    let started = Instant::now();
    let matches = eidolon_image::find(&files, &context, None)?;
    eprintln!(
        "cold extraction+verification: {:.2} ms/image ({} images, embeddings={})",
        started.elapsed().as_secs_f64() * 1000.0 / files.len() as f64,
        files.len(),
        embedder.is_some()
    );
    for (a, b, relation, label) in [
        (2, 8, Relation::Reencoded, "resized JPEG"),
        (12, 8, Relation::CropOf, "60%-area crop"),
        (16, 8, Relation::CropOf, "small crop"),
        (4, 8, Relation::Edited, "watermark"),
        (8, 18, Relation::Edited, "opaque logo"),
        (20, 8, Relation::CropOf, "cropped low-quality logo"),
    ] {
        let found = matches
            .iter()
            .find(|item| item.a == FileId(a) && item.b == FileId(b))
            .with_context(|| format!("missing {label}: {matches:?}"))?;
        assert_eq!(found.relation, relation, "{label}: {found:?}");
    }
    assert!(
        !matches
            .iter()
            .any(|item| item.a == FileId(6) || item.b == FileId(6)),
        "unrelated desktop pictures matched: {matches:?}"
    );
    assert!(matches.iter().all(|item| item.score.is_finite()
        && (0.8..=1.0).contains(&item.score)
        && !item.evidence.is_empty()));
    if embedder.is_some() {
        assert!(
            matches
                .iter()
                .all(|item| item.evidence.iter().any(|signal| signal.signal == "dinov2"))
        );
    }
    let cached = eidolon_image::find(&files, &context, None)?;
    assert_eq!(
        serde_json::to_string(&matches)?,
        serde_json::to_string(&cached)?
    );
    let mut strict = settings.clone();
    strict.min_score = 1.0;
    assert!(
        eidolon_image::find(
            &files,
            &Ctx {
                refiner: None,
                config: &strict,
                cache: &cache,
                embedder
            },
            None
        )?
        .is_empty()
    );
    let mut disabled = settings;
    disabled.no_embeddings = true;
    let disabled_matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &disabled,
            cache: &cache,
            embedder,
        },
        None,
    )?;
    assert!(!disabled_matches.is_empty());
    assert!(
        disabled_matches
            .iter()
            .all(|item| item.evidence.iter().all(|signal| signal.signal != "dinov2"))
    );
    Ok(())
}

fn assert_low_texture_relations(embedder: Option<&dyn Embedder>) -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-smooth-")
        .tempdir_in("/tmp")?;
    let original = photo(temp.path(), "Sonoma")?;
    let path = temp.path().join("smooth.png");
    original.save(&path)?;
    let original_entry = entry(30, path)?;
    let mut files = Vec::new();
    for index in 0..12 {
        let path = temp.path().join(format!("whole-{index}.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&path)?, 25 + index * 5)
            .encode_image(&original)?;
        files.push(entry(50 + usize::from(index), path)?);
    }
    for (id, area) in [(3, 0.8_f64), (5, 0.6), (7, 0.4)] {
        let width = (f64::from(original.width()) * area.sqrt()).round() as u32;
        let height = (f64::from(original.height()) * area.sqrt()).round() as u32;
        let cropped = image::imageops::crop_imm(
            &original,
            (original.width() - width) * 3 / 4,
            (original.height() - height) / 3,
            width,
            height,
        )
        .to_image();
        let path = temp.path().join(format!("smooth-{id}.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&path)?, 40)
            .encode_image(&cropped)?;
        files.push(entry(id, path)?);
    }
    let unrelated = photo(temp.path(), "Big Sur Coastline")?;
    let path = temp.path().join("unrelated.png");
    unrelated.save(&path)?;
    files.push(entry(40, path)?);
    files.push(original_entry);
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder,
        },
        None,
    )?;
    for id in [3, 5, 7] {
        let found = matches
            .iter()
            .find(|item| item.a == FileId(id) && item.b == FileId(30))
            .with_context(|| format!("no low-texture crop {id}: {matches:?}"))?;
        assert_eq!(found.relation, Relation::CropOf, "{found:?}");
    }
    assert!(
        matches
            .iter()
            .all(|item| item.a != FileId(40) && item.b != FileId(40)),
        "{matches:?}"
    );
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn real_photo_relations_without_embeddings() -> anyhow::Result<()> {
    assert_relations(None)
}

#[test]
#[cfg(target_os = "macos")]
fn low_texture_crops_without_embeddings() -> anyhow::Result<()> {
    assert_low_texture_relations(None)
}

#[test]
#[ignore = "explicit model download and real ONNX Runtime acceptance"]
#[cfg(target_os = "macos")]
fn real_photo_relations_with_fetched_model() -> anyhow::Result<()> {
    eidolon_image::models::fetch(&model_dir())?;
    let started = Instant::now();
    let embedder =
        eidolon_image::embed::OnnxEmbedder::load(&model_dir())?.context("fetched model missing")?;
    eprintln!(
        "model load including SHA-256: {:.2} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    assert_relations(Some(&embedder))?;
    assert_low_texture_relations(Some(&embedder))
}
