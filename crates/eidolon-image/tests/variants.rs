//! Global restyling remains a version; flat canvas padding remains duplicate content.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::Context as _;
use eidolon_core::{Config, Ctx, Embedder, FileEntry, FileId, Kind, Relation, cache::FeatureCache};
use image::Pixel as _;

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

fn assert_global_versions(embedder: Option<&dyn Embedder>) -> anyhow::Result<()> {
    for source in [
        "/System/Library/Desktop Pictures/.thumbnails/Big Sur Coastline.heic",
        "/System/Library/Desktop Pictures/.thumbnails/Mac Blue.heic",
    ] {
        assert_global_versions_for(source, embedder)?;
    }
    Ok(())
}

fn padded_canvas(original: &image::RgbImage) -> image::RgbImage {
    let mut canvas = image::RgbImage::from_pixel(
        original.width(),
        original.height(),
        *original.get_pixel(0, 0),
    );
    let resized = image::imageops::resize(
        original,
        original.width() * 9 / 10,
        original.height() * 9 / 10,
        image::imageops::FilterType::Lanczos3,
    );
    image::imageops::overlay(
        &mut canvas,
        &resized,
        i64::from(original.width() / 10),
        i64::from(original.height() / 20),
    );
    canvas
}

fn assert_global_versions_for(source: &str, embedder: Option<&dyn Embedder>) -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-variant-")
        .tempdir_in("/tmp")?;
    let original = eidolon_image::decode(Path::new(source))?.to_rgb8();
    let path = temp.path().join("original.png");
    original.save(&path)?;
    let mut files = vec![entry(17, path)?];
    let recoloured = image::RgbImage::from_fn(original.width(), original.height(), |x, y| {
        let pixel = original.get_pixel(x, y);
        image::Rgb([pixel[1], pixel[2], pixel[0]])
    });
    let relit = image::RgbImage::from_fn(original.width(), original.height(), |x, y| {
        original
            .get_pixel(x, y)
            .map(|value| value.saturating_add(28))
    });
    let dark = image::RgbImage::from_fn(original.width(), original.height(), |x, y| {
        original.get_pixel(x, y).map(|value| 255 - value)
    });
    let reframed = padded_canvas(&original);
    let inverted_recolour =
        image::RgbImage::from_fn(original.width(), original.height(), |x, y| {
            let pixel = original.get_pixel(x, y);
            image::Rgb([255 - pixel[2], 255 - pixel[0], 255 - pixel[1]])
        });
    let cropped = image::imageops::crop_imm(
        &original,
        original.width() / 6,
        original.height() / 6,
        original.width() * 2 / 3,
        original.height() * 2 / 3,
    )
    .to_image();
    for (id, name, image) in [
        (3, "recoloured", recoloured),
        (11, "relit", relit),
        (25, "dark", dark),
        (9, "crop", cropped),
        (31, "reframed", reframed),
        (33, "inverted_recolour", inverted_recolour),
    ] {
        let path = temp.path().join(format!("{name}.png"));
        image.save(&path)?;
        files.push(entry(id, path)?);
    }
    let cache = FeatureCache::in_memory()?;
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.80,
        model_dir: PathBuf::new(),
        no_embeddings: embedder.is_none(),
        no_keypoints: false,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    };
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder,
        },
        None,
    )?;
    for id in [3, 11, 25, 33] {
        let found = matches
            .iter()
            .find(|item| item.a == FileId(id.min(17)) && item.b == FileId(id.max(17)))
            .with_context(|| format!("missing global variant {id}: {matches:?}"))?;
        assert_eq!(found.relation, Relation::Variant, "{found:?}");
        assert!(found.evidence.iter().any(|item| item.signal == "variant"));
    }
    let crop_to_canvas = matches
        .iter()
        .find(|item| item.a == FileId(9) && item.b == FileId(31))
        .with_context(|| format!("missing cropped-to-reframed version: {matches:?}"))?;
    assert_eq!(
        crop_to_canvas.relation,
        Relation::CropOf,
        "{crop_to_canvas:?}"
    );
    let crop_to_original = matches
        .iter()
        .find(|item| item.a == FileId(9) && item.b == FileId(17))
        .with_context(|| format!("missing ordinary crop: {matches:?}"))?;
    assert_eq!(
        crop_to_original.relation,
        Relation::CropOf,
        "{crop_to_original:?}"
    );
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn added_flat_canvas_preserves_duplicate_content() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-flat-canvas-")
        .tempdir_in("/tmp")?;
    let original = eidolon_image::decode(Path::new(
        "/System/Library/Desktop Pictures/.thumbnails/Big Sur Coastline.heic",
    ))?
    .to_rgb8();
    let whole = temp.path().join("canvas.png");
    let part = temp.path().join("original.png");
    padded_canvas(&original).save(&whole)?;
    original.save(&part)?;
    let cache = FeatureCache::in_memory()?;
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.80,
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: false,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    };
    let matches = eidolon_image::find(
        &[entry(31, whole)?, entry(17, part)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    assert_eq!(matches.len(), 1, "{matches:?}");
    assert_eq!(matches[0].relation, Relation::CropOf, "{matches:?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(17), FileId(31)));
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn shifted_scene_windows_are_versions_not_containment() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-scene-window-")
        .tempdir_in("/tmp")?;
    for name in ["Big Sur Coastline", "Mac Blue", "Grid Green Light"] {
        let scene = eidolon_image::decode(
            &Path::new("/System/Library/Desktop Pictures/.thumbnails").join(format!("{name}.heic")),
        )?
        .to_rgb8();
        let width = scene.width() * 3 / 4;
        let left = temp.path().join("left.png");
        let right = temp.path().join("right.jpg");
        image::imageops::crop_imm(&scene, 0, 0, width, scene.height())
            .to_image()
            .save(&left)?;
        let shifted =
            image::imageops::crop_imm(&scene, scene.width() - width, 0, width, scene.height())
                .to_image();
        for gamma in [1.0_f64, 0.35, 2.8] {
            let transformed =
                image::RgbImage::from_fn(shifted.width(), shifted.height(), |x, y| {
                    shifted
                        .get_pixel(x, y)
                        .map(|value| (255.0 * (f64::from(value) / 255.0).powf(gamma)).round() as u8)
                });
            image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&right)?, 25)
                .encode_image(&transformed)?;
            let cache = FeatureCache::in_memory()?;
            let config = Config {
                timings: false,
                remote_overlay: true,
                min_score: 0.80,
                model_dir: PathBuf::new(),
                no_embeddings: true,
                no_keypoints: false,
                jobs: 2,
                fast: false,
                video_fps: 2.0,
            };
            let matches = eidolon_image::find(
                &[entry(17, left.clone())?, entry(3, right.clone())?],
                &Ctx {
                    refiner: None,
                    config: &config,
                    cache: &cache,
                    embedder: None,
                },
                None,
            )?;
            if matches.is_empty() {
                let retained = temp.keep();
                anyhow::bail!("{name}, gamma {gamma}: fixtures at {}", retained.display());
            }
            assert_eq!(matches.len(), 1, "{name}, gamma {gamma}: {matches:?}");
            assert_eq!(
                matches[0].relation,
                Relation::Variant,
                "{name}, gamma {gamma}: {matches:?}"
            );
            assert_eq!((matches[0].a, matches[0].b), (FileId(3), FileId(17)));
        }
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn full_frame_tone_versions_keep_geometry_across_encodings() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-tone-versions-")
        .tempdir_in("/tmp")?;
    let mut files = Vec::new();
    for (source, name) in ["iMac Blue", "iMac Green"].into_iter().enumerate() {
        let canonical = temp.path().join(format!("{source}-canonical.jpg"));
        let output = std::process::Command::new("sips")
            .args(["-s", "format", "jpeg", "-Z", "640"])
            .arg(Path::new("/System/Library/Desktop Pictures").join(format!("{name}.heic")))
            .arg("--out")
            .arg(&canonical)
            .output()?;
        anyhow::ensure!(output.status.success(), "sips failed: {output:?}");
        let original = image::open(&canonical)?.to_rgb8();
        for (version, gamma) in [1.0_f64, 0.35, 2.8].into_iter().enumerate() {
            let transformed =
                image::RgbImage::from_fn(original.width(), original.height(), |x, y| {
                    original
                        .get_pixel(x, y)
                        .map(|value| (255.0 * (f64::from(value) / 255.0).powf(gamma)).round() as u8)
                });
            let path = temp.path().join(format!("{source}-{version}.jpg"));
            image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&path)?, 30)
                .encode_image(&transformed)?;
            files.push(entry(files.len(), path)?);
        }
    }
    let cache = FeatureCache::in_memory()?;
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.80,
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: false,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    };
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    for first in 0..files.len() {
        for second in first + 1..files.len() {
            let found = matches
                .iter()
                .find(|found| found.a == FileId(first) && found.b == FileId(second))
                .with_context(|| format!("missing tone version {first}/{second}: {matches:?}"))?;
            assert_eq!(
                found.relation,
                Relation::Variant,
                "{first}/{second}: {found:?}"
            );
        }
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn global_recolour_relighting_and_dark_versions_are_variants() -> anyhow::Result<()> {
    assert_global_versions(None)
}

#[test]
#[ignore = "requires an already-fetched DINOv2 model; no network requests"]
#[cfg(target_os = "macos")]
fn global_versions_with_fetched_embeddings_are_variants() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
        .join("Library/Caches/eidolon/models");
    let model =
        eidolon_image::embed::OnnxEmbedder::load(&model_dir)?.context("fetch model first")?;
    assert_global_versions(Some(&model))
}

#[test]
#[cfg(target_os = "macos")]
fn reframed_and_relit_grid_versions_are_variants_not_crops() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-grid-variant-")
        .tempdir_in("/tmp")?;
    let mut files = Vec::new();
    for (id, name) in [(18, "Grid Magenta Light"), (4, "Grid Magenta")] {
        let input =
            Path::new("/System/Library/Desktop Pictures/.thumbnails").join(format!("{name}.heic"));
        let path = temp.path().join(format!("{id}.png"));
        eidolon_image::decode(&input)?.save(&path)?;
        files.push(entry(id, path)?);
    }
    let cache = FeatureCache::in_memory()?;
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.80,
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: false,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    };
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    assert_eq!(matches.len(), 1, "{matches:?}");
    assert_eq!(matches[0].relation, Relation::Variant, "{:?}", matches[0]);
    assert_eq!((matches[0].a, matches[0].b), (FileId(4), FileId(18)));
    Ok(())
}
