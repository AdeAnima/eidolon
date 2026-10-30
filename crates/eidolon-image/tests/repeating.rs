//! Repeated real-image geometry must not turn full-frame resizing into a crop.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use anyhow::ensure;
use eidolon_core::{Config, Ctx, FileEntry, FileId, Kind, Relation, cache::FeatureCache};
use image::imageops::FilterType;

fn entry(id: usize, path: PathBuf) -> anyhow::Result<FileEntry> {
    let metadata = path.metadata()?;
    Ok(FileEntry {
        id: FileId(id),
        path,
        kind: Kind::Image,
        size: metadata.len(),
        mtime_ns: i128::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
    })
}

#[test]
#[cfg(target_os = "macos")]
fn repeated_grid_resizing_is_reencoded_not_a_crop() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-grid-")
        .tempdir_in("/tmp")?;
    for name in ["Grid Green Light", "Grid Magenta Light"] {
        let source =
            Path::new("/System/Library/Desktop Pictures/.thumbnails").join(format!("{name}.heic"));
        let original = eidolon_image::decode(&source)?.to_rgb8();
        let path = temp.path().join(format!("{name}-original.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&path)?, 95)
            .encode_image(&original)?;
        let original = eidolon_image::decode(&path)?.to_rgb8();
        let resized = image::imageops::resize(
            &original,
            original.width() / 2,
            original.height() / 2,
            FilterType::Lanczos3,
        );
        let smaller = temp.path().join(format!("{name}-small.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&smaller)?, 90)
            .encode_image(&resized)?;
        let files = [entry(11, path)?, entry(3, smaller)?];
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
        assert_eq!(matches.len(), 1, "{name}: {matches:?}");
        assert_eq!(
            matches[0].relation,
            Relation::Reencoded,
            "{name}: {:?}",
            matches[0]
        );
        assert_eq!((matches[0].a, matches[0].b), (FileId(3), FileId(11)));
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn cropped_grids_with_low_quality_logos_keep_containment_direction() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-grid-crop-")
        .tempdir_in("/tmp")?;
    for name in ["Grid Green Light", "Grid Magenta Light"] {
        let source =
            Path::new("/System/Library/Desktop Pictures/.thumbnails").join(format!("{name}.heic"));
        let original = eidolon_image::decode(&source)?.to_rgb8();
        let whole = temp.path().join(format!("{name}-whole.png"));
        original.save(&whole)?;
        let width = (f64::from(original.width()) * 0.6_f64.sqrt()).round() as u32;
        let height = (f64::from(original.height()) * 0.6_f64.sqrt()).round() as u32;
        let mut crop = image::imageops::crop_imm(
            &original,
            (original.width() - width) / 3,
            (original.height() - height) / 2,
            width,
            height,
        )
        .to_image();
        for y in height * 7 / 10..height * 9 / 10 {
            for x in width / 10..width * 9 / 10 {
                let color = if (x / 6 + y / 8) % 2 == 0 {
                    [255, 0, 160]
                } else {
                    [255, 255, 255]
                };
                crop.put_pixel(x, y, image::Rgb(color));
            }
        }
        let part = temp.path().join(format!("{name}-crop.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&part)?, 25)
            .encode_image(&crop)?;
        let files = [entry(11, whole)?, entry(3, part)?];
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
        assert_eq!(matches.len(), 1, "{name}: {matches:?}");
        assert_eq!(
            matches[0].relation,
            Relation::CropOf,
            "{name}: {:?}",
            matches[0]
        );
        assert_eq!((matches[0].a, matches[0].b), (FileId(3), FileId(11)));
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn differently_lit_grids_do_not_establish_crop_containment() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-grid-lighting-")
        .tempdir_in("/tmp")?;
    for color in ["Green", "Magenta"] {
        let mut files = Vec::with_capacity(2);
        for (suffix, id) in [(" Light", 11), ("", 3)] {
            let name = format!("Grid {color}{suffix}");
            let source = Path::new("/System/Library/Desktop Pictures/.thumbnails")
                .join(format!("{name}.heic"));
            let canonical = temp.path().join(format!("{name}-canonical.jpg"));
            let output = Command::new("sips")
                .args(["-s", "format", "jpeg", "-Z", "960"])
                .arg(&source)
                .arg("--out")
                .arg(&canonical)
                .output()?;
            ensure!(
                output.status.success(),
                "sips failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let image = image::open(&canonical)?;
            image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&canonical)?, 92)
                .encode_image(&image)?;
            let image = image::open(&canonical)?;
            let png = temp.path().join(format!("{name}.png"));
            image.save(&png)?;
            let image = image::open(png)?;
            let roundtrip = temp.path().join(format!("{name}-roundtrip.jpg"));
            image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&roundtrip)?, 85)
                .encode_image(&image)?;
            files.push(entry(id, roundtrip)?);
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
        assert!(
            matches.iter().all(|item| item.relation != Relation::CropOf),
            "{color}: lighting changes must not count as crop identity: {matches:?}"
        );
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn recompressed_center_grid_crops_keep_identity() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-grid-recompressed-")
        .tempdir_in("/tmp")?;
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
    let ctx = Ctx {
        refiner: None,
        config: &config,
        cache: &cache,
        embedder: None,
    };
    let mut failures = Vec::new();
    for name in ["Grid Green Light", "Grid Magenta Light"] {
        let source =
            Path::new("/System/Library/Desktop Pictures/.thumbnails").join(format!("{name}.heic"));
        let canonical = temp.path().join(format!("{name}-whole.jpg"));
        let output = Command::new("sips")
            .args(["-s", "format", "jpeg", "-Z", "960"])
            .arg(source)
            .arg("--out")
            .arg(&canonical)
            .output()?;
        ensure!(
            output.status.success(),
            "sips failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let image = image::open(&canonical)?;
        image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&canonical)?, 92)
            .encode_image(&image)?;
        let image = image::open(&canonical)?.to_rgb8();
        for area in [0.4_f64, 0.6] {
            let width = (f64::from(image.width()) * area.sqrt()).round() as u32;
            let height = (f64::from(image.height()) * area.sqrt()).round() as u32;
            // Keep region-hash retrieval in range to isolate photometric
            // identity after JPEG compression, rather than candidate recall.
            let x = (image.width() - width).div_ceil(2);
            let y = (image.height() - height).div_ceil(2);
            let crop = image::imageops::crop_imm(&image, x, y, width, height).to_image();
            for quality in [25, 92] {
                let path = temp.path().join(format!("{name}-{area}-{quality}.jpg"));
                image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&path)?, quality)
                    .encode_image(&crop)?;
                let matches = eidolon_image::find(
                    &[entry(11, canonical.clone())?, entry(3, path)?],
                    &ctx,
                    None,
                )?;
                if matches.len() != 1
                    || matches[0].relation != Relation::CropOf
                    || (matches[0].a, matches[0].b) != (FileId(3), FileId(11))
                {
                    failures.push(format!("{name}, area {area}, q{quality}: {matches:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
