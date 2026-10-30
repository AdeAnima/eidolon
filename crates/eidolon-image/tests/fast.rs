//! Fast matching must prove image content, not only a shared layout or embedding.

use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use anyhow::Context as _;
use eidolon_core::{Config, Ctx, FileEntry, FileId, Kind, Match, Relation, cache::FeatureCache};

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

fn find(files: &[FileEntry]) -> anyhow::Result<Vec<Match>> {
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.75,
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: false,
        jobs: 2,
        fast: true,
        video_fps: 2.0,
    };
    let cache = FeatureCache::in_memory()?;
    eidolon_image::find(
        files,
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: None,
        },
        None,
    )
}

#[test]
fn rejects_unrelated_foregrounds_on_the_same_empty_background() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-fast-proof-")
        .tempdir_in("/tmp")?;
    let mut first = image::RgbImage::new(320, 320);
    let mut second = first.clone();
    for y in 160..270 {
        for x in 60..240 {
            if (i64::from(x) - 150).pow(2) + (i64::from(y) - 215).pow(2) < 55_i64.pow(2) {
                first.put_pixel(x, y, image::Rgb([170, 120, 75]));
            }
            if x > 115 && x < 180 {
                second.put_pixel(x, y, image::Rgb([95, 140, 180]));
            }
        }
    }
    let mut files = Vec::new();
    for (id, image) in [first, second].into_iter().enumerate() {
        let path = temp.path().join(format!("{id}.png"));
        image.save(&path)?;
        files.push(entry(id, path)?);
    }
    let matches = find(&files)?;
    assert!(
        matches.is_empty(),
        "empty background is not shared content: {matches:?}"
    );
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn proves_real_photo_resize_crop_and_local_watermark() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-fast-versions-")
        .tempdir_in("/tmp")?;
    let original = eidolon_image::decode(std::path::Path::new(
        "/System/Library/Desktop Pictures/.thumbnails/Big Sur Coastline.heic",
    ))?
    .to_rgb8();
    let resized =
        image::imageops::resize(&original, 178, 178, image::imageops::FilterType::Triangle);
    let cropped = image::imageops::crop_imm(&original, 35, 41, 276, 276).to_image();
    let mut marked = original.clone();
    for y in 260..300 {
        for x in 60..220 {
            let pixel = marked.get_pixel_mut(x, y);
            for channel in &mut pixel.0 {
                *channel = u16::midpoint(u16::from(*channel), 255) as u8;
            }
        }
    }
    let original_path = temp.path().join("original.png");
    original.save(&original_path)?;
    let whole = entry(17, original_path)?;
    for (name, image, expected) in [
        ("resized", resized, Relation::Reencoded),
        ("cropped", cropped, Relation::CropOf),
        ("watermarked", marked, Relation::Edited),
    ] {
        let path = temp.path().join(format!("{name}.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&path)?, 30)
            .encode_image(&image)?;
        let matches = find(&[whole.clone(), entry(3, path)?])?;
        let found = matches.first().with_context(|| format!("{name} missing"))?;
        assert_eq!(
            (found.a, found.b, found.relation),
            (FileId(3), FileId(17), expected),
            "{found:?}"
        );
        assert!(found.evidence.iter().any(|item| item.signal == "thumbnail"));
        assert!(
            !found
                .evidence
                .iter()
                .any(|item| item.signal == "unverified")
        );
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn shared_watermark_does_not_prove_unrelated_photo_content() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-fast-decoration-")
        .tempdir_in("/tmp")?;
    let mut files = Vec::new();
    for (id, name) in ["Big Sur Dark", "Catalina Dark"].into_iter().enumerate() {
        let image = eidolon_image::decode(
            &std::path::Path::new("/System/Library/Desktop Pictures/.thumbnails")
                .join(format!("{name}.heic")),
        )?
        .to_rgb8();
        let mut image =
            image::imageops::crop_imm(&image, 0, 0, image.width() * 3 / 4, image.height())
                .to_image();
        for y in image.height() * 7 / 10..image.height() * 9 / 10 {
            for x in image.width() / 10..image.width() * 9 / 10 {
                let color = if (x / (image.width() / 24).max(1) + y / (image.height() / 16).max(1))
                    .is_multiple_of(2)
                {
                    [255, 0, 160]
                } else {
                    [255, 255, 255]
                };
                image.put_pixel(x, y, image::Rgb(color));
            }
        }
        let path = temp.path().join(format!("{id}.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&path)?, 25)
            .encode_image(&image)?;
        files.push(entry(id, path)?);
    }
    let matches = find(&files)?;
    assert!(
        matches.is_empty(),
        "shared logo is not shared scene: {matches:?}"
    );
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn preserves_low_quality_watermarked_crops_at_subpixel_offsets() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-fast-crop-grid-")
        .tempdir_in("/tmp")?;
    let original = eidolon_image::decode(std::path::Path::new(
        "/System/Library/Desktop Pictures/.thumbnails/Grid Magenta Light.heic",
    ))?
    .resize_exact(960, 720, image::imageops::FilterType::Lanczos3)
    .to_rgb8();
    let path = temp.path().join("original.jpg");
    image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&path)?, 92)
        .encode_image(&original)?;
    let whole = entry(17, path)?;
    for (index, (x, y)) in [(17, 23), (86, 56), (151, 108), (211, 153)]
        .into_iter()
        .enumerate()
    {
        let mut crop = image::imageops::crop_imm(&original, x, y, 744, 558).to_image();
        for y in crop.height() * 7 / 10..crop.height() * 9 / 10 {
            for x in crop.width() / 10..crop.width() * 9 / 10 {
                let color =
                    if (x / (crop.width() / 24) + y / (crop.height() / 16)).is_multiple_of(2) {
                        [255, 0, 160]
                    } else {
                        [255, 255, 255]
                    };
                crop.put_pixel(x, y, image::Rgb(color));
            }
        }
        let path = temp.path().join(format!("crop-{index}.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&path)?, 25)
            .encode_image(&crop)?;
        let matches = find(&[whole.clone(), entry(3, path)?])?;
        let found = matches
            .first()
            .with_context(|| format!("watermarked crop at ({x}, {y}) missing"))?;
        assert_eq!(
            (found.a, found.b, found.relation),
            (FileId(3), FileId(17), Relation::CropOf),
            "{found:?}"
        );
    }
    Ok(())
}

#[test]
fn equal_thumbnail_values_do_not_prove_exact_decoded_equality() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-fast-depth-")
        .tempdir_in("/tmp")?;
    let mut files = Vec::new();
    for (id, value) in [0x7788_u16, 0x7789_u16].into_iter().enumerate() {
        let image = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_pixel(
            160,
            120,
            image::Rgb([value, value, value]),
        );
        let path = temp.path().join(format!("{id}.png"));
        image.save(&path)?;
        files.push(entry(id, path)?);
    }
    let matches = find(&files)?;
    let found = matches
        .first()
        .context("near-equal 16-bit versions missing")?;
    assert_eq!(found.relation, Relation::Reencoded, "{found:?}");
    assert!(found.evidence.iter().any(|item| item.signal == "thumbnail"));
    Ok(())
}
