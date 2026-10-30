//! Shared decoration and empty backgrounds must not prove a shared scene.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::Context as _;
use eidolon_core::{Config, Ctx, FileEntry, FileId, Kind, Relation, cache::FeatureCache};
use image::imageops::FilterType;

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

fn find_pair(
    first: &image::RgbImage,
    second: &image::RgbImage,
) -> anyhow::Result<Vec<eidolon_core::Match>> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-spatial-proof-")
        .tempdir_in("/tmp")?;
    let mut files = Vec::new();
    for (id, image) in [first, second].into_iter().enumerate() {
        let path = temp.path().join(format!("{id}.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&path)?, 25)
            .encode_image(image)?;
        files.push(entry(id, path)?);
    }
    find_files(&files)
}

fn find_files(files: &[FileEntry]) -> anyhow::Result<Vec<eidolon_core::Match>> {
    find_files_with_mode(files, false)
}

fn find_files_with_mode(
    files: &[FileEntry],
    fast: bool,
) -> anyhow::Result<Vec<eidolon_core::Match>> {
    let cache = FeatureCache::in_memory()?;
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.80,
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: false,
        jobs: 2,
        fast,
        video_fps: 2.0,
    };
    eidolon_image::find(
        files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )
}

fn checker(image: &image::RgbImage) -> image::RgbImage {
    let mut result = image.clone();
    for y in image.height() * 7 / 10..image.height() * 9 / 10 {
        for x in image.width() / 10..image.width() * 9 / 10 {
            let color = if (x / (image.width() / 24).max(1) + y / (image.height() / 16).max(1))
                .is_multiple_of(2)
            {
                [255, 0, 160]
            } else {
                [255, 255, 255]
            };
            result.put_pixel(x, y, image::Rgb(color));
        }
    }
    result
}

fn canvas(image: &image::RgbImage) -> image::RgbImage {
    let mut result =
        image::RgbImage::from_pixel(image.width(), image.height(), image::Rgb([24, 24, 24]));
    let content = image::imageops::resize(
        image,
        image.width() * 9 / 10,
        image.height() * 9 / 10,
        FilterType::Lanczos3,
    );
    image::imageops::overlay(
        &mut result,
        &content,
        i64::from(image.width() / 10),
        i64::from(image.height() / 20),
    );
    result
}

#[cfg(target_os = "macos")]
fn wallpaper_paths<const N: usize>(names: [&str; N]) -> anyhow::Result<Option<[PathBuf; N]>> {
    let directory = std::env::var_os("EIDOLON_TEST_WALLPAPER_DIR");
    let root = directory
        .as_deref()
        .map_or(Path::new("/System/Library/Desktop Pictures"), Path::new);
    let paths = names.map(|name| root.join(name));
    for path in &paths {
        if !path.try_exists()? {
            eprintln!(
                "SKIP: macOS wallpaper fixture unavailable: {}",
                path.display()
            );
            return Ok(None);
        }
    }
    Ok(Some(paths))
}

#[test]
#[cfg(target_os = "macos")]
fn missing_macos_wallpapers_skip_optional_photograph_test() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    for test in [
        "common_checker_and_canvas_do_not_match_unrelated_photographs",
        "broad_gradients_do_not_turn_off_center_crops_into_variants",
        "independently_encoded_ridge_recolours_are_versions",
        "native_density_crops_survive_crowded_encodings",
        "jpeg_sampling_noise_does_not_restyle_a_textured_crop",
    ] {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", test, "--nocapture"])
            .env(
                "EIDOLON_TEST_WALLPAPER_DIR",
                directory.path().join("absent"),
            )
            .output()?;
        assert!(
            output.status.success(),
            "{test}: missing optional fixtures must not fail: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("SKIP: macOS wallpaper fixture unavailable"),
            "{test}: missing fixtures must produce a clear skip message"
        );
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn common_checker_and_canvas_do_not_match_unrelated_photographs() -> anyhow::Result<()> {
    let Some([first, second, pink, orange]) = wallpaper_paths([
        ".thumbnails/Big Sur Dark.heic",
        ".thumbnails/Catalina Dark.heic",
        "Mac Pink.heic",
        "iMac Orange.heic",
    ])?
    else {
        return Ok(());
    };
    let first = eidolon_image::decode(&first)?.to_rgb8();
    let second = eidolon_image::decode(&second)?.to_rgb8();
    let first =
        image::imageops::crop_imm(&first, 0, 0, first.width() * 3 / 4, first.height()).to_image();
    let second = image::imageops::crop_imm(&second, 0, 0, second.width() * 3 / 4, second.height())
        .to_image();
    let pink = eidolon_image::decode(&pink)?.to_rgb8();
    let orange = eidolon_image::decode(&orange)?.to_rgb8();
    let pink = image::imageops::resize(&pink, 480, 640, FilterType::Triangle);
    let orange = image::imageops::resize(&orange, 480, 640, FilterType::Triangle);
    for (name, a, b) in [
        ("checker", checker(&first), checker(&second)),
        ("canvas", canvas(&pink), canvas(&orange)),
    ] {
        let matches = find_pair(&a, &b)?;
        assert!(
            matches.is_empty(),
            "shared {name} must not prove shared scene: {matches:?}"
        );
    }
    let matches = find_pair(&first, &checker(&first))?;
    let found = matches
        .first()
        .context("a real local overlay must remain detectable")?;
    assert_eq!(found.relation, Relation::Edited, "{matches:?}");
    Ok(())
}

#[test]
fn unrelated_foregrounds_on_empty_black_backgrounds_are_not_edits() -> anyhow::Result<()> {
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
    let matches = find_pair(&first, &second)?;
    assert!(
        matches.is_empty(),
        "empty background supplies no shared foreground proof: {matches:?}"
    );
    Ok(())
}

fn screenshot_page(alternate: bool) -> image::RgbImage {
    image::RgbImage::from_fn(768, 576, |x, y| {
        let chrome = y < 64 || x < 160;
        let glyph = if y < 64 {
            (x / 7 + y / 5).is_multiple_of(3) && y > 20 && y < 40
        } else {
            (x / 9 + y / 7).is_multiple_of(4) && x > 20 && x < 130 && y % 48 < 14
        };
        let content = if alternate {
            (210..510).contains(&x)
                && (130..400).contains(&y)
                && ((y % 53 < 7 && x % 61 < 42) || (x % 97 < 3 && y % 53 < 35))
        } else {
            (210..530).contains(&x)
                && (110..430).contains(&y)
                && ((y % 39 < 6 && x % 71 < 55) || (y % 103 < 3 && x < 490))
        };
        image::Rgb(if chrome && glyph {
            [60, 70, 85]
        } else if chrome {
            [235, 241, 250]
        } else if content {
            [95, 100, 110]
        } else {
            [255, 255, 255]
        })
    })
}

#[test]
fn shared_ui_chrome_does_not_prove_different_pages_are_duplicates() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-screenshot-pages-")
        .tempdir_in("/tmp")?;
    let mut files = Vec::new();
    for (id, image) in [screenshot_page(false), screenshot_page(true)]
        .iter()
        .enumerate()
    {
        let path = temp.path().join(format!("{id}.png"));
        image.save(&path)?;
        files.push(entry(id, path)?);
    }
    for fast in [false, true] {
        let matches = find_files_with_mode(&files, fast)?;
        assert!(
            matches.is_empty(),
            "different page content must not match through common chrome (fast={fast}): {matches:?}"
        );
    }
    Ok(())
}

#[test]
fn a_small_local_page_edit_remains_an_edited_duplicate() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-screenshot-edit-")
        .tempdir_in("/tmp")?;
    let original = screenshot_page(false);
    let mut edited = original.clone();
    for y in 240..268 {
        for x in 270..366 {
            edited.put_pixel(x, y, image::Rgb([30, 115, 210]));
        }
    }
    let mut files = Vec::new();
    for (id, image) in [&original, &edited].into_iter().enumerate() {
        let path = temp.path().join(format!("{id}.png"));
        image.save(&path)?;
        files.push(entry(id, path)?);
    }
    for fast in [false, true] {
        let matches = find_files_with_mode(&files, fast)?;
        let found = matches.first().with_context(|| {
            format!("a same-page local edit must remain detectable (fast={fast})")
        })?;
        assert_eq!(found.relation, Relation::Edited, "fast={fast}: {matches:?}");
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn broad_gradients_do_not_turn_off_center_crops_into_variants() -> anyhow::Result<()> {
    let Some([path]) = wallpaper_paths([".thumbnails/Big Sur Graphic Dark.heic"])? else {
        return Ok(());
    };
    let original = eidolon_image::decode(&path)?.to_rgb8();
    let original =
        image::imageops::crop_imm(&original, 0, 0, original.width() * 3 / 4, original.height())
            .to_image();
    let width = (f64::from(original.width()) * 0.8_f64.sqrt()).round() as u32;
    let height = (f64::from(original.height()) * 0.8_f64.sqrt()).round() as u32;
    for x in [0, (original.width() - width) / 2, original.width() - width] {
        for y in [
            0,
            (original.height() - height) / 2,
            original.height() - height,
        ] {
            let crop = image::imageops::crop_imm(&original, x, y, width, height).to_image();
            let matches = find_pair(&original, &crop)?;
            let found = matches
                .first()
                .with_context(|| format!("crop {x}/{y} must be found"))?;
            assert_eq!(
                found.relation,
                Relation::CropOf,
                "crop {x}/{y}: {matches:?}"
            );
            assert_eq!((found.a, found.b), (FileId(1), FileId(0)), "{matches:?}");
        }
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn independently_encoded_ridge_recolours_are_versions() -> anyhow::Result<()> {
    let Some([first, second]) = wallpaper_paths(["iMac Blue.heic", "iMac Green.heic"])? else {
        return Ok(());
    };
    let first = eidolon_image::decode(&first)?.to_rgb8();
    let second = eidolon_image::decode(&second)?.to_rgb8();
    let first = image::imageops::crop_imm(&first, 0, 0, first.width() * 3 / 4, first.height());
    let second = image::imageops::crop_imm(&second, 0, 0, second.width() * 3 / 4, second.height());
    let first = image::imageops::resize(&*first, 480, 640, FilterType::Triangle);
    let second = image::imageops::resize(&*second, 480, 640, FilterType::Triangle);
    let matches = find_pair(&first, &second)?;
    let found = matches
        .first()
        .context("same-ridge recolours must be found without embeddings")?;
    assert_eq!(found.relation, Relation::Variant, "{matches:?}");
    let temp = tempfile::Builder::new()
        .prefix("eidolon-crowded-versions-")
        .tempdir_in("/tmp")?;
    let mut files = Vec::new();
    for (family, image) in [&first, &second].into_iter().enumerate() {
        for quality in 32..68_u8 {
            let path = temp.path().join(format!("{family}-{quality}.jpg"));
            image::codecs::jpeg::JpegEncoder::new_with_quality(
                std::fs::File::create(&path)?,
                quality,
            )
            .encode_image(image)?;
            files.push(entry(files.len(), path)?);
        }
    }
    let matches = find_files(&files)?;
    let found = matches
        .iter()
        .find(|item| item.a == FileId(0) && item.b == FileId(36))
        .context("near-identical siblings must not evict a genuine cross-source recolour")?;
    assert_eq!(found.relation, Relation::Variant, "{found:?}");
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn native_density_crops_survive_crowded_encodings() -> anyhow::Result<()> {
    let Some([path]) = wallpaper_paths([".thumbnails/iMac Blue.heic"])? else {
        return Ok(());
    };
    let temp = tempfile::Builder::new()
        .prefix("eidolon-crowded-crops-")
        .tempdir_in("/tmp")?;
    let source = eidolon_image::decode(&path)?.to_rgb8();
    let whole = image::imageops::resize(&source, 480, 640, FilterType::Triangle);
    let part = image::imageops::crop_imm(&whole, 43, 67, 304, 405).to_image();
    let mut files = Vec::new();
    for (family, image) in [&whole, &part].into_iter().enumerate() {
        for quality in 32..68_u8 {
            let path = temp.path().join(format!("{family}-{quality}.jpg"));
            image::codecs::jpeg::JpegEncoder::new_with_quality(
                std::fs::File::create(&path)?,
                quality,
            )
            .encode_image(image)?;
            files.push(entry(files.len(), path)?);
        }
    }
    let matches = find_files(&files)?;
    for whole in 0..36 {
        for part in 36..72 {
            let Some(found) = matches
                .iter()
                .find(|item| item.a == FileId(part) && item.b == FileId(whole))
            else {
                let retained = temp.keep();
                anyhow::bail!(
                    "missing crop {part} -> whole {whole}, fixtures at {}",
                    retained.display()
                );
            };
            assert_eq!(found.relation, Relation::CropOf, "{found:?}");
        }
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn jpeg_sampling_noise_does_not_restyle_a_textured_crop() -> anyhow::Result<()> {
    let Some([path]) = wallpaper_paths([".thumbnails/Catalina Rock.heic"])? else {
        return Ok(());
    };
    let temp = tempfile::Builder::new()
        .prefix("eidolon-textured-crop-")
        .tempdir_in("/tmp")?;
    let source = eidolon_image::decode(&path)?.to_rgb8();
    let scaled = image::imageops::resize(&source, 960, 960, FilterType::Triangle);
    let whole = image::imageops::crop_imm(&scaled, 0, 0, 720, 960).to_image();
    let part = image::imageops::crop_imm(&whole, 27, 47, 455, 607).to_image();
    let mut files = Vec::new();
    for (id, image, quality) in [(0, &whole, 92), (1, &part, 25)] {
        let path = temp.path().join(format!("{id}.jpg"));
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&path)?, quality)
            .encode_image(image)?;
        files.push(entry(id, path)?);
    }
    let matches = find_files(&files)?;
    let found = matches.first().context("textured crop must be found")?;
    assert_eq!(found.relation, Relation::CropOf, "{found:?}");
    assert_eq!((found.a, found.b), (FileId(1), FileId(0)), "{found:?}");
    Ok(())
}
