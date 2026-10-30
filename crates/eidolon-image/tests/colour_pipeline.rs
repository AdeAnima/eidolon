//! Real colour-management and encoder changes must preserve crop identity.
#![cfg(target_os = "macos")]

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use anyhow::{Context as _, ensure};
use eidolon_core::{Config, Ctx, Embedder, FileEntry, FileId, Kind, Relation, cache::FeatureCache};
use image::ImageDecoder as _;

const PROFILES: &str = "/System/Library/ColorSync/Profiles";

fn sips(input: &Path, output: &Path, args: &[&str]) -> anyhow::Result<()> {
    let result = Command::new("sips")
        .args(args)
        .arg(input)
        .arg("--out")
        .arg(output)
        .output()?;
    ensure!(
        result.status.success(),
        "sips failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

fn entry(id: usize, path: &Path) -> anyhow::Result<FileEntry> {
    let metadata = path.metadata()?;
    Ok(FileEntry {
        id: FileId(id),
        path: path.to_path_buf(),
        kind: Kind::Image,
        size: metadata.len(),
        mtime_ns: i128::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
    })
}

fn has_profile(path: &Path) -> anyhow::Result<bool> {
    Ok(image::ImageReader::open(path)?
        .into_decoder()?
        .icc_profile()?
        .is_some())
}

fn jpeg_encoders(
    temp: &Path,
    image: &image::RgbImage,
    part: &Path,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let image_jpeg = temp.join("whole-image-encoder.jpg");
    image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&image_jpeg)?, 92)
        .encode_image(image)?;
    let native_jpeg = temp.join("crop-native-420.jpg");
    sips(
        part,
        &native_jpeg,
        &["-s", "format", "jpeg", "-s", "formatOptions", "25"],
    )?;
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(&native_jpeg)
        .output()?;
    ensure!(
        probe.status.success() && String::from_utf8_lossy(&probe.stdout).trim() == "yuvj420p",
        "native JPEG control must use 4:2:0 chroma: {}",
        String::from_utf8_lossy(&probe.stdout)
    );
    Ok((image_jpeg, native_jpeg))
}

fn fixtures(temp: &Path) -> anyhow::Result<Vec<(&'static str, PathBuf, PathBuf)>> {
    let original = temp.join("native.png");
    sips(
        Path::new("/System/Library/Desktop Pictures/.thumbnails/Big Sur Coastline.heic"),
        &original,
        &["-s", "format", "png"],
    )?;
    let image = image::open(original)?.to_rgb8();
    let whole = temp.join("whole.png");
    image.save(&whole)?;
    let width = (f64::from(image.width()) * 0.6_f64.sqrt()).round() as u32;
    let height = (f64::from(image.height()) * 0.6_f64.sqrt()).round() as u32;
    let crop = image::imageops::crop_imm(&image, 13, 41, width, height).to_image();
    let part = temp.join("crop.png");
    crop.save(&part)?;

    let whole_p3 = temp.join("whole-p3.png");
    let crop_p3 = temp.join("crop-p3.png");
    let p3 = format!("{PROFILES}/Display P3.icc");
    sips(&whole, &whole_p3, &["-e", &p3])?;
    sips(&part, &crop_p3, &["-e", &p3])?;
    let srgb = temp.join("crop-srgb.png");
    sips(
        &crop_p3,
        &srgb,
        &["-m", &format!("{PROFILES}/sRGB Profile.icc")],
    )?;
    ensure!(
        image::open(&srgb)?.to_rgb8() != crop,
        "native P3 conversion must change actual RGB samples"
    );
    let stripped = temp.join("crop-stripped.png");
    sips(&crop_p3, &stripped, &["-d", "profile"])?;
    ensure!(
        has_profile(&whole_p3)? && has_profile(&crop_p3)? && !has_profile(&stripped)?,
        "ICC control must actually remove an embedded profile"
    );

    let gamma = temp.join("crop-gamma.png");
    let mut encoder = png::Encoder::new(File::create(&gamma)?, crop.width(), crop.height());
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(0.45455));
    encoder.write_header()?.write_image_data(crop.as_raw())?;
    ensure!(
        png::Decoder::new(BufReader::new(File::open(&gamma)?))
            .read_info()?
            .info()
            .gama_chunk
            .is_some(),
        "PNG control must actually contain a gamma chunk"
    );

    let (image_jpeg, native_jpeg) = jpeg_encoders(temp, &image, &part)?;

    let saturated = image::RgbImage::from_fn(image.width(), image.height(), |x, y| {
        let pixel = image.get_pixel(x, y);
        let mean = pixel.0.iter().map(|&value| f32::from(value)).sum::<f32>() / 3.0;
        image::Rgb(pixel.0.map(|value| {
            (mean + 1.8 * (f32::from(value) - mean))
                .round()
                .clamp(0.0, 255.0) as u8
        }))
    });
    let saturated_whole = temp.join("whole-saturated.png");
    saturated.save(&saturated_whole)?;
    let saturated_crop = image::imageops::crop_imm(&saturated, 13, 41, width, height).to_image();
    let saturated_part = temp.join("crop-saturated-q25.jpg");
    image::codecs::jpeg::JpegEncoder::new_with_quality(File::create(&saturated_part)?, 25)
        .encode_image(&saturated_crop)?;
    let saturated_p3 = temp.join("whole-saturated-p3.png");
    sips(&saturated_whole, &saturated_p3, &["-e", &p3])?;
    let saturated_crop_png = temp.join("crop-saturated.png");
    saturated_crop.save(&saturated_crop_png)?;
    let saturated_crop_p3 = temp.join("crop-saturated-p3.png");
    sips(&saturated_crop_png, &saturated_crop_p3, &["-e", &p3])?;
    let saturated_srgb = temp.join("crop-saturated-srgb.png");
    sips(
        &saturated_crop_p3,
        &saturated_srgb,
        &["-m", &format!("{PROFILES}/sRGB Profile.icc")],
    )?;
    let saturated_heic = temp.join("whole-saturated-p3.heic");
    sips(&saturated_p3, &saturated_heic, &["-s", "format", "heic"])?;
    Ok(vec![
        ("Display P3 to sRGB", whole_p3.clone(), srgb),
        ("ICC stripped versus kept", whole_p3, stripped),
        ("PNG gamma chunk", whole.clone(), gamma),
        (
            "different JPEG encoder and 4:2:0 chroma",
            image_jpeg,
            native_jpeg,
        ),
        (
            "high-saturation crop at JPEG q25",
            saturated_whole,
            saturated_part,
        ),
        (
            "high-saturation Display P3 to sRGB",
            saturated_p3,
            saturated_srgb.clone(),
        ),
        ("HEIC Display P3 to sRGB", saturated_heic, saturated_srgb),
    ])
}

fn assert_colour_pipeline_crops(embedder: Option<&dyn Embedder>) -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-colour-pipeline-")
        .tempdir_in("/tmp")?;
    let pairs = fixtures(temp.path())?;
    let cache = FeatureCache::in_memory()?;
    let mut config = Config {
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
    let mut failures = Vec::new();
    for fast in [false, true] {
        config.fast = fast;
        for (name, whole, part) in &pairs {
            let files = [entry(11, whole)?, entry(3, part)?];
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
            if let Some(pixels) = matches.first().and_then(|found| {
                found
                    .evidence
                    .iter()
                    .find(|item| item.signal == if fast { "thumbnail" } else { "pixels" })
            }) {
                eprintln!("{name}, fast={fast}: {}", pixels.note);
            }
            if matches.len() != 1
                || matches[0].relation != Relation::CropOf
                || (matches[0].a, matches[0].b) != (FileId(3), FileId(11))
            {
                failures.push(format!("{name}, fast={fast}: {matches:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn colour_pipeline_changes_preserve_crops_without_embeddings() -> anyhow::Result<()> {
    assert_colour_pipeline_crops(None)
}

#[test]
#[ignore = "requires an already-fetched DINOv2 model; no network requests"]
fn colour_pipeline_changes_preserve_crops_with_fetched_embeddings() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
        .join("Library/Caches/eidolon/models");
    let model =
        eidolon_image::embed::OnnxEmbedder::load(&model_dir)?.context("fetch model first")?;
    assert_colour_pipeline_crops(Some(&model))
}
