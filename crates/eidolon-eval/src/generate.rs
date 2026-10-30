//! Seeded generation from desktop pictures, system manuals, licenses and ffmpeg.

/// Offline generation from archived, attributed and hash-verified source manifests.
pub mod licensed;

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, bail, ensure};
use clap::ValueEnum;
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView as _, Rgba, imageops::FilterType};
use serde::Serialize;
use sha2::{Digest as _, Sha256};

use crate::distinct::{Distinctness, TextFingerprint, audit, image_signature, media_conflicts};
use crate::model::{ExpectedPair, PathPair, Source, Truth, TruthFile};

/// Default seed, stable across generator releases within manifest version one.
pub const DEFAULT_SEED: u64 = 0x0e1d_0100_2026;

/// Requested file budget: small is 228 files, medium is 2,280, including design versions and thumbnails.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Scale {
    /// Six originals of each kind, suitable for local CI.
    Small,
    /// Sixty originals of each kind for a broader evaluation.
    Medium,
}

impl Scale {
    fn families(self) -> usize {
        match self {
            Self::Small => 6,
            Self::Medium => 60,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
        }
    }
}

/// Counts after decoding all images, probing all video streams, and reading all text files.
#[derive(Debug, Serialize)]
pub struct Validation {
    /// Total files validated.
    pub files: usize,
    /// Counts by `kind/variant`, including originals.
    pub categories: BTreeMap<String, usize>,
    /// Cross-family leakage checks on all originals and annotated text hard negatives.
    pub distinctness: Distinctness,
}

/// Build a new corpus without overwriting any existing directory contents.
///
/// All inputs are local. Sources are retained before transformation; generation is
/// deterministic for the seed, local source bytes, and installed tool versions.
/// The final manifest is written only after every output has been validated.
///
/// # Errors
/// Fails for a nonempty destination, unavailable sources/tools, failed conversions,
/// or invalid generated media. A partial directory is retained on error.
#[allow(clippy::too_many_lines)] // Source selection and generation order are the reproducibility contract.
pub fn generate(out: &Path, scale: Scale, seed: u64) -> anyhow::Result<(Truth, Validation)> {
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
    let mut truth = Truth {
        version: 1,
        seed,
        root,
        scale: scale.name().into(),
        sources: Vec::new(),
        files: Vec::new(),
        pairs: Vec::new(),
        hard_negatives: Vec::new(),
    };
    let mut random = Random(seed);
    let count = scale.families();
    let desktop = desktop_sources()?;
    ensure!(
        desktop.len() >= count,
        "need {count} distinct desktop pictures; found {}",
        desktop.len()
    );
    let mut image_signatures: Vec<Vec<u8>> = Vec::new();
    for (source_index, source) in desktop.iter().enumerate() {
        let index = image_signatures.len();
        if index == count {
            break;
        }
        let retained = PathBuf::from(format!(
            "sources/desktop_candidate_{source_index:03}.{}",
            source
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("heic")
        ));
        fs::copy(source, truth.root.join(&retained))?;
        truth.sources.push(Source {
            origin: source.display().to_string(),
            sha256: hash_file(&truth.root.join(&retained))?,
            retained: retained.clone(),
        });
        let original = PathBuf::from(format!("media/image/{index:03}_original.jpg"));
        command(
            Command::new("sips")
                .args(["-s", "format", "jpeg", "-Z", "960"])
                .arg(truth.root.join(&retained))
                .arg("--out")
                .arg(truth.root.join(&original)),
        )?;
        // Re-save through one codec to remove source metadata and canonicalize JPEG options.
        let scene = image::open(truth.root.join(&original))?;
        if scene.width() < 4 || scene.height() < 4 {
            continue;
        }
        save_original_window(&truth, index, &scene)?;
        let image = image::open(truth.root.join(&original))?;
        let signature = image_signature(&image);
        if image_signatures
            .iter()
            .any(|previous| media_conflicts(&signature, previous))
        {
            continue;
        }
        image_signatures.push(signature);
        truth.sources.push(Source {
            origin: source.display().to_string(),
            retained: original.clone(),
            sha256: hash_file(&truth.root.join(&original))?,
        });
        truth.sources.push(Source {
            origin: source.display().to_string(),
            retained: scene_path(index),
            sha256: hash_file(&truth.root.join(scene_path(index)))?,
        });
        image_family(&mut truth, index, &original, &mut random)?;
    }
    ensure!(
        image_signatures.len() == count,
        "need {count} distinct decoded desktop pictures; found {}",
        image_signatures.len()
    );
    let text_sources = text_sources(&mut truth, count)?;
    for (index, text) in text_sources.iter().enumerate() {
        text_family(&mut truth, index, text, &mut random)?;
    }
    for index in 0..count {
        video_family(&mut truth, index, &mut random)?;
    }
    let registration = crate::relations::annotate(&mut truth)?;
    annotate_hard_negatives(&mut truth);
    let frames = crate::frame_windows::annotate(&mut truth)?;
    let validation = validate(&truth)?;
    write_new_json(&truth.root.join("registration.json"), &registration)?;
    write_new_json(&truth.root.join("frame-windows.json"), &frames)?;
    fs::write(
        truth.root.join("distinctness.json"),
        serde_json::to_vec_pretty(&validation.distinctness)?,
    )?;
    fs::write(
        truth.root.join("truth.json"),
        serde_json::to_vec_pretty(&truth)?,
    )?;
    Ok((truth, validation))
}

/// Rebuild design relations and original-source negatives without rewriting media.
///
/// Writes corrected truth, distinctness, registration evidence and a refresh receipt
/// into an empty metadata directory. The manifest retains its original media root;
/// still-video clip labels additionally use recorded pan/zoom crop-window overlap.
///
/// # Errors
/// Fails on an occupied destination, invalid media, source leakage, or write errors.
pub fn refresh_truth(mut truth: Truth, out: &Path) -> anyhow::Result<(Truth, Validation)> {
    fs::create_dir_all(out)?;
    ensure!(
        fs::read_dir(out)?.next().is_none(),
        "metadata destination must be empty: {}",
        out.display()
    );
    if truth.root.is_relative() {
        truth.root = fs::canonicalize(&truth.root)?;
    }
    let registration = crate::relations::annotate(&mut truth)?;
    annotate_hard_negatives(&mut truth);
    let frames = crate::frame_windows::annotate(&mut truth)?;
    let validation = validate(&truth)?;
    write_new_json(&out.join("distinctness.json"), &validation.distinctness)?;
    write_new_json(&out.join("registration.json"), &registration)?;
    write_new_json(&out.join("frame-windows.json"), &frames)?;
    write_new_json(
        &out.join("refresh.json"),
        &serde_json::json!({
            "purpose": "annotation-refresh",
            "media_root": truth.root,
            "seed": truth.seed
        }),
    )?;
    write_new_json(&out.join("truth.json"), &truth)?;
    Ok((truth, validation))
}

/// Refresh only parameter-derived still-video clip labels, without reading media.
///
/// Writes truth and crop-window evidence exclusively into an empty metadata directory.
/// Other positive edges, source hashes and source-design registrations are untouched.
///
/// # Errors
/// Fails for malformed provenance, an occupied destination, or metadata write errors.
pub fn refresh_frame_truth(
    mut truth: Truth,
    out: &Path,
) -> anyhow::Result<(Truth, crate::frame_windows::Audit)> {
    fs::create_dir_all(out)?;
    ensure!(
        fs::read_dir(out)?.next().is_none(),
        "metadata destination must be empty: {}",
        out.display()
    );
    let frames = crate::frame_windows::annotate(&mut truth)?;
    write_new_json(&out.join("frame-windows.json"), &frames)?;
    write_new_json(&out.join("truth.json"), &truth)?;
    Ok((truth, frames))
}

fn write_new_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.flush()?;
    Ok(())
}

fn annotate_hard_negatives(truth: &mut Truth) {
    truth.hard_negatives.clear();
    let ancestry = crate::relations::video_ancestry(truth);
    for kind in ["image", "text", "video"] {
        let originals: Vec<_> = truth
            .files
            .iter()
            .filter(|file| file.kind == kind && file.variant == "original")
            .filter(|file| kind != "video" || !ancestry.contains_key(&file.family))
            .collect();
        let uncertain_collection = |file: &TruthFile| {
            truth
                .sources
                .iter()
                .find(|source| source.retained == file.path)
                .and_then(|source| {
                    crate::relations::ambiguous_collection(Path::new(&source.origin))
                })
        };
        for [a, b] in originals.as_chunks::<2>().0 {
            let uncertain = uncertain_collection(a);
            if a.design == b.design || uncertain.is_some() && uncertain == uncertain_collection(b) {
                continue;
            }
            truth.hard_negatives.push(PathPair {
                a: a.path.clone(),
                b: b.path.clone(),
                direct_only: false,
            });
        }
    }
}

/// Decode every image, ffprobe every video, read every text, and verify all file hashes.
///
/// # Errors
/// Returns an error if bytes changed, a decoder fails, or a video lacks dimensions/duration.
pub fn validate(truth: &Truth) -> anyhow::Result<Validation> {
    let mut result = Validation {
        files: 0,
        categories: BTreeMap::new(),
        distinctness: Distinctness::default(),
    };
    for file in &truth.files {
        let path = truth.root.join(&file.path);
        ensure!(
            hash_file(&path)? == file.sha256,
            "hash mismatch: {}",
            path.display()
        );
        match file.kind.as_str() {
            "image" => {
                let image =
                    image::open(&path).with_context(|| format!("decoding {}", path.display()))?;
                ensure!(image.width() > 0 && image.height() > 0, "empty image");
            }
            "video" => {
                let output = command(
                    Command::new("ffprobe")
                        .args([
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=width,height",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "json",
                        ])
                        .arg(&path),
                )?;
                let probe: serde_json::Value = serde_json::from_slice(&output)?;
                ensure!(
                    probe["streams"][0]["width"].as_u64().unwrap_or(0) > 0
                        && probe["streams"][0]["height"].as_u64().unwrap_or(0) > 0
                        && probe["format"]["duration"]
                            .as_str()
                            .unwrap_or("0")
                            .parse::<f64>()?
                            > 0.0,
                    "invalid video: {}",
                    path.display()
                );
            }
            "text" => {
                ensure!(!fs::read_to_string(&path)?.trim().is_empty(), "empty text");
            }
            _ => bail!("unexpected corpus kind {}", file.kind),
        }
        *result
            .categories
            .entry(format!("{}/{}", file.kind, file.variant))
            .or_default() += 1;
        result.files += 1;
    }
    result.distinctness = audit(truth)?;
    Ok(result)
}

fn desktop_sources() -> anyhow::Result<Vec<PathBuf>> {
    let root = Path::new("/System/Library/Desktop Pictures");
    let mut candidates = Vec::new();
    let mut names = HashSet::new();
    // Prefer full-sized installed assets over same-name thumbnail copies.
    for directory in [root.to_path_buf(), root.join(".thumbnails")] {
        if !directory.is_dir() {
            continue;
        }
        let mut entries: Vec<_> = fs::read_dir(directory)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<_, _>>()?;
        entries.sort();
        for path in entries {
            let supported = path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| {
                    ["heic", "jpg", "jpeg", "png"]
                        .iter()
                        .any(|known| ext.eq_ignore_ascii_case(known))
                });
            if supported && names.insert(path.file_stem().unwrap_or_default().to_os_string()) {
                candidates.push(path);
            }
        }
    }
    let priority = [
        "iMac Blue",
        "iMac Green",
        "Big Sur Aerial",
        "Big Sur Coastline",
        "Grid Green Light",
        "Grid Magenta Light",
        "Grid Magenta",
    ];
    candidates.sort_by_key(|path| {
        priority
            .iter()
            .position(|name| path.file_stem().and_then(|stem| stem.to_str()) == Some(name))
            .unwrap_or(priority.len())
    });
    Ok(candidates)
}

fn image_family(
    truth: &mut Truth,
    index: usize,
    original: &Path,
    random: &mut Random,
) -> anyhow::Result<()> {
    let image = image::open(truth.root.join(original))?;
    record(
        truth,
        "image",
        index,
        "original",
        original.to_path_buf(),
        None,
    )?;
    truth
        .files
        .last_mut()
        .context("missing original image")?
        .scene = Some(crate::model::SceneWindow {
        source: scene_path(index).to_string_lossy().into_owned(),
        x: 0,
        y: 0,
        width: image.width(),
        height: image.height(),
    });
    image_encodings(truth, index, original, &image)?;
    image_subtle_versions(truth, index, original, &image)?;
    for (variant, area) in [("crop_80", 0.8_f64), ("crop_60", 0.6), ("crop_40", 0.4)] {
        let cropped = crop(&image, area, random);
        let path = image_path(index, variant, "jpg");
        save_jpeg(&cropped, &truth.root.join(&path), 92)?;
        record(
            truth,
            "image",
            index,
            variant,
            path,
            Some((original, "crop_of")),
        )?;
    }
    let mut marked = image.clone();
    watermark(&mut marked);
    let path = image_path(index, "watermark", "jpg");
    save_jpeg(&marked, &truth.root.join(&path), 92)?;
    record(
        truth,
        "image",
        index,
        "watermark",
        path,
        Some((original, "edited")),
    )?;
    let mut combined = crop(&image, 0.6, random);
    watermark(&mut combined);
    let path = image_path(index, "crop_watermark_q25", "jpg");
    save_jpeg(&combined, &truth.root.join(&path), 25)?;
    record(
        truth,
        "image",
        index,
        "crop_watermark_q25",
        path,
        Some((original, "crop_of")),
    )?;
    image_versions(truth, index, original, &image)
}

fn image_versions(
    truth: &mut Truth,
    index: usize,
    original: &Path,
    image: &DynamicImage,
) -> anyhow::Result<()> {
    let scene = image::open(truth.root.join(scene_path(index)))?;
    let original_window = truth
        .files
        .iter()
        .find(|file| file.path == original)
        .and_then(|file| file.scene.clone())
        .context("missing original scene window")?;
    for variant in [
        "variant_hue",
        "variant_light_dark",
        "variant_relit",
        "variant_reframe",
    ] {
        let mut window = original_window.clone();
        let base = if variant == "variant_reframe" {
            window.x = scene.width() - image.width();
            let non_shared = f64::from(window.x) / f64::from(window.width);
            ensure!(
                non_shared >= 0.25,
                "reframe must expose at least 25% non-shared scene"
            );
            scene.crop_imm(window.x, 0, window.width, window.height)
        } else {
            image.clone()
        };
        let mut transformed = match variant {
            "variant_hue" => base.huerotate(135),
            "variant_light_dark" => light_dark(&base),
            _ => relight(&base, if index.is_multiple_of(2) { 0.35 } else { 2.8 }),
        };
        let path = image_path(index, variant, "jpg");
        save_jpeg(&transformed, &truth.root.join(&path), 92)?;
        let measure = || -> anyhow::Result<f64> {
            Ok(crate::visibility::aligned_mae(
                image,
                Some(&original_window),
                &image::open(truth.root.join(&path))?,
                Some(&window),
            ))
        };
        let mut difference = measure()?;
        if difference < crate::visibility::MIN_VISIBLE {
            transformed = strong_contrast(&base);
            save_jpeg(&transformed, &truth.root.join(&path), 92)?;
            difference = measure()?;
        }
        ensure!(
            difference >= crate::visibility::MIN_VISIBLE,
            "{variant} RGB visibility {difference:.3} is below20/255"
        );
        record(
            truth,
            "image",
            index,
            variant,
            path,
            Some((original, "variant")),
        )?;
        truth
            .files
            .last_mut()
            .context("missing generated version")?
            .scene = Some(window);
        truth
            .pairs
            .last_mut()
            .context("missing generated relation")?
            .visible_rgb_mae = Some(difference);
    }
    let path = image_path(index, "canvas_reframe", "jpg");
    let (w, h) = image.dimensions();
    let mut canvas = image::RgbImage::from_pixel(w, h, image::Rgb([24, 24, 24]));
    image::imageops::overlay(
        &mut canvas,
        &image
            .resize_exact(w * 9 / 10, h * 9 / 10, FilterType::Lanczos3)
            .into_rgb8(),
        i64::from(w / 10),
        i64::from(h / 20),
    );
    save_jpeg(
        &DynamicImage::ImageRgb8(canvas),
        &truth.root.join(&path),
        92,
    )?;
    record(
        truth,
        "image",
        index,
        "canvas_reframe",
        path,
        Some((original, "reencoded")),
    )?;
    truth
        .pairs
        .last_mut()
        .context("missing canvas relation")?
        .accepted_relations
        .push("crop_of".into());
    Ok(())
}

fn scene_path(index: usize) -> PathBuf {
    PathBuf::from(format!("sources/scene_{index:03}.png"))
}

fn save_original_window(truth: &Truth, index: usize, scene: &DynamicImage) -> anyhow::Result<()> {
    fs::create_dir_all(truth.root.join("sources"))?;
    scene.save(truth.root.join(scene_path(index)))?;
    save_jpeg(
        &scene.crop_imm(0, 0, scene.width() * 3 / 4, scene.height()),
        &truth.root.join(image_path(index, "original", "jpg")),
        92,
    )
}

fn relight(image: &DynamicImage, gamma: f32) -> DynamicImage {
    let curve: [u8; 256] =
        std::array::from_fn(|value| ((value as f32 / 255.0).powf(gamma) * 255.0).round() as u8);
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(
        image.width(),
        image.height(),
        |x, y| {
            let pixel = image.get_pixel(x, y).0;
            image::Rgb([pixel[0], pixel[1], pixel[2]].map(|value| curve[usize::from(value)]))
        },
    ))
}

fn light_dark(image: &DynamicImage) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(
        image.width(),
        image.height(),
        |x, y| {
            let pixel = image.get_pixel(x, y).0;
            let rgb = [pixel[0], pixel[1], pixel[2]];
            let offset = 255_i16
                - i16::from(*rgb.iter().min().unwrap_or(&0))
                - i16::from(*rgb.iter().max().unwrap_or(&0));
            image::Rgb(rgb.map(|value| (i16::from(value) + offset).clamp(0, 255) as u8))
        },
    ))
}

fn strong_contrast(image: &DynamicImage) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(
        image.width(),
        image.height(),
        |x, y| {
            let pixel = image.get_pixel(x, y).0;
            image::Rgb(
                [pixel[0], pixel[1], pixel[2]].map(|value| if value < 128 { 255 } else { 0 }),
            )
        },
    ))
}

fn image_subtle_versions(
    truth: &mut Truth,
    index: usize,
    original: &Path,
    image: &DynamicImage,
) -> anyhow::Result<()> {
    for variant in ["subtle_relight", "subtle_hue"] {
        let changed = if variant == "subtle_relight" {
            image.brighten(6)
        } else {
            let hue = image.huerotate(3);
            DynamicImage::ImageRgb8(image::RgbImage::from_fn(
                image.width(),
                image.height(),
                |x, y| {
                    let before = image.get_pixel(x, y).0;
                    let mut after = hue.get_pixel(x, y).0;
                    after[0] = after[0].saturating_add(4);
                    after[2] = after[2].saturating_sub(4);
                    image::Rgb(std::array::from_fn(|channel| {
                        after[channel].clamp(
                            before[channel].saturating_sub(6),
                            before[channel].saturating_add(6),
                        )
                    }))
                },
            ))
        };
        let path = image_path(index, variant, "jpg");
        save_jpeg(&changed, &truth.root.join(&path), 92)?;
        let decoded = image::open(truth.root.join(&path))?;
        let difference = crate::visibility::aligned_mae(image, None, &decoded, None);
        ensure!(
            difference <= 11.0,
            "subtle colour change exceeds 11/255: {difference}"
        );
        record(
            truth,
            "image",
            index,
            variant,
            path,
            Some((original, "reencoded")),
        )?;
        truth
            .pairs
            .last_mut()
            .context("missing subtle relation")?
            .visible_rgb_mae = Some(difference);
    }
    Ok(())
}

fn image_encodings(
    truth: &mut Truth,
    index: usize,
    original: &Path,
    image: &DynamicImage,
) -> anyhow::Result<()> {
    let exact = image_path(index, "exact", "jpg");
    fs::copy(truth.root.join(original), truth.root.join(&exact))?;
    record(
        truth,
        "image",
        index,
        "exact",
        exact,
        Some((original, "identical")),
    )?;
    let resize = image_path(index, "resize_50", "jpg");
    save_jpeg(
        &image.resize_exact(
            (image.width() / 2).max(1),
            (image.height() / 2).max(1),
            FilterType::Lanczos3,
        ),
        &truth.root.join(&resize),
        90,
    )?;
    record(
        truth,
        "image",
        index,
        "resize_50",
        resize,
        Some((original, "reencoded")),
    )?;
    let thumbnail = image_path(index, "thumb_320", "jpg");
    save_jpeg(
        &image.resize(320, 320, FilterType::Lanczos3),
        &truth.root.join(&thumbnail),
        70,
    )?;
    record(
        truth,
        "image",
        index,
        "thumb_320",
        thumbnail,
        Some((original, "reencoded")),
    )?;
    let low = image_path(index, "jpeg_q25", "jpg");
    save_jpeg(image, &truth.root.join(&low), 25)?;
    record(
        truth,
        "image",
        index,
        "jpeg_q25",
        low,
        Some((original, "reencoded")),
    )?;
    let transcode = image_path(index, "png_transcode", "png");
    image.save(truth.root.join(&transcode))?;
    let roundtrip = image_path(index, "png_to_jpeg", "jpg");
    save_jpeg(
        &image::open(truth.root.join(&transcode))?,
        &truth.root.join(&roundtrip),
        85,
    )?;
    record(
        truth,
        "image",
        index,
        "png_transcode",
        transcode,
        Some((original, "same_content")),
    )?;
    record(
        truth,
        "image",
        index,
        "png_to_jpeg",
        roundtrip,
        Some((original, "reencoded")),
    )?;
    Ok(())
}

fn image_path(index: usize, variant: &str, extension: &str) -> PathBuf {
    PathBuf::from(format!("media/image/{index:03}_{variant}.{extension}"))
}

fn save_jpeg(image: &DynamicImage, path: &Path, quality: u8) -> anyhow::Result<()> {
    let mut writer = std::io::BufWriter::new(fs::File::create(path)?);
    JpegEncoder::new_with_quality(&mut writer, quality).encode_image(image)?;
    writer.flush()?;
    Ok(())
}

fn crop(image: &DynamicImage, area: f64, random: &mut Random) -> DynamicImage {
    let (width, height) = image.dimensions();
    let w = ((f64::from(width) * area.sqrt()).round() as u32).max(1);
    let h = ((f64::from(height) * area.sqrt()).round() as u32).max(1);
    let x = random.below(u64::from(width - w + 1)) as u32;
    let y = random.below(u64::from(height - h + 1)) as u32;
    image.crop_imm(x, y, w, h)
}

fn watermark(image: &mut DynamicImage) {
    let mut pixels = image.to_rgba8();
    let (w, h) = pixels.dimensions();
    // Visible magenta-and-white checkerboard logo.
    for y in h * 7 / 10..h * 9 / 10 {
        for x in w / 10..w * 9 / 10 {
            let color = if ((x / (w / 24).max(1)) + (y / (h / 16).max(1))) % 2 == 0 {
                Rgba([255, 0, 160, 255])
            } else {
                Rgba([255, 255, 255, 255])
            };
            pixels.put_pixel(x, y, color);
        }
    }
    *image = DynamicImage::ImageRgba8(pixels);
}

fn text_sources(truth: &mut Truth, count: usize) -> anyhow::Result<Vec<String>> {
    let mut results = Vec::new();
    let mut fingerprints = Vec::new();
    let mut licenses = license_sources()?;
    licenses.sort();
    // Start with two independently checked licenses. Remaining families use manuals;
    // package names and byte hashes alone do not distinguish shared license bodies.
    for (index, path) in licenses.iter().enumerate() {
        if results.len() == 2 {
            break;
        }
        let bytes = fs::read(path)?;
        let retained = PathBuf::from(format!("sources/license_candidate_{index:03}.txt"));
        fs::write(truth.root.join(&retained), &bytes)?;
        truth.sources.push(Source {
            origin: path.display().to_string(),
            retained,
            sha256: digest(&bytes),
        });
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        accept_text(&text.replace("\r\n", "\n"), &mut results, &mut fingerprints);
    }
    ensure!(
        results.len() == 2,
        "not enough distinct local license texts"
    );
    for (index, page) in manual_sources()?.into_iter().enumerate() {
        if results.len() == count {
            break;
        }
        let output = Command::new("man")
            .args(["-P", "cat"])
            .arg(&page)
            .env("LC_ALL", "C")
            .env("MANWIDTH", "80")
            .output()?;
        // Retain command output before normalization, including rejected candidates.
        let retained = PathBuf::from(format!("sources/man_candidate_{index:04}.txt"));
        fs::write(truth.root.join(&retained), &output.stdout)?;
        truth.sources.push(Source {
            origin: format!("man -P cat {} (LC_ALL=C MANWIDTH=80)", page.display()),
            sha256: digest(&output.stdout),
            retained,
        });
        if !output.status.success() {
            continue;
        }
        let clean = clean_man(&String::from_utf8_lossy(&output.stdout));
        accept_text(&clean, &mut results, &mut fingerprints);
    }
    ensure!(
        results.len() == count,
        "need {count} distinct local text sources; found {}",
        results.len()
    );
    Ok(results)
}

fn accept_text(text: &str, results: &mut Vec<String>, fingerprints: &mut Vec<TextFingerprint>) {
    // A 10% excerpt must contain at least 30 source words, not shared short boilerplate.
    if text.split_whitespace().count() < 300 {
        return;
    }
    let fingerprint = TextFingerprint::new(text);
    if fingerprints
        .iter()
        .any(|previous| fingerprint.conflicts(previous))
    {
        return;
    }
    fingerprints.push(fingerprint);
    results.push(text.to_owned());
}

fn manual_sources() -> anyhow::Result<Vec<PathBuf>> {
    let mut result = [
        "printf", "echo", "ls", "find", "cat", "head", "tail", "sort", "uniq", "wc", "cut",
        "paste", "tr", "sed", "awk", "grep", "diff", "cmp", "comm", "join", "xargs", "tee",
        "touch", "mkdir", "rmdir", "cp", "mv", "rm", "ln", "chmod", "chown", "stat", "file", "du",
        "df", "date", "sleep", "env", "pwd", "basename", "dirname", "readlink", "realpath",
        "mktemp", "tar", "gzip", "gunzip", "bzip2", "zip", "unzip", "sh", "bash", "zsh", "curl",
        "ssh", "scp", "rsync", "ps", "kill", "top",
    ]
    .map(PathBuf::from)
    .to_vec();
    for root in [
        "/usr/share/man/man1",
        "/opt/homebrew/share/man/man1",
        "/usr/share/man/man5",
        "/usr/share/man/man8",
        "/opt/homebrew/share/man/man5",
        "/opt/homebrew/share/man/man8",
    ] {
        if !Path::new(root).is_dir() {
            continue;
        }
        let mut pages: Vec<_> = fs::read_dir(root)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<_, _>>()?;
        pages.retain(|path| path.is_file());
        pages.sort();
        result.extend(pages);
    }
    Ok(result)
}

fn license_sources() -> anyhow::Result<Vec<PathBuf>> {
    let root = Path::new("/opt/homebrew/Cellar");
    let mut result = Vec::new();
    for package in fs::read_dir(root).context("reading Homebrew license sources")? {
        let package = package?.path();
        if !package.is_dir() {
            continue;
        }
        for version in fs::read_dir(package)? {
            let version = version?.path();
            if !version.is_dir() {
                continue;
            }
            for file in fs::read_dir(version)? {
                let file = file?.path();
                let name = file
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_ascii_lowercase();
                if file.is_file() && (name.starts_with("license") || name.starts_with("copying")) {
                    result.push(file);
                }
            }
        }
    }
    Ok(result)
}

fn clean_man(text: &str) -> String {
    let mut result = String::new();
    for character in text.chars() {
        if character == '\u{8}' {
            result.pop();
        } else if character != '\u{c}' {
            result.push(character);
        }
    }
    result
}

fn text_family(
    truth: &mut Truth,
    index: usize,
    text: &str,
    random: &mut Random,
) -> anyhow::Result<()> {
    let original = PathBuf::from(format!("media/text/{index:03}_original.txt"));
    fs::write(truth.root.join(&original), text)?;
    record(truth, "text", index, "original", original.clone(), None)?;
    let words: Vec<_> = text.split_whitespace().collect();
    let mut variants = vec![
        ("exact", text.to_owned(), "identical"),
        ("crlf", text.replace('\n', "\r\n"), "same_content"),
        ("rewrapped", wrap(&words, 54), "same_content"),
    ];
    for (name, percent) in [("excerpt_10", 10), ("excerpt_20", 20), ("excerpt_30", 30)] {
        let length = (words.len() * percent / 100).max(1);
        let start = random.below((words.len() - length + 1) as u64) as usize;
        variants.push((name, words[start..start + length].join(" "), "excerpt_of"));
    }
    let edited = words
        .iter()
        .enumerate()
        .map(|(index, word)| if index % 43 == 21 { "revised" } else { word })
        .collect::<Vec<_>>();
    variants.push(("light_edits", wrap(&edited, 72), "edited"));
    for (variant, content, relation) in variants {
        let path = PathBuf::from(format!("media/text/{index:03}_{variant}.txt"));
        fs::write(truth.root.join(&path), content)?;
        record(
            truth,
            "text",
            index,
            variant,
            path,
            Some((&original, relation)),
        )?;
    }
    Ok(())
}

fn wrap(words: &[&str], width: usize) -> String {
    let mut output = String::new();
    let mut column = 0;
    for word in words {
        if column != 0 {
            if column + 1 + word.len() > width {
                output.push('\n');
                column = 0;
            } else {
                output.push(' ');
                column += 1;
            }
        }
        output.push_str(word);
        column += word.len();
    }
    output.push('\n');
    output
}

fn video_family(truth: &mut Truth, index: usize, random: &mut Random) -> anyhow::Result<()> {
    let original = PathBuf::from(format!("media/video/{index:03}_original.mp4"));
    let mut args = ffmpeg();
    let origin;
    if index % 3 == 2 {
        let source = truth.root.join(image_path(index, "original", "jpg"));
        origin = format!(
            "ffmpeg ken-burns source=media/image/{index:03}_original.jpg zoom=1+0.0015*on pan=sin(on/40)*20 duration=80 size=960x540 fps=8 clip_10_start=5 clip_25_start=40"
        );
        args.arg("-i").arg(source).args(["-vf", "scale=960:-1,zoompan=z='1+0.0015*on':x='iw/2-iw/zoom/2+sin(on/40)*20':y='ih/2-ih/zoom/2':d=640:s=960x540:fps=8", "-t", "80"]);
    } else {
        let seed = random.below(2_000_000_000);
        origin = format!(
            "ffmpeg lavfi life size=160x90 rate=8 random_seed={seed} ratio=0.45 death_color=0x102030 life_color=0xE8C070 scale=960x540 duration=80"
        );
        args.args(["-f", "lavfi", "-i", &format!("life=size=160x90:rate=8:random_seed={seed}:ratio=0.45:death_color=0x102030:life_color=0xE8C070,scale=960:540:flags=neighbor"), "-t", "80"]);
    }
    encode(&mut args, &truth.root.join(&original), false)?;
    record(truth, "video", index, "original", original.clone(), None)?;
    truth.sources.push(Source {
        origin,
        retained: original.clone(),
        sha256: hash_file(&truth.root.join(&original))?,
    });
    video_derivatives(
        truth,
        index,
        &original,
        VideoTiming {
            duration: 80.0,
            clip_10_start: 5.0,
            clip_25_start: 40.0,
        },
        random,
    )
}

#[derive(Clone, Copy)]
struct VideoTiming {
    duration: f64,
    clip_10_start: f64,
    clip_25_start: f64,
}

fn video_derivatives(
    truth: &mut Truth,
    index: usize,
    original: &Path,
    timing: VideoTiming,
    random: &mut Random,
) -> anyhow::Result<()> {
    let extension = original
        .extension()
        .and_then(|extension| extension.to_str())
        .context("video extension")?;
    let exact = PathBuf::from(format!("media/video/{index:03}_exact.{extension}"));
    fs::copy(truth.root.join(original), truth.root.join(&exact))?;
    record(
        truth,
        "video",
        index,
        "exact",
        exact,
        Some((original, "identical")),
    )?;
    for (variant, filter, start, duration, low) in [
        (
            "low_bitrate_360p",
            "scale=640:360",
            0.0,
            timing.duration,
            true,
        ),
        ("clip_10s", "null", timing.clip_10_start, 10.0, false),
        (
            "clip_25pct",
            "null",
            timing.clip_25_start,
            timing.duration * 0.25,
            false,
        ),
        (
            "variant_hue",
            "hue=h=135,eq=gamma=0.35",
            0.0,
            timing.duration,
            false,
        ),
        (
            "watermark",
            "drawbox=x=iw/10:y=ih*0.75:w=iw*0.8:h=ih*0.15:color=magenta@0.85:t=fill",
            0.0,
            timing.duration,
            false,
        ),
        (
            "letterboxed",
            "scale=640:300,pad=640:360:0:30:black",
            0.0,
            timing.duration,
            false,
        ),
    ] {
        let path = PathBuf::from(format!("media/video/{index:03}_{variant}.mp4"));
        let mut command = ffmpeg();
        command
            .args(["-ss", &start.to_string()])
            .arg("-i")
            .arg(truth.root.join(original))
            .args([
                "-t",
                &duration.to_string(),
                "-vf",
                &format!("{filter},fps=8,scale=w='min(960,trunc(iw/2)*2)':h=-2"),
            ]);
        encode(&mut command, &truth.root.join(&path), low)?;
        let visibility = if variant == "variant_hue" {
            Some(visible_video_version(&truth.root, original, &path)?)
        } else {
            None
        };
        let relation = if variant.starts_with("clip") {
            "clip_of"
        } else if variant == "variant_hue" {
            "variant"
        } else if variant == "watermark" {
            "edited"
        } else {
            "reencoded"
        };
        record(
            truth,
            "video",
            index,
            variant,
            path,
            Some((original, relation)),
        )?;
        truth
            .pairs
            .last_mut()
            .context("missing video relation")?
            .visible_rgb_mae = visibility;
    }
    video_thumbnails(truth, index, original, timing, random)
}

fn visible_video_version(root: &Path, original: &Path, path: &Path) -> anyhow::Result<f64> {
    let mut difference =
        crate::visibility::video_difference(&root.join(original), &root.join(path))?;
    if difference < crate::visibility::MIN_VISIBLE {
        let mut stronger = ffmpeg_output(true);
        stronger.arg("-i").arg(root.join(original))
            .args(["-vf", "lutrgb=r='if(gte(val,128),0,255)':g='if(gte(val,128),0,255)':b='if(gte(val,128),0,255)'"]);
        encode(&mut stronger, &root.join(path), false)?;
        difference = crate::visibility::video_difference(&root.join(original), &root.join(path))?;
    }
    ensure!(
        difference >= crate::visibility::MIN_VISIBLE,
        "video variant RGB visibility {difference:.3} is below20/255"
    );
    Ok(difference)
}

fn video_thumbnails(
    truth: &mut Truth,
    index: usize,
    original: &Path,
    timing: VideoTiming,
    random: &mut Random,
) -> anyhow::Result<()> {
    let timestamp =
        (8 + random.below(((timing.duration - 2.0) * 8.0).max(1.0) as u64)) as f64 / 8.0;
    let family = format!("video-{index:03}");
    let bytes = command(
        ffmpeg()
            .args(["-ss", &format!("{timestamp:.3}"), "-i"])
            .arg(truth.root.join(original))
            .args(["-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"]),
    )?;
    let frame = image::load_from_memory(&bytes)?;
    for (variant, quality) in [
        ("thumb_frame", 70),
        ("thumb_overlay", 70),
        ("thumb_poster", 85),
    ] {
        let thumbnail = match variant {
            "thumb_frame" => frame.resize_exact(320, 180, FilterType::Lanczos3),
            "thumb_overlay" => {
                let side = frame.width().min(frame.height());
                let mut square = frame
                    .crop_imm(
                        (frame.width() - side) / 2,
                        (frame.height() - side) / 2,
                        side,
                        side,
                    )
                    .resize_exact(320, 320, FilterType::Lanczos3);
                title_bar(&mut square);
                square
            }
            _ => frame.resize(1280, 1280, FilterType::Lanczos3),
        };
        let path = PathBuf::from(format!("media/image/video_{index:03}_{variant}.jpg"));
        save_jpeg(&thumbnail, &truth.root.join(&path), quality)?;
        record(
            truth,
            "image",
            index,
            variant,
            path.clone(),
            Some((original, "frame_of")),
        )?;
        truth
            .files
            .last_mut()
            .context("missing thumbnail")?
            .family
            .clone_from(&family);
        for video in truth
            .files
            .iter()
            .filter(|file| file.kind == "video" && file.family == family)
        {
            let contains_frame = match video.variant.as_str() {
                "exact" | "low_bitrate_360p" | "watermark" | "letterboxed" => true,
                "clip_10s" => {
                    (timing.clip_10_start..timing.clip_10_start + 10.0).contains(&timestamp)
                }
                "clip_25pct" => (timing.clip_25_start
                    ..timing.clip_25_start + timing.duration * 0.25)
                    .contains(&timestamp),
                _ => false,
            };
            if contains_frame {
                truth.pairs.push(ExpectedPair {
                    a: path.clone(),
                    b: video.path.clone(),
                    relation: "frame_of".into(),
                    kind: "image".into(),
                    variant: variant.into(),
                    visible_rgb_mae: None,
                    accepted_relations: Vec::new(),
                });
            }
        }
        truth.sources.push(Source { origin: format!("frame source={} timestamp={timestamp:.3}s variant={variant} jpeg_quality={quality}", original.display()),
            retained: path.clone(), sha256: hash_file(&truth.root.join(path))? });
    }
    Ok(())
}

fn title_bar(image: &mut DynamicImage) {
    let mut pixels = image.to_rgb8();
    for y in 284..320 {
        for x in 0..320 {
            pixels.put_pixel(x, y, image::Rgb([16, 16, 24]));
        }
    }
    for (column, letter) in b"EIDOLON FRAME".iter().enumerate() {
        let glyph = match letter {
            b'E' => [31, 16, 16, 30, 16, 16, 31],
            b'I' => [31, 4, 4, 4, 4, 4, 31],
            b'D' => [30, 17, 17, 17, 17, 17, 30],
            b'O' => [14, 17, 17, 17, 17, 17, 14],
            b'L' => [16, 16, 16, 16, 16, 16, 31],
            b'N' => [17, 25, 25, 21, 19, 19, 17],
            b'F' => [31, 16, 16, 30, 16, 16, 16],
            b'R' => [30, 17, 17, 30, 20, 18, 17],
            b'A' => [14, 17, 17, 31, 17, 17, 17],
            b'M' => [17, 27, 21, 21, 17, 17, 17],
            _ => [0; 7],
        };
        for (row, bits) in glyph.iter().enumerate() {
            for bit in 0..5 {
                if bits & (1 << (4 - bit)) == 0 {
                    continue;
                }
                for dy in 0..3 {
                    for dx in 0..3 {
                        pixels.put_pixel(
                            10 + column as u32 * 18 + bit * 3 + dx,
                            292 + row as u32 * 3 + dy,
                            image::Rgb([255, 255, 255]),
                        );
                    }
                }
            }
        }
    }
    *image = DynamicImage::ImageRgb8(pixels);
}

fn ffmpeg() -> Command {
    ffmpeg_output(false)
}

fn ffmpeg_output(overwrite: bool) -> Command {
    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        if overwrite { "-y" } else { "-n" },
        "-threads",
        "1",
        "-filter_threads",
        "1",
    ]);
    command
}

fn encode(args: &mut Command, path: &Path, low: bool) -> anyhow::Result<()> {
    args.args([
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
    ]);
    if low {
        args.args(["-b:v", "90k", "-maxrate", "90k", "-bufsize", "180k"]);
    } else {
        args.args(["-crf", "20"]);
    }
    command(args.arg(path))?;
    Ok(())
}

fn record(
    truth: &mut Truth,
    kind: &str,
    index: usize,
    variant: &str,
    path: PathBuf,
    expected: Option<(&Path, &str)>,
) -> anyhow::Result<()> {
    let scene = if kind == "image" && crate::visibility::full_frame_variant(kind, variant) {
        expected
            .and_then(|(original, _)| truth.files.iter().find(|file| file.path == original))
            .and_then(|file| file.scene.clone())
    } else {
        None
    };
    truth.files.push(TruthFile {
        sha256: hash_file(&truth.root.join(&path))?,
        path: path.clone(),
        kind: kind.into(),
        variant: variant.into(),
        family: if expected.is_some_and(|(_, relation)| relation == "variant") {
            format!("{kind}-{index:03}/{variant}")
        } else {
            format!("{kind}-{index:03}")
        },
        design: None,
        scene,
    });
    if let Some((original, relation)) = expected {
        let (a, b) = if matches!(relation, "crop_of" | "clip_of" | "excerpt_of" | "frame_of") {
            (path, original.to_path_buf())
        } else {
            (original.to_path_buf(), path)
        };
        truth.pairs.push(ExpectedPair {
            a,
            b,
            relation: relation.into(),
            kind: kind.into(),
            variant: variant.into(),
            visible_rgb_mae: None,
            accepted_relations: Vec::new(),
        });
    }
    Ok(())
}

fn command(command: &mut Command) -> anyhow::Result<Vec<u8>> {
    let output = command
        .output()
        .with_context(|| format!("starting {command:?}"))?;
    ensure!(
        output.status.success(),
        "{command:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output.stdout)
}

fn hash_file(path: &Path) -> anyhow::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Random(u64);
impl Random {
    fn below(&mut self, upper: u64) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (value ^ (value >> 31)) % upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> anyhow::Result<(tempfile::TempDir, Truth)> {
        let directory = tempfile::Builder::new()
            .prefix("eidolon-eval-test-")
            .tempdir_in("/tmp")?;
        for kind in ["image", "text", "video"] {
            fs::create_dir_all(directory.path().join("media").join(kind))?;
        }
        let truth = Truth {
            version: 1,
            seed: DEFAULT_SEED,
            root: directory.path().canonicalize()?,
            scale: "test".into(),
            sources: Vec::new(),
            files: Vec::new(),
            pairs: Vec::new(),
            hard_negatives: Vec::new(),
        };
        Ok((directory, truth))
    }

    #[test]
    fn subtle_low_saturation_colour_changes_remain_duplicate_positives() -> anyhow::Result<()> {
        let (_directory, mut truth) = fixture()?;
        let scene = DynamicImage::ImageRgb8(image::RgbImage::from_fn(192, 96, |x, y| {
            image::Rgb([120 + (x % 8) as u8, 122 + (y % 8) as u8, 125])
        }));
        save_original_window(&truth, 0, &scene)?;
        let original = image_path(0, "original", "jpg");
        image_family(&mut truth, 0, &original, &mut Random(42))?;
        crate::relations::annotate(&mut truth)?;
        let decoded = image::open(truth.root.join(&original))?;
        for variant in ["subtle_relight", "subtle_hue"] {
            let path = image_path(0, variant, "jpg");
            let changed = image::open(truth.root.join(&path))?;
            let difference = crate::visibility::aligned_mae(&decoded, None, &changed, None);
            assert!(
                difference > 0.0 && difference <= 11.0,
                "{variant}: {difference}"
            );
            assert!(truth.pairs.iter().any(|pair| pair.variant == variant
                && pair.relation == "reencoded"
                && pair.a == original
                && pair.b == path));
            assert_ne!(
                hash_file(&truth.root.join(&original))?,
                hash_file(&truth.root.join(path))?
            );
        }
        Ok(())
    }

    #[test]
    fn real_images_preserve_encodings_crop_areas_and_seeded_offsets() -> anyhow::Result<()> {
        let (_directory, mut truth) = fixture()?;
        let pixels = image::RgbImage::from_fn(640, 320, |x, y| {
            image::Rgb([
                (x % 256) as u8,
                (y % 256) as u8,
                ((x * 7 + y * 11) % 256) as u8,
            ])
        });
        let original = image_path(0, "original", "jpg");
        save_original_window(&truth, 0, &DynamicImage::ImageRgb8(pixels))?;
        image_family(&mut truth, 0, &original, &mut Random(42))?;
        assert_eq!(validate(&truth)?.files, 19);
        assert_eq!(truth.pairs.len(), 18);
        assert_eq!(
            fs::read(truth.root.join(&original))?,
            fs::read(truth.root.join(image_path(0, "exact", "jpg")))?
        );
        let decoded = image::open(truth.root.join(&original))?;
        let png = image::open(truth.root.join(image_path(0, "png_transcode", "png")))?;
        assert_eq!(decoded.to_rgb8(), png.to_rgb8());
        let resized = image::open(truth.root.join(image_path(0, "resize_50", "jpg")))?;
        assert_eq!(resized.dimensions(), (240, 160));
        for (name, area) in [("crop_80", 0.8), ("crop_60", 0.6), ("crop_40", 0.4)] {
            let cropped = image::open(truth.root.join(image_path(0, name, "jpg")))?;
            let actual_area = f64::from(cropped.width() * cropped.height()) / (480.0 * 320.0);
            assert!((actual_area - area).abs() < 0.01);
        }
        let marked = image::open(truth.root.join(image_path(0, "watermark", "jpg")))?.to_rgb8();
        let original_pixels = decoded.to_rgb8();
        let changed = marked
            .pixels()
            .zip(original_pixels.pixels())
            .filter(|(a, b)| a.0.iter().zip(b.0).any(|(x, y)| x.abs_diff(y) > 60))
            .count();
        assert!(
            changed > 480 * 320 / 10,
            "watermark must be visibly different"
        );
        let first_hashes: Vec<_> = truth.files.iter().map(|file| file.sha256.clone()).collect();
        truth.files.clear();
        truth.pairs.clear();
        image_family(&mut truth, 0, &original, &mut Random(42))?;
        assert_eq!(
            first_hashes,
            truth
                .files
                .iter()
                .map(|file| file.sha256.clone())
                .collect::<Vec<_>>()
        );
        assert_ne!(
            crop(&decoded, 0.4, &mut Random(1)).to_rgb8(),
            crop(&decoded, 0.4, &mut Random(2)).to_rgb8()
        );
        Ok(())
    }

    #[test]
    fn low_saturation_versions_use_a_visible_decoded_fallback() -> anyhow::Result<()> {
        let (_directory, mut truth) = fixture()?;
        let scene = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            64,
            64,
            image::Rgb([128, 128, 128]),
        ));
        save_original_window(&truth, 0, &scene)?;
        let original = image_path(0, "original", "jpg");
        image_family(&mut truth, 0, &original, &mut Random(42))?;
        let original_file = truth
            .files
            .iter()
            .find(|file| file.path == original)
            .context("original")?;
        let decoded = image::open(truth.root.join(&original))?;
        let versions: Vec<_> = truth
            .files
            .iter()
            .filter(|file| file.variant.starts_with("variant_"))
            .collect();
        assert_eq!(versions.len(), 4);
        for version in versions {
            let pixels = image::open(truth.root.join(&version.path))?;
            assert!(
                crate::visibility::aligned_mae(
                    &decoded,
                    original_file.scene.as_ref(),
                    &pixels,
                    version.scene.as_ref()
                ) >= 20.0,
                "{}",
                version.variant
            );
        }
        Ok(())
    }

    #[test]
    fn invisible_video_change_is_replaced_without_modifying_original() -> anyhow::Result<()> {
        let (_directory, truth) = fixture()?;
        let original = Path::new("media/video/black.mp4");
        let version = Path::new("media/video/version.mp4");
        encode(
            ffmpeg().args(["-f", "lavfi", "-i", "color=c=black:s=64x36:r=1:d=40"]),
            &truth.root.join(original),
            false,
        )?;
        encode(
            ffmpeg().arg("-i").arg(truth.root.join(original)),
            &truth.root.join(version),
            false,
        )?;
        let original_hash = hash_file(&truth.root.join(original))?;
        assert!(
            crate::visibility::video_difference(
                &truth.root.join(original),
                &truth.root.join(version)
            )? < 20.0
        );
        assert!(visible_video_version(&truth.root, original, version)? >= 20.0);
        assert_eq!(hash_file(&truth.root.join(original))?, original_hash);
        Ok(())
    }

    #[test]
    fn decoded_design_versions_are_visibly_changed_and_not_duplicate_families() -> anyhow::Result<()>
    {
        let (_directory, mut truth) = fixture()?;
        let source = DynamicImage::ImageRgb8(image::RgbImage::from_fn(256, 192, |x, y| {
            image::Rgb([x as u8, (y * 2 % 256) as u8, ((x + y) % 256) as u8])
        }));
        let original = image_path(0, "original", "jpg");
        save_original_window(&truth, 0, &source)?;
        image_family(&mut truth, 0, &original, &mut Random(42))?;
        let original_pixels = image::open(truth.root.join(&original))?;
        let original_window = truth
            .files
            .iter()
            .find(|file| file.path == original)
            .and_then(|file| file.scene.as_ref())
            .context("original geometry")?;
        for variant in [
            "variant_hue",
            "variant_light_dark",
            "variant_relit",
            "variant_reframe",
        ] {
            let path = image_path(0, variant, "jpg");
            let decoded = image::open(truth.root.join(&path))?;
            assert_eq!(decoded.dimensions(), original_pixels.dimensions());
            let window = truth
                .files
                .iter()
                .find(|file| file.path == path)
                .and_then(|file| file.scene.as_ref())
                .context("version geometry")?;
            assert!(
                crate::visibility::aligned_mae(
                    &original_pixels,
                    Some(original_window),
                    &decoded,
                    Some(window)
                ) >= 20.0
            );
            if variant == "variant_reframe" {
                assert!(
                    f64::from(window.x.abs_diff(original_window.x)) / f64::from(window.width)
                        >= 0.25
                );
            }
            assert!(
                truth
                    .files
                    .iter()
                    .any(|file| file.path == path && file.family != "image-000")
            );
            assert!(
                truth
                    .pairs
                    .iter()
                    .any(|pair| pair.a == original && pair.b == path && pair.relation == "variant")
            );
        }
        crate::relations::annotate(&mut truth)?;
        assert!(validate(&truth)?.distinctness.expected_variant_pairs > 0);
        Ok(())
    }

    #[test]
    fn real_text_variants_keep_normalized_content_and_contiguous_excerpts() -> anyhow::Result<()> {
        let (_directory, mut truth) = fixture()?;
        let source = (0..1_000)
            .map(|index| format!("distinct-word-{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let source = format!("Manual title\n{source}\nCopyright notice\n");
        text_family(&mut truth, 0, &source, &mut Random(7))?;
        assert_eq!(validate(&truth)?.files, 8);
        let words: Vec<_> = source.split_whitespace().collect();
        for variant in ["exact", "crlf", "rewrapped"] {
            let content =
                fs::read_to_string(truth.root.join(format!("media/text/000_{variant}.txt")))?;
            assert_eq!(content.split_whitespace().collect::<Vec<_>>(), words);
        }
        let normalized = words.join(" ");
        for (variant, percent) in [("excerpt_10", 10), ("excerpt_20", 20), ("excerpt_30", 30)] {
            let content =
                fs::read_to_string(truth.root.join(format!("media/text/000_{variant}.txt")))?;
            assert!(normalized.contains(&content));
            assert_eq!(
                content.split_whitespace().count(),
                words.len() * percent / 100
            );
            let pair = truth
                .pairs
                .iter()
                .find(|pair| pair.variant == variant)
                .context("excerpt annotation")?;
            assert!(pair.a.to_string_lossy().contains(variant));
            assert_eq!(pair.relation, "excerpt_of");
        }
        let edited = fs::read_to_string(truth.root.join("media/text/000_light_edits.txt"))?;
        assert_ne!(edited.split_whitespace().collect::<Vec<_>>(), words);
        assert!(edited.contains("revised"));
        Ok(())
    }

    #[test]
    fn real_video_variants_decode_with_correct_clip_duration_and_resolution() -> anyhow::Result<()>
    {
        let (_directory, mut truth) = fixture()?;
        video_family(&mut truth, 0, &mut Random(17))?;
        assert_eq!(validate(&truth)?.files, 11);
        for (variant, expected_duration) in [("clip_10s", 10.0), ("clip_25pct", 20.0)] {
            let path = truth.root.join(format!("media/video/000_{variant}.mp4"));
            let output = command(
                Command::new("ffprobe")
                    .args([
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=nw=1:nk=1",
                    ])
                    .arg(path),
            )?;
            let duration: f64 = std::str::from_utf8(&output)?.trim().parse()?;
            assert!((duration - expected_duration).abs() < 0.15);
            let pair = truth
                .pairs
                .iter()
                .find(|pair| pair.variant == variant)
                .context("clip annotation")?;
            assert!(pair.a.to_string_lossy().contains(variant));
            assert_eq!(pair.relation, "clip_of");
        }
        let output = command(
            Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=height",
                    "-of",
                    "default=nw=1:nk=1",
                ])
                .arg(truth.root.join("media/video/000_low_bitrate_360p.mp4")),
        )?;
        assert_eq!(std::str::from_utf8(&output)?.trim(), "360");
        assert_eq!(
            fs::read(truth.root.join("media/video/000_original.mp4"))?,
            fs::read(truth.root.join("media/video/000_exact.mp4"))?
        );
        for (variant, dimensions) in [
            ("thumb_frame", (320, 180)),
            ("thumb_overlay", (320, 320)),
            ("thumb_poster", (1280, 720)),
        ] {
            let file = truth
                .files
                .iter()
                .find(|file| file.variant == variant)
                .context("thumbnail file")?;
            let image = image::open(truth.root.join(&file.path))?;
            assert_eq!(image.dimensions(), dimensions);
            let relation = truth
                .pairs
                .iter()
                .find(|pair| pair.variant == variant)
                .context("frame relation")?;
            assert_eq!(relation.a, file.path);
            assert_eq!(relation.b, Path::new("media/video/000_original.mp4"));
            assert_eq!(relation.relation, "frame_of");
            if variant == "thumb_overlay" {
                assert!(
                    image.get_pixel(11, 293).0[..3]
                        .iter()
                        .all(|channel| *channel > 180)
                );
                assert!(
                    image.get_pixel(318, 318).0[..3]
                        .iter()
                        .all(|channel| *channel < 60)
                );
            }
        }
        // Decode every frame, not only container metadata, on actual generated variants.
        for file in &truth.files {
            command(
                Command::new("ffmpeg")
                    .args(["-v", "error", "-nostdin", "-threads", "1", "-i"])
                    .arg(truth.root.join(&file.path))
                    .args(["-f", "null", "-"]),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod distinctness_regressions {
    use super::*;

    #[test]
    fn generated_medium_text_families_do_not_share_fifteen_word_spans() -> anyhow::Result<()> {
        let directory = tempfile::Builder::new()
            .prefix("eidolon-text-families-")
            .tempdir_in("/tmp")?;
        fs::create_dir(directory.path().join("sources"))?;
        let mut truth = Truth {
            version: 1,
            seed: DEFAULT_SEED,
            root: directory.path().to_path_buf(),
            scale: "medium".into(),
            sources: Vec::new(),
            files: Vec::new(),
            pairs: Vec::new(),
            hard_negatives: Vec::new(),
        };
        let sources = text_sources(&mut truth, 60)?;
        let windows: Vec<HashSet<String>> = sources
            .iter()
            .map(|source| {
                let words: Vec<_> = source
                    .split(|character: char| !character.is_alphanumeric())
                    .filter(|word| !word.is_empty())
                    .map(str::to_lowercase)
                    .collect();
                words.windows(15).map(|words| words.join(" ")).collect()
            })
            .collect();
        for a in 0..windows.len() {
            for b in a + 1..windows.len() {
                assert!(
                    windows[a].is_disjoint(&windows[b]),
                    "text families {a:03}/{b:03} share at least fifteen words"
                );
            }
        }
        assert_eq!(sources.len(), 60);
        assert!(
            truth.sources.iter().any(
                |source| source.origin.contains("LICENSE") || source.origin.contains("COPYING")
            )
        );
        Ok(())
    }

    #[test]
    fn video_audit_accepts_different_seeds_and_rejects_duplicate_frames_or_parameters()
    -> anyhow::Result<()> {
        let directory = tempfile::Builder::new()
            .prefix("eidolon-video-families-")
            .tempdir_in("/tmp")?;
        fs::create_dir_all(directory.path().join("media/video"))?;
        fs::create_dir_all(directory.path().join("media/image"))?;
        let mut truth = Truth {
            version: 1,
            seed: DEFAULT_SEED,
            root: directory.path().to_path_buf(),
            scale: "test".into(),
            sources: Vec::new(),
            files: Vec::new(),
            pairs: Vec::new(),
            hard_negatives: Vec::new(),
        };
        let mut random = Random(17);
        video_family(&mut truth, 0, &mut random)?;
        video_family(&mut truth, 1, &mut random)?;
        let result = audit(&truth)?;
        assert_eq!(result.video_families, 2);
        assert_eq!(result.video_pairs_checked, 1);
        let copy = truth
            .files
            .iter_mut()
            .find(|file| file.path == Path::new("media/video/000_exact.mp4"))
            .context("video copy")?;
        copy.variant = "original".into();
        copy.family = "video-duplicate".into();
        let copy_path = copy.path.clone();
        let error =
            audit(&truth).expect_err("byte-identical video must not become a negative family");
        assert!(
            error.to_string().contains("near-identical video originals"),
            "{error}"
        );
        let original_source = truth.sources.first().context("video provenance")?;
        truth.sources.push(Source {
            origin: original_source.origin.clone(),
            retained: copy_path,
            sha256: original_source.sha256.clone(),
        });
        let error =
            audit(&truth).expect_err("same lavfi parameters must not become a negative family");
        assert!(
            error.to_string().contains("duplicate video parameters"),
            "{error}"
        );
        Ok(())
    }
}
