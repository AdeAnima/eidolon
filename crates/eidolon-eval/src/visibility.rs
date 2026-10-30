//! Conservative RGB visibility checks on aligned, decoded media.

use std::path::Path;

use anyhow::{Context as _, ensure};
use image::{DynamicImage, imageops::FilterType};

use crate::model::{SceneWindow, Truth, TruthFile};

pub(crate) const MIN_VISIBLE: f64 = 20.0;

pub(crate) fn full_frame(file: &TruthFile) -> bool {
    full_frame_variant(&file.kind, &file.variant)
}

pub(crate) fn full_frame_variant(kind: &str, variant: &str) -> bool {
    match kind {
        "image" => matches!(
            variant,
            "original"
                | "exact"
                | "resize_50"
                | "thumb_320"
                | "jpeg_q25"
                | "png_transcode"
                | "png_to_jpeg"
                | "subtle_hue"
                | "subtle_relight"
                | "variant_hue"
                | "variant_light_dark"
                | "variant_relit"
                | "variant_reframe"
        ),
        "video" => matches!(
            variant,
            "original" | "exact" | "low_bitrate_360p" | "variant_hue"
        ),
        _ => false,
    }
}

pub(crate) struct Sample {
    frames: Vec<DynamicImage>,
    scene: Option<SceneWindow>,
}

impl Sample {
    pub(crate) fn load(truth: &Truth, file: &TruthFile) -> anyhow::Result<Self> {
        let path = truth.root.join(&file.path);
        let frames = if file.kind == "video" {
            video_frames(&path)?
        } else {
            vec![
                image::open(&path)
                    .with_context(|| format!("loading visibility input {}", path.display()))?
                    .thumbnail(192, 192),
            ]
        };
        Ok(Self {
            frames,
            scene: file.scene.clone(),
        })
    }

    pub(crate) fn difference(&self, other: &Self) -> f64 {
        let count = self.frames.len().min(other.frames.len());
        self.frames
            .iter()
            .zip(&other.frames)
            .map(|(a, b)| aligned_mae(a, self.scene.as_ref(), b, other.scene.as_ref()))
            .sum::<f64>()
            / count as f64
    }
}

pub(crate) fn video_frames(path: &Path) -> anyhow::Result<Vec<DynamicImage>> {
    let bytes = crate::distinct::video_signature(path)?;
    let (frames, remainder) = bytes.as_chunks::<{ 64 * 36 * 3 }>();
    ensure!(
        !frames.is_empty() && remainder.is_empty(),
        "invalid RGB sample payload"
    );
    frames
        .iter()
        .map(|bytes| {
            image::RgbImage::from_raw(64, 36, bytes.to_vec())
                .map(DynamicImage::ImageRgb8)
                .context("invalid decoded RGB frame")
        })
        .collect()
}

pub(crate) fn video_difference(a: &Path, b: &Path) -> anyhow::Result<f64> {
    let a = video_frames(a)?;
    let b = video_frames(b)?;
    ensure!(
        !a.is_empty() && a.len() == b.len(),
        "unequal video visibility samples"
    );
    Ok(a.iter()
        .zip(&b)
        .map(|(a, b)| aligned_mae(a, None, b, None))
        .sum::<f64>()
        / a.len() as f64)
}

pub(crate) fn aligned_mae(
    a: &DynamicImage,
    a_scene: Option<&SceneWindow>,
    b: &DynamicImage,
    b_scene: Option<&SceneWindow>,
) -> f64 {
    let (a, b) = if let (Some(sa), Some(sb)) = (a_scene, b_scene)
        && sa.source == sb.source
    {
        let left = sa.x.max(sb.x);
        let top = sa.y.max(sb.y);
        let right = (sa.x + sa.width).min(sb.x + sb.width);
        let bottom = (sa.y + sa.height).min(sb.y + sb.height);
        if right <= left || bottom <= top {
            return 0.0;
        }
        (
            scene_crop(a, sa, left, top, right, bottom),
            scene_crop(b, sb, left, top, right, bottom),
        )
    } else {
        // Independently supplied design assets can have different centre windows.
        // Align their aspect-preserving common centre, never stretch the extra border.
        let scale = f64::from(a.width().min(b.width())) / f64::from(a.width().max(b.width()));
        let (a, b) = if a.width() > b.width() {
            (
                a.resize_exact(
                    b.width(),
                    (f64::from(a.height()) * scale).round().max(1.0) as u32,
                    FilterType::Triangle,
                ),
                b.clone(),
            )
        } else {
            (
                a.clone(),
                b.resize_exact(
                    a.width(),
                    (f64::from(b.height()) * scale).round().max(1.0) as u32,
                    FilterType::Triangle,
                ),
            )
        };
        let width = a.width().min(b.width());
        let height = a.height().min(b.height());
        (
            a.crop_imm(
                (a.width() - width) / 2,
                (a.height() - height) / 2,
                width,
                height,
            ),
            b.crop_imm(
                (b.width() - width) / 2,
                (b.height() - height) / 2,
                width,
                height,
            ),
        )
    };
    let (width, height) = (a.width().min(b.width()), a.height().min(b.height()));
    let a = a
        .resize_exact(width, height, FilterType::Triangle)
        .into_rgb8();
    let b = b
        .resize_exact(width, height, FilterType::Triangle)
        .into_rgb8();
    a.as_raw()
        .iter()
        .zip(b.as_raw())
        .map(|(a, b)| u64::from(a.abs_diff(*b)))
        .sum::<u64>() as f64
        / (u64::from(width) * u64::from(height) * 3) as f64
}

fn scene_crop(
    image: &DynamicImage,
    scene: &SceneWindow,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
) -> DynamicImage {
    let x = (((f64::from(left - scene.x) / f64::from(scene.width)) * f64::from(image.width()))
        .round() as u32)
        .min(image.width() - 1);
    let y = (((f64::from(top - scene.y) / f64::from(scene.height)) * f64::from(image.height()))
        .round() as u32)
        .min(image.height() - 1);
    let width = (((f64::from(right - left) / f64::from(scene.width)) * f64::from(image.width()))
        .round() as u32)
        .max(1)
        .min(image.width() - x);
    let height = (((f64::from(bottom - top) / f64::from(scene.height)) * f64::from(image.height()))
        .round() as u32)
        .max(1)
        .min(image.height() - y);
    image.crop_imm(x, y, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_real_scene_windows_before_comparing_colour() {
        let scene = DynamicImage::ImageRgb8(image::RgbImage::from_fn(400, 200, |x, y| {
            image::Rgb([(x / 2) as u8, y as u8, ((x + y) / 3) as u8])
        }));
        let left = scene.crop_imm(0, 0, 300, 200);
        let right = scene.crop_imm(100, 0, 300, 200);
        let a = SceneWindow {
            source: "scene".into(),
            x: 0,
            y: 0,
            width: 300,
            height: 200,
        };
        let b = SceneWindow {
            x: 100,
            ..a.clone()
        };
        assert_eq!(aligned_mae(&left, Some(&a), &right, Some(&b)), 0.0);
        assert!(aligned_mae(&left, Some(&a), &right.brighten(60), Some(&b)) >= MIN_VISIBLE);
        assert!(aligned_mae(&left, Some(&a), &right.brighten(5), Some(&b)) < MIN_VISIBLE);
    }
}
