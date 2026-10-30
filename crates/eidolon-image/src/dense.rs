//! Bounded RGB template registration for images without usable local keypoints.

use image::{ImageBuffer, Luma, Pixel, Rgb, imageops::FilterType};
use opencv::{core, imgproc, prelude::*};

use crate::features::{PREVIEW, REGISTRATION, Visual};
use crate::geometry;

use super::{interpolate, interpolate_pixels, matched_preview, structure};

pub(super) fn register(
    a: &Visual,
    b: &Visual,
    structural: bool,
    native_only: bool,
) -> anyhow::Result<Option<[f64; 9]>> {
    if native_only
        && (a.width > b.width
            || a.height > b.height
            || (a.width == b.width && a.height == b.height))
    {
        return Ok(None);
    }
    if contrast(a) < 0.08 || contrast(b) < 0.08 || (!structural && !compatible_colors(a, b)) {
        return Ok(None);
    }
    if structural {
        let source = ImageBuffer::<Luma<u8>, _>::from_raw(
            REGISTRATION,
            REGISTRATION,
            a.structure.as_slice(),
        )
        .ok_or_else(|| anyhow::anyhow!("invalid structure preview"))?;
        search(a, b, &source, &b.structure, true, native_only)
    } else {
        let source = ImageBuffer::<Rgb<u8>, _>::from_raw(
            REGISTRATION,
            REGISTRATION,
            a.registration.as_slice(),
        )
        .ok_or_else(|| anyhow::anyhow!("invalid RGB registration preview"))?;
        search(a, b, &source, &b.registration, false, native_only)
    }
}

fn search<P: Pixel<Subpixel = u8> + 'static>(
    a: &Visual,
    b: &Visual,
    source: &ImageBuffer<P, &[u8]>,
    target_bytes: &[u8],
    structural: bool,
    native_only: bool,
) -> anyhow::Result<Option<[f64; 9]>> {
    let channels = i32::from(P::CHANNEL_COUNT);
    let target_flat = core::Mat::from_slice(target_bytes)?;
    let target = target_flat.reshape(channels, REGISTRATION.cast_signed())?;
    let base_x = f64::from(a.width) / f64::from(b.width);
    let base_y = f64::from(a.height) / f64::from(b.height);
    let extent = base_x.max(base_y);
    // Try unchanged source pixel density first; also support resized crops.
    let steps = if native_only { 4..4 } else { 4..21 };
    let scales = std::iter::once(1.0).chain(steps.map(|step| f64::from(step) / (20.0 * extent)));
    let mut best = None;
    let mut best_error = if structural { 0.14 } else { 0.025 };
    let mut cost = core::Mat::default();
    for scale in scales {
        let sx = base_x * scale;
        let sy = base_y * scale;
        if sx > 1.0 || sy > 1.0 || sx.min(sy) < 0.15 || sx * sy >= 0.97 {
            continue;
        }
        let width = (sx * f64::from(REGISTRATION)).round() as u32;
        let height = (sy * f64::from(REGISTRATION)).round() as u32;
        let resized = image::imageops::resize(source, width, height, FilterType::Triangle);
        let flat = core::Mat::from_slice(resized.as_raw())?;
        let matrix = flat.reshape(channels, height.cast_signed())?;
        let matched = (!structural)
            .then(|| matched_preview(&a.preview, [sx, sy]))
            .flatten();
        let preview = matched
            .as_ref()
            .map_or(a.preview.as_slice(), |image| image.as_raw().as_slice());
        let patch_width = width / 3;
        let patch_height = height / 3;
        let mut votes = [core::Point::default(); 9];
        let mut vote_count = 0;
        for row in 0..3 {
            for column in 0..3 {
                let x = column * patch_width;
                let y = row * patch_height;
                let patch = matrix.roi(core::Rect::new(
                    x.cast_signed(),
                    y.cast_signed(),
                    patch_width.cast_signed(),
                    patch_height.cast_signed(),
                ))?;
                let search = target.roi(core::Rect::new(
                    x.cast_signed(),
                    y.cast_signed(),
                    (REGISTRATION - width + patch_width).cast_signed(),
                    (REGISTRATION - height + patch_height).cast_signed(),
                ))?;
                if let Some(point) = template_vote(&search, &patch, &mut cost, structural)? {
                    votes[vote_count] = point;
                    vote_count += 1;
                }
            }
        }
        for point in &votes[..vote_count] {
            // Scale is already fixed by the search; these votes estimate only
            // translation. Collinear patches are therefore not degenerate.
            let support = votes[..vote_count]
                .iter()
                .filter(|other| (other.x - point.x).abs() <= 2 && (other.y - point.y).abs() <= 2)
                .count();
            if support < 3 {
                continue;
            }
            let transform = [
                sx,
                0.0,
                f64::from(point.x) / f64::from(REGISTRATION),
                0.0,
                sy,
                f64::from(point.y) / f64::from(REGISTRATION),
                0.0,
                0.0,
                1.0,
            ];
            let error = residual(a, b, &transform, structural, preview);
            if error < best_error {
                best_error = error;
                best = Some(transform);
            }
        }
        // The exact-density solution is both cheaper and more accurate than a scale sweep.
        if best_error < if structural { 0.08 } else { 0.008 } {
            break;
        }
    }
    let Some(mut transform) = best else {
        return Ok(None);
    };
    let matched = (!structural)
        .then(|| matched_preview(&a.preview, [transform[0], transform[4]]))
        .flatten();
    let preview = matched
        .as_ref()
        .map_or(a.preview.as_slice(), |image| image.as_raw().as_slice());
    best_error = residual(a, b, &transform, structural, preview);
    refine(a, b, &mut transform, &mut best_error, structural, preview);
    Ok((best_error < if structural { 0.08 } else { 0.018 }).then_some(transform))
}

fn template_vote(
    search: &impl core::ToInputArray,
    patch: &impl core::ToInputArray,
    cost: &mut core::Mat,
    structural: bool,
) -> anyhow::Result<Option<core::Point>> {
    imgproc::match_template_def(
        search,
        patch,
        cost,
        if structural {
            imgproc::TM_CCOEFF_NORMED
        } else {
            imgproc::TM_SQDIFF_NORMED
        },
    )?;
    let (mut minimum, mut maximum) = (1.0, 0.0);
    let (mut point, mut maximum_point) = (core::Point::default(), core::Point::default());
    core::min_max_loc(
        cost,
        Some(&mut minimum),
        Some(&mut maximum),
        Some(&mut point),
        Some(&mut maximum_point),
        &core::no_array(),
    )?;
    let error = if structural {
        if maximum >= -minimum {
            point = maximum_point;
            1.0 - maximum
        } else {
            1.0 + minimum
        }
    } else {
        minimum
    };
    Ok((error < 0.08).then_some(point))
}

pub(super) fn refine_crop(a: &Visual, b: &Visual, transform: &[f64; 9]) -> [f64; 9] {
    let [Some(tl), Some(tr), Some(bl), Some(br)] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        .map(|(x, y)| geometry::project(transform, x, y))
    else {
        return *transform;
    };
    let Some(center) = geometry::project(transform, 0.5, 0.5) else {
        return *transform;
    };
    let left = f64::midpoint(tl[0], bl[0]);
    let top = f64::midpoint(tl[1], tr[1]);
    let sx = f64::from(a.width) / f64::from(b.width);
    let sy = f64::from(a.height) / f64::from(b.height);
    // Sparse repeated geometry can give a noisy projective fit. Try its
    // axis-aligned extent and unchanged pixel density, retaining only an
    // improving fit; the caller still verifies every preview pixel and edge.
    let seeds = [
        [
            f64::midpoint(tr[0], br[0]) - left,
            0.0,
            left,
            0.0,
            f64::midpoint(bl[1], br[1]) - top,
            top,
            0.0,
            0.0,
            1.0,
        ],
        [
            sx,
            0.0,
            center[0] - sx / 2.0,
            0.0,
            sy,
            center[1] - sy / 2.0,
            0.0,
            0.0,
            1.0,
        ],
    ];
    let mut best = *transform;
    let matched = matched_preview(&a.preview, [transform[0].abs(), transform[4].abs()]);
    let preview = matched
        .as_ref()
        .map_or(a.preview.as_slice(), |image| image.as_raw().as_slice());
    let mut best_error = residual(a, b, transform, false, preview);
    for mut candidate in seeds {
        if geometry::contained_area(&candidate).is_none() {
            continue;
        }
        let mut error = residual(a, b, &candidate, false, preview);
        refine(a, b, &mut candidate, &mut error, false, preview);
        if error < best_error {
            best_error = error;
            best = candidate;
        }
    }
    best
}

fn compatible_colors(a: &Visual, b: &Visual) -> bool {
    let mut seen = [false; 512];
    let mut palette = [false; 1000];
    for pixel in b.preview.as_chunks::<3>().0.iter().step_by(4) {
        let [r, g, b] = pixel.map(|value| usize::from(value / 32));
        let index = r * 64 + g * 8 + b;
        if seen[index] {
            continue;
        }
        seen[index] = true;
        for dr in 0..3 {
            for dg in 0..3 {
                for db in 0..3 {
                    palette[(r + dr) * 100 + (g + dg) * 10 + b + db] = true;
                }
            }
        }
    }
    let mut missing = 0;
    let mut count = 0;
    for pixel in a.preview.as_chunks::<3>().0.iter().step_by(16) {
        let [r, g, b] = pixel.map(|value| usize::from(value / 32) + 1);
        count += 1;
        missing += usize::from(!palette[r * 100 + g * 10 + b]);
    }
    missing * 10 <= count * 3
}

fn contrast(image: &Visual) -> f64 {
    let mut sums = [0.0; 3];
    let mut squares = [0.0; 3];
    for pixel in image.preview.as_chunks::<3>().0 {
        for channel in 0..3 {
            let value = f64::from(pixel[channel]) / 255.0;
            sums[channel] += value;
            squares[channel] += value * value;
        }
    }
    let count = f64::from(PREVIEW * PREVIEW);
    ((0..3)
        .map(|channel| squares[channel] / count - (sums[channel] / count).powi(2))
        .sum::<f64>()
        / 3.0)
        .sqrt()
}

fn residual(a: &Visual, b: &Visual, transform: &[f64; 9], structural: bool, preview: &[u8]) -> f64 {
    if structural {
        return structure::registration_error(a, b, transform);
    }
    // Coarse 16-square sampling aliases grid artwork and can prefer a wrong scale.
    let mut errors = [1.0_f64; 1024];
    for row in 0..32 {
        for column in 0..32 {
            let x = (column as f64 + 0.5) / 32.0;
            let y = (row as f64 + 0.5) / 32.0;
            let Some(point) = geometry::project(transform, x, y) else {
                return 1.0;
            };
            let (Some(source), Some(target)) =
                (interpolate_pixels(preview, [x, y]), interpolate(b, point))
            else {
                return 1.0;
            };
            errors[row * 32 + column] = (0..3)
                .map(|channel| (source[channel] - target[channel]).abs())
                .sum::<f64>()
                / 3.0;
        }
    }
    // Up to one fifth may be an overlay; the caller still verifies all pixels and edges.
    errors.select_nth_unstable_by(819, f64::total_cmp);
    errors[..819].iter().sum::<f64>() / 819.0
}

fn refine(
    a: &Visual,
    b: &Visual,
    transform: &mut [f64; 9],
    error: &mut f64,
    structural: bool,
    preview: &[u8],
) {
    // Normal resizing preserves the crop's aspect ratio. Independent axis
    // updates can fit repeated edges or JPEG ringing by stretching the scene.
    let aspect = transform[4] / transform[0];
    let indices: &[usize] = if structural {
        &[0, 4, 2, 5]
    } else {
        &[0, 2, 5]
    };
    for step in [2.0, 1.0, 0.5, 0.25, 0.125] {
        let step = step / f64::from(PREVIEW);
        for _ in 0..8 {
            let mut improved = false;
            for &index in indices {
                for direction in [-1.0, 1.0] {
                    let mut candidate = *transform;
                    candidate[index] += direction * step;
                    if !structural && index == 0 {
                        candidate[4] += direction * step * aspect;
                    }
                    if geometry::contained_area(&candidate).is_none() {
                        continue;
                    }
                    let candidate_error = residual(a, b, &candidate, structural, preview);
                    if candidate_error + 1e-7 < *error {
                        *transform = candidate;
                        *error = candidate_error;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }
}
