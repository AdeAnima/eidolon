//! Conservative evidence of common 2D composition through similarity registration.
//!
//! The search preserves aspect ratio and fits rotation, uniform scale, and translation.
//! Certification requires both luminance and gradient correlation >=0.99, >=80% mutual
//! coverage, luminance RMS contrast >=8/255, and gradient RMS >=1/255 per pixel.

use image::{DynamicImage, GrayImage, imageops::FilterType};
use serde::Serialize;

/// Required normalized correlation for both luminance and spatial gradients.
pub const MIN_CORRELATION: f64 = 0.99;
/// Minimum mutually covered fraction of the two original scene windows.
pub const MIN_COVERAGE: f64 = 0.8;

/// Per-source evidence and counts for a rule-based annotation refresh.
#[derive(Debug, Serialize)]
pub struct Audit {
    /// Structural threshold applied uniformly, without detector predictions.
    pub minimum_correlation: f64,
    /// Required geometric overlap in both directions.
    pub minimum_coverage: f64,
    /// Every candidate source-original link, including rejected registrations.
    pub source_links: Vec<SourceLink>,
}

/// One source-family pair and the fate of its propagated Variant edges.
#[derive(Debug, Serialize)]
pub struct SourceLink {
    /// First original source family.
    pub a_family: String,
    /// Second original source family.
    pub b_family: String,
    /// Registration of the decoded originals, not of detector output.
    pub proof: Proof,
    /// Visibility-eligible pairs excluded as unproven, split by media kind.
    pub unscored_pairs: std::collections::BTreeMap<String, usize>,
    /// Previously positive pairs actually removed by this refresh, split by kind.
    pub removed_positive_pairs: std::collections::BTreeMap<String, usize>,
}

/// Auditable registration on aspect-preserving, 64-pixel-long-edge thumbnails.
#[derive(Clone, Debug, Serialize)]
pub struct Proof {
    /// Whether the fixed structural and coverage thresholds were met.
    pub accepted: bool,
    /// Minimum absolute luminance/gradient normalized correlation, in 0..1.
    pub correlation: f64,
    /// Smaller of the two covered image fractions, in 0..1.
    pub coverage: f64,
    /// Uniform scale from the first thumbnail's coordinates to the second's.
    pub scale: f64,
    /// Rotation in degrees, around thumbnail centres.
    pub rotation_degrees: f64,
    /// Horizontal displacement in second-thumbnail pixels.
    pub dx: f64,
    /// Vertical displacement in second-thumbnail pixels.
    pub dy: f64,
}

/// Compare decoded originals without treating common colours as geometric evidence.
///
/// A rejected registration means unproven, not unrelated. Search bounds are +/-6
/// degrees (7.5 with refinement), scales 0.75..1.25, and translations +/-15 pixels.
#[must_use]
pub fn compare(a: &DynamicImage, b: &DynamicImage) -> Proof {
    compare_prepared(&Prepared::new(a), &Prepared::new(b))
}

pub(crate) struct Prepared {
    coarse: Plane,
    fine: Plane,
}

impl Prepared {
    pub(crate) fn new(image: &DynamicImage) -> Self {
        let fine = image.resize(64, 64, FilterType::Triangle).to_luma8();
        let coarse = image::imageops::resize(
            &fine,
            (fine.width() / 2).max(1),
            (fine.height() / 2).max(1),
            FilterType::Triangle,
        );
        Self {
            coarse: Plane::new(&image::imageops::blur(&coarse, 0.8)),
            fine: Plane::new(&image::imageops::blur(&fine, 0.8)),
        }
    }
}

pub(crate) fn compare_prepared(a: &Prepared, b: &Prepared) -> Proof {
    let mut best = candidate(&a.coarse, &b.coarse, 1.0, 0.0, 0.0, 0.0);
    for rotation in [-6.0, -3.0, 0.0, 3.0, 6.0] {
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2] {
            for dx in [-6.0, -3.0, 0.0, 3.0, 6.0] {
                for dy in [-6.0, -3.0, 0.0, 3.0, 6.0] {
                    select(
                        &mut best,
                        candidate(&a.coarse, &b.coarse, scale, rotation, dx, dy),
                    );
                }
            }
        }
    }
    let coarse = best;
    let mut best = candidate(
        &a.fine,
        &b.fine,
        coarse.scale,
        coarse.rotation_degrees,
        coarse.dx * 2.0,
        coarse.dy * 2.0,
    );
    for rotation in [-1.5, -0.75, 0.0, 0.75, 1.5] {
        for scale in [-0.05, -0.025, 0.0, 0.025, 0.05] {
            for dx in [-3.0, -1.5, 0.0, 1.5, 3.0] {
                for dy in [-3.0, -1.5, 0.0, 1.5, 3.0] {
                    select(
                        &mut best,
                        candidate(
                            &a.fine,
                            &b.fine,
                            coarse.scale + scale,
                            coarse.rotation_degrees + rotation,
                            coarse.dx * 2.0 + dx,
                            coarse.dy * 2.0 + dy,
                        ),
                    );
                }
            }
        }
    }
    best.accepted = best.correlation >= MIN_CORRELATION && best.coverage >= MIN_COVERAGE;
    best
}

fn select(best: &mut Proof, candidate: Proof) {
    if candidate.coverage >= MIN_COVERAGE
        && (best.coverage < MIN_COVERAGE || candidate.correlation > best.correlation)
    {
        *best = candidate;
    }
}

struct Plane {
    width: usize,
    height: usize,
    // Luminance and its central horizontal/vertical derivatives.
    pixels: Vec<[f64; 3]>,
}

impl Plane {
    fn new(image: &GrayImage) -> Self {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let mut pixels = vec![[0.0; 3]; width * height];
        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let at = |x, y| f64::from(image.as_raw()[y * width + x]);
                pixels[y * width + x] = [
                    at(x, y),
                    (at(x + 1, y) - at(x - 1, y)) / 2.0,
                    (at(x, y + 1) - at(x, y - 1)) / 2.0,
                ];
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    fn sample(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        if x < 1.0 || y < 1.0 || x >= self.width as f64 - 2.0 || y >= self.height as f64 - 2.0 {
            return None;
        }
        let ix = x as usize;
        let iy = y as usize;
        let fx = x - ix as f64;
        let fy = y - iy as f64;
        let top_left = self.pixels[iy * self.width + ix];
        let top_right = self.pixels[iy * self.width + ix + 1];
        let bottom_left = self.pixels[(iy + 1) * self.width + ix];
        let bottom_right = self.pixels[(iy + 1) * self.width + ix + 1];
        Some(std::array::from_fn(|channel| {
            (top_left[channel] * (1.0 - fx) + top_right[channel] * fx) * (1.0 - fy)
                + (bottom_left[channel] * (1.0 - fx) + bottom_right[channel] * fx) * fy
        }))
    }
}

#[allow(clippy::too_many_lines)] // The accumulators belong to one no-allocation registration pass.
fn candidate(a: &Plane, b: &Plane, scale: f64, rotation: f64, dx: f64, dy: f64) -> Proof {
    let mut proof = Proof {
        accepted: false,
        correlation: 0.0,
        coverage: 0.0,
        scale,
        rotation_degrees: rotation,
        dx,
        dy,
    };
    if a.width < 5 || a.height < 5 || b.width < 5 || b.height < 5 {
        return proof;
    }
    let (sin, cos) = rotation.to_radians().sin_cos();
    let mut count = 0.0;
    let (mut sum_a, mut sum_b, mut squares_a, mut squares_b, mut cross_luminance) =
        (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut gradient_a, mut gradient_b, mut cross_gradient) = (0.0, 0.0, 0.0);
    for y in 1..a.height - 1 {
        for x in 1..a.width - 1 {
            let px = x as f64 - (a.width - 1) as f64 / 2.0;
            let py = y as f64 - (a.height - 1) as f64 / 2.0;
            let bx = scale * (cos * px - sin * py) + (b.width - 1) as f64 / 2.0 + dx;
            let by = scale * (sin * px + cos * py) + (b.height - 1) as f64 / 2.0 + dy;
            let Some(second) = b.sample(bx, by) else {
                continue;
            };
            let first = a.pixels[y * a.width + x];
            let gx = cos * second[1] + sin * second[2];
            let gy = -sin * second[1] + cos * second[2];
            count += 1.0;
            sum_a += first[0];
            sum_b += second[0];
            squares_a += first[0] * first[0];
            squares_b += second[0] * second[0];
            cross_luminance += first[0] * second[0];
            gradient_a += first[1] * first[1] + first[2] * first[2];
            gradient_b += gx * gx + gy * gy;
            cross_gradient += first[1] * gx + first[2] * gy;
        }
    }
    if count < 64.0 {
        return proof;
    }
    proof.coverage = (count / ((a.width - 2) * (a.height - 2)) as f64)
        .min(count * scale * scale / ((b.width - 2) * (b.height - 2)) as f64)
        .min(1.0);
    let variance_a = squares_a - sum_a * sum_a / count;
    let variance_b = squares_b - sum_b * sum_b / count;
    if variance_a / count < 64.0
        || variance_b / count < 64.0
        || gradient_a / count < 1.0
        || gradient_b / count < 1.0
    {
        return proof;
    }
    let luminance = (cross_luminance - sum_a * sum_b / count) / (variance_a * variance_b).sqrt();
    let gradient = cross_gradient / (gradient_a * gradient_b).sqrt();
    if luminance * gradient > 0.0 {
        proof.correlation = luminance.abs().min(gradient.abs()).min(1.0);
    }
    proof
}
