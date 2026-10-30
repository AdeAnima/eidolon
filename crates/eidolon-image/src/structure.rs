//! Colour-independent, aligned structure for globally restyled image versions.

use opencv::{core, imgproc, prelude::*};

use crate::features::{ChannelSample, PREVIEW, REGISTRATION, ToneMap, Visual};
use crate::geometry;

use super::{interpolate, pixel};

const CHANNEL_PAIRS: [(usize, usize); 5] = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 1)];

#[derive(Clone, Copy, Default)]
struct Moments {
    count: f64,
    first: f64,
    second: f64,
    first_squared: f64,
    second_squared: f64,
    product: f64,
}

impl Moments {
    fn push(&mut self, first: f64, second: f64) {
        self.count += 1.0;
        self.first += first;
        self.second += second;
        self.first_squared += first * first;
        self.second_squared += second * second;
        self.product += first * second;
    }

    fn correlation(self) -> f64 {
        let variance_a = self.count * self.first_squared - self.first * self.first;
        let variance_b = self.count * self.second_squared - self.second * self.second;
        let denominator = (variance_a.max(0.0) * variance_b.max(0.0)).sqrt();
        if denominator <= f64::EPSILON {
            return 0.0;
        }
        ((self.count * self.product - self.first * self.second) / denominator).clamp(-1.0, 1.0)
    }

    fn contrast(self) -> f64 {
        let count = self.count.max(1.0);
        (self.first_squared / count - (self.first / count).powi(2))
            .max(0.0)
            .sqrt()
    }

    fn shared_contrast(self) -> f64 {
        let count = self.count.max(1.0);
        self.contrast().min(
            (self.second_squared / count - (self.second / count).powi(2))
                .max(0.0)
                .sqrt(),
        )
    }
}

#[derive(Clone, Copy, Default)]
struct Samples {
    pixels: Moments,
    gradients: Moments,
}

impl Samples {
    fn push(&mut self, source: &[[f64; 3]; 5], target: &[[f64; 3]; 5], a: usize, b: usize) {
        self.pixels.push(source[0][a], target[0][b]);
        self.gradients
            .push(source[2][a] - source[1][a], target[2][b] - target[1][b]);
        self.gradients
            .push(source[4][a] - source[3][a], target[4][b] - target[3][b]);
    }

    fn push_cached(&mut self, source: &ChannelSample, target: &ChannelSample, a: usize, b: usize) {
        self.pixels.push(source.center[a], target.center[b]);
        self.gradients
            .push(source.horizontal[a], target.horizontal[b]);
        self.gradients.push(source.vertical[a], target.vertical[b]);
    }
}

#[derive(Debug)]
pub(super) struct Structure {
    pub correlation: f64,
    pub gradients: f64,
    pub tone_normalized: bool,
    contrast: f64,
    gradient_energy: f64,
    spatial_support: f64,
    spatial_regions: u16,
}

impl Structure {
    pub fn supported(&self, registered: bool) -> bool {
        self.globally_supported(registered)
            && self.spatial_support >= 0.50
            && self.spatial_regions.count_ones() >= 6
            && (0..4)
                .filter(|row| self.spatial_regions & (0xF << (row * 4)) != 0)
                .count()
                >= 3
            && (0..4)
                .filter(|column| self.spatial_regions & (0x1111 << column) != 0)
                .count()
                >= 3
    }

    fn globally_supported(&self, registered: bool) -> bool {
        self.informative()
            && self.correlation >= if registered { 0.92 } else { 0.985 }
            && self.gradients >= if registered { 0.84 } else { 0.94 }
    }

    fn informative(&self) -> bool {
        self.contrast >= 0.08 && self.gradient_energy >= 0.01
    }

    pub fn confidence(&self) -> f64 {
        (0.88 + 0.07 * self.correlation + 0.05 * self.gradients).clamp(0.0, 0.995)
    }
}

fn channels(rgb: [f64; 3]) -> [f64; 3] {
    [
        0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2],
        rgb[0].max(rgb[1]).max(rgb[2]),
        rgb[0].min(rgb[1]).min(rgb[2]),
    ]
}

fn tone_map(image: &Visual) -> &ToneMap {
    image.tone_map.get_or_init(|| {
        let mut counts = [[0_u32; 256]; 3];
        for pixel in image.preview.as_chunks::<3>().0 {
            let rgb = pixel.map(|value| f64::from(value) / 255.0);
            for (channel, value) in channels(rgb).into_iter().enumerate() {
                counts[channel][(value * 255.0).round() as usize] += 1;
            }
        }
        let entropy = std::array::from_fn(|channel| {
            counts[channel]
                .iter()
                .filter(|&&count| count != 0)
                .map(|&count| {
                    let probability = f64::from(count) / f64::from(PREVIEW * PREVIEW);
                    -probability * probability.log2()
                })
                .sum()
        });
        let cdf = counts.map(|histogram| {
            let mut before = 0;
            histogram.map(|count| {
                let rank =
                    (f64::from(before) + 0.5 * f64::from(count)) / f64::from(PREVIEW * PREVIEW);
                before += count;
                rank
            })
        });
        ToneMap { cdf, entropy }
    })
}

fn normalized(values: [f64; 3], map: &ToneMap) -> [f64; 3] {
    std::array::from_fn(|channel| {
        let value = (values[channel] * 255.0).clamp(0.0, 255.0);
        let low = value.floor() as usize;
        let high = (low + 1).min(255);
        let fraction = value.fract();
        map.cdf[channel][low] * (1.0 - fraction) + map.cdf[channel][high] * fraction
    })
}

fn target(image: &Visual, h: &[f64; 9], x: usize, y: usize) -> Option<[f64; 3]> {
    let point = geometry::project(
        h,
        (x as f64 + 0.5) / f64::from(PREVIEW),
        (y as f64 + 0.5) / f64::from(PREVIEW),
    )?;
    interpolate(image, point).map(channels)
}

pub(super) fn compare(a: &Visual, b: &Visual, h: &[f64; 9]) -> Structure {
    if *h == geometry::IDENTITY {
        compare_identity(a, b)
    } else {
        compare_transformed(a, b, h)
    }
}

fn compare_identity(a: &Visual, b: &Visual) -> Structure {
    let first = sampled(a);
    let second = sampled(b);
    let a_tone = tone_map(a);
    let b_tone = tone_map(b);
    let mut samples = [Samples::default(); 2 * CHANNEL_PAIRS.len()];
    for (source, target) in first.iter().zip(second) {
        for (mode, sample) in samples.iter_mut().enumerate() {
            let (a, b) = CHANNEL_PAIRS[mode % CHANNEL_PAIRS.len()];
            let tone = mode / CHANNEL_PAIRS.len();
            sample.push_cached(&source[tone], &target[tone], a, b);
        }
    }
    let mut modes = std::array::from_fn(|mode| {
        summarize(
            samples[mode],
            &[Samples::default(); 16],
            mode,
            a_tone,
            b_tone,
        )
    });
    // Global correlation, contrast, and gradient gates precede interior proof.
    let eligible: [_; 2 * CHANNEL_PAIRS.len()] =
        std::array::from_fn(|mode| modes[mode].globally_supported(true));
    if eligible.iter().any(|value| *value) {
        let mut regions = [[Samples::default(); 16]; 2 * CHANNEL_PAIRS.len()];
        let side = (PREVIEW as usize - 4) / 4;
        for (index, (source, target)) in first.iter().zip(second).enumerate() {
            let Some(region) = interior_region(2 + index % side * 4, 2 + index / side * 4) else {
                continue;
            };
            for (mode, regions) in regions
                .iter_mut()
                .enumerate()
                .filter(|(mode, _)| eligible[*mode])
            {
                let (a, b) = CHANNEL_PAIRS[mode % CHANNEL_PAIRS.len()];
                let tone = mode / CHANNEL_PAIRS.len();
                regions[region].push_cached(&source[tone], &target[tone], a, b);
            }
        }
        for (mode, structure) in modes
            .iter_mut()
            .enumerate()
            .filter(|(mode, _)| eligible[*mode])
        {
            *structure = summarize(samples[mode], &regions[mode], mode, a_tone, b_tone);
        }
    }
    strongest(modes)
}

fn sampled(image: &Visual) -> &[[ChannelSample; 2]] {
    image.structural_samples.get_or_init(|| {
        let tone = tone_map(image);
        let side = (PREVIEW as usize - 4) / 4;
        let mut samples = Vec::with_capacity(side * side);
        for y in (2..PREVIEW as usize - 2).step_by(4) {
            for x in (2..PREVIEW as usize - 2).step_by(4) {
                let raw = [
                    channels(pixel(image, x, y)),
                    channels(pixel(image, x - 1, y)),
                    channels(pixel(image, x + 1, y)),
                    channels(pixel(image, x, y - 1)),
                    channels(pixel(image, x, y + 1)),
                ];
                samples.push([
                    channel_sample(&raw),
                    channel_sample(&raw.map(|value| normalized(value, tone))),
                ]);
            }
        }
        samples
    })
}

fn channel_sample(points: &[[f64; 3]; 5]) -> ChannelSample {
    ChannelSample {
        center: points[0],
        horizontal: std::array::from_fn(|channel| points[2][channel] - points[1][channel]),
        vertical: std::array::from_fn(|channel| points[4][channel] - points[3][channel]),
    }
}

fn interior_region(x: usize, y: usize) -> Option<usize> {
    let margin = PREVIEW as usize / 8;
    let span = PREVIEW as usize - 2 * margin;
    (margin..PREVIEW as usize - margin)
        .contains(&x)
        .then_some(y)
        .filter(|y| (margin..PREVIEW as usize - margin).contains(y))
        .map(|y| ((y - margin) * 4 / span) * 4 + (x - margin) * 4 / span)
}

fn compare_transformed(a: &Visual, b: &Visual, h: &[f64; 9]) -> Structure {
    let mut samples = [Samples::default(); 2 * CHANNEL_PAIRS.len()];
    let mut regions = [[Samples::default(); 16]; 2 * CHANNEL_PAIRS.len()];
    let a_tone = tone_map(a);
    let b_tone = tone_map(b);
    for y in (2..PREVIEW as usize - 2).step_by(4) {
        for x in (2..PREVIEW as usize - 2).step_by(4) {
            // Common frames and logos must not dominate the structural proof.
            let region = interior_region(x, y);
            let (Some(center), Some(left), Some(right), Some(top), Some(bottom)) = (
                target(b, h, x, y),
                target(b, h, x - 1, y),
                target(b, h, x + 1, y),
                target(b, h, x, y - 1),
                target(b, h, x, y + 1),
            ) else {
                continue;
            };
            let source = [
                channels(pixel(a, x, y)),
                channels(pixel(a, x - 1, y)),
                channels(pixel(a, x + 1, y)),
                channels(pixel(a, x, y - 1)),
                channels(pixel(a, x, y + 1)),
            ];
            let destination = [center, left, right, top, bottom];
            for normalized_tone in [false, true] {
                let (source, destination) = if normalized_tone {
                    (
                        source.map(|value| normalized(value, a_tone)),
                        destination.map(|value| normalized(value, b_tone)),
                    )
                } else {
                    (source, destination)
                };
                let offset = usize::from(normalized_tone) * CHANNEL_PAIRS.len();
                for (mode, &(first, second)) in CHANNEL_PAIRS.iter().enumerate() {
                    let mode = mode + offset;
                    samples[mode].push(&source, &destination, first, second);
                    if let Some(region) = region {
                        regions[mode][region].push(&source, &destination, first, second);
                    }
                }
            }
        }
    }
    let modes: [Structure; 2 * CHANNEL_PAIRS.len()] =
        std::array::from_fn(|mode| summarize(samples[mode], &regions[mode], mode, a_tone, b_tone));
    strongest(modes)
}

fn strongest(modes: [Structure; 2 * CHANNEL_PAIRS.len()]) -> Structure {
    modes
        .into_iter()
        .max_by(|first, second| {
            first
                .supported(true)
                .cmp(&second.supported(true))
                .then_with(|| first.informative().cmp(&second.informative()))
                .then_with(|| {
                    first
                        .correlation
                        .min(first.gradients)
                        .total_cmp(&second.correlation.min(second.gradients))
                })
        })
        .expect("the luminance mode is always present")
}

fn summarize(
    samples: Samples,
    regions: &[Samples; 16],
    mode: usize,
    a_tone: &ToneMap,
    b_tone: &ToneMap,
) -> Structure {
    let (first, second) = CHANNEL_PAIRS[mode % CHANNEL_PAIRS.len()];
    let normalized = mode >= CHANNEL_PAIRS.len();
    // Equalization cannot turn a flat input into proof: require sixteen
    // effective 8-bit intensity bins in both compared channels.
    let informative =
        !normalized || (a_tone.entropy[first] >= 4.0 && b_tone.entropy[second] >= 4.0);
    let sign = samples.pixels.correlation().signum();
    let mut informative_regions = 0_u32;
    let mut spatial_regions = 0_u16;
    for (index, region) in regions.iter().enumerate() {
        if region.pixels.shared_contrast() >= 0.02 {
            informative_regions += 1;
            if region.pixels.correlation() * sign >= 0.85
                && region.gradients.correlation() * sign >= 0.65
            {
                spatial_regions |= 1 << index;
            }
        }
    }
    Structure {
        correlation: samples.pixels.correlation().abs(),
        gradients: samples.gradients.correlation().abs(),
        tone_normalized: normalized,
        contrast: if informative {
            samples.pixels.contrast()
        } else {
            0.0
        },
        gradient_energy: (samples.gradients.first_squared / samples.gradients.count.max(1.0))
            .sqrt(),
        spatial_support: f64::from(spatial_regions.count_ones())
            / f64::from(informative_regions.max(1)),
        spatial_regions,
    }
}

pub(super) fn register(
    a: &Visual,
    b: &Visual,
    mut full_structure: Option<Structure>,
) -> anyhow::Result<Option<([f64; 9], Structure, f64)>> {
    let a_planes = gradient_planes(a);
    let b_planes = gradient_planes(b);
    let count = (REGISTRATION * REGISTRATION) as usize;
    for (first, second) in CHANNEL_PAIRS {
        let first = core::Mat::from_slice(&a_planes[first * count..(first + 1) * count])?;
        let second = core::Mat::from_slice(&b_planes[second * count..(second + 1) * count])?;
        let first = first.reshape(1, REGISTRATION.cast_signed())?;
        let second = second.reshape(1, REGISTRATION.cast_signed())?;
        let mut response = 0.0;
        let shift =
            imgproc::phase_correlate(&first, &second, &core::no_array(), Some(&mut response))?;
        // A dominant Fourier peak independently proves full-frame geometry.
        if response >= 0.75
            && shift.x.hypot(shift.y) <= 0.5
            && let Some(structure) = full_structure.take()
        {
            return Ok(Some((
                geometry::IDENTITY,
                structure,
                response.clamp(0.0, 1.0),
            )));
        }
        // A lower peak only proposes a translated window. Its actual shared
        // pixels and signed gradients must still prove the complete overlap.
        if response < 0.25 {
            continue;
        }
        let dx = shift.x.round() as i32;
        let dy = shift.y.round() as i32;
        let side = REGISTRATION.cast_signed();
        let width = side - dx.abs();
        let height = side - dy.abs();
        if width < side / 2 || height < side / 2 {
            continue;
        }
        let source = first.roi(core::Rect::new((-dx).max(0), (-dy).max(0), width, height))?;
        let target = second.roi(core::Rect::new(dx.max(0), dy.max(0), width, height))?;
        let residual = imgproc::phase_correlate_iterative_def(&source, &target)?;
        if residual.x.abs().max(residual.y.abs()) > 2.0 {
            continue;
        }
        let transform = [
            1.0,
            0.0,
            (f64::from(dx) + residual.x) / f64::from(REGISTRATION),
            0.0,
            1.0,
            (f64::from(dy) + residual.y) / f64::from(REGISTRATION),
            0.0,
            0.0,
            1.0,
        ];
        if geometry::reframed_windows(&transform) {
            let structure = compare(a, b, &transform);
            if structure.supported(true) {
                return Ok(Some((transform, structure, response.clamp(0.0, 1.0))));
            }
        }
    }
    Ok(None)
}

fn gradient_planes(image: &Visual) -> &[f32] {
    image.phase_gradients.get_or_init(|| {
        let tone = tone_map(image);
        let side = REGISTRATION as usize;
        let count = side * side;
        let mut planes = vec![0.0; 3 * count];
        for y in 0..side {
            for x in 0..side {
                let mut horizontal = [0.0_f64; 3];
                let mut vertical = [0.0_f64; 3];
                for (dx, dy, wx, wy) in [
                    (-1, -1, -1.0, -1.0),
                    (0, -1, 0.0, -2.0),
                    (1, -1, 1.0, -1.0),
                    (-1, 0, -2.0, 0.0),
                    (1, 0, 2.0, 0.0),
                    (-1, 1, -1.0, 1.0),
                    (0, 1, 0.0, 2.0),
                    (1, 1, 1.0, 1.0),
                ] {
                    let x = x.saturating_add_signed(dx).min(side - 1);
                    let y = y.saturating_add_signed(dy).min(side - 1);
                    let index = (y * side + x) * 3;
                    let rgb = std::array::from_fn(|channel| {
                        f64::from(image.registration[index + channel]) / 255.0
                    });
                    for (channel, value) in normalized(channels(rgb), tone).into_iter().enumerate()
                    {
                        horizontal[channel] += wx * value;
                        vertical[channel] += wy * value;
                    }
                }
                for channel in 0..3 {
                    planes[channel * count + y * side + x] =
                        horizontal[channel].hypot(vertical[channel]) as f32;
                }
            }
        }
        planes
    })
}

pub(super) fn registration_error(a: &Visual, b: &Visual, transform: &[f64; 9]) -> f64 {
    let mut moments = [Moments::default(); CHANNEL_PAIRS.len()];
    for row in 0..16 {
        for column in 0..16 {
            let x = (f64::from(column) + 0.5) / 16.0;
            let y = (f64::from(row) + 0.5) / 16.0;
            let Some(point) = geometry::project(transform, x, y) else {
                return 1.0;
            };
            let (Some(first), Some(second)) = (interpolate(a, [x, y]), interpolate(b, point))
            else {
                return 1.0;
            };
            let first = channels(first);
            let second = channels(second);
            for (mode, &(source, target)) in CHANNEL_PAIRS.iter().enumerate() {
                moments[mode].push(first[source], second[target]);
            }
        }
    }
    1.0 - moments
        .iter()
        .map(|moments| moments.correlation().abs())
        .fold(0.0, f64::max)
}
