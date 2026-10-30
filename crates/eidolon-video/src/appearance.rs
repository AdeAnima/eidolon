//! Spatial identity, local overlays, and coherent global color variants.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::frames::{self, Frame};

const SIDE: usize = 32;

type Projection = [[f64; 3]; 64];
const COARSE_VIEWS: usize = 225;
const IMAGE_VIEWS: usize = 16_384;

/// Reuse exact coarse crop samples across queries against one native timeline.
///
/// Image identities remain valid for the cache's lifetime. Source pixels are
/// bounded to one frame; image projections use at most 24 MiB plus map overhead.
pub(crate) struct FrameViews<'a> {
    source: &'a Frame,
    samples: Vec<Projection>,
    valid: [u64; 4],
    images: HashMap<(usize, usize), Box<Projection>>,
}

impl<'a> FrameViews<'a> {
    /// Create a bounded projection cache for a native timeline.
    pub(crate) fn new(source: &'a Frame) -> Self {
        Self {
            source,
            samples: Vec::new(),
            valid: [0; 4],
            images: HashMap::new(),
        }
    }

    fn select(&mut self, source: &'a Frame) {
        if !std::ptr::eq(self.source, source) {
            self.source = source;
            self.valid.fill(0);
        }
    }

    fn projection(&mut self, image: &'a Frame, geometry: Geometry) -> Option<&Projection> {
        let index = coarse_index(geometry.rect)?;
        if geometry.reverse {
            let key = (std::ptr::from_ref(image) as usize, index);
            if self.images.len() >= IMAGE_VIEWS && !self.images.contains_key(&key) {
                return None;
            }
            return Some(
                self.images
                    .entry(key)
                    .or_insert_with(|| Box::new(project(image, geometry))),
            );
        }
        if self.samples.is_empty() {
            self.samples.resize(COARSE_VIEWS, [[0.0; 3]; 64]);
        }
        let bit = 1 << (index % 64);
        if self.valid[index / 64] & bit == 0 {
            self.samples[index] = project(self.source, geometry);
            self.valid[index / 64] |= bit;
        }
        Some(&self.samples[index])
    }
}

fn coarse_index(rect: Rect) -> Option<usize> {
    if [rect.width, rect.height]
        .into_iter()
        .any(|side| !(16..=32).contains(&side) || side % 4 != 0)
        || rect.x < 0
        || rect.y < 0
        || rect.x + rect.width > 32
        || rect.y + rect.height > 32
        || rect.x % 4 != 0
        || rect.y % 4 != 0
    {
        return None;
    }
    let offset = [0, 5, 9, 12, 14];
    let horizontal = offset[((rect.width - 16) / 4) as usize] + rect.x as usize / 4;
    let vertical = offset[((rect.height - 16) / 4) as usize] + rect.y as usize / 4;
    Some(horizontal * 15 + vertical)
}

fn project(frame: &Frame, geometry: Geometry) -> Projection {
    let sampler = FrameSampler::sampled(frame, frame_mapping(geometry, [0.0; 4]), 4);
    std::array::from_fn(|index| sampler.sample(index % 8 * 4 + 2, index / 8 * 4 + 2))
}

// An unresolved photometric comparison returns no vote, rather than "unchanged".
pub(crate) enum ColorChange {
    Stable,
    Variant(f64),
}

#[derive(Clone, Copy)]
struct Preview<'a> {
    gray: &'a [u8],
    rgb: &'a [[u8; 3]],
    hashes: &'a [u64],
}

impl<'a> Preview<'a> {
    fn new(frame: &'a Frame) -> Self {
        Self {
            gray: &frame.thumbnail,
            rgb: &frame.rgb_thumbnail,
            hashes: frame
                .color_hashes
                .as_ref()
                .map_or(&[], |hashes| &hashes[..]),
        }
    }
}

pub(crate) fn color_change(a: &Frame, b: &Frame) -> Option<ColorChange> {
    compare_color(Preview::new(a), Preview::new(b))
}

fn compare_color(a: Preview<'_>, b: Preview<'_>) -> Option<ColorChange> {
    if a.gray.len() != SIDE * SIDE
        || b.gray.len() != SIDE * SIDE
        || a.rgb.len() != SIDE * SIDE
        || b.rgb.len() != SIDE * SIDE
    {
        return None;
    }
    let mut changes = [0.0; 16];
    for (tile, a, b) in rgb_pairs(a, b, u16::MAX) {
        for (a, b) in a.iter().zip(b) {
            changes[tile] += (f64::from(*a) - f64::from(*b)).powi(2);
        }
    }
    if changes.iter().sum::<f64>() < (SIDE * SIDE * 3) as f64 * 144.0 {
        return Some(ColorChange::Stable);
    }
    let mut mask = 0_u16;
    let mut textures = 0_u16;
    let mut qualities = [0.0; 16];
    for (tile, quality) in qualities.iter_mut().enumerate() {
        let (correlation, is_textured) = tile_correlation(a.gray, b.gray, tile);
        if correlation >= 0.9 {
            mask |= 1 << tile;
            *quality = correlation;
            if is_textured {
                textures |= 1 << tile;
            }
        }
    }
    let count = mask.count_ones();
    let change = changes
        .iter()
        .enumerate()
        .filter(|(tile, _)| mask & (1 << tile) != 0)
        .map(|(_, change)| change)
        .sum::<f64>();
    if count >= 8
        && (mask & textures).count_ones() >= 4
        && qualities.iter().sum::<f64>() >= f64::from(count) * 0.95
        && change < f64::from(count) * 64.0 * 3.0 * 144.0
    {
        return Some(ColorChange::Stable);
    }
    // Hue changes can alter luminance structure in otherwise valid pixels.
    // Measure the global change footprint before selecting affine-fit tiles.
    let mut rows = 0_u8;
    let mut columns = 0_u8;
    for (tile, change) in changes.iter().enumerate() {
        if *change >= 64.0 * 3.0 * 144.0 {
            rows |= 1 << (tile / 4);
            columns |= 1 << (tile % 4);
        }
    }
    if rows.count_ones() < 3 || columns.count_ones() < 3 {
        return None;
    }
    // Failure to resolve a transform is not evidence that colors are unchanged:
    // heavy codec noise can hide a genuine global change in part of a sequence.
    directional_variant(a, b, mask, textures, &changes, &qualities)
        .or_else(|| directional_variant(b, a, mask, textures, &changes, &qualities))
        .or_else(|| nonlinear_variant(a, b, &changes))
        .or_else(|| nonlinear_variant(b, a, &changes))
        .map(ColorChange::Variant)
}

fn directional_variant(
    a: Preview<'_>,
    b: Preview<'_>,
    mut mask: u16,
    textures: u16,
    changes: &[f64; 16],
    qualities: &[f64; 16],
) -> Option<f64> {
    // A local overlay may contaminate two otherwise structurally aligned tiles.
    // Never trim enough of the frame to turn a local tint into a global change.
    for _ in 0..3 {
        let count = mask.count_ones();
        if count < 8 || (mask & textures).count_ones() < 4 {
            return None;
        }
        let mut change = 0.0;
        let mut quality = 0.0;
        let mut columns = 0_u8;
        let mut rows = 0_u8;
        for tile in 0..16 {
            if mask & (1 << tile) != 0 {
                change += changes[tile];
                quality += qualities[tile];
                if changes[tile] >= 64.0 * 3.0 * 144.0 {
                    columns |= 1 << (tile % 4);
                    rows |= 1 << (tile / 4);
                }
            }
        }
        // Keep the original structural-fit coverage gate as well as the
        // full-image change-footprint gate above.
        if change < f64::from(count) * 64.0 * 3.0 * 144.0
            || (columns.count_ones() < 3 && rows.count_ones() < 3)
        {
            return None;
        }
        let quality = quality / f64::from(count);
        let errors = affine_error(a, b, mask)?;
        if quality >= 0.95 && errors.iter().sum::<f64>() <= change * 0.1 {
            return Some(quality);
        }
        let worst = (0..16)
            .filter(|tile| mask & (1 << tile) != 0)
            .max_by(|a, b| errors[*a].total_cmp(&errors[*b]))?;
        mask &= !(1 << worst);
    }
    None
}

fn rgb_pairs<'a>(
    a: Preview<'a>,
    b: Preview<'a>,
    mask: u16,
) -> impl Iterator<Item = (usize, &'a [u8; 3], &'a [u8; 3])> {
    a.rgb
        .iter()
        .zip(b.rgb)
        .enumerate()
        .filter_map(move |(index, (a, b))| {
            let tile = (index / (SIDE * 8)) * 4 + (index % SIDE) / 8;
            (mask & (1 << tile) != 0).then_some((tile, a, b))
        })
}

fn affine_error(a: Preview<'_>, b: Preview<'_>, mask: u16) -> Option<[f64; 16]> {
    // Four input coefficients (RGB and bias), with three output channels.
    let mut matrix = [[0.0_f64; 7]; 4];
    for (_, a, b) in rgb_pairs(a, b, mask) {
        let input = [
            f64::from(a[0]) / 255.0,
            f64::from(a[1]) / 255.0,
            f64::from(a[2]) / 255.0,
            1.0,
        ];
        for (row, weight) in matrix.iter_mut().zip(&input) {
            for (entry, value) in row[..4].iter_mut().zip(&input) {
                *entry += weight * value;
            }
            for (entry, value) in row[4..].iter_mut().zip(b) {
                *entry += weight * f64::from(*value) / 255.0;
            }
        }
    }
    solve(&mut matrix)?;
    let mut error = [0.0; 16];
    for (tile, a, b) in rgb_pairs(a, b, mask) {
        let input = [
            f64::from(a[0]) / 255.0,
            f64::from(a[1]) / 255.0,
            f64::from(a[2]) / 255.0,
            1.0,
        ];
        for (channel, value) in b.iter().enumerate() {
            let predicted = matrix
                .iter()
                .zip(&input)
                .map(|(row, input)| row[channel + 4] * input)
                .sum::<f64>();
            error[tile] += (predicted.clamp(0.0, 1.0) * 255.0 - f64::from(*value)).powi(2);
        }
    }
    Some(error)
}

fn solve(matrix: &mut [[f64; 7]; 4]) -> Option<()> {
    // Weak ridge regularization handles palettes with dependent RGB channels.
    for (index, row) in matrix.iter_mut().enumerate().take(3) {
        row[index] += 1e-6;
    }
    for column in 0..4 {
        let pivot = (column..4)
            .max_by(|a, b| {
                matrix[*a][column]
                    .abs()
                    .total_cmp(&matrix[*b][column].abs())
            })
            .unwrap_or(column);
        matrix.swap(column, pivot);
        let scale = matrix[column][column];
        if scale.abs() < 1e-12 {
            return None;
        }
        for entry in &mut matrix[column][column..] {
            *entry /= scale;
        }
        let (before, rest) = matrix.split_at_mut(column);
        let (pivot, after) = rest.split_at_mut(1);
        let pivot = &pivot[0];
        for row in before.iter_mut().chain(after.iter_mut()) {
            let weight = row[column];
            for (entry, value) in row[column..].iter_mut().zip(&pivot[column..]) {
                *entry -= weight * value;
            }
        }
    }
    Some(())
}

struct Tone {
    lift: [f64; 256],
    encode: [u8; 4097],
}

// A bounded family of monotone tone curves, shared by every comparison.
// Searching both directions also covers the inverse (brightening) curves.
static TONES: LazyLock<[Tone; 9]> = LazyLock::new(|| {
    std::array::from_fn(|index| {
        let power = 2.0_f64.powf(-(index as f64) / 4.0);
        Tone {
            lift: std::array::from_fn(|value| (value as f64 / 255.0).powf(power)),
            encode: std::array::from_fn(|value| {
                ((value as f64 / 4096.0).powf(1.0 / power) * 255.0).round() as u8
            }),
        }
    })
});

fn nonlinear_variant(a: Preview<'_>, b: Preview<'_>, changes: &[f64; 16]) -> Option<f64> {
    if !a.hashes.iter().any(|first| {
        b.hashes.iter().any(|second| {
            let distance = ((first ^ second) & (u64::MAX >> 1)).count_ones();
            distance.min(63 - distance) <= 16
        })
    }) {
        return None;
    }
    let mut mean = [0.0; 3];
    for pixel in b.rgb {
        for (channel, value) in mean.iter_mut().enumerate() {
            *value += f64::from(pixel[channel]) / (SIDE * SIDE) as f64;
        }
    }
    let variance: f64 = b
        .rgb
        .iter()
        .map(|pixel| {
            pixel
                .iter()
                .zip(mean)
                .map(|(value, mean)| (f64::from(*value) - mean).powi(2))
                .sum::<f64>()
        })
        .sum();
    if variance < (SIDE * SIDE * 3) as f64 * 16.0 {
        return None;
    }
    let budget = 0.1 * variance.min(changes.iter().sum::<f64>());
    let indices: [usize; 64] =
        std::array::from_fn(|index| (index / 8 * 4 + 2) * SIDE + index % 8 * 4 + 2);
    let inputs: [[f64; 4]; 64] = indices.map(|index| {
        let pixel = a.rgb[index];
        [
            f64::from(pixel[0]) / 255.0,
            f64::from(pixel[1]) / 255.0,
            f64::from(pixel[2]) / 255.0,
            1.0,
        ]
    });
    let mut gram = [[0.0; 7]; 4];
    for input in &inputs {
        for (row, weight) in gram.iter_mut().zip(input) {
            for (entry, value) in row[..4].iter_mut().zip(input) {
                *entry += weight * value;
            }
        }
    }
    let mut prediction = [[0_u8; 3]; SIDE * SIDE];
    'tone: for tone in &*TONES {
        let mut matrix = gram;
        for (index, input) in indices.iter().zip(&inputs) {
            for (row, weight) in matrix.iter_mut().zip(input) {
                for (entry, value) in row[4..].iter_mut().zip(b.rgb[*index]) {
                    *entry += weight * tone.lift[usize::from(value)];
                }
            }
        }
        solve(&mut matrix)?;
        let mut error = 0.0;
        // Fit only 64 dispersed samples, then verify every retained pixel.
        for ((input, target), predicted) in a.rgb.iter().zip(b.rgb).zip(&mut prediction) {
            for (channel, value) in predicted.iter_mut().enumerate() {
                let linear = matrix[3][channel + 4]
                    + (0..3)
                        .map(|index| matrix[index][channel + 4] * f64::from(input[index]) / 255.0)
                        .sum::<f64>();
                *value = tone.encode[(linear.clamp(0.0, 1.0) * 4096.0).round() as usize];
                error += (f64::from(*value) - f64::from(target[channel])).powi(2);
            }
            if error > budget {
                continue 'tone;
            }
        }
        if let Some(quality) = tone_structure(&prediction, b.rgb) {
            return Some(quality);
        }
    }
    None
}

fn tone_structure(prediction: &[[u8; 3]], actual: &[[u8; 3]]) -> Option<f64> {
    let mut regions = [[0.0; 3]; 16];
    for index in 0..SIDE * SIDE {
        let tile = (index / (SIDE * 8)) * 4 + (index % SIDE) / 8;
        for neighbor in [
            index.checked_sub(1).filter(|_| index % SIDE != 0),
            index.checked_sub(SIDE),
        ]
        .into_iter()
        .flatten()
        {
            for (channel, predicted) in prediction[index].iter().enumerate() {
                let predicted = f64::from(*predicted) - f64::from(prediction[neighbor][channel]);
                let observed =
                    f64::from(actual[index][channel]) - f64::from(actual[neighbor][channel]);
                regions[tile][0] += predicted * observed;
                regions[tile][1] += predicted * predicted;
                regions[tile][2] += observed * observed;
            }
        }
    }
    let total = regions.iter().fold([0.0; 3], |mut total, region| {
        for (total, value) in total.iter_mut().zip(region) {
            *total += value;
        }
        total
    });
    let quality = total[0] / (total[1] * total[2]).sqrt().max(f64::EPSILON);
    if quality < 0.9 {
        return None;
    }
    let mut support = 0_u16;
    for (tile, [dot, first, second]) in regions.into_iter().enumerate() {
        if first.min(second) >= 64.0 * 3.0 && dot >= 0.8 * (first * second).sqrt() {
            support |= 1 << tile;
        }
    }
    (support.count_ones() >= 6
        && (0..4)
            .filter(|row| support & (0xF << (row * 4)) != 0)
            .count()
            >= 3
        && (0..4)
            .filter(|column| support & (0x1111 << column) != 0)
            .count()
            >= 3)
        .then_some(quality.min(1.0))
}

pub(crate) fn aligned(a: &Frame, b: &Frame) -> Option<f64> {
    if a.thumbnail.len() != SIDE * SIDE || b.thumbnail.len() != SIDE * SIDE {
        return None;
    }
    let mut matched = 0;
    let mut textured = 0;
    let mut quality = 0.0;
    for tile in 0..16 {
        let remaining = 16 - tile;
        if matched + remaining < 8 || textured + remaining < 4 {
            return None;
        }
        if a.tiles[tile]
            .iter()
            .zip(b.tiles[tile])
            .map(|(a, b)| u32::from(a.abs_diff(b)))
            .sum::<u32>()
            > 60
        {
            continue;
        }
        let (correlation, is_textured) = tile_correlation(&a.thumbnail, &b.thumbnail, tile);
        if correlation >= 0.85 {
            matched += 1;
            textured += usize::from(is_textured);
            quality += correlation;
        }
    }
    (matched >= 8 && textured >= 4).then_some(quality / 16.0)
}

fn tile_correlation(a: &[u8], b: &[u8], tile: usize) -> (f64, bool) {
    let mut sums = [0.0; 5];
    let left = (tile % 4) * 8;
    let top = (tile / 4) * 8;
    for y in top..top + 8 {
        for x in left..left + 8 {
            let a = f64::from(a[y * SIDE + x]);
            let b = f64::from(b[y * SIDE + x]);
            sums[0] += a;
            sums[1] += b;
            sums[2] += a * a;
            sums[3] += b * b;
            sums[4] += a * b;
        }
    }
    let variance_a = sums[2] - sums[0] * sums[0] / 64.0;
    let variance_b = sums[3] - sums[1] * sums[1] / 64.0;
    if variance_a < 256.0 && variance_b < 256.0 {
        (1.0, false)
    } else if variance_a >= 256.0 && variance_b >= 256.0 {
        let correlation = (sums[4] - sums[0] * sums[1] / 64.0) / (variance_a * variance_b).sqrt();
        (correlation.clamp(-1.0, 1.0), true)
    } else {
        (0.0, false)
    }
}

#[derive(Clone, Copy)]
struct Rect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    rect: Rect,
    reverse: bool,
}

impl Geometry {
    pub(crate) fn correlation(self, a: &[u8], b: &[u8]) -> f64 {
        let (full, cropped) = if self.reverse { (b, a) } else { (a, b) };
        if full.len() != SIDE * SIDE || cropped.len() != SIDE * SIDE {
            return -1.0;
        }
        let mut sums = [0.0; 5];
        for y in 0..8 {
            for x in 0..8 {
                let a = sample(full, f64::from(x * 4) + 1.5, f64::from(y * 4) + 1.5);
                let b = sample(
                    cropped,
                    f64::from(self.rect.x)
                        + (f64::from(x) + 0.5) * f64::from(self.rect.width) / 8.0
                        - 0.5,
                    f64::from(self.rect.y)
                        + (f64::from(y) + 0.5) * f64::from(self.rect.height) / 8.0
                        - 0.5,
                );
                sums[0] += a;
                sums[1] += b;
                sums[2] += a * a;
                sums[3] += b * b;
                sums[4] += a * b;
            }
        }
        let a_variance = sums[2] - sums[0] * sums[0] / 64.0;
        let b_variance = sums[3] - sums[1] * sums[1] / 64.0;
        // Nearly flat patches cannot independently identify a scene.
        if a_variance < 256.0 || b_variance < 256.0 {
            return -1.0;
        }
        ((sums[4] - sums[0] * sums[1] / 64.0) / (a_variance * b_variance).sqrt()).clamp(-1.0, 1.0)
    }
}

fn sample(gray: &[u8], x: f64, y: f64) -> f64 {
    let x = x.clamp(0.0, 31.0);
    let y = y.clamp(0.0, 31.0);
    let left = x as usize;
    let top = y as usize;
    let right = (left + 1).min(31);
    let bottom = (top + 1).min(31);
    let dx = x - left as f64;
    let dy = y - top as f64;
    let upper =
        f64::from(gray[top * SIDE + left]) * (1.0 - dx) + f64::from(gray[top * SIDE + right]) * dx;
    let lower = f64::from(gray[bottom * SIDE + left]) * (1.0 - dx)
        + f64::from(gray[bottom * SIDE + right]) * dx;
    upper * (1.0 - dy) + lower * dy
}

pub(crate) fn fit(a: &[u8], b: &[u8]) -> Option<Geometry> {
    search(|geometry, _| geometry.correlation(a, b), 0.90)
}

fn search(mut measure: impl FnMut(Geometry, f64) -> f64, minimum: f64) -> Option<Geometry> {
    let mut best = Geometry {
        rect: Rect {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        },
        reverse: false,
    };
    let mut quality = measure(best, 0.0);
    if quality >= 0.98 {
        return Some(best);
    }
    // One mapping is estimated for the sequence, not a different crop per frame.
    // Search at coarse resolution, then refine the strongest spatial hypothesis.
    for reverse in [false, true] {
        for width in (16..=32).step_by(4) {
            for height in (16..=32).step_by(4) {
                for x in (0..=32 - width).step_by(4) {
                    for y in (0..=32 - height).step_by(4) {
                        let candidate = Geometry {
                            rect: Rect {
                                x,
                                y,
                                width,
                                height,
                            },
                            reverse,
                        };
                        let value = measure(candidate, quality);
                        if value > quality {
                            best = candidate;
                            quality = value;
                        }
                    }
                }
            }
        }
    }
    let coarse = best;
    for width in coarse.rect.width - 2..=coarse.rect.width + 2 {
        for height in coarse.rect.height - 2..=coarse.rect.height + 2 {
            for x in coarse.rect.x - 2..=coarse.rect.x + 2 {
                for y in coarse.rect.y - 2..=coarse.rect.y + 2 {
                    if width < 16
                        || height < 16
                        || x < 0
                        || y < 0
                        || x + width > 32
                        || y + height > 32
                    {
                        continue;
                    }
                    let candidate = Geometry {
                        rect: Rect {
                            x,
                            y,
                            width,
                            height,
                        },
                        reverse: coarse.reverse,
                    };
                    let value = measure(candidate, quality);
                    if value > quality {
                        best = candidate;
                        quality = value;
                    }
                }
            }
        }
    }
    (quality >= minimum).then_some(best)
}

/// Verify cached 32×32 RGB pixels under one aligned or cropped frame mapping.
pub(crate) fn thumbnail_match(a: &Frame, b: &Frame) -> Option<f64> {
    if a.overlay.is_some() {
        // Caption exclusions need the independent native-pixel proof below.
        return None;
    }
    if a.thumbnail.len() != SIDE * SIDE
        || b.thumbnail.len() != SIDE * SIDE
        || a.rgb_thumbnail.len() != SIDE * SIDE
        || b.rgb_thumbnail.len() != SIDE * SIDE
    {
        return None;
    }
    let identity = Geometry {
        rect: Rect {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        },
        reverse: false,
    };
    aligned(a, b)
        .and_then(|_| thumbnail_geometry(identity, a, b))
        .or_else(|| thumbnail_geometry(fit(&a.thumbnail, &b.thumbnail)?, a, b))
}

/// Verify cached 32×32 RGB/texture pixels under an independently fixed crop mapping.
pub(crate) fn thumbnail_geometry(geometry: Geometry, a: &Frame, b: &Frame) -> Option<f64> {
    let (full, cropped) = if geometry.reverse { (b, a) } else { (a, b) };
    let sampler = FrameSampler {
        frame: cropped,
        mapping: frame_mapping(geometry, [0.0; 4]),
        axes: None,
        preview: None,
    };
    let support = pixel_identity(full, &sampler, false)?;
    (support.matched >= 8 && support.textured >= 4).then_some(support.quality / 16.0)
}

/// Verify a native-resolution frame mapping, including its full RGB rectangle.
pub(crate) fn frame_match(a: &Frame, b: &Frame) -> Option<f64> {
    frame_match_with(a, b, None)
}

/// Verify a frame while reusing exact crop samples from a shared native timeline.
pub(crate) fn frame_match_cached<'a>(
    a: &'a Frame,
    b: &'a Frame,
    views: &mut FrameViews<'a>,
) -> Option<f64> {
    views.select(b);
    frame_match_with(a, b, Some(views))
}

fn frame_match_with<'a>(
    a: &'a Frame,
    b: &'a Frame,
    mut views: Option<&mut FrameViews<'a>>,
) -> Option<f64> {
    if a.pixels.is_none() || b.pixels.is_none() {
        return None;
    }
    if a.overlay.is_some() {
        return masked_frame_match(a, b);
    }
    if let Some(quality) = aligned(a, b) {
        let geometry = Geometry {
            rect: Rect {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
            },
            reverse: false,
        };
        if frame_identity(geometry, [0.0; 4], a, b).is_some() {
            return Some(quality);
        }
    }
    if a.thumbnail.len() != SIDE * SIDE
        || b.thumbnail.len() != SIDE * SIDE
        || a.rgb_thumbnail.len() != SIDE * SIDE
        || b.rgb_thumbnail.len() != SIDE * SIDE
    {
        return None;
    }
    for reverse in [false, true] {
        for side in (16..=32).rev().step_by(2) {
            if let Some(geometry) = centered_frame_geometry(a, b, reverse, side)
                && coarse_support(geometry, a, b, 0.0, views.as_deref_mut()).rank() >= 0.35
                && let Some(quality) = frame_geometry_match(geometry, a, b)
            {
                return Some(quality);
            }
        }
    }
    let geometry = search(
        |geometry, minimum| coarse_support(geometry, a, b, minimum, views.as_deref_mut()).rank(),
        0.35,
    )?;
    frame_geometry_match(geometry, a, b)
}

/// Rank a caption thumbnail without comparing its occluded hashes or pixels.
pub(crate) fn overlay_candidate(image: &Frame, frame: &Frame, aspect: f64) -> Option<f64> {
    let ratio = image.pixels.as_ref()?.aspect / aspect;
    let width = (32.0 * ratio.min(1.0)).round() as i32;
    let height = (32.0 / ratio.max(1.0)).round() as i32;
    let mut best = None;
    for rect in [
        Rect {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        },
        Rect {
            x: (32 - width) / 2,
            y: (32 - height) / 2,
            width,
            height,
        },
    ] {
        if rect.width < 16 || rect.height < 16 {
            continue;
        }
        let fit = masked_support::<4>(
            Geometry {
                rect,
                reverse: false,
            },
            [0.0; 4],
            image,
            frame,
        );
        if fit.correlation() >= 0.5 {
            let rank = fit
                .correlation()
                .midpoint((1.0 - fit.error / (fit.count as f64 * 3.0 * 24.0_f64.powi(2))).max(0.0));
            if rank >= 0.45 && best.is_none_or(|previous| rank > previous) {
                best = Some(rank);
            }
        }
    }
    best
}

/// Sparse source frames may need a translated or scaled crop before native refinement.
pub(crate) fn overlay_remote_candidate(image: &Frame, frame: &Frame, aspect: f64) -> Option<f64> {
    overlay_candidate(image, frame, aspect).or_else(|| {
        let rank = |geometry| {
            let fit = masked_support::<4>(geometry, [0.0; 4], image, frame);
            if fit.correlation() < 0.5 {
                return 0.0;
            }
            fit.correlation()
                .midpoint((1.0 - fit.error / (fit.count as f64 * 3.0 * 24.0_f64.powi(2))).max(0.0))
        };
        search(|geometry, _| rank(geometry), 0.45).map(rank)
    })
}

struct MaskedSupport {
    spatial: FrameSupport,
    sums: [f64; 5],
    count: usize,
    error: f64,
    outliers: usize,
}

impl MaskedSupport {
    fn correlation(&self) -> f64 {
        let count = self.count as f64;
        if count == 0.0 {
            return -1.0;
        }
        let [a, b, aa, bb, ab] = self.sums;
        let va = aa - a * a / count;
        let vb = bb - b * b / count;
        if va < count * 4.0 || vb < count * 4.0 {
            return -1.0;
        }
        ((ab - a * b / count) / (va * vb).sqrt()).clamp(-1.0, 1.0)
    }

    fn proven(&self) -> bool {
        self.count * 10 >= SIDE * SIDE * 7
            && self.spatial.matched >= 8
            && self.spatial.textured >= 4
            && self.correlation() >= 0.97
            && self.error < self.count as f64 * 3.0 * 144.0
            && self.outliers <= self.count / 32
    }
}

fn masked_support<const STRIDE: usize>(
    geometry: Geometry,
    offsets: [f64; 4],
    image: &Frame,
    frame: &Frame,
) -> MaskedSupport {
    let mapping = frame_mapping(geometry, offsets);
    let identity = [0.0, 0.0, 1.0, 1.0];
    let query = FrameSampler::sampled(
        image,
        if geometry.reverse { mapping } else { identity },
        STRIDE,
    );
    let source = FrameSampler::sampled(
        frame,
        if geometry.reverse { identity } else { mapping },
        STRIDE,
    );
    let source_mask = frame.source_overlay();
    let mut result = MaskedSupport {
        spatial: FrameSupport {
            matched: 0,
            textured: 0,
            quality: 0.0,
            identity: 0.0,
        },
        sums: [0.0; 5],
        count: 0,
        error: 0.0,
        outliers: 0,
    };
    let samples = (8 / STRIDE) * (8 / STRIDE);
    for tile in 0..16 {
        let mut sums = [0.0; 5];
        let mut count = 0;
        let mut error = 0.0;
        for y in (tile / 4 * 8 + STRIDE / 2..tile / 4 * 8 + 8).step_by(STRIDE) {
            if image.overlay.is_some_and(|mask| {
                !mask.valid(
                    query.mapping[1] + y as f64 * query.mapping[3],
                    query.mapping[3],
                )
            }) || source_mask.is_some_and(|mask| {
                !mask.valid(
                    source.mapping[1] + y as f64 * source.mapping[3],
                    source.mapping[3],
                )
            }) {
                continue;
            }
            for x in (tile % 4 * 8 + STRIDE / 2..tile % 4 * 8 + 8).step_by(STRIDE) {
                let a = query.sample(x, y);
                let b = source.sample(x, y);
                let gray = |p: [f64; 3]| p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114;
                let ga = gray(a);
                let gb = gray(b);
                for (sum, value) in sums.iter_mut().zip([ga, gb, ga * ga, gb * gb, ga * gb]) {
                    *sum += value;
                }
                let difference = a
                    .into_iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                error += difference;
                result.outliers += usize::from(difference > 3.0 * 24.0_f64.powi(2));
                count += 1;
            }
        }
        for (sum, tile_sum) in result.sums.iter_mut().zip(sums) {
            *sum += tile_sum;
        }
        result.error += error;
        result.count += count;
        if count * 2 < samples {
            continue;
        }
        let n = count as f64;
        result.spatial.identity += (1.0 - error / (n * 3.0 * 24.0_f64.powi(2))).max(0.0);
        if error >= n * 3.0 * 144.0 {
            continue;
        }
        let va = sums[2] - sums[0] * sums[0] / n;
        let vb = sums[3] - sums[1] * sums[1] / n;
        let textured = va >= n * 4.0 && vb >= n * 4.0;
        let correlation = if textured {
            (sums[4] - sums[0] * sums[1] / n) / (va * vb).sqrt()
        } else if va < n * 4.0 && vb < n * 4.0 {
            1.0
        } else {
            -1.0
        };
        if correlation >= 0.9 {
            result.spatial.matched += 1;
            result.spatial.textured += usize::from(textured);
            result.spatial.quality += correlation.min(1.0);
        }
    }
    result
}

fn masked_frame_match(image: &Frame, frame: &Frame) -> Option<f64> {
    let prove = |geometry| {
        let fit = masked_support::<1>(geometry, [0.0; 4], image, frame);
        if fit.proven() {
            return Some(fit.correlation());
        }
        let mut offsets = [0.0; 4];
        let mut rank = masked_support::<2>(geometry, offsets, image, frame)
            .spatial
            .rank();
        for step in [0.5, 0.25, 0.125, 0.0625, 0.03125] {
            for dimension in 0..4 {
                for direction in [-1.0, 1.0] {
                    let mut candidate = offsets;
                    let delta = step * direction;
                    candidate[dimension] += delta;
                    if dimension < 2 {
                        candidate[dimension + 2] -= delta;
                    }
                    let [x, y, width, height] = frame_mapping(geometry, candidate);
                    if x < 0.0
                        || y < 0.0
                        || width < 0.5
                        || height < 0.5
                        || x + width * 32.0 > 32.0
                        || y + height * 32.0 > 32.0
                    {
                        continue;
                    }
                    let fit = masked_support::<2>(geometry, candidate, image, frame);
                    let next = fit.spatial.rank();
                    if next > rank {
                        rank = next;
                        offsets = candidate;
                    }
                }
            }
        }
        let fit = masked_support::<1>(geometry, offsets, image, frame);
        fit.proven().then(|| fit.correlation())
    };
    for reverse in [false, true] {
        for side in (16..=32).rev().step_by(2) {
            if let Some(geometry) = centered_frame_geometry(image, frame, reverse, side)
                && masked_support::<4>(geometry, [0.0; 4], image, frame)
                    .spatial
                    .rank()
                    >= 0.35
                && let Some(quality) = prove(geometry)
            {
                return Some(quality);
            }
        }
    }
    let geometry = search(
        |geometry, _| {
            masked_support::<4>(geometry, [0.0; 4], image, frame)
                .spatial
                .rank()
        },
        0.35,
    )?;
    prove(geometry)
}

fn centered_frame_geometry(a: &Frame, b: &Frame, reverse: bool, side: i32) -> Option<Geometry> {
    let (full, cropped) = if reverse { (b, a) } else { (a, b) };
    let ratio = full.pixels.as_ref()?.aspect / cropped.pixels.as_ref()?.aspect;
    let width = (f64::from(side) * ratio.min(1.0)).round() as i32;
    let height = (f64::from(side) / ratio.max(1.0)).round() as i32;
    (width >= 16 && height >= 16).then_some(Geometry {
        rect: Rect {
            x: (32 - width) / 2,
            y: (32 - height) / 2,
            width,
            height,
        },
        reverse,
    })
}

fn frame_geometry_match(geometry: Geometry, a: &Frame, b: &Frame) -> Option<f64> {
    if !possible_frame_identity(geometry, a, b) {
        return None;
    }
    if let Some(support) = frame_identity(geometry, [0.0; 4], a, b)
        && support.matched >= 8
        && support.textured >= 4
    {
        return Some(support.quality / 16.0);
    }
    let (support, _) = refine_frame_geometry::<2>(geometry, a, b);
    if support.matched < 8 || support.textured < 4 {
        return None;
    }
    let (_, offsets) = refine_frame_geometry::<1>(geometry, a, b);
    let support = frame_identity(geometry, offsets, a, b)?;
    (support.matched >= 8 && support.textured >= 4).then_some(support.quality / 16.0)
}

fn possible_frame_identity(geometry: Geometry, a: &Frame, b: &Frame) -> bool {
    let (full, cropped) = if geometry.reverse { (b, a) } else { (a, b) };
    let Some(pixels) = &cropped.pixels else {
        return false;
    };
    let [x, y, width, height] = frame_mapping(geometry, [0.0; 4]);
    let (left, top, right, bottom) = pixels.bounds;
    let horizontal = uncertain_ranges(left, right, x, width);
    let vertical = uncertain_ranges(top, bottom, y, height);
    let full = FrameSampler::new(full, [0.0, 0.0, 1.0, 1.0]);
    let mut impossible = 0;
    for (y, rows) in vertical.iter().enumerate() {
        for (x, columns) in horizontal.iter().enumerate() {
            let [minimum, maximum] = pixels.range(rows.clone(), columns.clone());
            let difference = full
                .sample(x, y)
                .iter()
                .enumerate()
                .map(|(channel, value)| {
                    (value - value.clamp(f64::from(minimum[channel]), f64::from(maximum[channel])))
                        .powi(2)
                })
                .sum::<f64>();
            impossible += usize::from(difference > 3.0 * 24.0_f64.powi(2) + 1e-6);
            if impossible > SIDE * SIDE / 4 + SIDE * SIDE / 32 {
                return false;
            }
        }
    }
    true
}

fn uncertain_ranges(
    start: u32,
    end: u32,
    origin: f64,
    scale: f64,
) -> [std::ops::Range<usize>; SIDE] {
    // All refinement steps sum to less than one thumbnail pixel per edge.
    // Native min/max envelopes include every possible subpixel crop, so this
    // rejection cannot discard a rectangle that the complete pixel proof accepts.
    let size = f64::from(end - start) / SIDE as f64;
    std::array::from_fn(|index| {
        let first = f64::from(start) + (origin + index as f64 * scale - 1.0).max(0.0) * size;
        let last = (f64::from(start) + (origin + (index + 1) as f64 * scale + 1.0) * size)
            .min(f64::from(end));
        first.floor() as usize / 3..(last.ceil() as usize).div_ceil(3).min(SIDE)
    })
}

struct FrameSupport {
    matched: usize,
    textured: usize,
    quality: f64,
    identity: f64,
}

impl FrameSupport {
    fn rank(&self) -> f64 {
        let spatial =
            (self.matched as f64 * 0.8 + self.textured as f64 * 0.1 + self.quality * 0.1) / 16.0;
        spatial.midpoint(self.identity / 16.0)
    }
}

fn refine_frame_geometry<const STRIDE: usize>(
    geometry: Geometry,
    a: &Frame,
    b: &Frame,
) -> (FrameSupport, [f64; 4]) {
    let mut offsets = [0.0; 4];
    let mut best = frame_support::<STRIDE>(geometry, offsets, a, b, 0.0);
    // A thumbnail pixel can cover many source pixels. Integer thumbnail crops
    // are insufficient: refine the spatial mapping, not the acceptance gate.
    for step in [0.5, 0.25, 0.125, 0.0625, 0.03125] {
        for dimension in 0..4 {
            for direction in [-1.0, 1.0] {
                let mut candidate = offsets;
                let delta = step * direction;
                candidate[dimension] += delta;
                if dimension < 2 {
                    // Move one edge without also displacing its opposite edge.
                    candidate[dimension + 2] -= delta;
                }
                let x = f64::from(geometry.rect.x) + candidate[0];
                let y = f64::from(geometry.rect.y) + candidate[1];
                let width = f64::from(geometry.rect.width) + candidate[2];
                let height = f64::from(geometry.rect.height) + candidate[3];
                if x < 0.0
                    || y < 0.0
                    || width < 16.0
                    || height < 16.0
                    || x + width > 32.0
                    || y + height > 32.0
                {
                    continue;
                }
                let support = frame_support::<STRIDE>(geometry, candidate, a, b, best.rank());
                if support.rank() > best.rank() {
                    best = support;
                    offsets = candidate;
                }
            }
        }
    }
    (best, offsets)
}

fn frame_mapping(geometry: Geometry, offsets: [f64; 4]) -> [f64; 4] {
    [
        f64::from(geometry.rect.x) + offsets[0],
        f64::from(geometry.rect.y) + offsets[1],
        (f64::from(geometry.rect.width) + offsets[2]) / 32.0,
        (f64::from(geometry.rect.height) + offsets[3]) / 32.0,
    ]
}

fn frame_identity(
    geometry: Geometry,
    offsets: [f64; 4],
    a: &Frame,
    b: &Frame,
) -> Option<FrameSupport> {
    let (full, cropped) = if geometry.reverse { (b, a) } else { (a, b) };
    let sampler = FrameSampler::new(cropped, frame_mapping(geometry, offsets));
    pixel_identity(full, &sampler, true)
}

fn pixel_identity(full: &Frame, sampler: &FrameSampler<'_>, native: bool) -> Option<FrameSupport> {
    let pixels: [_; SIDE * SIDE] =
        std::array::from_fn(|index| sampler.sample(index % SIDE, index / SIDE));
    let sampler = FrameSampler {
        frame: sampler.frame,
        mapping: sampler.mapping,
        axes: None,
        preview: Some(&pixels),
    };
    let mut gray = [0; SIDE * SIDE];
    let mut stable = 0;
    let mut textured = 0;
    for tile in 0..16 {
        let remaining = 16 - tile;
        if stable + remaining < 8 || textured + remaining < 4 {
            return None;
        }
        let left = (tile % 4) * 8;
        let top = (tile / 4) * 8;
        let mut error = 0.0;
        'pixels: for y in top..top + 8 {
            for x in left..left + 8 {
                let other = sampler.sample(x, y);
                gray[y * SIDE + x] =
                    (other[0] * 0.299 + other[1] * 0.587 + other[2] * 0.114).round() as u8;
                for (channel, value) in other.into_iter().enumerate() {
                    let delta = f64::from(full.rgb_thumbnail[y * SIDE + x][channel]) - value;
                    error += delta * delta;
                }
                if error >= 64.0 * 3.0 * 144.0 {
                    break 'pixels;
                }
            }
        }
        if error >= 64.0 * 3.0 * 144.0 {
            continue;
        }
        let (correlation, texture) = tile_correlation(&full.thumbnail, &gray, tile);
        if correlation >= 0.9 {
            stable += 1;
            textured += usize::from(texture);
        }
    }
    if stable < 8
        || textured < 4
        || !(if native {
            native_frame_identity(full, &sampler)
        } else {
            rectangle_identity(|x, y| {
                (
                    full.rgb_thumbnail[y * SIDE + x].map(f64::from),
                    sampler.sample(x, y),
                )
            })
        })
    {
        return None;
    }
    Some(sample_support::<1>(full, |x, y| sampler.sample(x, y), 0.0))
}

fn native_frame_identity(full: &Frame, cropped: &FrameSampler<'_>) -> bool {
    if full.pixels.is_none() || cropped.frame.pixels.is_none() {
        return false;
    }
    let full = FrameSampler::new(full, [0.0, 0.0, 1.0, 1.0]);
    rectangle_identity(|x, y| (full.sample(x, y), cropped.sample(x, y)))
}

fn rectangle_identity(mut pixels: impl FnMut(usize, usize) -> ([f64; 3], [f64; 3])) -> bool {
    let mut errors = [[0_u16; SIDE + 1]; SIDE + 1];
    let noise = SIDE * SIDE / 32;
    let occlusion = SIDE * SIDE / 4;
    for y in 0..SIDE {
        let mut row = 0;
        for x in 0..SIDE {
            let (a, b) = pixels(x, y);
            let difference = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
            row += u16::from(difference > 3.0 * 24.0_f64.powi(2));
            errors[y + 1][x + 1] = errors[y][x + 1] + row;
        }
        if usize::from(errors[y + 1][SIDE]) > occlusion + noise {
            return false;
        }
    }
    let total = usize::from(errors[SIDE][SIDE]);
    if total <= noise {
        return true;
    }
    // Independent pixels must agree across the claimed rectangle, not merely
    // in eight scattered tiles. Allow one localized logo or caption, plus a
    // small number of resampling outliers outside that occlusion.
    for height in 1..=SIDE {
        let width = (occlusion / height).min(SIDE);
        for top in 0..=SIDE - height {
            let bottom = top + height;
            for left in 0..=SIDE - width {
                let right = left + width;
                let inside = errors[bottom][right] + errors[top][left]
                    - errors[top][right]
                    - errors[bottom][left];
                if total - usize::from(inside) <= noise {
                    return true;
                }
            }
        }
    }
    false
}

fn coarse_support<'a>(
    geometry: Geometry,
    a: &'a Frame,
    b: &Frame,
    minimum: f64,
    views: Option<&mut FrameViews<'a>>,
) -> FrameSupport {
    let (full, _) = if geometry.reverse { (b, a) } else { (a, b) };
    if let Some(projection) = views.and_then(|views| views.projection(a, geometry)) {
        sample_support::<4>(full, |x, y| projection[y / 4 * 8 + x / 4], minimum)
    } else {
        frame_support::<4>(geometry, [0.0; 4], a, b, minimum)
    }
}

fn frame_support<const STRIDE: usize>(
    geometry: Geometry,
    offsets: [f64; 4],
    a: &Frame,
    b: &Frame,
    minimum_rank: f64,
) -> FrameSupport {
    let (full, cropped) = if geometry.reverse { (b, a) } else { (a, b) };
    let mapping = frame_mapping(geometry, offsets);
    let preview = FrameSampler::sampled(cropped, mapping, STRIDE);
    sample_support::<STRIDE>(full, |x, y| preview.sample(x, y), minimum_rank)
}

fn sample_support<const STRIDE: usize>(
    full: &Frame,
    sample: impl Fn(usize, usize) -> [f64; 3],
    minimum_rank: f64,
) -> FrameSupport {
    let samples = ((8 / STRIDE) * (8 / STRIDE)) as f64;
    let mut support = FrameSupport {
        matched: 0,
        textured: 0,
        quality: 0.0,
        identity: 0.0,
    };
    for tile in 0..16 {
        // Each remaining tile contributes at most 1/16 to the final rank.
        // A losing refinement cannot recover, even with perfect unseen pixels.
        if support.rank() + (16 - tile) as f64 / 16.0 + 1e-12 <= minimum_rank {
            return support;
        }
        let left = (tile % 4) * 8;
        let top = (tile / 4) * 8;
        let mut sums = [0.0; 5];
        let mut colors = [0.0; 3];
        for y in (top + STRIDE / 2..top + 8).step_by(STRIDE) {
            for x in (left + STRIDE / 2..left + 8).step_by(STRIDE) {
                let other = sample(x, y);
                let a = f64::from(full.thumbnail[y * SIDE + x]);
                let b = other[0] * 0.299 + other[1] * 0.587 + other[2] * 0.114;
                sums[0] += a;
                sums[1] += b;
                sums[2] += a * a;
                sums[3] += b * b;
                sums[4] += a * b;
                let mut color_error = 0.0;
                for (channel, difference) in colors.iter_mut().enumerate() {
                    let delta =
                        f64::from(full.rgb_thumbnail[y * SIDE + x][channel]) - other[channel];
                    *difference += delta.abs();
                    color_error += delta * delta;
                }
                support.identity +=
                    (1.0 - color_error / (3.0 * 24.0_f64.powi(2))).max(0.0) / samples;
            }
        }
        if colors.iter().sum::<f64>() > samples * 60.0 {
            continue;
        }
        let a_variance = sums[2] - sums[0] * sums[0] / samples;
        let b_variance = sums[3] - sums[1] * sums[1] / samples;
        let minimum_variance = samples * 4.0;
        let (correlation, texture) =
            if a_variance < minimum_variance && b_variance < minimum_variance {
                (1.0, false)
            } else if a_variance >= minimum_variance && b_variance >= minimum_variance {
                (
                    (sums[4] - sums[0] * sums[1] / samples) / (a_variance * b_variance).sqrt(),
                    true,
                )
            } else {
                (0.0, false)
            };
        if correlation >= 0.90 {
            support.matched += 1;
            support.textured += usize::from(texture);
            support.quality += correlation.min(1.0);
        }
    }
    support
}

#[derive(Clone, Copy)]
struct SampleAxis {
    first: usize,
    count: usize,
    weights: [f64; 4],
}

impl SampleAxis {
    fn new(start: u32, end: u32, origin: f64, scale: f64, index: usize) -> Self {
        let size = f64::from(end - start) / SIDE as f64;
        let first = f64::from(start) + (origin + index as f64 * scale) * size;
        let last = (first + scale * size).min(f64::from(end));
        let pixel = first.floor() as usize;
        let count = last.ceil() as usize - pixel;
        debug_assert!(count <= 4);
        Self {
            first: pixel,
            count,
            weights: std::array::from_fn(|offset| {
                (last.min((pixel + offset + 1) as f64) - first.max((pixel + offset) as f64))
                    .max(0.0)
                    / (last - first)
            }),
        }
    }
}

struct FrameSampler<'a> {
    frame: &'a Frame,
    mapping: [f64; 4],
    axes: Option<([SampleAxis; SIDE], [SampleAxis; SIDE])>,
    preview: Option<&'a [[f64; 3]]>,
}

impl<'a> FrameSampler<'a> {
    fn new(frame: &'a Frame, mapping: [f64; 4]) -> Self {
        Self::sampled(frame, mapping, 1)
    }

    fn sampled(frame: &'a Frame, mapping: [f64; 4], stride: usize) -> Self {
        // This is an exact cache identity, not an approximate geometry comparison.
        let identity = mapping.map(f64::to_bits) == [0.0, 0.0, 1.0, 1.0].map(f64::to_bits);
        if identity
            && let Some(preview) = frame
                .pixels
                .as_ref()
                .and_then(|pixels| pixels.preview.get())
        {
            return Self {
                frame,
                mapping,
                axes: None,
                preview: Some(preview),
            };
        }
        let [x, y, width, height] = mapping;
        let axes = frame.pixels.as_ref().map(|pixels| {
            let (left, top, right, bottom) = pixels.bounds;
            (
                std::array::from_fn(|index| {
                    if identity || index % stride == stride / 2 {
                        SampleAxis::new(left, right, x, width, index)
                    } else {
                        SampleAxis {
                            first: 0,
                            count: 0,
                            weights: [0.0; 4],
                        }
                    }
                }),
                std::array::from_fn(|index| {
                    if identity || index % stride == stride / 2 {
                        SampleAxis::new(top, bottom, y, height, index)
                    } else {
                        SampleAxis {
                            first: 0,
                            count: 0,
                            weights: [0.0; 4],
                        }
                    }
                }),
            )
        });
        let mut sampler = Self {
            frame,
            mapping,
            axes,
            preview: None,
        };
        if identity && let Some(pixels) = &frame.pixels {
            sampler.preview = Some(pixels.preview.get_or_init(|| {
                (0..SIDE * SIDE)
                    .map(|index| sampler.sample(index % SIDE, index / SIDE))
                    .collect()
            }));
        }
        sampler
    }

    #[inline]
    fn sample(&self, x: usize, y: usize) -> [f64; 3] {
        if let Some(preview) = self.preview {
            return preview[y * SIDE + x];
        }
        let (Some(pixels), Some((horizontal, vertical))) = (&self.frame.pixels, &self.axes) else {
            let [x_origin, y_origin, x_scale, y_scale] = self.mapping;
            return sample_rgb(
                &self.frame.rgb_thumbnail,
                x_origin + (x as f64 + 0.5) * x_scale - 0.5,
                y_origin + (y as f64 + 0.5) * y_scale - 0.5,
            );
        };
        // Crop before reducing resolution. Reuse each axis's overlap weights
        // across the rectangle instead of recomputing them for every RGB pixel.
        let horizontal = horizontal[x];
        let vertical = vertical[y];
        let mut sum = [0.0; 3];
        for row in 0..vertical.count {
            let start = ((vertical.first + row) * frames::WIDTH as usize + horizontal.first) * 3;
            for column in 0..horizontal.count {
                let weight = vertical.weights[row] * horizontal.weights[column];
                let rgb = &pixels.rgb[start + column * 3..][..3];
                for (channel, value) in sum.iter_mut().enumerate() {
                    *value += f64::from(rgb[channel]) * weight;
                }
            }
        }
        sum
    }
}

fn sample_rgb(rgb: &[[u8; 3]], x: f64, y: f64) -> [f64; 3] {
    let x = x.clamp(0.0, 31.0);
    let y = y.clamp(0.0, 31.0);
    let left = x as usize;
    let top = y as usize;
    let right = (left + 1).min(31);
    let bottom = (top + 1).min(31);
    let dx = x - left as f64;
    let dy = y - top as f64;
    std::array::from_fn(|channel| {
        let upper = f64::from(rgb[top * SIDE + left][channel]) * (1.0 - dx)
            + f64::from(rgb[top * SIDE + right][channel]) * dx;
        let lower = f64::from(rgb[bottom * SIDE + left][channel]) * (1.0 - dx)
            + f64::from(rgb[bottom * SIDE + right][channel]) * dx;
        upper * (1.0 - dy) + lower * dy
    })
}
