use std::collections::HashSet;

use eidolon_core::{Ctx, Evidence, FileEntry, Match, Relation};
use rayon::prelude::*;

use crate::features::{Features, PREVIEW, Retrieval, Visual};
use crate::geometry::{self, IDENTITY};

#[path = "content.rs"]
mod content;

#[path = "dense.rs"]
mod dense;

#[path = "fast.rs"]
mod fast;

#[path = "structure.rs"]
mod structure;

#[path = "candidates.rs"]
mod candidate_index;

const CROP_BACKGROUND_BIAS: f64 = 1.0 / 255.0;

pub(crate) fn candidates(
    features: &[Retrieval],
    frame_embeddings: &[Option<&[f32]>],
) -> Vec<(usize, usize)> {
    candidate_index::generate(features, frame_embeddings)
}

fn embedding_neighbors<'a>(
    embeddings: impl Iterator<Item = Option<&'a [f32]>>,
    index: usize,
    vector: &[f32],
) -> Vec<usize> {
    let mut neighbors = embeddings
        .enumerate()
        .filter_map(|(other, candidate)| {
            if other == index {
                return None;
            }
            let similarity = cosine(vector, candidate?);
            (similarity >= 0.82).then_some((similarity, other))
        })
        .collect::<Vec<_>>();
    if neighbors.len() > 64 {
        neighbors.select_nth_unstable_by(64, |first, second| {
            second
                .0
                .total_cmp(&first.0)
                .then_with(|| first.1.cmp(&second.1))
        });
        neighbors.truncate(64);
    }
    neighbors.into_iter().map(|(_, index)| index).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    f64::from(
        a.iter()
            .zip(b)
            .map(|(first, second)| first * second)
            .sum::<f32>(),
    )
    .clamp(-1.0, 1.0)
}

pub(crate) fn dense_candidates(
    features: &[Features],
    candidates: &[(usize, usize)],
) -> HashSet<(usize, usize)> {
    // RGB search is a fallback for sparse local geometry, not a second verifier
    // for every semantic neighbour. Keep a fixed work budget for each sparse image.
    let mut nearest = vec![[None::<(u32, usize)>; 8]; features.len()];
    for &(first, second) in candidates {
        let a = &features[first];
        let b = &features[second];
        let sparse_a = a.orb.as_ref().is_some_and(|orb| orb.points.len() < 512);
        let sparse_b = b.orb.as_ref().is_some_and(|orb| orb.points.len() < 512);
        if !(sparse_a || sparse_b) || a.visual.digest == b.visual.digest {
            continue;
        }
        let distance = a
            .visual
            .regions
            .iter()
            .map(|hash| (hash ^ b.visual.phash).count_ones())
            .chain(
                b.visual
                    .regions
                    .iter()
                    .map(|hash| (hash ^ a.visual.phash).count_ones()),
            )
            .chain(std::iter::once(
                (a.visual.phash ^ b.visual.phash).count_ones(),
            ))
            .min()
            .unwrap_or(64);
        for (index, other, sparse) in [(first, second, sparse_a), (second, first, sparse_b)] {
            if !sparse {
                continue;
            }
            let neighbors = &mut nearest[index];
            let candidate = (distance, other);
            if let Some((worst, previous)) = neighbors
                .iter()
                .enumerate()
                .max_by_key(|(_, item)| item.unwrap_or((u32::MAX, usize::MAX)))
                && previous.is_none_or(|previous| candidate < previous)
            {
                neighbors[worst] = Some(candidate);
            }
        }
    }
    nearest
        .into_iter()
        .enumerate()
        .flat_map(|(index, neighbors)| {
            neighbors
                .into_iter()
                .flatten()
                .map(move |(_, other)| (index.min(other), index.max(other)))
        })
        .collect()
}

/// Verify ordinary pairs first so they cannot consume the fallback-registration budget.
pub(crate) fn verify(
    files: &[&FileEntry],
    features: &[Features],
    candidates: &[(usize, usize)],
    ctx: &Ctx<'_>,
) -> anyhow::Result<Vec<Match>> {
    let results = candidates
        .par_iter()
        .map(|&pair| compare(files, features, pair, false, ctx))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut matches = Vec::new();
    let mut rejected = Vec::new();
    for (&pair, result) in candidates.iter().zip(results) {
        if let Some(found) = result {
            matches.push(found);
        } else {
            rejected.push(pair);
        }
    }
    if !ctx.config.no_keypoints {
        let dense = dense_candidates(features, &rejected);
        matches.extend(
            dense
                .par_iter()
                .map(|&pair| compare(files, features, pair, true, ctx))
                .collect::<anyhow::Result<Vec<_>>>()?
                .into_iter()
                .flatten(),
        );
    }
    Ok(matches)
}

pub(crate) fn compare(
    files: &[&FileEntry],
    features: &[Features],
    pair: (usize, usize),
    allow_dense: bool,
    ctx: &Ctx<'_>,
) -> anyhow::Result<Option<Match>> {
    if ctx.config.fast {
        compare_sampled::<4>(files, features, pair, allow_dense, ctx)
    } else {
        compare_sampled::<1>(files, features, pair, allow_dense, ctx)
    }
}

fn compare_sampled<const STEP: usize>(
    files: &[&FileEntry],
    features: &[Features],
    pair: (usize, usize),
    allow_dense: bool,
    ctx: &Ctx<'_>,
) -> anyhow::Result<Option<Match>> {
    let (first, second) = pair;
    let a = &features[first];
    let b = &features[second];
    let full_distance = (a.visual.phash ^ b.visual.phash).count_ones();
    let gradient_distance = (a.visual.dhash ^ b.visual.dhash).count_ones();
    let mut evidence = signals(a, b, full_distance, gradient_distance);
    let equal = decoded_equal(&a.visual, &b.visual);
    let proven = preverified::<STEP>(&a.visual, &b.visual, full_distance, ctx, &mut evidence)?;
    let alignment = match (&a.orb, &b.orb) {
        (Some(first), Some(second)) if !equal && proven.is_none() => {
            geometry::verify(first, second, ctx.config.fast)?
        }
        _ => None,
    };
    let (mut placement, mut pixels, mut variant) = if let Some(verified) = proven {
        verified
    } else {
        let Some((placement, pixels)) = initial_comparison::<STEP>(
            &a.visual,
            &b.visual,
            alignment.as_ref(),
            full_distance,
            &mut evidence,
        ) else {
            return Ok(None);
        };
        (placement, pixels, None)
    };
    let (source, target) = if placement.reverse_crop {
        (&b.visual, &a.visual)
    } else {
        (&a.visual, &b.visual)
    };
    if placement.crop_area.is_some() && alignment.is_some() && !pixels.crop_supported() {
        pixels = refine_crop::<STEP>(source, target, &mut placement);
    }
    variant = variant.or_else(|| {
        global_variant(
            source,
            target,
            &placement.transform,
            &pixels,
            alignment.is_some(),
            placement.crop_area,
        )
    });
    let mut dense_registration = false;
    let rejected_crop = placement.crop_area.is_some() && !pixels.crop_supported();
    let ambiguous_overlay = alignment.is_none()
        && !ctx.config.no_keypoints
        && needs_crop_registration(&a.visual, &b.visual, &pixels);
    if !equal
        && variant.is_none()
        && (alignment.is_none() || !pixels.supported() || rejected_crop)
        && !ctx.config.no_keypoints
        && !pixels.strict_full(full_distance)
        && (!pixels.localized() || rejected_crop || ambiguous_overlay)
        && let Some((registered, result, global)) =
            dense_placement::<STEP>(&a.visual, &b.visual, !allow_dense)?
    {
        placement = registered;
        pixels = result;
        variant = global;
        dense_registration = true;
        evidence.push(registration_signal::<STEP>(
            &pixels,
            variant.is_some(),
            allow_dense,
        ));
    }
    evidence.push(pixel_signal::<STEP>(&pixels));
    let registered = alignment.is_some() || dense_registration;
    if !equal
        && variant.is_none()
        && ((!registered && ambiguous_overlay)
            || !pixels.verified(&placement, registered, full_distance))
    {
        return Ok(None);
    }
    let (reverse_crop, crop_area) = (placement.reverse_crop, placement.crop_area);
    let relation = classify(equal, &pixels, crop_area, variant.as_ref(), &mut evidence);
    let score = confidence(
        equal,
        alignment.as_ref().filter(|_| !dense_registration),
        &pixels,
        variant.as_ref(),
    );
    Ok((score >= ctx.config.min_score).then(|| {
        finish_match(
            files,
            features,
            pair,
            relation,
            reverse_crop,
            score,
            evidence,
        )
    }))
}

fn decoded_equal(a: &Visual, b: &Visual) -> bool {
    a.width == b.width && a.height == b.height && a.digest == b.digest
}

fn pixel_signal<const STEP: usize>(pixels: &Pixels) -> Evidence {
    let sampling = if STEP == 1 {
        ""
    } else if pixels.sample_step == 1 {
        "; fast thumbnail proof: adaptive full cached 128x128 RGB preview"
    } else {
        "; fast thumbnail proof: 32x32 distributed samples, stride 4 on cached 128x128 RGB"
    };
    Evidence {
        signal: if STEP == 1 { "pixels" } else { "thumbnail" },
        value: pixels.mean,
        note: format!(
            "aligned RGB MAE; unchanged {:.3}, edges {:.3}, background MAE {:.4}, background bias {:.4}, background edges {:.3}, interior content {}/{}, compared {:.3}, scale-matched {}, ICC-normalized {}{sampling}",
            pixels.unchanged,
            pixels.edge_agreement,
            pixels.background_mean,
            pixels.background_bias,
            pixels.background_edge_agreement,
            pixels.content.regions.count_ones(),
            pixels.content.informative,
            pixels.coverage,
            pixels.scale_matched,
            pixels.profile_normalized
        ),
    }
}

fn preverified<const STEP: usize>(
    a: &Visual,
    b: &Visual,
    hash_distance: u32,
    ctx: &Ctx<'_>,
    evidence: &mut Vec<Evidence>,
) -> anyhow::Result<Option<(Placement, Pixels, Option<structure::Structure>)>> {
    if STEP != 1
        && let Some((placement, pixels)) =
            fast::full_frame(a, b, hash_distance, ctx.config.min_score)
    {
        return Ok(Some((placement, pixels, None)));
    }
    if decoded_equal(a, b) || ctx.config.no_keypoints {
        return Ok(None);
    }
    Ok(registered_variant::<STEP>(a, b, evidence)?
        .map(|(placement, pixels, structure)| (placement, pixels, Some(structure))))
}

fn registration_signal<const STEP: usize>(
    pixels: &Pixels,
    structural: bool,
    multiscale: bool,
) -> Evidence {
    Evidence {
        signal: if structural { "dense_structure" } else { "dense_rgb" },
        value: pixels.background_mean,
        note: if STEP != 1 && multiscale {
            "bounded multi-scale patch-consensus registration; independently verified on distributed thumbnail samples"
        } else if STEP != 1 {
            "native-pixel-density patch-consensus registration; independently verified on distributed thumbnail samples"
        } else if multiscale {
            "bounded multi-scale patch-consensus registration; independently verified over the full preview"
        } else {
            "native-pixel-density patch-consensus registration; independently verified over the full preview"
        }.into(),
    }
}

fn needs_crop_registration(a: &Visual, b: &Visual, pixels: &Pixels) -> bool {
    let first = u64::from(a.width) * u64::from(a.height);
    let second = u64::from(b.width) * u64::from(b.height);
    // A rescaled sparse scene can leave a localized residual without an edit.
    pixels.localized() && first.abs_diff(second) > first.min(second) / 10
}

fn finish_match(
    files: &[&FileEntry],
    features: &[Features],
    pair: (usize, usize),
    relation: Relation,
    reverse_crop: bool,
    score: f64,
    mut evidence: Vec<Evidence>,
) -> Match {
    let (first, second) = pair;
    for index in [first, second] {
        let visual = &features[index].visual;
        evidence.push(Evidence {
            signal: "quality",
            value: 0.0,
            note: format!(
                "file={} pixels={}",
                files[index].id.0,
                u64::from(visual.width) * u64::from(visual.height)
            ),
        });
    }
    let (a_id, b_id) = if relation == Relation::CropOf {
        if reverse_crop {
            (files[second].id, files[first].id)
        } else {
            (files[first].id, files[second].id)
        }
    } else {
        (
            files[first].id.min(files[second].id),
            files[first].id.max(files[second].id),
        )
    };
    Match {
        a: a_id,
        b: b_id,
        relation,
        score,
        evidence,
    }
}

fn refine_crop<const STEP: usize>(a: &Visual, b: &Visual, placement: &mut Placement) -> Pixels {
    placement.transform = dense::refine_crop(a, b, &placement.transform);
    let mut pixels = photometry::<STEP>(a, b, &placement.transform);
    if !pixels.crop_supported()
        && let Some((refined, aligned)) = refine_pixels::<STEP>(a, b, &placement.transform)
    {
        placement.transform = refined;
        pixels = aligned;
    }
    placement.crop_area = geometry::contained_area(&placement.transform);
    pixels
}

fn dense_placement<const STEP: usize>(
    a: &Visual,
    b: &Visual,
    native_only: bool,
) -> anyhow::Result<Option<(Placement, Pixels, Option<structure::Structure>)>> {
    for structural in [false, true] {
        for (reverse_crop, source, target) in [(false, a, b), (true, b, a)] {
            if let Some(mut transform) = dense::register(source, target, structural, native_only)?
                && let Some(area) = geometry::contained_area(&transform)
            {
                let mut pixels = photometry::<STEP>(source, target, &transform);
                if !structural
                    && !pixels.crop_supported()
                    && let Some((refined, aligned)) =
                        refine_pixels::<STEP>(source, target, &transform)
                {
                    transform = refined;
                    pixels = aligned;
                }
                let variant = global_variant(source, target, &transform, &pixels, true, Some(area));
                if variant.is_some() || (!structural && pixels.crop_supported()) {
                    return Ok(Some((
                        Placement {
                            transform,
                            reverse_crop,
                            crop_area: Some(area),
                        },
                        pixels,
                        variant,
                    )));
                }
            }
        }
    }
    Ok(None)
}

fn refine_pixels<const STEP: usize>(
    a: &Visual,
    b: &Visual,
    transform: &[f64; 9],
) -> Option<([f64; 9], Pixels)> {
    // Sparse registration can leave a fraction-of-a-pixel phase error at sharp
    // edges. Refine translation at the selected proof density, keeping all gates.
    let matched = matched_preview(&a.preview, [transform[0].abs(), transform[4].abs()]);
    let source = matched
        .as_ref()
        .map_or(a.preview.as_slice(), |image| image.as_raw().as_slice());
    let offsets = [-0.25, -0.125, 0.0, 0.125, 0.25];
    for (y, &dy) in offsets.iter().enumerate() {
        for (x, &dx) in offsets.iter().enumerate() {
            if (x, y) == (2, 2) {
                continue;
            }
            let mut candidate = *transform;
            candidate[2] += dx / f64::from(PREVIEW);
            candidate[5] += dy / f64::from(PREVIEW);
            if geometry::contained_area(&candidate).is_none() {
                continue;
            }
            let mut pixels = photometry_pixels::<STEP>(source, &b.preview, &candidate);
            if pixels.crop_supported() {
                pixels.scale_matched = matched.is_some();
                return Some((candidate, pixels));
            }
        }
    }
    None
}

fn registered_variant<const STEP: usize>(
    a: &Visual,
    b: &Visual,
    evidence: &mut Vec<Evidence>,
) -> anyhow::Result<Option<(Placement, Pixels, structure::Structure)>> {
    let full_structure = structure::compare(a, b, &IDENTITY);
    let full_pixels = full_structure
        .supported(true)
        .then(|| photometry::<STEP>(a, b, &IDENTITY));
    if full_pixels
        .as_ref()
        .is_some_and(|pixels| !pixels.global_change())
    {
        return Ok(None);
    }
    let full_structure = full_pixels.as_ref().map(|_| full_structure);
    let Some((transform, structure, response)) = structure::register(a, b, full_structure)? else {
        return Ok(None);
    };
    let pixels = if geometry::full_frame(&transform) {
        full_pixels.unwrap_or_else(|| photometry::<STEP>(a, b, &transform))
    } else {
        photometry::<STEP>(a, b, &transform)
    };
    if !pixels.visible_global_change()
        && !(geometry::reframed_windows(&transform)
            && (pixels.same_overlap() || structure.supported(false)))
    {
        return Ok(None);
    }
    evidence.push(Evidence {
        signal: "phase",
        value: response,
        note: "gradient-magnitude phase registration; shared structure and signed gradients independently verified".into(),
    });
    Ok(Some((
        Placement {
            transform,
            reverse_crop: false,
            crop_area: None,
        },
        pixels,
        structure,
    )))
}

fn global_variant(
    a: &Visual,
    b: &Visual,
    h: &[f64; 9],
    pixels: &Pixels,
    registered: bool,
    crop_area: Option<f64>,
) -> Option<structure::Structure> {
    // Distributed JPEG/resampling noise is not a restyle of a verified crop.
    if crop_area.is_some() && pixels.crop_supported() {
        return None;
    }
    let reframed = registered && geometry::reframed_windows(h);
    if !pixels.global_change() && !reframed {
        return None;
    }
    let structure = structure::compare(a, b, h);
    structure.supported(registered).then_some(structure)
}

fn confidence(
    equal: bool,
    alignment: Option<&geometry::Alignment>,
    pixels: &Pixels,
    variant: Option<&structure::Structure>,
) -> f64 {
    if let Some(variant) = variant {
        variant.confidence()
    } else if equal {
        1.0
    } else if let Some(alignment) = alignment {
        let (error, agreement) = if pixels.localized() {
            (pixels.background_mean, pixels.background_edge_agreement)
        } else {
            (pixels.mean, pixels.edge_agreement)
        };
        (0.86 + 0.07 * alignment.ratio + 0.04 * agreement + 0.03 * (1.0 - error / 0.065))
            .clamp(0.0, 0.995)
    } else {
        let error = if pixels.localized() {
            pixels.background_mean
        } else {
            pixels.mean
        };
        (0.90 + 0.08 * (1.0 - error / 0.03)).clamp(0.0, 0.98)
    }
}

fn signals(
    a: &Features,
    b: &Features,
    full_distance: u32,
    gradient_distance: u32,
) -> Vec<Evidence> {
    let region_distance = a
        .visual
        .regions
        .iter()
        .map(|hash| (hash ^ b.visual.phash).count_ones())
        .chain(
            b.visual
                .regions
                .iter()
                .map(|hash| (hash ^ a.visual.phash).count_ones()),
        )
        .min()
        .unwrap_or(64);
    let mut evidence = vec![
        Evidence {
            signal: "phash",
            value: f64::from(full_distance),
            note: format!(
                "full-frame Hamming {full_distance}/64; best crop-region Hamming {region_distance}/64"
            ),
        },
        Evidence {
            signal: "dhash",
            value: f64::from(gradient_distance),
            note: "64-bit horizontal gradient Hamming distance".into(),
        },
    ];
    if let (Some(first), Some(second)) = (&a.embedding, &b.embedding) {
        evidence.push(Evidence {
            signal: "dinov2",
            value: cosine(first, second),
            note: "L2-normalized DINOv2 CLS cosine; candidate signal, not proof of duplication"
                .into(),
        });
    }
    evidence
}

struct Placement {
    transform: [f64; 9],
    reverse_crop: bool,
    crop_area: Option<f64>,
}

fn initial_comparison<const STEP: usize>(
    a: &Visual,
    b: &Visual,
    alignment: Option<&geometry::Alignment>,
    full_distance: u32,
    evidence: &mut Vec<Evidence>,
) -> Option<(Placement, Pixels)> {
    if full_distance <= 10
        && alignment.is_some_and(|alignment| !geometry::full_frame(&alignment.transform))
    {
        let pixels = photometry::<STEP>(a, b, &IDENTITY);
        if pixels.strict_full(full_distance) {
            evidence.push(Evidence {
                signal: "full_frame",
                value: pixels.mean,
                note: "strict normalized full-frame pixel agreement overrides ambiguous repeated-pattern geometry".into(),
            });
            return Some((
                Placement {
                    transform: IDENTITY,
                    reverse_crop: false,
                    crop_area: None,
                },
                pixels,
            ));
        }
    }
    let placement = placement(alignment, evidence)?;
    let (source, target) = if placement.reverse_crop {
        (b, a)
    } else {
        (a, b)
    };
    let pixels = photometry::<STEP>(source, target, &placement.transform);
    Some((placement, pixels))
}

fn placement(
    alignment: Option<&geometry::Alignment>,
    evidence: &mut Vec<Evidence>,
) -> Option<Placement> {
    let Some(alignment) = alignment else {
        return Some(Placement {
            transform: IDENTITY,
            reverse_crop: false,
            crop_area: None,
        });
    };
    evidence.push(Evidence {
        signal: "orb",
        value: alignment.ratio,
        note: format!(
            "{} mutual ratio-test RANSAC inliers; grid coverage {:.3}",
            alignment.inliers, alignment.coverage
        ),
    });
    if geometry::full_frame(&alignment.transform) {
        // Avoid inventing residuals through interpolation of aligned full frames.
        Some(Placement {
            transform: IDENTITY,
            reverse_crop: false,
            crop_area: None,
        })
    } else if let Some(area) = geometry::contained_area(&alignment.transform) {
        Some(Placement {
            transform: alignment.transform,
            reverse_crop: false,
            crop_area: Some(area),
        })
    } else {
        let inverse = geometry::inverse(&alignment.transform)?;
        if let Some(area) = geometry::contained_area(&inverse) {
            Some(Placement {
                transform: inverse,
                reverse_crop: true,
                crop_area: Some(area),
            })
        } else if geometry::reframed_windows(&alignment.transform) {
            Some(Placement {
                transform: alignment.transform,
                reverse_crop: false,
                crop_area: None,
            })
        } else {
            None
        }
    }
}

fn classify(
    equal: bool,
    pixels: &Pixels,
    crop_area: Option<f64>,
    variant: Option<&structure::Structure>,
    evidence: &mut Vec<Evidence>,
) -> Relation {
    if let Some(global) = variant {
        evidence.push(Evidence {
            signal: "variant", value: global.correlation,
            note: format!("global colour/lighting or reframed-canvas version; structural correlation {:.3}, signed-gradient agreement {:.3}, tone-normalized {}, changed blocks {:.3}, projected area {:?}", global.correlation, global.gradients, global.tone_normalized, pixels.global_changed, crop_area),
        });
        Relation::Variant
    } else if equal {
        Relation::SameContent
    } else if let Some(area) = crop_area {
        evidence.push(Evidence {
            signal: "containment",
            value: area,
            note: "homography projects the part into this fraction of the whole image".into(),
        });
        Relation::CropOf
    } else if pixels.localized() {
        Relation::Edited
    } else if pixels.mean <= 0.035 {
        Relation::Reencoded
    } else {
        Relation::Similar
    }
}

#[derive(Debug)]
struct Pixels {
    mean: f64,
    unchanged: f64,
    edge_agreement: f64,
    changed_fraction: f64,
    concentration: f64,
    background_mean: f64,
    background_bias: f64,
    background_edge_agreement: f64,
    background_regions: usize,
    content: content::Support,
    coverage: f64,
    global_changed: f64,
    scale_matched: bool,
    profile_normalized: bool,
    sample_step: usize,
}

impl Pixels {
    fn global_change(&self) -> bool {
        self.coverage >= 0.90 && self.visible_global_change()
    }

    fn visible_global_change(&self) -> bool {
        self.mean > 0.035 && self.global_changed >= 0.35 && !self.localized()
    }

    fn localized(&self) -> bool {
        (0.003..0.25).contains(&self.changed_fraction)
            && self.concentration >= 0.90
            && self.unchanged >= 0.64
            && self.background_mean <= 0.030
            && self.background_edge_agreement >= 0.90
            && self.background_regions >= 4
    }

    fn supported(&self) -> bool {
        self.coverage >= 0.90 && self.appearance_supported()
    }

    fn same_overlap(&self) -> bool {
        self.coverage >= 0.50
            && self.background_bias <= CROP_BACKGROUND_BIAS
            && self.appearance_supported()
    }

    fn content_supported(&self) -> bool {
        // Equal RGB samples cover the whole comparison even on a flat image.
        // They can prove reencoding, never exact high-bit-depth equality.
        self.mean <= f64::EPSILON || self.content.supported()
    }

    fn appearance_supported(&self) -> bool {
        self.content_supported()
            && self.unchanged >= 0.64
            && if self.localized() {
                self.mean <= 0.16
            } else {
                self.mean <= 0.065 && self.edge_agreement >= 0.70
            }
    }

    fn crop_supported(&self) -> bool {
        self.supported() && self.background_bias <= CROP_BACKGROUND_BIAS
    }

    fn verified(&self, placement: &Placement, registered: bool, hash_distance: u32) -> bool {
        self.supported()
            && (placement.crop_area.is_none() || self.background_bias <= CROP_BACKGROUND_BIAS)
            && (registered || self.strict_full(hash_distance) || self.localized())
    }

    fn strict_full(&self, hash_distance: u32) -> bool {
        self.content_supported()
            && hash_distance <= 10
            && self.mean <= 0.025
            && self.edge_agreement >= 0.90
            && self.unchanged >= 0.88
    }
}

fn pixel(image: &Visual, x: usize, y: usize) -> [f64; 3] {
    pixel_bytes(&image.preview, x, y)
}

fn pixel_bytes(preview: &[u8], x: usize, y: usize) -> [f64; 3] {
    let index = (y * PREVIEW as usize + x) * 3;
    [
        f64::from(preview[index]) / 255.0,
        f64::from(preview[index + 1]) / 255.0,
        f64::from(preview[index + 2]) / 255.0,
    ]
}

fn interpolate(image: &Visual, point: [f64; 2]) -> Option<[f64; 3]> {
    interpolate_pixels(&image.preview, point)
}

fn interpolate_pixels(preview: &[u8], point: [f64; 2]) -> Option<[f64; 3]> {
    let x = point[0] * f64::from(PREVIEW) - 0.5;
    let y = point[1] * f64::from(PREVIEW) - 0.5;
    if x < 0.0 || y < 0.0 || x > f64::from(PREVIEW - 1) || y > f64::from(PREVIEW - 1) {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(PREVIEW as usize - 1);
    let y1 = (y0 + 1).min(PREVIEW as usize - 1);
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    let top_left = pixel_bytes(preview, x0, y0);
    let top_right = pixel_bytes(preview, x1, y0);
    let bottom_left = pixel_bytes(preview, x0, y1);
    let bottom_right = pixel_bytes(preview, x1, y1);
    Some(std::array::from_fn(|channel| {
        (top_left[channel] * (1.0 - dx) + top_right[channel] * dx) * (1.0 - dy)
            + (bottom_left[channel] * (1.0 - dx) + bottom_right[channel] * dx) * dy
    }))
}

fn photometry<const STEP: usize>(a: &Visual, b: &Visual, transform: &[f64; 9]) -> Pixels {
    let original = photometry_previews::<STEP>(&a.preview, &b.preview, transform);
    if original.background_bias > CROP_BACKGROUND_BIAS
        && geometry::contained_area(transform).is_some()
        && (a.srgb_preview.is_some() || b.srgb_preview.is_some())
    {
        let mut normalized = photometry_previews::<STEP>(
            a.srgb_preview.as_deref().unwrap_or(&a.preview),
            b.srgb_preview.as_deref().unwrap_or(&b.preview),
            transform,
        );
        if normalized.crop_supported() {
            normalized.profile_normalized = true;
            return normalized;
        }
    }
    original
}

fn photometry_previews<const STEP: usize>(a: &[u8], b: &[u8], transform: &[f64; 9]) -> Pixels {
    let original = photometry_pixels::<STEP>(a, b, transform);
    if STEP != 1
        && !original.crop_supported()
        && original.unchanged >= 0.60
        && geometry::contained_area(transform).is_some()
    {
        // Signed colour bias is deliberately strict. A sparse sample can move
        // a genuine JPEG crop across that boundary; prove it at full density
        // instead of loosening the gate or rejecting it because of sampling.
        return photometry_previews::<1>(a, b, transform);
    }
    if (original.supported() && original.background_bias <= CROP_BACKGROUND_BIAS)
        || geometry::contained_area(transform).is_none()
    {
        return original;
    }
    let [Some(origin), Some(right), Some(bottom)] =
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)].map(|(x, y)| geometry::project(transform, x, y))
    else {
        return original;
    };
    let extent = [
        (right[0] - origin[0]).hypot(right[1] - origin[1]),
        (bottom[0] - origin[0]).hypot(bottom[1] - origin[1]),
    ];
    let Some(matched) = matched_preview(a, extent) else {
        return original;
    };
    let mut adjusted = photometry_pixels::<STEP>(matched.as_raw(), b, transform);
    if adjusted.crop_supported() || (!original.supported() && adjusted.supported()) {
        adjusted.scale_matched = true;
        adjusted
    } else {
        original
    }
}

fn matched_preview(a: &[u8], extent: [f64; 2]) -> Option<image::RgbImage> {
    let [width, height] = extent.map(|length| {
        (length * f64::from(PREVIEW))
            .round()
            .clamp(16.0, f64::from(PREVIEW)) as u32
    });
    if width == PREVIEW && height == PREVIEW {
        return None;
    }
    // A crop has more preview samples per scene pixel than the whole image.
    // Match effective resolution without weakening photometric verification.
    let source = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(PREVIEW, PREVIEW, a)
        .expect("cached RGB previews have fixed dimensions");
    let reduced = image::imageops::resize(
        &source,
        width,
        height,
        image::imageops::FilterType::Triangle,
    );
    Some(image::imageops::resize(
        &reduced,
        PREVIEW,
        PREVIEW,
        image::imageops::FilterType::Triangle,
    ))
}

fn gradient(source: &[u8], x: usize, y: usize) -> f64 {
    let [left, right, top, bottom] =
        [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)].map(|(x, y)| pixel_bytes(source, x, y));
    (0..3)
        .map(|channel| {
            (left[channel] - right[channel]).abs() + (top[channel] - bottom[channel]).abs()
        })
        .sum::<f64>()
        / 3.0
}

fn photometry_pixels<const STEP: usize>(source: &[u8], b: &[u8], transform: &[f64; 9]) -> Pixels {
    let mut total = 0.0;
    let mut count = 0_u32;
    let mut unchanged = 0_u32;
    let mut changed = 0_u32;
    let mut edges = 0_u32;
    let mut agreed_edges = 0_u32;
    let mut blocks = [0_u32; 64];
    let mut block_counts = [0_u32; 64];
    let mut block_errors = [0.0; 64];
    let mut block_bias = [[0.0; 3]; 64];
    let mut block_edges = [0_u32; 64];
    let mut block_agreed_edges = [0_u32; 64];
    let mut content = content::Content::new();
    // Both modes use the same alignment and rejection rules. Fast mode samples
    // a distributed 32-by-32 grid instead of examining every cached preview pixel.
    let side = (PREVIEW as usize - 2).div_ceil(STEP);
    for y in (1..PREVIEW as usize - 1).step_by(STEP) {
        for x in (1..PREVIEW as usize - 1).step_by(STEP) {
            let Some(point) = geometry::project(
                transform,
                (x as f64 + 0.5) / f64::from(PREVIEW),
                (y as f64 + 0.5) / f64::from(PREVIEW),
            ) else {
                continue;
            };
            let Some(target) = interpolate_pixels(b, point) else {
                continue;
            };
            let source_pixel = pixel_bytes(source, x, y);
            let residual: [f64; 3] =
                std::array::from_fn(|channel| source_pixel[channel] - target[channel]);
            let difference = residual.iter().map(|value| value.abs()).sum::<f64>() / 3.0;
            content.observe(x, y, source_pixel, target);
            total += difference;
            count += 1;
            let block = (y / 16) * 8 + x / 16;
            block_counts[block] += 1;
            block_errors[block] += difference;
            for (channel, residual) in residual.into_iter().enumerate() {
                block_bias[block][channel] += residual;
            }
            if difference < 0.04 {
                unchanged += 1;
            }
            if difference > 0.10 {
                changed += 1;
                blocks[block] += 1;
            }
            if gradient(source, x, y) > 0.09 {
                edges += 1;
                block_edges[block] += 1;
                if difference < 0.08 {
                    agreed_edges += 1;
                    block_agreed_edges[block] += 1;
                }
            }
        }
    }
    let (mask, localized_changes) = localized_region(&blocks);
    let background = background_proof(
        mask,
        &block_counts,
        &block_errors,
        &block_bias,
        &block_edges,
        &block_agreed_edges,
    );
    Pixels {
        mean: if count == 0 {
            1.0
        } else {
            total / f64::from(count)
        },
        unchanged: f64::from(unchanged) / f64::from(count.max(1)),
        edge_agreement: if edges == 0 {
            1.0
        } else {
            f64::from(agreed_edges) / f64::from(edges)
        },
        changed_fraction: f64::from(changed) / f64::from(count.max(1)),
        concentration: f64::from(localized_changes) / f64::from(changed.max(1)),
        background_mean: background.mean,
        background_bias: background.bias,
        background_edge_agreement: background.edge_agreement,
        background_regions: background.regions,
        content: content.finish(),
        coverage: f64::from(count) / (side * side) as f64,
        global_changed: std::iter::zip(&block_errors, &block_counts)
            .filter(|&(error, count)| *error > 0.025 * f64::from(*count))
            .count() as f64
            / 64.0,
        scale_matched: false,
        profile_normalized: false,
        sample_step: STEP,
    }
}

struct Background {
    mean: f64,
    bias: f64,
    edge_agreement: f64,
    regions: usize,
}

fn background_proof(
    mask: u64,
    counts: &[u32; 64],
    errors: &[f64; 64],
    biases: &[[f64; 3]; 64],
    edges: &[u32; 64],
    agreed_edges: &[u32; 64],
) -> Background {
    let (mut count, mut error, mut edge_count, mut agreed, mut regions) =
        (0_u32, 0.0, 0_u32, 0_u32, 0);
    let mut bias = [0.0_f64; 3];
    for index in 0..64 {
        if mask & (1_u64 << index) == 0 {
            count += counts[index];
            error += errors[index];
            edge_count += edges[index];
            agreed += agreed_edges[index];
            regions += usize::from(agreed_edges[index] >= 2);
            for (channel, bias) in bias.iter_mut().enumerate() {
                *bias += biases[index][channel];
            }
        }
    }
    Background {
        mean: error / f64::from(count.max(1)),
        // JPEG/resampling noise cancels; a coherent colour or lighting change does not.
        bias: bias.iter().map(|value| value.abs()).sum::<f64>() / (3.0 * f64::from(count.max(1))),
        // Empty background and a few coincident edges are not shared-scene proof.
        edge_agreement: if edge_count == 0 {
            0.0
        } else {
            f64::from(agreed) / f64::from(edge_count)
        },
        regions,
    }
}

fn localized_region(blocks: &[u32; 64]) -> (u64, u32) {
    let mut sums = [0_u32; 81];
    for y in 0..8 {
        for x in 0..8 {
            sums[(y + 1) * 9 + x + 1] =
                blocks[y * 8 + x] + sums[y * 9 + x + 1] + sums[(y + 1) * 9 + x] - sums[y * 9 + x];
        }
    }
    let (mut best, mut rectangle, mut area) = (0, [0; 4], 65);
    for top in 0..8 {
        for left in 0..8 {
            for bottom in top + 1..=8 {
                for right in left + 1..=8 {
                    let size = (bottom - top) * (right - left);
                    if size > 24 {
                        continue;
                    }
                    let count = sums[bottom * 9 + right] + sums[top * 9 + left]
                        - sums[bottom * 9 + left]
                        - sums[top * 9 + right];
                    if count > best || (count == best && size < area) {
                        best = count;
                        rectangle = [top, left, bottom, right];
                        area = size;
                    }
                }
            }
        }
    }
    let mut mask = 0_u64;
    if best > 0 {
        for y in rectangle[0]..rectangle[2] {
            for x in rectangle[1]..rectangle[3] {
                mask |= 1_u64 << (y * 8 + x);
            }
        }
    }
    (mask, best)
}

#[cfg(test)]
#[path = "benchmark.rs"]
mod benchmark;

#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
