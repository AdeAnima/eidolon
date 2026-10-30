//! Offset voting and ordered-run verification keep scene resemblance from becoming a match.

use std::collections::HashMap;

use eidolon_core::{Evidence, FileEntry, Match, Relation};

use crate::candidates::{Anchor, similarity};
use crate::frames::{Frame, Video, cosine};

struct Run {
    offset: f64,
    coverage: f64,
    continuity: f64,
    quality: f64,
    span: f64,
    count: usize,
    embedded: bool,
    local_change: f64,
    scene_change: f64,
    variant_fraction: f64,
}

fn classify(run: &Run, short: &Video, long: &Video) -> Option<Relation> {
    if run.variant_fraction >= 0.5 {
        return Some(Relation::Variant);
    }
    // Temporal or grayscale agreement cannot establish duplicate pixels when
    // scene-wide color differences have no coherent global explanation.
    if run.scene_change >= 0.5 {
        return None;
    }
    let step = short.step.max(long.step);
    let duration_tolerance = (long.duration * 0.005)
        .max(1.0 / short.fps + 1.0 / long.fps)
        .max(step);
    if long.duration - short.duration > duration_tolerance
        && run.offset >= -step
        && run.offset + short.duration <= long.duration + step
    {
        Some(Relation::ClipOf)
    } else if long.duration - short.duration <= duration_tolerance && run.offset.abs() <= step {
        if run.local_change >= 0.25 || run.scene_change >= 0.05 {
            Some(Relation::Edited)
        } else {
            Some(Relation::Reencoded)
        }
    } else {
        Some(Relation::Similar)
    }
}

pub(crate) fn compare(
    a: &FileEntry,
    a_video: &Video,
    b: &FileEntry,
    b_video: &Video,
    anchors: &[Anchor],
    min_score: f64,
    fast: bool,
) -> Option<Match> {
    let (short, long, reverse) = if a_video.duration <= b_video.duration {
        (a_video, b_video, false)
    } else {
        (b_video, a_video, true)
    };
    let step = short.step.max(long.step);
    let run = best_run(short, long, reverse, anchors, fast)?;
    let score = (0.55 + run.coverage * 0.25 + run.quality * 0.20).clamp(0.0, 1.0);
    if score < min_score {
        return None;
    }
    let relation = classify(&run, short, long)?;
    let (a_id, b_id) = if relation == Relation::ClipOf {
        if reverse { (b.id, a.id) } else { (a.id, b.id) }
    } else if a.id < b.id {
        (a.id, b.id)
    } else {
        (b.id, a.id)
    };
    let mut evidence = vec![
        Evidence {
            signal: "temporal_alignment",
            value: run.coverage,
            note: format!(
                "offset_seconds={:.3}; matched_frames={}; coverage={:.3}; longest_run_seconds={:.3}; source_seconds={:.3}; part_seconds={:.3}",
                run.offset.max(0.0),
                run.count,
                run.coverage,
                run.span,
                long.duration,
                short.duration
            ),
        },
        Evidence {
            signal: if run.embedded {
                "frame_embedding"
            } else {
                "frame_visual"
            },
            value: run.quality,
            note: format!(
                "aligned frame similarity; sample_step_seconds={step:.3}; continuity={:.3}",
                run.continuity
            ),
        },
    ];
    if fast {
        evidence.push(Evidence {
            signal: "thumbnail",
            value: 32.0,
            note: "32x32 cached RGB and grayscale pixels corroborate temporal alignment".to_owned(),
        });
    }
    evidence.extend([
            Evidence {
                signal: "spatial_edit",
                value: run.local_change,
                note: "largest fraction of aligned frames with the same content-normalized color tile changing by more than 32/255 in any channel".to_owned(),
            },
            Evidence {
                signal: "scene_edit",
                value: run.scene_change,
                note: "fraction of aligned frames with at least half the color tiles changing significantly".to_owned(),
            },
            Evidence {
                signal: "global_color_transform",
                value: run.variant_fraction,
                note: "fraction of color-observable aligned frames with a broad shift at least 90% explained by a clipped affine RGB transform or independently verified monotone tone model; unresolved codec noise is not an unchanged-color vote".to_owned(),
            },
    ]);
    for (file, video) in [(a, a_video), (b, b_video)] {
        let bitrate = video
            .bitrate
            .map_or_else(String::new, |bitrate| format!(" bitrate={bitrate}"));
        evidence.push(Evidence {
            signal: "quality",
            value: 0.0,
            note: format!(
                "file={} pixels={}{} length={:.3} fps={:.3}",
                file.id.0,
                u64::from(video.width) * u64::from(video.height),
                bitrate,
                video.duration,
                video.fps
            ),
        });
    }
    Some(Match {
        a: a_id,
        b: b_id,
        relation,
        score,
        evidence,
    })
}

fn best_run(
    short: &Video,
    long: &Video,
    reverse: bool,
    anchors: &[Anchor],
    fast: bool,
) -> Option<Run> {
    let step = short.step.max(long.step);
    let mut votes: HashMap<i64, (f64, f64)> = HashMap::new();
    for anchor in anchors {
        let offset = if reverse {
            anchor.a_time - anchor.b_time
        } else {
            anchor.b_time - anchor.a_time
        };
        if offset < -step || offset + short.duration > long.duration + step {
            continue;
        }
        let bin = (offset / step).round() as i64;
        let weight = anchor.similarity.powi(8) * if anchor.embedded { 1.0 } else { 1.5 };
        let vote = votes.entry(bin).or_default();
        vote.0 += weight;
        vote.1 += offset * weight;
    }
    let mut votes: Vec<_> = votes.into_iter().collect();
    votes.sort_unstable_by(|(a_bin, a), (b_bin, b)| {
        b.0.total_cmp(&a.0)
            .then_with(|| a_bin.abs().cmp(&b_bin.abs()))
    });
    let short_frames: Vec<_> = short.frames.iter().collect();
    let long_frames: Vec<_> = long.frames.iter().collect();
    let short_embeddings: Vec<_> = short
        .frames
        .iter()
        .filter(|frame| frame.embedding.is_some())
        .collect();
    let long_embeddings: Vec<_> = long
        .frames
        .iter()
        .filter(|frame| frame.embedding.is_some())
        .collect();
    let mut best: Option<Run> = None;
    for (_, (weight, sum)) in votes.into_iter().take(8) {
        let offset = sum / weight;
        for (short_frames, long_frames, embedded) in [
            (&short_frames, &long_frames, false),
            (&short_embeddings, &long_embeddings, true),
        ] {
            if let Some(mut run) = verify(short_frames, long_frames, offset, step, embedded, fast) {
                if !eligible(&run, short.duration) {
                    continue;
                }
                if embedded {
                    if !spatially_verified(short_frames, long_frames, offset, step, fast) {
                        continue;
                    }
                } else {
                    let motion = motion_agreement(short_frames, long_frames, offset, step);
                    if motion < 0.5 {
                        continue;
                    }
                    run.quality = run.quality.midpoint(motion);
                }
                // Coverage gates out coincidences; visual fit resolves repeated scenes.
                if best.as_ref().is_none_or(|best| run.quality > best.quality) {
                    best = Some(run);
                }
            }
        }
    }
    if let Some(run) = best.as_mut() {
        let mut evaluated = 0_usize;
        let mut variants = 0;
        for frame in &short_frames {
            let other = long_frames[nearest(&long_frames, frame.time + run.offset)];
            if (other.time - frame.time - run.offset).abs() > step * 0.55 + 1e-5 {
                continue;
            }
            if let Some(change) = crate::appearance::color_change(frame, other) {
                evaluated += 1;
                variants +=
                    usize::from(matches!(change, crate::appearance::ColorChange::Variant(_)));
            }
        }
        run.variant_fraction = if evaluated >= 3 {
            variants as f64 / evaluated as f64
        } else {
            0.0
        };
    }
    best
}

fn eligible(run: &Run, duration: f64) -> bool {
    run.count >= 3
        && run.coverage >= 0.75
        && run.continuity >= 0.65
        && run.span >= (duration * 0.55).min(4.0)
}

fn motion_agreement(short: &[&Frame], long: &[&Frame], offset: f64, step: f64) -> f64 {
    let mut previous: Option<(&Frame, &Frame)> = None;
    let mut energy_a = 0.0;
    let mut energy_b = 0.0;
    let mut dot = 0.0;
    let mut samples = 0;
    for frame in short {
        let other = long[nearest(long, frame.time + offset)];
        if (other.time - frame.time - offset).abs() > step * 0.55 + 1e-5 {
            continue;
        }
        if let Some((before, other_before)) = previous {
            if frame.time - before.time < step.max(1.0) - 1e-5 {
                continue;
            }
            for (((a, before), b), other_before) in frame
                .thumbnail
                .iter()
                .zip(&before.thumbnail)
                .zip(&other.thumbnail)
                .zip(&other_before.thumbnail)
            {
                let a = f64::from(i16::from(*a) - i16::from(*before));
                let b = f64::from(i16::from(*b) - i16::from(*other_before));
                energy_a += a * a;
                energy_b += b * b;
                dot += a * b;
                samples += 1;
            }
        }
        previous = Some((frame, other));
    }
    // Static shared borders must not outvote motion with the wrong temporal order.
    // Two genuinely stationary sequences have no motion to corroborate or contradict.
    if energy_a < f64::from(samples) && energy_b < f64::from(samples) {
        1.0
    } else if energy_a > 0.0 && energy_b > 0.0 {
        (dot / (energy_a * energy_b).sqrt()).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn spatially_verified(
    short: &[&Frame],
    long: &[&Frame],
    offset: f64,
    step: f64,
    fast: bool,
) -> bool {
    let tolerance = step.max(3.0) * 0.55 + 1e-5;
    let mut geometry = None;
    let mut count = 0_usize;
    for frame in short {
        let other = long[nearest(long, frame.time + offset)];
        if (other.time - frame.time - offset).abs() > tolerance {
            continue;
        }
        let (Some(a), Some(b)) = (&frame.embedding, &other.embedding) else {
            continue;
        };
        if cosine(a, b).is_none_or(|value| value < 0.95) {
            continue;
        }
        let transform = if let Some(transform) = geometry {
            transform
        } else {
            let Some(transform) = crate::appearance::fit(&frame.thumbnail, &other.thumbnail) else {
                return false;
            };
            geometry = Some(transform);
            transform
        };
        let color_verified = !fast
            || crate::appearance::thumbnail_geometry(transform, frame, other).is_some()
            || matches!(
                crate::appearance::color_change(frame, other),
                Some(crate::appearance::ColorChange::Variant(_))
            );
        if color_verified && transform.correlation(&frame.thumbnail, &other.thumbnail) >= 0.85 {
            count += 1;
        }
    }
    count >= 3 && count as f64 / short.len() as f64 >= 0.75
}

fn verify(
    short: &[&Frame],
    long: &[&Frame],
    offset: f64,
    step: f64,
    embedded: bool,
    fast: bool,
) -> Option<Run> {
    if short.is_empty() || long.is_empty() {
        return None;
    }
    let spacing = if embedded { step.max(3.0) } else { step };
    let tolerance = spacing * 0.55 + 1e-5;
    let mut count = 0_usize;
    let mut evaluated = 0_usize;
    let mut quality = 0.0;
    let mut local_changes = [0_usize; 16];
    let mut scene_changes = 0_usize;
    let mut contrast = 0.0;
    let (mut run_count, mut longest_count) = (0_usize, 0_usize);
    let (mut start, mut end) = (0.0, 0.0);
    let mut longest_span: f64 = 0.0;
    let mut last_match: Option<f64> = None;
    let mut last_index = None;
    for frame in short {
        let target = frame.time + offset;
        let index = nearest(long, target);
        if Some(index) == last_index {
            continue;
        }
        last_index = Some(index);
        evaluated += 1;
        let other = long[index];
        if (other.time - target).abs() > tolerance {
            continue;
        }
        let mut changed_tiles = 0;
        for (index, (a, b)) in frame.tiles.iter().zip(&other.tiles).enumerate() {
            if a.iter().zip(b).any(|(a, b)| a.abs_diff(*b) > 32) {
                local_changes[index] += 1;
                changed_tiles += 1;
            }
        }
        if changed_tiles >= 8 {
            scene_changes += 1;
        }
        let matched = if embedded {
            cosine(frame.embedding.as_ref()?, other.embedding.as_ref()?)
                .filter(|value| *value >= 0.95)
        } else {
            similarity(frame, other, fast)
                .filter(|(_, embedded)| !embedded)
                .map(|(quality, _)| quality)
        };
        let matched = matched.filter(|_| {
            !fast
                || embedded
                || crate::appearance::aligned(frame, other).is_some()
                || matches!(
                    crate::appearance::color_change(frame, other),
                    Some(crate::appearance::ColorChange::Variant(_))
                )
        });
        if let Some(value) = matched {
            if last_match.is_none_or(|previous| frame.time - previous > (spacing * 2.1).max(3.0)) {
                run_count = 0;
                start = frame.time;
            }
            run_count += 1;
            longest_count = longest_count.max(run_count);
            end = frame.time;
            longest_span = longest_span.max(end - start);
            last_match = Some(frame.time);
            count += 1;
            quality += value;
            if embedded {
                let alternative = nearest(long, other.time + spacing * 3.0);
                if alternative != index {
                    let alternative_quality = cosine(
                        frame.embedding.as_ref()?,
                        long[alternative].embedding.as_ref()?,
                    )
                    .unwrap_or(0.0);
                    contrast += value - alternative_quality;
                }
            }
        }
    }
    if count == 0 || evaluated == 0 {
        return None;
    }
    let quality = quality / count as f64;
    // Semantic scene similarity alone is insufficient: temporal changes must align too.
    if embedded && contrast / (count as f64) < 0.01 {
        return None;
    }
    Some(Run {
        offset,
        coverage: count as f64 / evaluated as f64,
        continuity: longest_count as f64 / evaluated as f64,
        quality,
        span: longest_span.max(end - start),
        count,
        embedded,
        local_change: *local_changes.iter().max()? as f64 / evaluated as f64,
        scene_change: scene_changes as f64 / evaluated as f64,
        variant_fraction: 0.0,
    })
}

fn nearest(frames: &[&Frame], time: f64) -> usize {
    let index = frames
        .partition_point(|frame| frame.time < time)
        .min(frames.len() - 1);
    if index > 0 && (frames[index - 1].time - time).abs() < (frames[index].time - time).abs() {
        index - 1
    } else {
        index
    }
}
