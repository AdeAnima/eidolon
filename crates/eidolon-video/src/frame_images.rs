//! Indexed image-to-video retrieval with independent pixels and local temporal refinement.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Context as _;
use eidolon_core::{Ctx, Embedder, Evidence, FileEntry, FrameRefiner, Match, Relation};
use image::{Rgb, RgbImage, imageops::FilterType};
use rayon::prelude::*;

use crate::candidates::{FrameRef, Index};
use crate::frames::{self, Frame, Remote, Video};

const IMAGE_FEATURE: &str = "video.image-frame/v6;video-descriptor=v5;resize=96-triangle-white;pixels=96-source-aspect;mask=text-band/v1";
const SEEDS: usize = 4;

struct Loaded<'a> {
    file: &'a FileEntry,
    video: Video,
    remote: Option<Remote>,
}

#[derive(Default)]
struct Clock {
    calls: AtomicU64,
    nanos: AtomicU64,
}

impl Clock {
    fn measure<T>(&self, work: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let result = work();
        self.nanos.fetch_add(
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        result
    }

    fn seconds(&self) -> f64 {
        self.nanos.load(Ordering::Relaxed) as f64 / 1e9
    }
}

#[derive(Default)]
struct RefinementTiming {
    provider: Clock,
    decode: Clock,
    proof: Clock,
    cache_hits: AtomicU64,
}

impl RefinementTiming {
    fn write(&self, file: &FileEntry) {
        if self.provider.calls.load(Ordering::Relaxed) == 0 {
            return;
        }
        eprintln!(
            "Video refinement timing: {{\"file_id\":{},\"provider_seconds\":{:.6},\"provider_calls\":{},\"decode_seconds\":{:.6},\"decode_calls\":{},\"proof_seconds\":{:.6},\"proof_calls\":{},\"cache_hits\":{}}}",
            file.id.0,
            self.provider.seconds(),
            self.provider.calls.load(Ordering::Relaxed),
            self.decode.seconds(),
            self.decode.calls.load(Ordering::Relaxed),
            self.proof.seconds(),
            self.proof.calls.load(Ordering::Relaxed),
            self.cache_hits.load(Ordering::Relaxed),
        );
    }
}

#[derive(Default)]
struct DecodedWindows {
    windows: HashMap<(u64, u64), Arc<Vec<Frame>>>,
    frames: usize,
    limit: usize,
}

#[derive(Default)]
struct Refinement {
    decoded: Mutex<DecodedWindows>,
    timing: RefinementTiming,
}

impl Refinement {
    fn new(limit: usize) -> Self {
        Self {
            decoded: Mutex::new(DecodedWindows {
                limit,
                ..DecodedWindows::default()
            }),
            timing: RefinementTiming::default(),
        }
    }

    fn refine(
        &self,
        file: &FileEntry,
        provider: &dyn FrameRefiner,
        center: f64,
        radius: f64,
    ) -> anyhow::Result<Option<Arc<Vec<Frame>>>> {
        // Keep one in-flight decode per source. Never retain refusals or errors:
        // a later request may fit the budget or reuse newly cached ranges.
        let mut decoded = self
            .decoded
            .lock()
            .map_err(|_| anyhow::anyhow!("native frame cache poisoned"))?;
        let key = (center.to_bits(), radius.to_bits());
        if let Some(frames) = decoded.windows.get(&key) {
            self.timing.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(Arc::clone(frames)));
        }
        let Some(window) = self
            .timing
            .provider
            .measure(|| provider.refine(file, center, radius))?
        else {
            return Ok(None);
        };
        let frames = Arc::new(
            self.timing
                .decode
                .measure(|| frames::refined_window(&window, center, radius))?,
        );
        // Both empty-window count and decoded-frame count stay bounded. The
        // source-local cache is dropped when its image queries finish.
        if decoded.windows.len() < decoded.limit && decoded.frames + frames.len() <= decoded.limit {
            decoded.frames += frames.len();
            decoded.windows.insert(key, Arc::clone(&frames));
        }
        Ok(Some(frames))
    }

    fn prove(
        &self,
        frames: &[Frame],
        image: &Frame,
        best: &mut Option<(f64, f64)>,
        min_score: f64,
    ) -> bool {
        self.timing.proof.measure(|| {
            frames.iter().any(|frame| {
                consider(image, frame, best);
                best.is_some_and(|(quality, _)| 0.75 + quality * 0.25 >= min_score)
            })
        })
    }
}

#[derive(Clone, Copy)]
struct Seed {
    frame: usize,
    rank: f64,
}

pub(crate) fn find(
    images: &[FileEntry],
    videos: &[FileEntry],
    ctx: &Ctx<'_>,
    skipped: &std::sync::Mutex<Vec<eidolon_core::InputError>>,
) -> anyhow::Result<Vec<Match>> {
    if images.is_empty() || videos.is_empty() {
        return Ok(Vec::new());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(ctx.config.jobs)
        .build()
        .context("creating image-to-video workers")?;
    let loaded = pool.install(|| load_videos(videos, ctx, skipped))?;
    if loaded.is_empty() {
        return Ok(Vec::new());
    }
    // Related recolors cannot establish FrameOf containment.
    let index = Index::new(loaded.iter().map(|loaded| &loaded.video), false);
    let images: Vec<_> = pool.install(|| {
        images
            .par_iter()
            .filter_map(|file| match load_image(file, ctx) {
                Ok(image) => Some(Ok((file, image))),
                Err(error) => match error
                    .downcast::<eidolon_core::InputError>()
                    .and_then(|input| crate::record_skip(skipped, input))
                {
                    Ok(()) => None,
                    Err(error) => Some(Err(error)),
                },
            })
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    let candidates: Vec<_> = pool.install(|| {
        images
            .par_iter()
            .enumerate()
            .map_init(HashSet::new, |hits, (image, (_, frame))| {
                (
                    image,
                    seeds(frame, &loaded, &index, hits, ctx.config.remote_overlay),
                )
            })
            .collect()
    });
    let mut queries: Vec<Vec<_>> = (0..loaded.len()).map(|_| Vec::new()).collect();
    for (image, candidates) in candidates {
        for (video, seeds) in candidates {
            queries[video].push((image, seeds));
        }
    }
    // Work-conserving source scheduling; each worker owns one native timeline.
    // Only a source set smaller than the pool may nest parallel query work.
    let parallel_queries = loaded.len() < pool.current_num_threads();
    let mut matches: Vec<_> = pool.install(|| {
        loaded
            .par_iter()
            .zip(queries.into_par_iter())
            .flat_map_iter(|(source, queries)| {
                match_video(source, &images, queries, parallel_queries, ctx)
            })
            .collect()
    });
    matches.sort_unstable_by_key(|item| (item.a, item.b));
    Ok(matches)
}

fn load_videos<'a>(
    videos: &'a [FileEntry],
    ctx: &Ctx<'_>,
    skipped: &std::sync::Mutex<Vec<eidolon_core::InputError>>,
) -> anyhow::Result<Vec<Loaded<'a>>> {
    videos
        .par_iter()
        .filter_map(|file| {
            let loaded = (|| {
                let remote =
                    Remote::load(&file.path).map_err(|error| eidolon_core::InputError {
                        file: file.id,
                        reason: format!("{error:#}"),
                    })?;
                let Some(video) = frames::load_with_remote(file, ctx, remote.as_ref())? else {
                    return Err(eidolon_core::InputError {
                        file: file.id,
                        reason: "no decodable video stream".to_owned(),
                    }
                    .into());
                };
                Ok::<_, anyhow::Error>(Some(Loaded {
                    file,
                    video,
                    remote,
                }))
            })();
            match loaded {
                Ok(loaded) => loaded.map(Ok),
                Err(error) => match error
                    .downcast::<eidolon_core::InputError>()
                    .and_then(|input| crate::record_skip(skipped, input))
                {
                    Ok(()) => None,
                    Err(error) => Some(Err(error)),
                },
            }
        })
        .collect()
}

pub(super) fn feature_key(model: &str) -> String {
    format!("{IMAGE_FEATURE};model={model}")
}

pub(super) fn uncached_embedding_work(file: &FileEntry, ctx: &Ctx<'_>) -> anyhow::Result<usize> {
    let Some(model) = ctx.embedder.filter(|_| !ctx.config.no_embeddings) else {
        return Ok(0);
    };
    for id in model.frame_cache_model_ids().into_iter().flatten() {
        if ctx.cache.contains(file, &feature_key(id))? {
            return Ok(0);
        }
    }
    Ok(1)
}

fn load_image(file: &FileEntry, ctx: &Ctx<'_>) -> anyhow::Result<Frame> {
    let embedder = ctx.embedder.filter(|_| !ctx.config.no_embeddings);
    let ids = embedder.map_or([Some("none"), None], Embedder::frame_cache_model_ids);
    for id in ids.into_iter().flatten() {
        if let Some(bytes) = ctx.cache.get(file, &feature_key(id))? {
            return serde_json::from_slice(&bytes).context("reading cached image-frame features");
        }
    }
    let decoded = eidolon_image::decode(&file.path).map_err(|error| eidolon_core::InputError {
        file: file.id,
        reason: format!("{error:#}"),
    })?;
    let aspect = f64::from(decoded.width()) / f64::from(decoded.height());
    let decoded = decoded.resize_exact(frames::WIDTH, frames::HEIGHT, FilterType::Triangle);
    let rgb = if decoded.color().has_alpha() {
        let rgba = decoded.into_rgba8();
        RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
            let pixel = rgba.get_pixel(x, y).0;
            let alpha = u16::from(pixel[3]);
            Rgb(std::array::from_fn(|channel| {
                ((u16::from(pixel[channel]) * alpha + 255 * (255 - alpha) + 127) / 255) as u8
            }))
        })
    } else {
        decoded.into_rgb8()
    };
    let mut frame = frames::describe(&rgb, 0.0);
    if let Some(model) = embedder {
        match model
            .prepare()
            .and_then(|()| model.embed_frame(&rgb))
            .and_then(frames::normalize)
        {
            Ok(vector) => frame.embedding = Some(vector),
            Err(error) => eprintln!(
                "eidolon-video: {}: image embedding failed; retaining hashes: {error:#}",
                file.path.display()
            ),
        }
    }
    frame.retain_pixels(rgb, aspect);
    if let Some(pixels) = &frame.pixels {
        frame.overlay = crate::overlay::Band::detect(pixels.rgb.as_chunks::<3>().0, pixels.bounds);
    }
    let key = feature_key(embedder.map_or("none", Embedder::frame_model_id));
    ctx.cache.put(file, &key, &serde_json::to_vec(&frame)?)?;
    Ok(frame)
}

fn rank(image: &Frame, frame: &Frame) -> f64 {
    let distance =
        (image.phash ^ frame.phash).count_ones() + (image.dhash ^ frame.dhash).count_ones();
    let hash = 1.0 - f64::from(distance) / 128.0;
    let embedding = image
        .embedding
        .as_ref()
        .zip(frame.embedding.as_ref())
        .and_then(|(a, b)| frames::cosine(a, b))
        .unwrap_or(hash);
    hash.midpoint(embedding)
}

fn seeds(
    image: &Frame,
    loaded: &[Loaded<'_>],
    index: &Index,
    hits: &mut HashSet<FrameRef>,
    remote_overlay: bool,
) -> HashMap<usize, Vec<Seed>> {
    let mut candidates: HashMap<usize, Vec<Seed>> = HashMap::new();
    if image.overlay.is_some() {
        // Missing pixels cannot enter a whole-image hash. A bounded masked
        // spatial signature nominates and ranks each video's sampled frames.
        for (video, source) in loaded.iter().enumerate() {
            let aspect = f64::from(source.video.width) / f64::from(source.video.height);
            for (frame_index, frame) in source.video.frames.iter().enumerate() {
                let rank = if remote_overlay && source.remote.is_some() {
                    crate::appearance::overlay_remote_candidate(image, frame, aspect)
                } else {
                    crate::appearance::overlay_candidate(image, frame, aspect)
                };
                if let Some(rank) = rank {
                    insert_seed(
                        candidates
                            .entry(video)
                            .or_insert_with(|| Vec::with_capacity(SEEDS)),
                        Seed {
                            frame: frame_index,
                            rank,
                        },
                    );
                }
            }
        }
        return candidates;
    }
    index.query(image, hits);
    for reference in hits.iter() {
        candidates
            .entry(reference.video)
            .or_insert_with(|| Vec::with_capacity(SEEDS));
    }
    // A bucket collision nominates a video, not its best source timestamp.
    // Rank its bounded samples independently before opening native windows.
    for (&video, candidates) in &mut candidates {
        for (frame_index, frame) in loaded[video].video.frames.iter().enumerate() {
            let rank = rank(image, frame);
            let seed = Seed {
                frame: frame_index,
                rank,
            };
            insert_seed(candidates, seed);
        }
    }
    candidates
}

fn insert_seed(candidates: &mut Vec<Seed>, seed: Seed) {
    let insertion = candidates.partition_point(|other| {
        other
            .rank
            .total_cmp(&seed.rank)
            .reverse()
            .then_with(|| other.frame.cmp(&seed.frame))
            .is_lt()
    });
    if insertion < SEEDS {
        if candidates.len() == SEEDS {
            candidates.pop();
        }
        candidates.insert(insertion, seed);
    }
}

fn windows(seeds: &[Seed], video: &Video, masked: bool) -> Vec<(f64, f64)> {
    let frame_duration = 1.0 / video.fps;
    let mut ranges: Vec<_> = seeds
        .iter()
        .map(|seed| {
            let frame = &video.frames[seed.frame];
            let spacing = if masked || frame.embedding.is_some() {
                video.step.max(frames::EMBED_INTERVAL * 0.5)
            } else {
                video.step
            };
            let radius = spacing + frame_duration;
            (
                (frame.time - radius).max(0.0),
                (frame.time + radius).min(video.duration),
            )
        })
        .collect();
    ranges.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        if let Some(previous) = merged.last_mut()
            && start <= previous.1
        {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
        .into_iter()
        .map(|(start, end)| (start.midpoint(end), (end - start) * 0.5))
        .collect()
}

fn remote_windows<'a>(
    seeds: &'a [Seed],
    video: &'a Video,
    masked: bool,
    enabled: bool,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let caption = masked && enabled;
    let ordinary = (!caption).then(|| windows(seeds, video, masked));
    let caption = caption.then(|| seed_windows(seeds, video));
    ordinary
        .into_iter()
        .flatten()
        .chain(caption.into_iter().flatten())
}

fn seed_windows<'a>(seeds: &'a [Seed], video: &'a Video) -> impl Iterator<Item = (f64, f64)> + 'a {
    let step = (1.0 / video.fps).max(0.5);
    let radius = video.step.max(frames::EMBED_INTERVAL * 0.5) + 1.0 / video.fps;
    let extent = (radius / step).ceil() as i64 + 1;
    // Keep the strongest proposals first, then expand each seed's neighborhood.
    // One absolute prefix grid lets overlapping seeds share requests without a set.
    (0..=extent).flat_map(move |distance| {
        (0..seeds.len() * 2).filter_map(move |slot| {
            if distance == 0 && slot % 2 != 0 {
                return None;
            }
            let position = slot / 2;
            let time = video.frames[seeds[position].frame].time;
            // A prefix starting before this keyframe needs the preceding long GOP.
            let center = (time / step).ceil() as i64;
            let index = center + if slot % 2 == 0 { -distance } else { distance };
            let start = index as f64 * step;
            let end = (start + step).min(video.duration);
            if index < 0 || start >= video.duration || end < time - radius || start > time + radius
            {
                return None;
            }
            if seeds.iter().enumerate().any(|(other, seed)| {
                let time = video.frames[seed.frame].time;
                let center = (time / step).ceil() as i64;
                end >= time - radius
                    && start <= time + radius
                    && (index.abs_diff(center), other) < (distance as u64, position)
            }) {
                return None;
            }
            Some((start.midpoint(end), (end - start) * 0.5))
        })
    })
}

fn match_video(
    source: &Loaded<'_>,
    images: &[(&FileEntry, Frame)],
    mut queries: Vec<(usize, Vec<Seed>)>,
    parallel_queries: bool,
    ctx: &Ctx<'_>,
) -> Vec<Match> {
    let mut matches = Vec::new();
    if ctx.config.fast {
        queries.retain(|(index, seeds)| {
            let (file, image) = &images[*index];
            if let Some((quality, time)) =
                thumbnail_seed(image, source, seeds, ctx.config.min_score)
            {
                matches.push(frame_match(file, source, quality, time, false, true));
                false
            } else {
                true
            }
        });
    }
    let caption_refinement = ctx.config.remote_overlay
        && source.remote.is_some()
        && queries
            .iter()
            .any(|(image, _)| images[*image].1.overlay.is_some());
    if caption_refinement {
        let rank = |query: &(usize, Vec<Seed>)| {
            if images[query.0].1.overlay.is_some() {
                query.1[0].rank
            } else {
                f64::NEG_INFINITY
            }
        };
        // Native allowance belongs to the source, not the input's position.
        queries.sort_unstable_by(|a, b| rank(b).total_cmp(&rank(a)).then(a.0.cmp(&b.0)));
    }
    if queries.is_empty() {
        return matches;
    }
    let pixels = if queries.len() > 1 {
        let decoded = if let Some(remote) = &source.remote {
            let indices: Vec<_> = (0..source.video.frames.len()).collect();
            frames::retained(source.file, &source.video, remote, &indices).map(Some)
        } else {
            frames::native(source.file, &source.video, ctx.cache)
        };
        match decoded {
            Ok(pixels) => pixels,
            Err(error) => {
                eprintln!(
                    "eidolon-video: shared pixels for {} failed: {error:#}",
                    source.file.path.display()
                );
                None
            }
        }
    } else {
        None
    };
    if source.remote.is_none()
        && let Some(pixels) = pixels.as_deref()
    {
        matches.extend(match_timeline(source, images, &queries, pixels, ctx));
        return matches;
    }
    let refinement = Refinement::new(source.video.max_frames);
    let find = |(image, seeds): &(usize, Vec<Seed>)| {
        let (file, image) = &images[*image];
        match_image(
            file,
            image,
            source,
            seeds,
            pixels.as_deref(),
            ctx,
            &refinement,
        )
    };
    if parallel_queries && !caption_refinement {
        matches.par_extend(queries.par_iter().filter_map(find));
    } else {
        matches.extend(queries.iter().filter_map(find));
    }
    if ctx.config.timings {
        refinement.timing.write(source.file);
    }
    matches
}

struct Query {
    image: usize,
    ranges: Vec<std::ops::Range<usize>>,
    best: Option<(f64, f64)>,
    complete: bool,
}

fn match_timeline(
    source: &Loaded<'_>,
    images: &[(&FileEntry, Frame)],
    queries: &[(usize, Vec<Seed>)],
    pixels: &[Frame],
    ctx: &Ctx<'_>,
) -> Vec<Match> {
    let Some(first) = pixels.first() else {
        return Vec::new();
    };
    let mut views = crate::appearance::FrameViews::new(first);
    let mut pending: Vec<_> = queries
        .iter()
        .map(|(index, seeds)| {
            let (_, image) = &images[*index];
            let mut best = None;
            for seed in seeds {
                consider(image, &source.video.frames[seed.frame], &mut best);
            }
            let ranges = windows(seeds, &source.video, image.overlay.is_some())
                .into_iter()
                .map(|(center, radius)| {
                    let first = pixels.partition_point(|frame| frame.time + 1e-7 < center - radius);
                    let last = pixels.partition_point(|frame| frame.time + 1e-7 < center + radius);
                    first..last
                })
                .collect();
            Query {
                image: *index,
                ranges,
                best,
                complete: false,
            }
        })
        .collect();
    for (index, frame) in pixels.iter().enumerate() {
        if pending.iter().all(|query| query.complete) {
            break;
        }
        for query in &mut pending {
            if query.complete || !query.ranges.iter().any(|range| range.contains(&index)) {
                continue;
            }
            let (_, image) = &images[query.image];
            if let Some(quality) = crate::appearance::frame_match_cached(image, frame, &mut views)
                && query.best.is_none_or(|(previous, _)| quality > previous)
            {
                query.best = Some((quality, frame.time));
            }
            query.complete = query
                .best
                .is_some_and(|(quality, _)| 0.75 + quality * 0.25 >= ctx.config.min_score);
        }
    }
    pending
        .into_iter()
        .filter_map(|query| {
            let (quality, time) = query.best?;
            let (file, _) = &images[query.image];
            (0.75 + quality * 0.25 >= ctx.config.min_score)
                .then(|| frame_match(file, source, quality, time, false, false))
        })
        .collect()
}

fn thumbnail_seed(
    image: &Frame,
    source: &Loaded<'_>,
    seeds: &[Seed],
    min_score: f64,
) -> Option<(f64, f64)> {
    for seed in seeds {
        let frame = &source.video.frames[seed.frame];
        if let Some(quality) = crate::appearance::thumbnail_match(image, frame)
            && 0.75 + quality * 0.25 >= min_score
        {
            return Some((quality, frame.time));
        }
    }
    None
}

fn match_image(
    file: &FileEntry,
    image: &Frame,
    source: &Loaded<'_>,
    seeds: &[Seed],
    pixels: Option<&[Frame]>,
    ctx: &Ctx<'_>,
    refinement: &Refinement,
) -> Option<Match> {
    let mut best = None;
    for seed in seeds {
        consider(image, &source.video.frames[seed.frame], &mut best);
    }
    let refined = refine_candidates(image, source, seeds, pixels, ctx, &mut best, refinement);
    let (quality, time) = best?;
    (0.75 + quality * 0.25 >= ctx.config.min_score)
        .then(|| frame_match(file, source, quality, time, refined, false))
}

fn frame_match(
    file: &FileEntry,
    source: &Loaded<'_>,
    quality: f64,
    time: f64,
    refined: bool,
    thumbnail: bool,
) -> Match {
    let mut evidence = vec![
        Evidence {
            signal: "frame_timestamp",
            value: time,
            note: format!(
                "timestamp_seconds={time:.6}; sparse_proxy={}",
                source.remote.is_some()
            ),
        },
        Evidence {
            signal: if thumbnail { "thumbnail" } else { "frame_spatial" },
            value: quality,
            note: if thumbnail {
                "32x32 cached RGB pixels verify the complete frame rectangle; semantic similarity alone is insufficient"
            } else {
                "independently verified spatial pixels; semantic similarity alone is insufficient"
            }
            .to_owned(),
        },
    ];
    if refined {
        evidence.push(Evidence {
            signal: "frame_refinement",
            value: 1.0,
            note:
                "verified native-cadence provider window; timestamp mapped to original source clock"
                    .to_owned(),
        });
    }
    Match {
        a: file.id,
        b: source.file.id,
        relation: Relation::FrameOf,
        score: 0.75 + quality * 0.25,
        evidence,
    }
}

fn consider(image: &Frame, frame: &Frame, best: &mut Option<(f64, f64)>) -> bool {
    if let Some(quality) = crate::appearance::frame_match(image, frame)
        && best.is_none_or(|(previous, _)| quality > previous)
    {
        *best = Some((quality, frame.time));
        return true;
    }
    false
}

fn refine_candidates(
    image: &Frame,
    source: &Loaded<'_>,
    seeds: &[Seed],
    pixels: Option<&[Frame]>,
    ctx: &Ctx<'_>,
    best: &mut Option<(f64, f64)>,
    refinement: &Refinement,
) -> bool {
    if let Some(remote) = &source.remote {
        if let Some(pixels) = pixels {
            for seed in seeds {
                consider(image, &pixels[seed.frame], best);
            }
        } else {
            let indices: Vec<_> = seeds.iter().map(|seed| seed.frame).collect();
            match frames::retained(source.file, &source.video, remote, &indices) {
                Ok(frames) => {
                    for frame in frames {
                        consider(image, &frame, best);
                    }
                }
                Err(error) => eprintln!(
                    "eidolon-video: retained pixels for {} failed: {error:#}",
                    source.file.path.display()
                ),
            }
        }
        // A retained frame needs no extra transfer only when its score is sufficient.
        let Some(provider) = ctx.refiner.filter(|_| {
            !best.is_some_and(|(quality, _)| 0.75 + quality * 0.25 >= ctx.config.min_score)
        }) else {
            return false;
        };
        // Prove bounded native prefixes before asking for any later packets.
        // Range-cache blocks retain decoder dependencies across overlapping queries.
        let step = (1.0 / source.video.fps).max(0.5);
        for (center, radius) in remote_windows(
            seeds,
            &source.video,
            image.overlay.is_some(),
            ctx.config.remote_overlay,
        ) {
            let mut start = (center - radius).max(0.0);
            let end = (center + radius).min(source.video.duration);
            while start < end - 1e-7 {
                let next = (start + step).min(end);
                let center = start.midpoint(next);
                let radius = (next - start) * 0.5;
                let result = refinement.refine(source.file, provider, center, radius);
                match result {
                    Ok(Some(frames)) => {
                        if refinement.prove(&frames, image, best, ctx.config.min_score) {
                            return true;
                        }
                    }
                    // Skip this window; another independent window may be cheaper or cached.
                    Ok(None) => break,
                    Err(error) => {
                        eprintln!(
                            "eidolon-video: frame refinement for {} failed: {error:#}",
                            source.file.path.display()
                        );
                        break;
                    }
                }
                start = next;
            }
        }
        return false;
    }
    for (center, radius) in windows(seeds, &source.video, image.overlay.is_some()) {
        if let Some(pixels) = pixels {
            let start = center - radius;
            let end = center + radius;
            let first = pixels.partition_point(|frame| frame.time + 1e-7 < start);
            let last = pixels.partition_point(|frame| frame.time + 1e-7 < end);
            for frame in &pixels[first..last] {
                consider(image, frame, best);
                if best.is_some_and(|(quality, _)| 0.75 + quality * 0.25 >= ctx.config.min_score) {
                    return false;
                }
            }
            continue;
        }
        match frames::refine(source.file, &source.video, center, radius) {
            Ok(frames) => {
                for frame in frames {
                    consider(image, &frame, best);
                    if best
                        .is_some_and(|(quality, _)| 0.75 + quality * 0.25 >= ctx.config.min_score)
                    {
                        return false;
                    }
                }
            }
            Err(error) => eprintln!(
                "eidolon-video: frame refinement for {} failed: {error:#}",
                source.file.path.display()
            ),
        }
    }
    false
}
