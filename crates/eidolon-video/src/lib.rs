//! Temporally verified video duplicates and spatially verified image-to-video frames.
//!
//! Requires `ffprobe` and `ffmpeg` on `PATH`; both are restricted to local
//! `file,pipe` protocols. Extraction uses a Rayon pool with `Config::jobs` workers
//! and one decoder thread per video. Local extraction honors `Config::video_fps`;
//! its cap is ten minutes of configured samples (600 at 1 fps, 1,200 at 2 fps).
//!
//! Each frame has global perceptual/difference hashes, four regional difference
//! hashes, 4×4 mean colors, and 32×32 grayscale/RGB thumbnails (4 KiB total).
//! Uniform near-black margins covering at most one quarter of each edge are
//! excluded; textured dark scene edges remain. Six rank-normalized luma/RGB
//! hashes also nominate hue/tone variants through shared projection tables,
//! probing both contrast signs and one-bit neighbors.
//! Optional injected embeddings are sampled every three seconds. Versioned cache
//! keys include sampling settings and model identity. Twenty-eight ordinary hash
//! LSH tables and eight embedding projection tables produce frame candidates.
//! Video and thumbnail-side retrieval use the embedder's frame-specific inference
//! and cache identities, independently of ordinary image-to-image preprocessing.
//!
//! Verification votes on temporal offsets, tests the eight strongest offsets,
//! and requires three matching frames, 75% coverage, and a consistent run covering
//! 65% of samples. Global hash gates allow at most 12/63 perceptual bits, 18/64
//! difference bits, and 0.13 normalized color distance. Local overlays may instead
//! match half the color-consistent spatial tiles, including four textured tiles
//! with correlation ≥0.85. Hash/spatial runs also require aligned temporal changes
//! with cosine ≥0.5 when there is motion; static backgrounds cannot establish order.
//! Semantic candidates require cosine ≥0.95, temporal contrast, and independent
//! spatial agreement under one fixed crop mapping.
//! Scores combine coverage and visual fit, then honor `Config::min_score`.
//!
//! A broad color/luma change is a related [`eidolon_core::Relation::Variant`]
//! when a clipped affine RGB transform explains at least 90% of a visible color
//! change (RMS ≥12/255) on eight structurally aligned tiles, including four
//! textured tiles; at most two contaminated tiles may be discarded.
//! Both color-model families require visible changes across three spatial rows
//! and three columns; a single edited band cannot establish a global variant.
//! For nonlinear tone changes, a bounded monotone tone family is fitted to 64
//! dispersed pixels and independently verified against all 1,024 RGB pixels.
//! It must explain ≥90% of both target variance and color change, preserve signed
//! RGB gradients (cosine ≥0.9), and support six textured regions across three rows
//! and columns. Fine RGB samples prevent opposite color changes averaging away.
//! At least three color-observable frames must resolve, with a variant in at least
//! half. Unresolved codec noise does not vote for unchanged colors.
//! Broad unresolved recoloring is rejected, never admitted as a duplicate clip.
//! Variants remain symmetric even when one sequence is shorter.
//! Otherwise, persistent localized changes indicate an edit; full-duration
//! matches without them are re-encodings. Compression-related hash misses do not
//! by themselves indicate an edit. Contained shorter sequences are directional
//! clips with source offsets in evidence.
//! Short clips in very long videos can fall between adaptive samples;
//! fewer than three aligned samples are insufficient evidence. No bytes are hashed for exact
//! duplicate detection, and no source file is written.
//! `Config::fast` keeps indexed hash/embedding candidates and temporal alignment,
//! with cached 32×32 RGB/texture corroboration. Semantic similarity alone never
//! establishes a match. Proven matches carry `thumbnail` evidence, not `unverified`.
//! With embeddings on both timelines, fast nomination defers global color-model
//! fitting until temporal verification. Hash-only timelines retain color-model
//! nomination; neither route bypasses the independent final proof.
//!
//! [`find_frames`] uses the same frame indexes to nominate videos, then ranks their
//! bounded samples independently: a bucket collision need not be the best timestamp.
//! Spatial color/texture consensus verifies resized frames, crops, branding, and
//! images containing a video viewport; semantic resemblance is never sufficient.
//! Original-aspect 96px previews are cropped before reducing to 32px descriptors.
//! Text-bearing uniform horizontal bands are detected on image previews. Such
//! queries use masked color/correlation nomination, not caption-bearing hashes or
//! embeddings. Dense RGB and masked normalized correlation independently verify
//! at least 70% visible support; text pixels and their resampling halo do not vote.
//! Padding without repeated contrast strokes does not create a caption mask.
//! Independent subpixel crop-edge refinement requires eight color-stable spatial
//! tiles, including four textured tiles, with grayscale correlation ≥0.9 and
//! untransformed RGB RMS error <12/255. The complete claimed rectangle must also
//! agree at 32×32 resolution, allowing one localized overlay of at most one-quarter
//! area and at most 1/32 scattered resampling outliers. Matching a shared subset
//! of a different composition, or a related recolor, does not prove containment.
//! Bounded native-FPS windows refine promising timestamps in local videos. Short,
//! constant-cadence sources share one decoded native timeline across still-image
//! queries; their actual presentation cadence is checked and version-cached.
//! Longer or variable-cadence sources retain window-specific decoding. Sources
//! run without batch barriers, with at most one resident native timeline per worker.
//! Candidate generation and still-image verification use the configured worker pool.
//! Conservative native RGB bounds and bounded coarse search reject losing crops
//! before dense subpixel fitting. Area-overlap weights, full previews, and exact
//! coarse crop projections are reused across queries; each native worker's image
//! projection cache is bounded to 24 MiB plus map overhead.
//! Native refinement frames omit nomination-only color-hash planes; their full
//! RGB/texture proofs are unchanged. Indexed feature-cache bytes stay compatible.
//! Verification stops at the first strictly proven timestamp meeting the score gate.
//! Sparse remote proxies use `<proxy>.remote.json` for their original duration,
//! source FPS, and retained frame PTS. Native VFR decoding never duplicates samples
//! into artificial observations. The configured sample cap applies; source-clock
//! changes invalidate their cache. An optional [`eidolon_core::FrameRefiner`] can
//! supply native-cadence windows when retained samples cannot establish containment.
//! Fast mode first verifies sampled 32×32 RGB rectangles, before native decoding.
//! It removes queries at the first independently proven sample meeting the score
//! gate. Unresolved cases retain normal native-pixel refinement, rather than
//! being accepted on hashes or semantic resemblance alone.

mod alignment;
mod appearance;
mod candidates;
mod frame_images;
mod frames;
mod overlay;

#[cfg(test)]
mod tests;

use anyhow::Context as _;
use eidolon_core::{Ctx, FileEntry, Match};
use rayon::prelude::*;

/// Estimate uncached embeddings for the video and optional image-to-video stages.
///
/// Queries the same fresh, provider-qualified keys used by extraction, without
/// decoding media or initializing inference. An uncached local video contributes
/// its configured maximum frame count; remote proxies use their retained count.
/// Each uncached image-to-video descriptor contributes one. Non-dispatched
/// stages and disabled embeddings contribute zero.
///
/// # Errors
/// Fails if the cache cannot be queried or the sampling rate is invalid.
pub fn uncached_embedding_work(
    files: &[FileEntry],
    ctx: &Ctx<'_>,
    thumbnails: bool,
) -> anyhow::Result<usize> {
    let videos = files
        .iter()
        .filter(|file| file.kind == eidolon_core::Kind::Video)
        .count();
    let images = files
        .iter()
        .filter(|file| file.kind == eidolon_core::Kind::Image)
        .count();
    let cross_media = thumbnails && videos > 0 && images > 0;
    let mut work = 0_usize;
    for file in files {
        let missing = match file.kind {
            eidolon_core::Kind::Video => frames::uncached_embedding_work(file, ctx)?,
            eidolon_core::Kind::Image if cross_media => {
                frame_images::uncached_embedding_work(file, ctx)?
            }
            _ => 0,
        };
        work = work.saturating_add(missing);
    }
    Ok(work)
}

/// Find temporally aligned re-encodings, local edits, related variants, and video clips.
///
/// Frames are streamed through `FFmpeg`, cached, and indexed before temporal
/// verification. Unreadable inputs are reported to stderr and skipped.
/// No input file is modified.
/// Sampling honors [`eidolon_core::Config::video_fps`]; rates have separate caches.
/// Fast mode corroborates candidates with cached 32×32 RGB/texture evidence.
///
/// # Errors
/// Returns cache, setup, or worker pool errors.
pub fn find(files: &[FileEntry], ctx: &Ctx<'_>) -> anyhow::Result<Vec<Match>> {
    let (matches, skipped) = find_with_skipped(files, ctx)?;
    for input in skipped {
        if let Some(file) = files.iter().find(|file| file.id == input.file) {
            eprintln!(
                "eidolon: skipped {}: {}",
                file.path.display(),
                input.reason.replace(['\r', '\n'], " ")
            );
        }
    }
    Ok(matches)
}

/// Find video matches and return per-input failures without printing them.
///
/// Failed inputs never participate in matches. Callers retain their stable IDs
/// for reporting and for excluding them from later image-to-video comparisons.
///
/// # Errors
/// Returns cache, setup, or worker pool errors; input failures are returned separately.
pub fn find_with_skipped(
    files: &[FileEntry],
    ctx: &Ctx<'_>,
) -> anyhow::Result<(Vec<Match>, Vec<eidolon_core::InputError>)> {
    let skipped = std::sync::Mutex::new(Vec::new());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(ctx.config.jobs)
        .build()
        .context("creating video extraction worker pool")?;
    let loaded: Vec<_> = pool.install(|| {
        files
            .par_iter()
            .filter_map(|file| {
                let input = match frames::load(file, ctx) {
                    Ok(Some(video)) => return Some(Ok((file, video))),
                    Ok(None) => eidolon_core::InputError {
                        file: file.id,
                        reason: "no decodable video stream".to_owned(),
                    },
                    Err(error) => match error.downcast::<eidolon_core::InputError>() {
                        Ok(input) => input,
                        Err(error) => return Some(Err(error)),
                    },
                };
                if let Err(error) = record_skip(&skipped, input) {
                    return Some(Err(error));
                }
                None
            })
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    let (files, videos): (Vec<_>, Vec<_>) = loaded.into_iter().unzip();
    let candidates = pool.install(|| candidates::generate(&videos, ctx.config.fast));
    let mut matches: Vec<_> = pool.install(|| {
        candidates
            .into_par_iter()
            .filter_map(|((a, b), anchors)| {
                alignment::compare(
                    files[a],
                    &videos[a],
                    files[b],
                    &videos[b],
                    &anchors,
                    ctx.config.min_score,
                    ctx.config.fast,
                )
            })
            .collect()
    });
    matches.sort_unstable_by_key(|item| (item.a, item.b));
    let mut skipped = skipped
        .into_inner()
        .map_err(|_| anyhow::anyhow!("video failure list poisoned"))?;
    skipped.sort_unstable_by_key(|input| input.file);
    Ok((matches, skipped))
}

pub(crate) fn record_skip(
    skipped: &std::sync::Mutex<Vec<eidolon_core::InputError>>,
    input: eidolon_core::InputError,
) -> anyhow::Result<()> {
    skipped
        .lock()
        .map_err(|_| anyhow::anyhow!("video failure list poisoned"))?
        .push(input);
    Ok(())
}

/// Find images that show a frame of a video, including scaled thumbnails and overlays.
///
/// Hash and injected embedding indexes nominate videos. Their bounded sampled
/// frames are ranked independently to select refinement times, including times
/// that did not collide with the query's index buckets.
/// Independent spatial pixels establish containment in both default and fast modes.
/// Local videos are decoded around promising timestamps to resolve frames between
/// samples. Sparse remote proxies first verify their actual retained samples, then
/// request native-cadence pixels from the optional [`Ctx::refiner`] when needed.
/// Every match is directional (`a` is the image, `b` the video), and includes
/// `frame_timestamp` evidence in source seconds. Default mode rejects global
/// recolors; `frame_refinement` marks improvements from a provider's native window.
/// Fast mode first tries cached low-resolution RGB proof, marked by `thumbnail`
/// evidence. Unresolved cases retain strict native-cadence pixel refinement.
/// Invalid inputs are reported and skipped.
///
/// # Errors
/// Returns cache, setup, or worker pool errors.
pub fn find_frames(
    images: &[FileEntry],
    videos: &[FileEntry],
    ctx: &Ctx<'_>,
) -> anyhow::Result<Vec<Match>> {
    let (matches, skipped) = find_frames_with_skipped(images, videos, ctx)?;
    for input in skipped {
        if let Some(file) = images
            .iter()
            .chain(videos)
            .find(|file| file.id == input.file)
        {
            eprintln!(
                "eidolon: skipped {}: {}",
                file.path.display(),
                input.reason.replace(['\r', '\n'], " ")
            );
        }
    }
    Ok(matches)
}

/// Find image-to-video matches and return per-input failures without printing them.
///
/// Failed inputs never participate in the returned matches.
/// Set [`eidolon_core::Config::timings`] to emit native-refinement timing diagnostics.
///
/// # Errors
/// Returns cache, setup, or worker pool errors; input failures are returned separately.
pub fn find_frames_with_skipped(
    images: &[FileEntry],
    videos: &[FileEntry],
    ctx: &Ctx<'_>,
) -> anyhow::Result<(Vec<Match>, Vec<eidolon_core::InputError>)> {
    let skipped = std::sync::Mutex::new(Vec::new());
    let matches = frame_images::find(images, videos, ctx, &skipped)?;
    let mut skipped = skipped
        .into_inner()
        .map_err(|_| anyhow::anyhow!("video failure list poisoned"))?;
    skipped.sort_unstable_by_key(|input| input.file);
    Ok((matches, skipped))
}
