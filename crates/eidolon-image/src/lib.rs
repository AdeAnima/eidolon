//! Report-only near-duplicate image detection with cached visual features.

pub mod embed;
pub mod models;

mod color;
mod detect;
mod features;
mod geometry;
mod heif;
#[cfg(target_os = "macos")]
mod imageio;
mod matching;
mod preview;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Once;

use anyhow::Context as _;
use eidolon_core::{Ctx, FileEntry, FileId, Match};
use image::ImageDecoder as _;
use rayon::prelude::*;

static HEIF_HOOKS: Once = Once::new();

/// Decode a supported still image and apply its embedded EXIF orientation.
///
/// HEIF primary-item irot/imir transformations take precedence over redundant
/// EXIF orientation, preventing iPhone photographs from being rotated twice.
///
/// # Errors
/// Returns an error when the input cannot be read or decoded.
pub fn decode(path: &Path) -> anyhow::Result<image::DynamicImage> {
    decode_with_profile(path, false).map(|(image, _)| image)
}

fn decode_with_profile(
    path: &Path,
    include_profile: bool,
) -> anyhow::Result<(image::DynamicImage, Option<moxcms::ColorProfile>)> {
    HEIF_HOOKS.call_once(libheif_rs::integration::image::register_all_decoding_hooks);
    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("decoding {}", path.display()))?;
    let profile = if include_profile {
        color::profile(path, &mut decoder)?
    } else {
        None
    };
    let mut orientation = decoder.orientation()?;
    // The HEIF hook applies irot/imir itself but does not expose EXIF metadata.
    if orientation == image::metadata::Orientation::NoTransforms {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        if let Ok(exif) = exif::Reader::new().read_from_container(&mut reader)
            && let Some(field) = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
            && let Some(value) = field.value.get_uint(0)
            && let Ok(value) = u8::try_from(value)
            && let Some(exif_orientation) = image::metadata::Orientation::from_exif(value)
            && exif_orientation != image::metadata::Orientation::NoTransforms
            && !heif::has_native_orientation(path)?
        {
            orientation = exif_orientation;
        }
    }
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok((image, profile))
}

/// Count still-image embedding cache misses without decoding or initializing inference.
///
/// Accepts mixed file kinds and honors disabled embeddings and equivalent
/// provider-qualified cache identities. Each uncached whole-image embedding
/// contributes one. `include_source_centers` also counts independent cache misses
/// for [`source_frame_embeddings`], used for still-image candidate retrieval.
///
/// # Errors
/// Fails if the feature cache cannot be queried.
pub fn uncached_embedding_work(
    files: &[FileEntry],
    ctx: &Ctx<'_>,
    include_source_centers: bool,
) -> anyhow::Result<usize> {
    features::uncached_embedding_work(files, ctx, include_source_centers)
}

/// Load normalized source-aspect center embeddings for additional image candidates.
///
/// Uses the still-image retrieval decode, retaining source aspect ratio, with the
/// embedder's frame preprocessing and independent preprocessing/provider-qualified
/// cache entries. Use for both image-only and mixed-media scans, independently
/// of thumbnail matching. Disabled or unavailable embeddings produce an empty
/// bank. Decode failures are omitted so [`find`] can report them normally.
///
/// # Errors
/// Cache, model inference, invalid vectors, and worker-pool failures remain fatal.
pub fn source_frame_embeddings(
    files: &[FileEntry],
    ctx: &Ctx<'_>,
) -> anyhow::Result<HashMap<FileId, Vec<f32>>> {
    if ctx.config.no_embeddings
        || ctx.embedder.is_none()
        || !files
            .iter()
            .any(|file| file.kind == eidolon_core::Kind::Image)
    {
        return Ok(HashMap::new());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(ctx.config.jobs)
        .build()?;
    pool.install(|| {
        files
            .par_iter()
            .filter(|file| file.kind == eidolon_core::Kind::Image)
            .filter_map(|file| match features::source_frame_embedding(file, ctx) {
                Ok(vector) => Some(Ok((file.id, vector))),
                Err(error) if error.downcast_ref::<eidolon_core::InputError>().is_some() => None,
                Err(error) => Some(Err(error)),
            })
            .collect()
    })
}

/// Find near duplicates without modifying any input file.
///
/// Versioned per-file features are cached. Hash/embedding candidates need
/// geometric and photometric confirmation. `config.fast` uses a distributed
/// 32-by-32 aligned RGB sample instead of the full cached 128-square preview,
/// with `thumbnail` evidence. Both modes reject proof-free semantic candidates
/// and retain crop, overlay, structural, and colour-profile checks.
/// `jobs` controls extraction and verification worker threads.
/// An optional normalized bank from [`source_frame_embeddings`] nominates
/// additional image neighbors without replacing whole-image candidates or
/// bypassing pixel proof. The CLI supplies this bank for every image scan with
/// embeddings enabled, independently of thumbnail matching.
///
/// # Errors
/// Returns [`eidolon_core::InputError`] for unreadable/undecodable inputs.
/// Invalid cache data, failed model inference, native feature extraction errors,
/// and unavailable worker threads remain run-wide errors.
#[expect(
    clippy::implicit_hasher,
    reason = "The bank has one concrete producer; generic hashers make None callers ambiguous."
)]
pub fn find(
    files: &[FileEntry],
    ctx: &Ctx<'_>,
    frame_embeddings: Option<&HashMap<FileId, Vec<f32>>>,
) -> anyhow::Result<Vec<Match>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(ctx.config.jobs)
        .build()?;
    pool.install(|| {
        let features = files
            .par_iter()
            .map(|file| features::extract(file, ctx))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let frame_vectors = frame_embeddings
            .filter(|bank| !ctx.config.no_embeddings && !bank.is_empty())
            .map(|bank| {
                files
                    .iter()
                    .map(|file| bank.get(&file.id).map(Vec::as_slice))
                    .collect::<Vec<_>>()
            });
        let mut candidates =
            detect::candidates(&features, frame_vectors.as_deref().unwrap_or_default());
        let (files, features) = features::prepare(files, features, &mut candidates, ctx)?;
        let mut matches = detect::verify(&files, &features, &candidates, ctx)?;
        matches.sort_unstable_by_key(|item| (item.a, item.b));
        Ok(matches)
    })
}
