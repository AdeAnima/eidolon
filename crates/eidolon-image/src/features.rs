use std::borrow::Cow;
use std::sync::{LazyLock, OnceLock};

use anyhow::{Context as _, ensure};
use eidolon_core::{Ctx, FileEntry};
use image::{RgbImage, imageops::FilterType};
use image_hasher::{HashAlg, HasherConfig};
use opencv::{core, features, prelude::*};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest as _, Sha256};

pub(crate) const PREVIEW: u32 = 128;
pub(crate) const REGISTRATION: u32 = 64;
const VISUAL_KEY: &str = "image.visual-phash-regions128-rgb-value64-white-icc/v7";

struct CacheKeys {
    identity: &'static str,
    retrieval: String,
    orb: String,
}

fn cache_keys() -> anyhow::Result<&'static CacheKeys> {
    static KEYS: LazyLock<Result<CacheKeys, String>> = LazyLock::new(|| {
        crate::preview::identity()
            .map(|identity| CacheKeys {
                identity,
                retrieval: format!("image.retrieval/{identity}/v1"),
                orb: format!("image.orb800-grid4000-1280-white/{identity}/v4"),
            })
            .map_err(|error| format!("{error:#}"))
    });
    KEYS.as_ref().map_err(|error| anyhow::anyhow!("{error}"))
}

// Rayon owns the concurrency budget; native parallel regions would oversubscribe it.
static OPENCV_THREADS: LazyLock<Result<(), String>> =
    LazyLock::new(|| core::set_num_threads(1).map_err(|error| error.to_string()));

#[derive(Serialize, Deserialize)]
pub(crate) struct Visual {
    pub width: u32,
    pub height: u32,
    pub digest: [u8; 32],
    pub phash: u64,
    pub dhash: u64,
    pub regions: Vec<u64>,
    pub preview: Vec<u8>,
    pub registration: Vec<u8>,
    pub structure: Vec<u8>,
    pub srgb_preview: Option<Vec<u8>>,
    // Derived lazily from the cached 64-square registration preview.
    #[serde(skip)]
    pub phase_gradients: OnceLock<Vec<f32>>,
    #[serde(skip)]
    pub tone_map: OnceLock<ToneMap>,
    // Derived once per run; stored preview bytes remain the persistent source.
    #[serde(skip)]
    pub structural_samples: OnceLock<Vec<[ChannelSample; 2]>>,
}

pub(crate) struct ToneMap {
    pub cdf: [[f64; 256]; 3],
    pub entropy: [f64; 3],
}

#[derive(Clone, Copy)]
pub(crate) struct ChannelSample {
    pub center: [f64; 3],
    pub horizontal: [f64; 3],
    pub vertical: [f64; 3],
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Orb {
    pub points: Vec<[f32; 2]>,
    pub descriptors: Vec<u8>,
    // Native-endian words are a run-local view of the persisted 32-byte descriptors.
    #[serde(skip)]
    pub packed: OnceLock<Vec<[u64; 4]>>,
    #[serde(skip)]
    pub byte_index: OnceLock<crate::matching::ByteIndex>,
}

pub(crate) struct Features {
    pub visual: Visual,
    pub orb: Option<Orb>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct RetrievalVisual {
    pub width: u32,
    pub height: u32,
    pub phash: u64,
    pub dhash: u64,
    pub regions: Vec<u64>,
}

// Reduced pixels cannot populate the optional canonical proof.
pub(crate) struct Retrieval {
    pub visual: RetrievalVisual,
    pub orb: Option<Orb>,
    pub embedding: Option<Vec<f32>>,
    pub(crate) proof: Option<Visual>,
}

fn parse<T: DeserializeOwned>(bytes: Option<&[u8]>) -> anyhow::Result<Option<T>> {
    bytes
        .map(bincode::deserialize)
        .transpose()
        .map_err(Into::into)
}

fn store<T: Serialize>(
    ctx: &Ctx<'_>,
    file: &FileEntry,
    key: &str,
    value: &T,
) -> anyhow::Result<()> {
    ctx.cache.put(file, key, &bincode::serialize(value)?)
}

fn embedding_key(model: &str, identity: &str) -> String {
    format!("image.embedding/{model}/white/{identity}/v3")
}

fn source_embedding_key(model: &str, identity: &str) -> String {
    format!("image.source-center/{model}/white/{identity}/v1")
}

pub(crate) fn source_frame_embedding(file: &FileEntry, ctx: &Ctx<'_>) -> anyhow::Result<Vec<f32>> {
    let model = ctx
        .embedder
        .context("source-center embedding requires a model")?;
    let identity = cache_keys()?.identity;
    for id in model.frame_cache_model_ids().into_iter().flatten() {
        if let Some(bytes) = ctx.cache.get(file, &source_embedding_key(id, identity))? {
            return parse(Some(&bytes))?.context("cached source-center embedding missing");
        }
    }
    let source = source_features(file, ctx)?;
    model.prepare()?;
    let mut vector = model.embed_frame(&source.small)?;
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    ensure!(
        norm.is_finite() && norm > f32::EPSILON,
        "invalid source-center embedding for {}",
        file.path.display()
    );
    for value in &mut vector {
        *value /= norm;
    }
    store(
        ctx,
        file,
        &source_embedding_key(model.frame_model_id(), identity),
        &vector,
    )?;
    Ok(vector)
}

pub(crate) fn uncached_embedding_work(
    files: &[FileEntry],
    ctx: &Ctx<'_>,
    include_source_centers: bool,
) -> anyhow::Result<usize> {
    let Some(model) = ctx.embedder.filter(|_| !ctx.config.no_embeddings) else {
        return Ok(0);
    };
    let identity = cache_keys()?.identity;
    let keys = model
        .cache_model_ids()
        .map(|id| id.map(|id| embedding_key(id, identity)));
    let source_keys = include_source_centers.then(|| {
        model
            .frame_cache_model_ids()
            .map(|id| id.map(|id| source_embedding_key(id, identity)))
    });
    let mut work = 0;
    for file in files
        .iter()
        .filter(|file| file.kind == eidolon_core::Kind::Image)
    {
        let mut cached = false;
        for key in keys.iter().flatten() {
            if ctx.cache.contains(file, key)? {
                cached = true;
                break;
            }
        }
        work += usize::from(!cached);
        if let Some(keys) = &source_keys {
            let mut cached = false;
            for key in keys.iter().flatten() {
                if ctx.cache.contains(file, key)? {
                    cached = true;
                    break;
                }
            }
            work += usize::from(!cached);
        }
    }
    Ok(work)
}

pub(crate) fn extract(file: &FileEntry, ctx: &Ctx<'_>) -> anyhow::Result<Retrieval> {
    OPENCV_THREADS
        .as_ref()
        .map_err(|error| anyhow::anyhow!("configuring OpenCV workers: {error}"))?;
    let keys = cache_keys()?;
    let visual_bytes = ctx.cache.get(file, &keys.retrieval)?;
    let orb_bytes = if ctx.config.no_keypoints {
        None
    } else {
        ctx.cache.get(file, &keys.orb)?
    };
    let embedder = ctx.embedder.filter(|_| !ctx.config.no_embeddings);
    let mut embedding_bytes = None;
    if let Some(model) = embedder {
        for id in model.cache_model_ids().into_iter().flatten() {
            embedding_bytes = ctx.cache.get(file, &embedding_key(id, keys.identity))?;
            if embedding_bytes.is_some() {
                break;
            }
        }
    }
    let visual: Option<RetrievalVisual> = parse(visual_bytes.as_deref())?;
    let orb: Option<Orb> = parse(orb_bytes.as_deref())?;
    let embedding: Option<Vec<f32>> = parse(embedding_bytes.as_deref())?;
    let needs_decode = visual.is_none()
        || (!ctx.config.no_keypoints && orb.is_none())
        || (embedder.is_some() && embedding.is_none());
    if !needs_decode {
        return Ok(Retrieval {
            visual: visual.context("cached retrieval feature missing")?,
            orb,
            embedding,
            proof: None,
        });
    }
    let source = source_features(file, ctx)?;
    let mut proof = source.proof;
    let small = source.small;
    let visual = if let Some(visual) = visual {
        visual
    } else {
        let (phash, dhash, regions) = if let Some(proof) = proof.as_mut() {
            (proof.phash, proof.dhash, std::mem::take(&mut proof.regions))
        } else {
            let thumbnail = if source.canonical {
                image::imageops::resize(&small, PREVIEW, PREVIEW, FilterType::Triangle)
            } else {
                crate::preview::resize(&small, PREVIEW, PREVIEW)?
            };
            hashes(&thumbnail)
        };
        let visual = RetrievalVisual {
            width: source.width,
            height: source.height,
            phash,
            dhash,
            regions,
        };
        store(ctx, file, &keys.retrieval, &visual)?;
        visual
    };
    let orb = match orb {
        Some(orb) => Some(orb),
        None if !ctx.config.no_keypoints => {
            let orb = orb_features(&small)?;
            store(ctx, file, &keys.orb, &orb)?;
            Some(orb)
        }
        None => None,
    };
    let embedding = match (embedding, embedder) {
        (Some(vector), _) => Some(vector),
        (None, Some(model)) => {
            model.prepare()?;
            let mut vector = model.embed(&small)?;
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            ensure!(
                norm.is_finite() && norm > f32::EPSILON,
                "invalid embedding for {}",
                file.path.display()
            );
            for value in &mut vector {
                *value /= norm;
            }
            store(
                ctx,
                file,
                &embedding_key(model.model_id(), keys.identity),
                &vector,
            )?;
            Some(vector)
        }
        _ => None,
    };
    Ok(Retrieval {
        visual,
        orb,
        embedding,
        proof,
    })
}

pub(crate) fn complete(
    file: &FileEntry,
    retrieval: Retrieval,
    ctx: &Ctx<'_>,
) -> anyhow::Result<Features> {
    let mut visual = if let Some(visual) = retrieval.proof {
        visual
    } else {
        canonical_proof(file, ctx)?
    };
    // Hashes order fallback work and explain candidate similarity only.
    // Every proof field above comes exclusively from the canonical full decode.
    visual.phash = retrieval.visual.phash;
    visual.dhash = retrieval.visual.dhash;
    visual.regions = retrieval.visual.regions;
    Ok(Features {
        visual,
        orb: retrieval.orb,
        embedding: retrieval.embedding,
    })
}

struct SourceFeatures {
    small: RgbImage,
    proof: Option<Visual>,
    width: u32,
    height: u32,
    canonical: bool,
}

fn source_features(file: &FileEntry, ctx: &Ctx<'_>) -> anyhow::Result<SourceFeatures> {
    let needs_proof = !ctx.cache.contains(file, VISUAL_KEY)?;
    let decoded = crate::preview::decode(&file.path, needs_proof).map_err(|error| {
        eidolon_core::InputError {
            file: file.id,
            reason: format!("{error:#}"),
        }
    })?;
    let digest = (decoded.canonical && needs_proof)
        .then(|| precise_digest(&decoded.image))
        .flatten();
    let image = rendered_rgb(decoded.image)?;
    ensure!(
        image.width() > 0 && image.height() > 0,
        "empty image: {}",
        file.path.display()
    );
    let mut proof = None;
    let small = if decoded.canonical {
        let small = downscale(&image, crate::preview::LONG_SIDE);
        // A full fallback decode already exists: reuse both its resampling and
        // canonical proof, rather than decoding or resampling it a second time.
        if needs_proof {
            let visual = canonical_visual(&image, &small, digest, decoded.profile.as_ref())?;
            store(ctx, file, VISUAL_KEY, &visual)?;
            proof = Some(visual);
        }
        match small {
            Cow::Owned(small) => small,
            Cow::Borrowed(_) => image,
        }
    } else {
        let longest = image.width().max(image.height());
        if longest > crate::preview::LONG_SIDE {
            crate::preview::resize(
                &image,
                (u64::from(image.width()) * u64::from(crate::preview::LONG_SIDE)
                    / u64::from(longest))
                .max(1) as u32,
                (u64::from(image.height()) * u64::from(crate::preview::LONG_SIDE)
                    / u64::from(longest))
                .max(1) as u32,
            )?
        } else {
            image
        }
    };
    Ok(SourceFeatures {
        small,
        proof,
        width: decoded.width,
        height: decoded.height,
        canonical: decoded.canonical,
    })
}

fn canonical_proof(file: &FileEntry, ctx: &Ctx<'_>) -> anyhow::Result<Visual> {
    let bytes = ctx.cache.get(file, VISUAL_KEY)?;
    if let Some(visual) = parse(bytes.as_deref())? {
        return Ok(visual);
    }
    let (image, profile) =
        crate::decode_with_profile(&file.path, true).map_err(|error| eidolon_core::InputError {
            file: file.id,
            reason: format!("{error:#}"),
        })?;
    let digest = precise_digest(&image);
    let image = rendered_rgb(image)?;
    let small = downscale(&image, crate::preview::LONG_SIDE);
    let visual = canonical_visual(&image, &small, digest, profile.as_ref())?;
    store(ctx, file, VISUAL_KEY, &visual)?;
    Ok(visual)
}

pub(crate) fn prepare<'a>(
    files: &'a [FileEntry],
    retrieval: Vec<Retrieval>,
    candidates: &mut [(usize, usize)],
    ctx: &Ctx<'_>,
) -> anyhow::Result<(Vec<&'a FileEntry>, Vec<Features>)> {
    let mut needed = vec![false; files.len()];
    for &(a, b) in candidates.iter() {
        needed[a] = true;
        needed[b] = true;
    }
    let mut indices = vec![0; files.len()];
    let mut selected = Vec::new();
    for (index, file) in files.iter().enumerate().filter(|(index, _)| needed[*index]) {
        indices[index] = selected.len();
        selected.push(file);
    }
    let features = retrieval
        .into_par_iter()
        .enumerate()
        .filter(|(index, _)| needed[*index])
        .map(|(index, retrieval)| complete(&files[index], retrieval, ctx))
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (a, b) in candidates {
        *a = indices[*a];
        *b = indices[*b];
    }
    Ok((selected, features))
}

fn precise_digest(image: &image::DynamicImage) -> Option<[u8; 32]> {
    let color = image.color();
    // Retain low bits for decoded equality of HDR/16-bit inputs.
    (color.bits_per_pixel() > u16::from(color.channel_count()) * 8)
        .then(|| Sha256::digest(image.as_bytes()).into())
}

fn canonical_visual(
    image: &RgbImage,
    small: &RgbImage,
    digest: Option<[u8; 32]>,
    profile: Option<&moxcms::ColorProfile>,
) -> anyhow::Result<Visual> {
    let mut visual = visual_features(image, small, digest);
    visual.srgb_preview = profile
        .map(|profile| crate::color::to_srgb(small, profile))
        .transpose()?
        .flatten();
    Ok(visual)
}

fn rendered_rgb(image: image::DynamicImage) -> anyhow::Result<RgbImage> {
    if !image.color().has_alpha() {
        return Ok(image.into_rgb8());
    }
    let (width, height) = (image.width(), image.height());
    // Composite on white, then compact RGBA into RGB in the same allocation.
    // Hidden RGB channels in fully transparent pixels must not match a photo.
    let mut bytes = image.into_rgba8().into_raw();
    let pixels = bytes.len() / 4;
    for index in 0..pixels {
        let offset = index * 4;
        let pixel = [
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ];
        let alpha = u32::from(pixel[3]);
        for channel in 0..3 {
            bytes[index * 3 + channel] =
                ((u32::from(pixel[channel]) * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
        }
    }
    bytes.truncate(pixels * 3);
    RgbImage::from_raw(width, height, bytes).context("invalid decoded pixel layout")
}

fn downscale(image: &RgbImage, long_side: u32) -> Cow<'_, RgbImage> {
    let longest = image.width().max(image.height());
    if longest <= long_side {
        Cow::Borrowed(image)
    } else {
        let width =
            (u64::from(image.width()) * u64::from(long_side) / u64::from(longest)).max(1) as u32;
        let height =
            (u64::from(image.height()) * u64::from(long_side) / u64::from(longest)).max(1) as u32;
        Cow::Owned(image::imageops::resize(
            image,
            width,
            height,
            FilterType::Triangle,
        ))
    }
}

fn visual_features(
    original: &RgbImage,
    small: &RgbImage,
    precise_digest: Option<[u8; 32]>,
) -> Visual {
    let thumbnail = image::imageops::resize(small, PREVIEW, PREVIEW, FilterType::Triangle);
    let (phash, dhash, regions) = hashes(&thumbnail);
    let registration =
        image::imageops::resize(&thumbnail, REGISTRATION, REGISTRATION, FilterType::Triangle);
    let structure = registration
        .pixels()
        .map(|pixel| pixel[0].max(pixel[1]).max(pixel[2]))
        .collect();
    Visual {
        width: original.width(),
        height: original.height(),
        digest: precise_digest.unwrap_or_else(|| Sha256::digest(original.as_raw()).into()),
        phash,
        dhash,
        regions,
        registration: registration.into_raw(),
        structure,
        preview: thumbnail.into_raw(),
        srgb_preview: None,
        phase_gradients: OnceLock::new(),
        tone_map: OnceLock::new(),
        structural_samples: OnceLock::new(),
    }
}

fn hashes(thumbnail: &RgbImage) -> (u64, u64, Vec<u64>) {
    let phasher = HasherConfig::with_bytes_type::<[u8; 8]>()
        .hash_alg(HashAlg::Mean)
        .preproc_dct()
        .to_hasher();
    let dhasher = HasherConfig::with_bytes_type::<[u8; 8]>()
        .hash_alg(HashAlg::Gradient)
        .to_hasher();
    let phash = u64::from_le_bytes(phasher.hash_image(thumbnail).into_inner());
    let dhash = u64::from_le_bytes(dhasher.hash_image(thumbnail).into_inner());
    let mut regions = Vec::with_capacity(27);
    // Region hashes seed crop candidates even without a neural model. They are
    // candidates only: no crop relation is inferred without geometric proof.
    for percent in [80_u32, 60, 45] {
        let width = (PREVIEW * percent / 100).max(1);
        let height = (PREVIEW * percent / 100).max(1);
        for y in 0..3 {
            for x in 0..3 {
                let crop = image::imageops::crop_imm(
                    thumbnail,
                    (PREVIEW - width) * x / 2,
                    (PREVIEW - height) * y / 2,
                    width,
                    height,
                );
                let crop = image::imageops::resize(&*crop, 32, 32, FilterType::Triangle);
                regions.push(u64::from_le_bytes(phasher.hash_image(&crop).into_inner()));
            }
        }
    }
    (phash, dhash, regions)
}

fn orb_features(image: &RgbImage) -> anyhow::Result<Orb> {
    let gray = image::imageops::grayscale(image);
    let flat = core::Mat::from_slice(gray.as_raw())?;
    let matrix = flat.reshape(1, i32::try_from(gray.height())?)?;
    let mut detector = features::ORB::create(
        4000,
        1.2,
        8,
        16,
        0,
        2,
        features::ORB_ScoreType::HARRIS_SCORE,
        31,
        10,
    )?;
    let mut points = core::Vector::<core::KeyPoint>::new();
    let mut descriptors = core::Mat::default();
    detector.detect_and_compute_def(&matrix, &core::no_array(), &mut points, &mut descriptors)?;
    if descriptors.empty() {
        return Ok(Orb {
            points: Vec::new(),
            descriptors: Vec::new(),
            packed: OnceLock::new(),
            byte_index: OnceLock::new(),
        });
    }
    // A high-contrast logo must not consume the entire global keypoint budget.
    // Keep a bounded descriptor set spread over the image, including its background.
    let raw = descriptors.data_bytes()?;
    let stride = usize::try_from(descriptors.cols())?;
    let mut cells = [0_u8; 64];
    let mut selected = Orb {
        points: Vec::with_capacity(1280.min(points.len())),
        descriptors: Vec::with_capacity(1280.min(points.len()) * stride),
        packed: OnceLock::new(),
        byte_index: OnceLock::new(),
    };
    for (index, keypoint) in points.iter().enumerate() {
        let point = keypoint.pt();
        let x = point.x / image.width() as f32;
        let y = point.y / image.height() as f32;
        let cell = (y * 8.0).min(7.0) as usize * 8 + (x * 8.0).min(7.0) as usize;
        if cells[cell] < 20 {
            cells[cell] += 1;
            selected.points.push([x, y]);
            selected
                .descriptors
                .extend_from_slice(&raw[index * stride..(index + 1) * stride]);
        }
    }
    Ok(selected)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{Instant, UNIX_EPOCH};

    use anyhow::Context as _;
    use eidolon_core::{Config, Ctx, Embedder, FileEntry, FileId, Kind, cache::FeatureCache};

    #[test]
    fn reduced_features_materialize_only_canonical_proof_bytes() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("large.jpg");
        image::RgbImage::from_fn(3200, 1600, |x, y| {
            image::Rgb([(x / 16) as u8, (y / 8) as u8, ((x + y) / 32) as u8])
        })
        .save(&path)?;
        let metadata = path.metadata()?;
        let file = FileEntry {
            id: FileId(37),
            path,
            size: metadata.len(),
            mtime_ns: i128::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
            kind: Kind::Image,
        };
        let cache = FeatureCache::in_memory()?;
        let config = Config {
            timings: false,
            remote_overlay: true,
            min_score: 0.75,
            model_dir: PathBuf::new(),
            no_embeddings: true,
            no_keypoints: true,
            jobs: 1,
            fast: false,
            video_fps: 2.0,
        };
        let ctx = Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        };
        let retrieval = super::extract(&file, &ctx)?;
        assert_eq!(
            (retrieval.visual.width, retrieval.visual.height),
            (3200, 1600)
        );
        assert!(
            !cache.contains(&file, super::VISUAL_KEY)?,
            "scaled retrieval must not populate canonical evidence"
        );
        let features = super::complete(&file, retrieval, &ctx)?;
        let original = super::rendered_rgb(crate::decode(&file.path)?)?;
        let small = super::downscale(&original, 800);
        let expected = super::visual_features(&original, &small, None);
        assert_eq!(features.visual.digest, expected.digest);
        assert_eq!(features.visual.preview, expected.preview);
        assert_eq!(features.visual.registration, expected.registration);
        assert_eq!(features.visual.structure, expected.structure);
        assert_eq!(
            cache
                .get(&file, super::VISUAL_KEY)?
                .context("proof missing")?,
            bincode::serialize(&expected)?
        );
        Ok(())
    }

    #[test]
    #[ignore = "real local model and per-image extraction benchmark"]
    fn measures_real_feature_extraction_and_cache_hits() -> anyhow::Result<()> {
        let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
            .join("Library/Caches/eidolon/models");
        let started = Instant::now();
        let model = crate::embed::OnnxEmbedder::load(&model_dir)?
            .context("fetch the model before measuring")?;
        eprintln!(
            "model load including SHA-256: {:.2} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
        for path in [
            Path::new("/System/Library/Desktop Pictures/.thumbnails/Big Sur Coastline.heic"),
            Path::new("/System/Library/Desktop Pictures/Sonoma.heic"),
        ] {
            let metadata = path.metadata()?;
            let file = FileEntry {
                id: FileId(0),
                path: path.into(),
                size: metadata.len(),
                mtime_ns: i128::try_from(
                    metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos(),
                )?,
                kind: Kind::Image,
            };
            for embeddings in [false, true] {
                let cache = FeatureCache::in_memory()?;
                let config = Config {
                    timings: false,
                    remote_overlay: true,
                    min_score: 0.8,
                    model_dir: model_dir.clone(),
                    no_embeddings: !embeddings,
                    no_keypoints: false,
                    jobs: 1,
                    fast: false,
                    video_fps: 2.0,
                };
                let ctx = Ctx {
                    refiner: None,
                    config: &config,
                    cache: &cache,
                    embedder: Some(&model as &dyn Embedder),
                };
                let started = Instant::now();
                let features = super::extract(&file, &ctx)?;
                let cold = started.elapsed().as_secs_f64() * 1000.0;
                let started = Instant::now();
                let cached = super::extract(&file, &ctx)?;
                let warm = started.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(features.visual.phash, cached.visual.phash);
                assert!(features.orb.is_some());
                assert_eq!(features.embedding.is_some(), embeddings);
                eprintln!(
                    "{} {}x{} embeddings={embeddings}: extraction {cold:.2} ms, cached {warm:.2} ms",
                    path.display(),
                    features.visual.width,
                    features.visual.height
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "offline old/new hash distribution; set EIDOLON_IMAGE_EQ_INPUT and EIDOLON_IMAGE_EQ_OUTPUT"]
    fn profiles_reduced_hash_equivalence() -> anyhow::Result<()> {
        let input = PathBuf::from(
            std::env::var_os("EIDOLON_IMAGE_EQ_INPUT").context("set EIDOLON_IMAGE_EQ_INPUT")?,
        );
        let output = PathBuf::from(
            std::env::var_os("EIDOLON_IMAGE_EQ_OUTPUT").context("set EIDOLON_IMAGE_EQ_OUTPUT")?,
        );
        let mut paths = if input.is_dir() {
            std::fs::read_dir(&input)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![input]
        };
        paths.sort();
        let cache = FeatureCache::in_memory()?;
        let config = Config {
            timings: false,
            remote_overlay: true,
            min_score: 0.75,
            model_dir: PathBuf::new(),
            no_embeddings: true,
            no_keypoints: true,
            jobs: 1,
            fast: false,
            video_fps: 2.0,
        };
        let ctx = Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        };
        let mut phash = [0_u64; 65];
        let mut dhash = [0_u64; 65];
        let mut region_hash = [0_u64; 65];
        let mut rows = Vec::new();
        for (index, path) in paths.into_iter().filter(|path| path.is_file()).enumerate() {
            let metadata = path.metadata()?;
            let file = FileEntry {
                id: FileId(index),
                path,
                size: metadata.len(),
                mtime_ns: i128::try_from(
                    metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos(),
                )?,
                kind: Kind::Image,
            };
            let original = crate::decode(&file.path)?;
            let digest = super::precise_digest(&original);
            let original = super::rendered_rgb(original)?;
            let small = super::downscale(&original, 800);
            let old = super::visual_features(&original, &small, digest);
            let new = super::extract(&file, &ctx)?;
            assert_eq!(
                (new.visual.width, new.visual.height),
                (old.width, old.height)
            );
            let p = (old.phash ^ new.visual.phash).count_ones();
            let d = (old.dhash ^ new.visual.dhash).count_ones();
            phash[p as usize] += 1;
            dhash[d as usize] += 1;
            let regions = old
                .regions
                .iter()
                .zip(&new.visual.regions)
                .map(|(a, b)| (a ^ b).count_ones())
                .collect::<Vec<_>>();
            for &distance in &regions {
                region_hash[distance as usize] += 1;
            }
            if let Some(bytes) = cache.get(&file, super::VISUAL_KEY)? {
                let proof: super::Visual = bincode::deserialize(&bytes)?;
                assert_eq!(proof.digest, old.digest);
                assert_eq!(proof.preview, old.preview);
                assert_eq!(proof.registration, old.registration);
                assert_eq!(proof.structure, old.structure);
            }
            rows.push(serde_json::json!({
                "path": file.path, "width": old.width, "height": old.height,
                "old_phash": format!("{:016x}", old.phash), "new_phash": format!("{:016x}", new.visual.phash),
                "old_dhash": format!("{:016x}", old.dhash), "new_dhash": format!("{:016x}", new.visual.dhash),
                "phash_hamming": p, "dhash_hamming": d, "region_hamming": regions,
            }));
        }
        let output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        serde_json::to_writer_pretty(
            output,
            &serde_json::json!({
                "decoder_identity": crate::preview::identity()?,
                "images": rows.len(),
                "phash_hamming_histogram": phash.as_slice(),
                "dhash_hamming_histogram": dhash.as_slice(),
                "region_hamming_histogram": region_hash.as_slice(),
                "files": rows,
            }),
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "offline per-stage benchmark; set EIDOLON_IMAGE_BENCH_FILE"]
    fn profiles_feature_stages() -> anyhow::Result<()> {
        let path = PathBuf::from(
            std::env::var_os("EIDOLON_IMAGE_BENCH_FILE").context("set EIDOLON_IMAGE_BENCH_FILE")?,
        );
        let started = Instant::now();
        let decoded = crate::decode(&path)?;
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let decoded = super::rendered_rgb(decoded)?;
        let small = super::downscale(&decoded, 800);
        let render_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let visual = super::visual_features(&decoded, &small, None);
        let hash_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let orb = super::orb_features(&small)?;
        let orb_ms = started.elapsed().as_secs_f64() * 1000.0;
        let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
            .join("Library/Caches/eidolon/models");
        let started = Instant::now();
        let model =
            crate::embed::OnnxEmbedder::load(&model_dir)?.context("fetch the model first")?;
        let load_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let embedding = model.embed(&small)?;
        let embed_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(visual.preview.len(), 128 * 128 * 3);
        assert!(orb.points.len() <= 1280);
        assert_eq!(embedding.len(), 384);
        eprintln!(
            "{}: decode {decode_ms:.2}ms; render/resize {render_ms:.2}ms; hashes {hash_ms:.2}ms; ORB {orb_ms:.2}ms; model load {load_ms:.2}ms; embedding {embed_ms:.2}ms",
            path.display()
        );
        Ok(())
    }
}
