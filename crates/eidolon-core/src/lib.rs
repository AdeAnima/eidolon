//! Shared contract between eidolon's detectors and its CLI.
//!
//! Every detector crate exposes one entry point with the shape
//! `pub fn find(files: &[FileEntry], ctx: &Ctx) -> anyhow::Result<Vec<Match>>`
//! and receives only the files of its [`Kind`]. Exception: `eidolon_image::find`
//! takes a third argument, an optional normalized source-aspect embedding bank
//! from `eidolon_image::source_frame_embeddings`, which the CLI supplies for image
//! scans unless `--image-source-bank off`, independently of thumbnail matching;
//! it adds retrieval candidates only.
//! The one cross-kind entry point,
//! `eidolon_video::find_frames(images, videos, ctx)`, reports
//! [`Relation::FrameOf`]. Detectors never touch files beyond reading them:
//! eidolon only reports.

pub mod cache;

use std::path::PathBuf;

use serde::Serialize;

/// Index of a file in the run's input list; stable for the whole run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct FileId(pub usize);

/// A single input that cannot be decoded or read for near-duplicate analysis.
///
/// Cache, model and other run-wide failures must not use this error type.
/// Callers may exclude this file from near analysis while retaining exact matches.
#[derive(Debug)]
pub struct InputError {
    /// Stable identity of the unreadable input.
    pub file: FileId,
    /// Full diagnostic explaining the input failure.
    pub reason: String,
}

impl std::fmt::Display for InputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for InputError {}

/// Media family a file was classified into (by content sniffing, not extension).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Decodable still image.
    Image,
    /// Container with a video stream.
    Video,
    /// UTF-8 (or UTF-16 with BOM) text.
    Text,
    /// Anything else; only exact matching applies.
    Other,
}

/// One input file.
#[derive(Clone, Debug, Serialize)]
pub struct FileEntry {
    /// Position in the input list.
    pub id: FileId,
    /// Canonical absolute path.
    pub path: PathBuf,
    /// Size in bytes.
    pub size: u64,
    /// Modification time, nanoseconds since the Unix epoch.
    pub mtime_ns: i128,
    /// Classified media family.
    pub kind: Kind,
}

/// How file `a` relates to file `b` in a [`Match`].
///
/// Containment relations are directional: `a` is the part, `b` the whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    /// Byte-identical files.
    Identical,
    /// Same decoded content in a different encoding or container
    /// (lossless re-save, normalized-equal text).
    SameContent,
    /// Same content at different quality, resolution, bitrate or format.
    Reencoded,
    /// Same content with local edits: watermark, logo, text or other overlay.
    Edited,
    /// Same design or composition in another version: recoloured, light vs
    /// dark, re-lit or re-framed. Related, but not a duplicate.
    Variant,
    /// Image `a` is a cropped region of image `b`.
    CropOf,
    /// Video `a` is a time segment of video `b`.
    ClipOf,
    /// Text `a` is contained in text `b`.
    ExcerptOf,
    /// Image `a` shows a frame of video `b` (thumbnail, poster or screenshot,
    /// possibly scaled, cropped or with overlays). The only cross-kind relation.
    FrameOf,
    /// Strongly similar, no specific relation established.
    Similar,
}

impl Relation {
    /// Whether `a`/`b` order carries meaning for this relation.
    #[must_use]
    pub fn is_directional(self) -> bool {
        matches!(
            self,
            Self::CropOf | Self::ClipOf | Self::ExcerptOf | Self::FrameOf
        )
    }
}

/// One signal that contributed to a match.
#[derive(Clone, Debug, Serialize)]
pub struct Evidence {
    /// Detector signal name, e.g. `"phash"`, `"orb"`, `"dinov2"`, `"minhash"`.
    pub signal: &'static str,
    /// Raw measured value (distance, similarity, inlier ratio, ...).
    pub value: f64,
    /// Human-readable explanation of the value, e.g. `"hamming 3/64"`.
    pub note: String,
}

/// A detected relation between two files.
#[derive(Clone, Debug, Serialize)]
pub struct Match {
    /// The part for directional relations, otherwise the lower id.
    pub a: FileId,
    /// The whole for directional relations, otherwise the higher id.
    pub b: FileId,
    /// Relation of `a` to `b`.
    pub relation: Relation,
    /// Confidence in `[0, 1]`.
    pub score: f64,
    /// Signals that produced this match; never empty.
    pub evidence: Vec<Evidence>,
}

/// Run-wide tuning shared by all detectors.
#[derive(Clone, Debug)]
// Independent detector options map to separate user-facing switches.
#[allow(clippy::struct_excessive_bools)]
pub struct Config {
    /// Matches below this score are dropped by detectors.
    pub min_score: f64,
    /// Directory holding downloaded models (e.g. `dinov2-small.onnx`).
    pub model_dir: PathBuf,
    /// Disable the embedding signal even if a model is present.
    pub no_embeddings: bool,
    /// Disable keypoint (ORB) verification.
    pub no_keypoints: bool,
    /// Worker threads; 0 = all cores.
    pub jobs: usize,
    /// Use cheaper verification: low-resolution pixel proof instead of the
    /// full-resolution checks. Faster; matches without any pixel proof are
    /// marked `unverified` and never join duplicate groups.
    pub fast: bool,
    /// Emit native video refinement timing diagnostics; off by default in the CLI.
    pub timings: bool,
    /// Video sampling rate in frames per second (positive, finite).
    pub video_fps: f64,
    /// Enable caption-overlay nomination/refinement for remote videos; on by default in the CLI.
    pub remote_overlay: bool,
}

/// Computes a fixed-length L2-normalized embedding for an RGB image.
///
/// Implemented by the image crate; injected into the video detector by the
/// CLI so frame embeddings reuse the same model.
pub trait Embedder: Send + Sync {
    /// Identifier stored with cached embeddings, e.g. `"dinov2-small@<sha256>"`.
    fn model_id(&self) -> &str;
    /// Qualified, equivalent model identities accepted for cache reads.
    ///
    /// The first identity is preferred. New values are always written using
    /// [`Self::model_id`], including after provider fallback.
    fn cache_model_ids(&self) -> [Option<&str>; 2] {
        [Some(self.model_id()), None]
    }
    /// Initialize inference after a detector has established a cache miss.
    ///
    /// This may resolve a fallback provider and change [`Self::model_id`], so
    /// callers obtain their write key again after preparation.
    ///
    /// # Errors
    /// Fails if the requested inference backend cannot be initialized.
    fn prepare(&self) -> anyhow::Result<()> {
        Ok(())
    }
    /// Embed one image.
    ///
    /// # Errors
    /// Fails if inference fails.
    fn embed(&self, image: &image::RgbImage) -> anyhow::Result<Vec<f32>>;
    /// Identifier for embeddings used in thumbnail-to-video-frame matching.
    ///
    /// Both sides of a `FrameOf` comparison (still thumbnails and decoded video
    /// frames) must use these frame-match methods so their vectors are
    /// comparable. Defaults to [`Self::model_id`].
    fn frame_model_id(&self) -> &str {
        self.model_id()
    }
    /// Cache read identities for frame-match embeddings; see
    /// [`Self::cache_model_ids`]. New values are written using
    /// [`Self::frame_model_id`]. Defaults to [`Self::cache_model_ids`];
    /// implementors overriding [`Self::frame_model_id`] must override this too.
    fn frame_cache_model_ids(&self) -> [Option<&str>; 2] {
        self.cache_model_ids()
    }
    /// Embed one image for thumbnail-to-video-frame matching.
    ///
    /// # Errors
    /// Fails if inference fails.
    fn embed_frame(&self, image: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        self.embed(image)
    }
}

/// A locally readable video window whose presentation clock maps to its source.
///
/// The provider owns the artifact and keeps it readable for the detector call.
/// The artifact must preserve original frame pixels and native frame cadence;
/// sparse held/repeated frames are not a refined window.
#[derive(Clone, Debug)]
pub struct RefinedWindow {
    /// Local decoded-media input that `ffmpeg` can read; no URL or network input.
    pub path: PathBuf,
    /// Source seconds corresponding to presentation timestamp zero in `path`.
    ///
    /// A decoded presentation timestamp `t` maps to `source_offset_s + t`.
    /// This is the actual media clock offset, not the requested window start.
    pub source_offset_s: f64,
}

/// Supplies optional native-cadence windows for otherwise sparse video inputs
/// (remote videos fetched as keyframe-only proxies).
///
/// Implementations may fetch additional source bytes and must account for that
/// transfer in the originating input's transport evidence. Calls may be concurrent.
/// Local videos need no provider: the detector decodes their windows directly.
pub trait FrameRefiner: Send + Sync {
    /// Return a local media window covering the requested source-time interval.
    ///
    /// `center_s` and `radius_s` are finite source seconds; the interval is
    /// `[max(0, center_s - radius_s), center_s + radius_s]`, clamped to the source.
    /// GOP-aligned extra context is permitted. Return `None` when no refinement
    /// is available; callers retain only independently verified sparse samples.
    ///
    /// # Errors
    /// Returns an error when retrieving or materializing the window fails.
    fn refine(
        &self,
        video: &FileEntry,
        center_s: f64,
        radius_s: f64,
    ) -> anyhow::Result<Option<RefinedWindow>>;
}

/// Everything a detector needs besides its files.
pub struct Ctx<'a> {
    /// Tuning.
    pub config: &'a Config,
    /// Persistent per-file feature cache.
    pub cache: &'a cache::FeatureCache,
    /// Shared embedder, if the model is available and enabled.
    pub embedder: Option<&'a dyn Embedder>,
    /// Optional native-cadence windows for sparse remote video inputs.
    pub refiner: Option<&'a dyn FrameRefiner>,
}
