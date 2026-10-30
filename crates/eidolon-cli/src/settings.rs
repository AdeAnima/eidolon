//! Effective performance settings shared by JSON and stderr reporting.

use std::io::Write;

use eidolon_image::embed::{ExecutionDevice, OnnxEmbedder};
use serde::Serialize;

use crate::args::{Args, ThumbnailMode};

/// Resolved run settings, including the selected embedding device and worker count.
// These independent flags stay booleans in the public JSON settings object.
#[allow(clippy::struct_excessive_bools)]
#[derive(Serialize)]
pub(crate) struct Settings {
    fast: bool,
    video_fps: f64,
    embed_device_requested: ExecutionDevice,
    embed_device: Option<ExecutionDevice>,
    thumbnails_requested: ThumbnailMode,
    /// Whether this run dispatches image-to-video frame matching.
    pub(crate) thumbnails: bool,
    remote_overlay: bool,
    image_source_bank: bool,
    max_fetch_fraction: f64,
    fetch_fresh: bool,
    timings: bool,
    min_score: f64,
    jobs: usize,
    no_embeddings: bool,
    no_keypoints: bool,
    remote_video_mode: &'static str,
}

impl Settings {
    /// Resolve requested flags against the loaded embedder and explicit inputs.
    pub(crate) fn new(
        args: &Args,
        embedder: Option<&OnnxEmbedder>,
        has_url_inputs: bool,
        jobs: usize,
    ) -> Self {
        Self {
            fast: args.fast,
            video_fps: args.video_fps,
            embed_device_requested: args.embed_device,
            embed_device: embedder.map(OnnxEmbedder::device),
            thumbnails_requested: args.thumbnails,
            thumbnails: args.thumbnails.enabled(has_url_inputs),
            remote_overlay: args.remote_overlay.enabled(),
            image_source_bank: args.image_source_bank.enabled(),
            max_fetch_fraction: args.max_fetch_fraction,
            fetch_fresh: args.fetch_fresh,
            timings: args.timings,
            min_score: args.min_score,
            jobs,
            no_embeddings: embedder.is_none(),
            no_keypoints: args.no_keypoints,
            remote_video_mode: match args.remote_video_mode {
                eidolon_fetch::VideoMode::Keyframes => "keyframes",
                eidolon_fetch::VideoMode::FullCadence => "full-cadence",
            },
        }
    }

    /// Write the same effective settings as one prefixed JSON line on stderr.
    pub(crate) fn write_line(&self, writer: &mut impl Write) -> anyhow::Result<()> {
        write!(writer, "Effective settings: ")?;
        serde_json::to_writer(&mut *writer, self)?;
        writeln!(writer)?;
        Ok(())
    }
}
