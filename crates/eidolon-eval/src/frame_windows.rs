//! Ken-Burns thumbnail/clip truth from recorded pan/zoom parameters only.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, ensure};
use serde::Serialize;

use crate::model::{ExpectedPair, PathPair, Truth, TruthFile};

/// An auditable crop in normalized coordinates of the original still image.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Window {
    /// Left edge, in 0..1.
    pub x: f64,
    /// Top edge, in 0..1.
    pub y: f64,
    /// Visible fraction of the still's width.
    pub width: f64,
    /// Visible fraction of the still's height.
    pub height: f64,
}

/// Parameter-derived evidence for one thumbnail/clip pair.
#[derive(Debug, Serialize)]
pub struct Comparison {
    /// Thumbnail path relative to the corpus root.
    pub thumbnail: PathBuf,
    /// Clip path relative to the corpus root.
    pub video: PathBuf,
    /// Original still referenced by the video's retained provenance.
    pub still: PathBuf,
    /// Thumbnail capture time on the original video timeline.
    pub thumbnail_seconds: f64,
    /// Whether the clip contains the exact captured source timestamp.
    pub timestamp_inside_clip: bool,
    /// Best crop-window intersection-over-union over actual source frame times.
    pub best_iou: f64,
    /// Source timeline position of the best clip frame.
    pub best_frame_seconds: f64,
    /// Actual thumbnail crop, including a center-square overlay crop.
    pub thumbnail_window: Window,
    /// Full clip-frame crop at the best source timestamp.
    pub clip_window: Window,
    /// `frame_of`, `unscored` or `negative`.
    pub decision: &'static str,
}

/// Metadata-only annotation changes; no media or detector output is read.
#[derive(Debug, Default, Serialize)]
pub struct Audit {
    /// Previously absent positive edges added by this annotation pass.
    pub added_positive_pairs: usize,
    /// Previously positive clip edges no longer supported by provenance.
    pub removed_positive_pairs: usize,
    /// Low-overlap pairs now explicitly negative for direct predictions only.
    pub negative_pairs: usize,
    /// Partial-overlap pairs deliberately excluded from scoring.
    pub unscored_pairs: usize,
    /// All evaluated thumbnail/clip windows and their decisions.
    pub comparisons: Vec<Comparison>,
}

/// Rebuild Ken-Burns thumbnail/clip labels without decoding or modifying media.
///
/// Exact source timestamps remain valid even for cropped thumbnails. Otherwise,
/// `IoU` >=0.8 certifies `FrameOf`, 0.3..0.8 is unscored, and `IoU` <0.3 is direct-only negative.
/// Native videos without explicit Ken-Burns provenance remain unchanged.
///
/// # Errors
/// Returns an error for malformed recorded parameters or missing still geometry.
pub fn annotate(truth: &mut Truth) -> anyhow::Result<Audit> {
    let files: HashMap<_, _> = truth
        .files
        .iter()
        .map(|file| (file.path.as_path(), file))
        .collect();
    let mut comparisons = Vec::new();
    for source in &truth.sources {
        if !source.origin.starts_with("ffmpeg ken-burns source=") {
            continue;
        }
        let original = files
            .get(source.retained.as_path())
            .context("Ken-Burns original absent from truth")?;
        if !truth.files.iter().any(|file| {
            file.family == original.family
                && file.kind == "image"
                && file.variant.starts_with("thumb_")
        }) || !truth.files.iter().any(|file| {
            file.family == original.family
                && file.kind == "video"
                && matches!(file.variant.as_str(), "clip_10s" | "clip_25pct")
        }) {
            continue;
        }
        let pan = Pan::parse(&source.origin, &files)?;
        for thumbnail in truth.files.iter().filter(|file| {
            file.kind == "image"
                && file.family == original.family
                && file.variant.starts_with("thumb_")
        }) {
            let capture = truth
                .sources
                .iter()
                .find(|entry| entry.retained == thumbnail.path)
                .context("thumbnail capture provenance missing")?;
            let (source_path, timestamp) = capture
                .origin
                .strip_prefix("frame source=")
                .and_then(|value| value.split_once(" timestamp="))
                .context("invalid thumbnail capture provenance")?;
            ensure!(
                Path::new(source_path) == source.retained,
                "thumbnail references a different original video"
            );
            let timestamp: f64 = timestamp
                .split_whitespace()
                .next()
                .and_then(|value| value.strip_suffix('s'))
                .context("missing capture timestamp")?
                .parse()?;
            ensure!(
                timestamp.is_finite() && timestamp >= 0.0 && timestamp < pan.duration,
                "capture time outside source video"
            );
            for clip in truth.files.iter().filter(|file| {
                file.kind == "video"
                    && file.family == original.family
                    && matches!(file.variant.as_str(), "clip_10s" | "clip_25pct")
            }) {
                comparisons.push(pan.compare(thumbnail, clip, timestamp));
            }
        }
    }
    Ok(apply(truth, comparisons))
}

fn key(a: &Path, b: &Path) -> (PathBuf, PathBuf) {
    if a < b {
        (a.into(), b.into())
    } else {
        (b.into(), a.into())
    }
}

fn apply(truth: &mut Truth, comparisons: Vec<Comparison>) -> Audit {
    let affected: HashSet<_> = comparisons
        .iter()
        .map(|row| key(&row.thumbnail, &row.video))
        .collect();
    let positive: HashSet<_> = comparisons
        .iter()
        .filter(|row| row.decision == "frame_of")
        .map(|row| key(&row.thumbnail, &row.video))
        .collect();
    let existing: HashSet<_> = truth
        .pairs
        .iter()
        .filter(|pair| pair.relation == "frame_of")
        .map(|pair| key(&pair.a, &pair.b))
        .collect();
    let mut audit = Audit::default();
    truth.pairs.retain(|pair| {
        let pair_key = key(&pair.a, &pair.b);
        let remove = pair.relation == "frame_of"
            && affected.contains(&pair_key)
            && !positive.contains(&pair_key);
        audit.removed_positive_pairs += usize::from(remove);
        !remove
    });
    truth
        .hard_negatives
        .retain(|pair| !affected.contains(&key(&pair.a, &pair.b)));
    for row in &comparisons {
        match row.decision {
            "frame_of" if !existing.contains(&key(&row.thumbnail, &row.video)) => {
                let variant = truth
                    .files
                    .iter()
                    .find(|file| file.path == row.thumbnail)
                    .expect("evaluated thumbnail exists")
                    .variant
                    .clone();
                truth.pairs.push(ExpectedPair {
                    a: row.thumbnail.clone(),
                    b: row.video.clone(),
                    relation: "frame_of".into(),
                    kind: "image".into(),
                    variant,
                    visible_rgb_mae: None,
                    accepted_relations: Vec::new(),
                });
                audit.added_positive_pairs += 1;
            }
            "negative" => {
                truth.hard_negatives.push(PathPair {
                    a: row.thumbnail.clone(),
                    b: row.video.clone(),
                    direct_only: true,
                });
                audit.negative_pairs += 1;
            }
            "unscored" => audit.unscored_pairs += 1,
            _ => {}
        }
    }
    audit.comparisons = comparisons;
    audit
}

struct Pan {
    still: PathBuf,
    width: f64,
    height: f64,
    output_width: f64,
    output_height: f64,
    zoom_step: f64,
    period: f64,
    amplitude: f64,
    fps: f64,
    duration: f64,
    clip_10_start: f64,
    clip_25_start: f64,
}

fn value<'a>(text: &'a str, prefix: &str) -> anyhow::Result<&'a str> {
    text.split_whitespace()
        .find_map(|word| word.strip_prefix(prefix))
        .with_context(|| format!("missing pan parameter {prefix}"))
}

impl Pan {
    fn parse(text: &str, files: &HashMap<&Path, &TruthFile>) -> anyhow::Result<Self> {
        let (still, _) = text
            .strip_prefix("ffmpeg ken-burns source=")
            .and_then(|text| text.split_once(" zoom="))
            .context("invalid still-video provenance")?;
        let scene = files
            .get(Path::new(still))
            .and_then(|file| file.scene.as_ref())
            .context("still image geometry missing from retained truth")?;
        ensure!(scene.width > 0 && scene.height > 0, "empty still geometry");
        let zoom_step = value(text, "zoom=")?
            .strip_prefix("1+")
            .and_then(|value| value.strip_suffix("*on"))
            .context("unsupported zoom expression")?
            .parse()?;
        let (period, amplitude) = value(text, "pan=")?
            .strip_prefix("sin(on/")
            .and_then(|value| value.split_once(")*"))
            .context("unsupported pan expression")?;
        let (output_width, output_height) = value(text, "size=")?
            .split_once('x')
            .context("invalid output size")?;
        let output_width: f64 = output_width.parse()?;
        let duration: f64 = value(text, "duration=")?.parse()?;
        let result = Self {
            still: still.into(),
            width: output_width,
            height: (output_width * f64::from(scene.height) / f64::from(scene.width)).round(),
            output_width,
            output_height: output_height.parse()?,
            zoom_step,
            period: period.parse()?,
            amplitude: amplitude.parse()?,
            fps: value(text, "fps=")?.parse()?,
            duration,
            clip_10_start: value(text, "clip_10_start=").map_or(Ok(5.0), str::parse)?,
            clip_25_start: value(text, "clip_25_start=").map_or(Ok(duration * 0.5), str::parse)?,
        };
        ensure!(
            [
                result.width,
                result.height,
                result.output_height,
                result.period,
                result.fps,
                result.duration
            ]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
                && result.zoom_step.is_finite()
                && result.zoom_step >= 0.0
                && result.amplitude.is_finite(),
            "invalid pan parameters"
        );
        Ok(result)
    }

    fn window(&self, frame: u64) -> Window {
        let zoom = (1.0 + self.zoom_step * frame as f64).clamp(1.0, 10.0);
        let width = (self.width / zoom).floor();
        let height = (self.height / zoom).floor();
        let x = (self.width / 2.0 - self.width / zoom / 2.0
            + (frame as f64 / self.period).sin() * self.amplitude)
            .clamp(0.0, self.width - width);
        let y = (self.height / 2.0 - self.height / zoom / 2.0).clamp(0.0, self.height - height);
        // zoompan rounds input crop origins to the source YUV420 chroma grid.
        Window {
            x: (x.floor() as u64 & !1) as f64 / self.width,
            y: (y.floor() as u64 & !1) as f64 / self.height,
            width: width / self.width,
            height: height / self.height,
        }
    }

    fn compare(&self, thumbnail: &TruthFile, video: &TruthFile, timestamp: f64) -> Comparison {
        let (start, duration) = if video.variant == "clip_10s" {
            (self.clip_10_start, 10.0)
        } else {
            (self.clip_25_start, self.duration * 0.25)
        };
        let mut thumbnail_window = self.window((timestamp * self.fps).floor() as u64);
        if thumbnail.variant == "thumb_overlay" {
            let side = self.output_width.min(self.output_height);
            let fraction_x = side / self.output_width;
            let fraction_y = side / self.output_height;
            thumbnail_window.x += thumbnail_window.width * (1.0 - fraction_x) / 2.0;
            thumbnail_window.y += thumbnail_window.height * (1.0 - fraction_y) / 2.0;
            thumbnail_window.width *= fraction_x;
            thumbnail_window.height *= fraction_y;
        }
        let mut best_iou = -1.0;
        let mut best_frame = (start * self.fps).ceil() as u64;
        for frame in (start * self.fps).ceil() as u64
            ..((start + duration).min(self.duration) * self.fps).ceil() as u64
        {
            let overlap = iou(thumbnail_window, self.window(frame));
            if overlap > best_iou {
                best_iou = overlap;
                best_frame = frame;
            }
        }
        let timestamp_inside_clip = (start..start + duration).contains(&timestamp);
        let decision = if timestamp_inside_clip || best_iou >= 0.8 {
            "frame_of"
        } else if best_iou >= 0.3 {
            "unscored"
        } else {
            "negative"
        };
        Comparison {
            thumbnail: thumbnail.path.clone(),
            video: video.path.clone(),
            still: self.still.clone(),
            thumbnail_seconds: timestamp,
            timestamp_inside_clip,
            best_iou,
            best_frame_seconds: best_frame as f64 / self.fps,
            thumbnail_window,
            clip_window: self.window(best_frame),
            decision,
        }
    }
}

fn iou(a: Window, b: Window) -> f64 {
    let width = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
    let height = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
    let intersection = width.max(0.0) * height.max(0.0);
    intersection / (a.width * a.height + b.width * b.height - intersection)
}
