//! Command-line argument parsing and explicit default-path resolution.

use std::path::PathBuf;

use anyhow::Context as _;
use clap::{Parser, Subcommand, ValueEnum};
use eidolon_image::embed::ExecutionDevice;
use serde::Serialize;

// Independent command-line switches map directly to flags, not a state machine.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Parser)]
#[command(
    name = "eidolon",
    version,
    about = "Report exact and near-duplicate images, videos and text without changing files.",
    arg_required_else_help = true
)]
pub(crate) struct Args {
    /// Local files, recursive directories or explicit HTTP(S) URLs.
    #[arg(value_name = "PATH_OR_URL")]
    pub(crate) paths: Vec<PathBuf>,
    /// Read additional local paths or HTTP(S) URLs from stdin, one per line.
    #[arg(long = "stdin")]
    pub(crate) read_stdin: bool,
    /// Read NUL-delimited stdin paths instead of newline-delimited paths.
    #[arg(short = '0', long = "null", requires = "read_stdin")]
    pub(crate) nul: bool,
    /// Include dotfiles and dot-directories.
    #[arg(long)]
    pub(crate) hidden: bool,
    /// Emit versioned JSON instead of a human-readable report.
    #[arg(long)]
    pub(crate) json: bool,
    /// Minimum confidence for reported near matches (0 to 1).
    #[arg(long, default_value = "0.75", value_parser = parse_score)]
    pub(crate) min_score: f64,
    /// Disable image and video embedding inference.
    #[arg(long)]
    pub(crate) no_embeddings: bool,
    /// Use sampled thumbnail confirmation; candidates without pixel proof remain unverified.
    #[arg(long)]
    pub(crate) fast: bool,
    /// Video samples per second; positive and finite. Lower values can miss thumbnail frames.
    #[arg(long, default_value = "2", value_name = "N", value_parser = parse_positive_float)]
    pub(crate) video_fps: f64,
    /// Embedding execution device: auto, cpu or coreml.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = parse_embed_device)]
    pub(crate) embed_device: ExecutionDevice,
    /// Image/video frame matching: auto enables it only for URL input scans.
    #[arg(long, value_enum, default_value = "auto", value_name = "MODE")]
    pub(crate) thumbnails: ThumbnailMode,
    /// Remote caption-overlay crop search and prioritized native refinement (slower).
    #[arg(long, value_enum, default_value = "on", value_name = "MODE")]
    pub(crate) remote_overlay: Toggle,
    /// Supplement whole-image candidates with source-center embeddings.
    #[arg(long, value_enum, default_value = "on", value_name = "MODE")]
    pub(crate) image_source_bank: Toggle,
    /// Disable image keypoint verification.
    #[arg(long)]
    pub(crate) no_keypoints: bool,
    /// Worker threads; 0 uses all available cores.
    #[arg(long, default_value = "0")]
    pub(crate) jobs: usize,
    /// Feature database (default: ~/Library/Caches/eidolon/features.sqlite).
    #[arg(long, value_name = "PATH")]
    pub(crate) cache: Option<PathBuf>,
    /// Use an in-memory feature cache without reading or writing the database.
    #[arg(long, conflicts_with = "cache")]
    pub(crate) no_cache: bool,
    /// Reject URL inputs and prevent explicit model downloads.
    #[arg(long, global = true)]
    pub(crate) offline: bool,
    /// Remote video transfer: keyframes (default) or full-cadence (more bytes).
    #[arg(long, default_value = "keyframes", value_name = "MODE", value_parser = parse_remote_video_mode)]
    pub(crate) remote_video_mode: eidolon_fetch::VideoMode,
    /// Native-refinement transfer fraction; 1 disables the fractional limit.
    #[arg(long, default_value_t = eidolon_fetch::FetchOptions::default().max_fetch_fraction, value_name = "FRACTION", value_parser = parse_fetch_fraction)]
    pub(crate) max_fetch_fraction: f64,
    /// Reuse fresh cached URL sources (30-second fallback without Cache-Control).
    #[arg(long)]
    pub(crate) fetch_fresh: bool,
    /// Emit per-video native-refinement timing diagnostics on stderr.
    #[arg(long)]
    pub(crate) timings: bool,
    /// Model directory (default: ~/Library/Caches/eidolon/models).
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) model_dir: Option<PathBuf>,
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

/// Policy for dispatching image-to-video frame matching.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ThumbnailMode {
    /// Enable only when an explicit URL input is present.
    Auto,
    /// Enable for local and remote scans.
    On,
    /// Never invoke image-to-video frame matching.
    Off,
}

impl ThumbnailMode {
    /// Resolve the policy against the run's explicit input sources.
    pub(crate) const fn enabled(self, has_url_inputs: bool) -> bool {
        match self {
            Self::Auto => has_url_inputs,
            Self::On => true,
            Self::Off => false,
        }
    }
}

/// Explicit on/off policy for optional retrieval work.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum Toggle {
    /// Perform the additional retrieval work.
    On,
    /// Skip the additional retrieval work.
    Off,
}

impl Toggle {
    /// Whether this option requests the additional retrieval work.
    pub(crate) const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Download and verify the pinned model; on macOS, prepare the Auto embedding backend.
    FetchModels,
}

pub(crate) fn default_path(name: &str) -> anyhow::Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("determine home directory; set --cache and --model-dir explicitly")?
        .join("Library/Caches/eidolon")
        .join(name))
}

fn parse_score(input: &str) -> Result<f64, String> {
    let score = input.parse::<f64>().map_err(|error| error.to_string())?;
    if score.is_finite() && (0.0..=1.0).contains(&score) {
        Ok(score)
    } else {
        Err("score must be a finite number between 0 and 1".into())
    }
}

fn parse_remote_video_mode(input: &str) -> Result<eidolon_fetch::VideoMode, String> {
    match input {
        "keyframes" => Ok(eidolon_fetch::VideoMode::Keyframes),
        "full-cadence" => Ok(eidolon_fetch::VideoMode::FullCadence),
        _ => Err("remote video mode must be keyframes or full-cadence".into()),
    }
}

fn parse_positive_float(input: &str) -> Result<f64, String> {
    let value = input.parse::<f64>().map_err(|error| error.to_string())?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("value must be positive and finite".into())
    }
}

fn parse_fetch_fraction(input: &str) -> Result<f64, String> {
    let value = parse_positive_float(input)?;
    if value <= 1.0 {
        Ok(value)
    } else {
        Err("fraction must be greater than 0 and at most 1".into())
    }
}

fn parse_embed_device(input: &str) -> Result<ExecutionDevice, String> {
    match input {
        "auto" => Ok(ExecutionDevice::Auto),
        "cpu" => Ok(ExecutionDevice::Cpu),
        "coreml" => Ok(ExecutionDevice::CoreMl),
        _ => Err("embedding device must be auto, cpu or coreml".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_scan_options_and_fetch_model_directory() {
        let args = Args::try_parse_from([
            "eidolon",
            "--stdin",
            "-0",
            "--json",
            "--jobs",
            "2",
            "--min-score",
            "0.8",
            "--no-embeddings",
            "--no-keypoints",
            "--no-cache",
        ])
        .expect("valid scan flags");
        assert!(args.read_stdin && args.nul && args.json && args.no_cache);
        assert_eq!(args.jobs, 2);
        assert_eq!(args.min_score, 0.8);
        assert!(Args::try_parse_from(["eidolon", "-0"]).is_err());
        let fetch = Args::try_parse_from([
            "eidolon",
            "fetch-models",
            "--model-dir",
            "/tmp/eidolon-models",
        ])
        .expect("fetch supports the explicit model directory");
        assert!(matches!(fetch.command, Some(Command::FetchModels)));
        assert_eq!(fetch.model_dir, Some("/tmp/eidolon-models".into()));
    }

    #[test]
    fn accepts_mixed_urls_and_paths_with_explicit_offline_mode() {
        let args = Args::try_parse_from([
            "eidolon",
            "--offline",
            "https://example.test/image.png",
            "/tmp/eidolon-local",
        ])
        .expect("URL inputs are accepted by the parser before offline validation");
        assert!(args.offline);
        assert_eq!(
            args.paths,
            vec![
                PathBuf::from("https://example.test/image.png"),
                PathBuf::from("/tmp/eidolon-local"),
            ]
        );
    }

    #[test]
    fn selects_explicit_remote_video_transfer_modes() {
        let default = Args::try_parse_from(["eidolon", "/tmp/eidolon-local"]).expect("local scan");
        assert!(matches!(
            default.remote_video_mode,
            eidolon_fetch::VideoMode::Keyframes
        ));
        let full = Args::try_parse_from([
            "eidolon",
            "--remote-video-mode",
            "full-cadence",
            "https://example.test/video.mp4",
        ])
        .expect("explicit full-cadence transfer");
        assert!(matches!(
            full.remote_video_mode,
            eidolon_fetch::VideoMode::FullCadence
        ));
        assert!(
            Args::try_parse_from([
                "eidolon",
                "--remote-video-mode",
                "unknown",
                "https://example.test/video.mp4",
            ])
            .is_err()
        );
    }

    #[test]
    fn accepts_explicit_performance_and_precision_settings() {
        assert!(
            Args::try_parse_from([
                "eidolon",
                "--fast",
                "--video-fps",
                "1.5",
                "--embed-device",
                "cpu",
                "--thumbnails",
                "off",
                "--max-fetch-fraction",
                "0.4",
                "/tmp/eidolon-input",
            ])
            .is_ok()
        );
    }

    #[test]
    fn defaults_preserve_confirmation_and_select_automatic_policies() {
        let args = Args::try_parse_from(["eidolon", "/tmp/eidolon-input"]).expect("default scan");
        assert!(!args.fast);
        assert_eq!(args.video_fps, 2.0);
        assert_eq!(args.embed_device, ExecutionDevice::Auto);
        assert_eq!(args.thumbnails, ThumbnailMode::Auto);
        assert_eq!(args.remote_overlay, Toggle::On);
        assert_eq!(args.image_source_bank, Toggle::On);
        assert_eq!(args.max_fetch_fraction, 0.25);
        assert!(!args.fetch_fresh);
        assert!(!args.timings);
    }

    #[test]
    fn accepts_opt_in_freshness_and_refinement_timings() {
        let args = Args::try_parse_from([
            "eidolon",
            "--fetch-fresh",
            "--timings",
            "/tmp/eidolon-input",
        ])
        .expect("opt-in diagnostics and freshness");
        assert!(args.fetch_fresh);
        assert!(args.timings);
    }

    #[test]
    fn rejects_nonpositive_and_nonfinite_video_rates() {
        for value in ["0", "-1", "NaN", "inf", "-inf"] {
            let option = format!("--video-fps={value}");
            assert!(
                Args::try_parse_from(["eidolon", &option, "/tmp/eidolon-input"]).is_err(),
                "accepted invalid video fps {value}"
            );
        }
    }

    #[test]
    fn validates_fractional_fetch_budgets() {
        for value in ["0", "-0.1", "1.01", "NaN", "inf"] {
            let option = format!("--max-fetch-fraction={value}");
            assert!(
                Args::try_parse_from(["eidolon", &option, "/tmp/eidolon-input"]).is_err(),
                "accepted invalid fetch fraction {value}"
            );
        }
        for value in ["0.001", "0.25", "1"] {
            assert!(
                Args::try_parse_from([
                    "eidolon",
                    "--max-fetch-fraction",
                    value,
                    "/tmp/eidolon-input",
                ])
                .is_ok()
            );
        }
    }

    #[test]
    fn accepts_only_supported_embedding_devices_and_thumbnail_policies() {
        for device in ["auto", "cpu", "coreml"] {
            assert!(
                Args::try_parse_from(["eidolon", "--embed-device", device, "/tmp/eidolon-input",])
                    .is_ok()
            );
        }
        for policy in ["auto", "on", "off"] {
            assert!(
                Args::try_parse_from(["eidolon", "--thumbnails", policy, "/tmp/eidolon-input",])
                    .is_ok()
            );
        }
        assert!(
            Args::try_parse_from(["eidolon", "--embed-device", "gpu", "/tmp/eidolon-input",])
                .is_err()
        );
        assert!(
            Args::try_parse_from(["eidolon", "--thumbnails", "sometimes", "/tmp/eidolon-input",])
                .is_err()
        );
    }

    #[test]
    fn rejects_nonfinite_and_out_of_range_scores() {
        for score in ["NaN", "inf", "-0.1", "1.1"] {
            assert!(
                Args::try_parse_from(["eidolon", "--min-score", score, "/tmp/eidolon-input"])
                    .is_err()
            );
        }
    }
}
