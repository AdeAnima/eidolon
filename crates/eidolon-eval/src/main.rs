//! Local corpus generation, scoring, and detector orchestration.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use anyhow::{Context as _, ensure};
use clap::{Parser, Subcommand};
use eidolon_eval::generate::{DEFAULT_SEED, Scale, Validation, generate, refresh_truth};
use eidolon_eval::model::{Output, Truth};
use eidolon_eval::score::{Report, evaluate};
use eidolon_eval::serve::{CorpusServer, Traffic, media_path, url_sample};

#[derive(Parser)]
#[command(
    version,
    about = "Generate and score a deterministic, local duplicate corpus"
)]
struct Args {
    #[command(subcommand)]
    command: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Generate local media and truth.json, decoding/probing every result.
    Generate {
        out_dir: PathBuf,
        #[arg(long, value_enum, default_value = "small")]
        scale: Scale,
        #[arg(long, default_value_t = DEFAULT_SEED)]
        seed: u64,
    },
    /// Generate offline from an attributed, hash-verified local source manifest.
    GenerateSources {
        manifest: PathBuf,
        out_dir: PathBuf,
        #[arg(long, default_value_t = DEFAULT_SEED)]
        seed: u64,
    },
    /// Rebuild truth in a separate metadata directory, leaving media untouched.
    RefreshTruth { truth: PathBuf, out_dir: PathBuf },
    /// Refresh still-video clip windows using metadata only; do not decode media.
    RefreshFrames { truth: PathBuf, out_dir: PathBuf },
    /// Compare expected pairs against eidolon's version-one JSON report.
    Score {
        truth: PathBuf,
        output: PathBuf,
        #[arg(long)]
        json: bool,
        /// Fail with exit code 2 when group recall is below this fraction (0..1).
        #[arg(long, value_parser = recall_threshold)]
        fail_under: Option<f64>,
        /// Counted HTTP traffic from a standalone corpus server.
        #[arg(long)]
        traffic: Option<PathBuf>,
    },
    /// Generate if truth.json is missing, invoke eidolon, and score its output.
    Run {
        out_dir: PathBuf,
        #[arg(long, value_enum, default_value = "small")]
        scale: Scale,
        #[arg(long, default_value_t = DEFAULT_SEED)]
        seed: u64,
        /// Detector executable; defaults to target/release/eidolon, then cargo run.
        #[arg(long)]
        eidolon: Option<PathBuf>,
        #[arg(long)]
        json: bool,
        #[arg(long, value_parser = recall_threshold)]
        fail_under: Option<f64>,
        /// Serve a fixed20% derivative sample as URLs, keeping originals local.
        #[arg(long)]
        urls: bool,
        /// Remote-video sampling policy; defaults to keyframes when --urls is enabled.
        #[arg(long, value_enum, requires = "urls")]
        remote_video_mode: Option<RemoteVideoMode>,
        /// Extra detector options after --; override runner defaults such as --thumbnails on.
        #[arg(last = true, allow_hyphen_values = true, value_name = "EIDOLON_ARG")]
        eidolon_args: Vec<String>,
    },
    /// Serve manifest-listed media over loopback HTTP with ranges and byte counters.
    Serve {
        corpus: PathBuf,
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum RemoteVideoMode {
    Keyframes,
    FullCadence,
}

impl RemoteVideoMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Keyframes => "keyframes",
            Self::FullCadence => "full-cadence",
        }
    }
}

fn main() -> ExitCode {
    match execute(Args::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn execute(args: Args) -> anyhow::Result<bool> {
    match args.command {
        Action::Generate {
            out_dir,
            scale,
            seed,
        } => {
            let (truth, validation) = generate(&out_dir, scale, seed)?;
            print_validation(&truth, validation, &truth.root.join("truth.json"))?;
            Ok(true)
        }
        Action::GenerateSources {
            manifest,
            out_dir,
            seed,
        } => {
            let (truth, validation) =
                eidolon_eval::generate::licensed::generate(&manifest, &out_dir, seed)?;
            print_validation(&truth, validation, &truth.root.join("truth.json"))?;
            Ok(true)
        }
        Action::RefreshTruth { truth, out_dir } => {
            let (truth, validation) = refresh_truth(load_truth(&truth)?, &out_dir)?;
            print_validation(&truth, validation, &out_dir.join("truth.json"))?;
            Ok(true)
        }
        Action::RefreshFrames { truth, out_dir } => {
            let (_, audit) =
                eidolon_eval::generate::refresh_frame_truth(load_truth(&truth)?, &out_dir)?;
            writeln!(
                io::stdout().lock(),
                "FrameOf added={} removed={} unscored={} negative={}",
                audit.added_positive_pairs,
                audit.removed_positive_pairs,
                audit.unscored_pairs,
                audit.negative_pairs
            )?;
            Ok(true)
        }
        Action::Score {
            truth,
            output,
            json,
            fail_under,
            traffic,
        } => {
            let truth = load_truth(&truth)?;
            let output: Output = serde_json::from_slice(&fs::read(output)?)?;
            let mut report = evaluate(&truth, &output)?;
            if let Some(path) = traffic {
                let counts: Traffic = serde_json::from_slice(&fs::read(path)?)?;
                let paths: HashSet<_> = output
                    .files
                    .iter()
                    .filter_map(|file| file.url.as_ref())
                    .map(|origin| media_path(&origin.url))
                    .collect::<anyhow::Result<_>>()?;
                ensure!(
                    !paths.is_empty(),
                    "--traffic requires URL-backed output files"
                );
                report.served_traffic = Some(counts.for_paths(&paths));
            }
            deliver(&report, json, fail_under)
        }
        Action::Run {
            out_dir,
            scale,
            seed,
            eidolon,
            json,
            fail_under,
            urls,
            remote_video_mode,
            eidolon_args,
        } => {
            let report = run_evaluation(
                &out_dir,
                scale,
                seed,
                eidolon.as_deref(),
                urls,
                remote_video_mode,
                &eidolon_args,
            )?;
            deliver(&report, json, fail_under)
        }
        Action::Serve { corpus, port } => serve_command(&corpus, port),
    }
}

fn print_validation(truth: &Truth, validation: Validation, manifest: &Path) -> anyhow::Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "Validated {} files; {} positive pairs; {} hard negatives",
        validation.files,
        truth.pairs.len(),
        truth.hard_negatives.len()
    )?;
    for (category, count) in validation.categories {
        writeln!(stdout, "{category}: {count}")?;
    }
    writeln!(
        stdout,
        "Distinctness: {}",
        serde_json::to_string(&validation.distinctness)?
    )?;
    writeln!(stdout, "Truth: {}", manifest.display())?;
    Ok(())
}

fn serve_command(corpus: &Path, port: u16) -> anyhow::Result<bool> {
    let manifest = if corpus.is_dir() {
        corpus.join("truth.json")
    } else {
        corpus.to_path_buf()
    };
    let truth = load_truth(&manifest)?;
    let server = CorpusServer::start(&truth, port)?;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let signal_stop = std::sync::Arc::clone(&stop);
    ctrlc::set_handler(move || signal_stop.store(true, std::sync::atomic::Ordering::Relaxed))?;
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "Serving http://{}/; counters: /__eidolon__/stats; stop with Ctrl-C",
        server.address()
    )?;
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let traffic = server.stop()?;
    writeln!(
        stdout,
        "Served {} body bytes; full objects {}; ratio {:.4}; {}",
        traffic.bytes_served,
        traffic.full_remote_bytes,
        traffic.bytes_fetched_ratio,
        truth.root.join("url-traffic.json").display()
    )?;
    Ok(true)
}

fn run_evaluation(
    out_dir: &Path,
    scale: Scale,
    seed: u64,
    executable: Option<&Path>,
    urls: bool,
    remote_video_mode: Option<RemoteVideoMode>,
    extra_args: &[String],
) -> anyhow::Result<Report> {
    let manifest = out_dir.join("truth.json");
    let truth = if manifest.is_file() {
        load_truth(&manifest)?
    } else {
        generate(out_dir, scale, seed)?.0
    };
    let server = if urls {
        Some(CorpusServer::start(&truth, 0)?)
    } else {
        None
    };
    let sample = url_sample(&truth);
    let inputs: Vec<OsString> = if let Some(server) = &server {
        truth
            .files
            .iter()
            .map(|file| {
                if sample.contains(&file.path) {
                    server.url_for(&file.path).map(OsString::from)
                } else {
                    Ok(truth.root.join(&file.path).into_os_string())
                }
            })
            .collect::<anyhow::Result<_>>()?
    } else {
        vec![truth.root.join("media").into_os_string()]
    };
    let mode = urls.then_some(remote_video_mode.unwrap_or(RemoteVideoMode::Keyframes));
    let options = detector_options(mode, extra_args);
    let output = run_detector(&truth.root, executable, &inputs, &options, urls)?;
    let parsed: Output =
        serde_json::from_slice(&output).context("eidolon returned invalid report JSON")?;
    let mut report = evaluate(&truth, &parsed)?;
    report.remote_video_mode = if urls {
        option_value(&options, "--remote-video-mode").map(str::to_owned)
    } else {
        None
    };
    report.eidolon_args = Some(options);
    if let Some(server) = server {
        report.served_traffic = Some(server.stop()?.for_paths(&sample));
        fs::write(
            truth.root.join("url-traffic.json"),
            serde_json::to_vec_pretty(&report.served_traffic)?,
        )?;
    }
    fs::write(
        truth.root.join("eidolon-score.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

fn load_truth(path: &Path) -> anyhow::Result<Truth> {
    let mut truth: Truth = serde_json::from_slice(&fs::read(path)?)?;
    if truth.root.is_relative() {
        truth.root = path
            .canonicalize()?
            .parent()
            .context("manifest has no parent")?
            .join(&truth.root);
    }
    Ok(truth)
}

fn detector_options(mode: Option<RemoteVideoMode>, extra: &[String]) -> Vec<String> {
    let mut options = Vec::new();
    if !has_option(extra, "--json") {
        options.push("--json".into());
    }
    if !has_option(extra, "--thumbnails") {
        options.extend(["--thumbnails".into(), "on".into()]);
    }
    if let Some(mode) = mode
        && !has_option(extra, "--remote-video-mode")
    {
        options.extend(["--remote-video-mode".into(), mode.as_str().into()]);
    }
    options.extend_from_slice(extra);
    if mode.is_some() {
        for flag in ["--stdin", "--null"] {
            if !has_option(extra, flag) {
                options.push(flag.into());
            }
        }
    }
    options
}

fn has_option(options: &[String], name: &str) -> bool {
    options.iter().any(|option| {
        option == name
            || option
                .strip_prefix(name)
                .is_some_and(|suffix| suffix.starts_with('='))
    })
}

fn option_value<'a>(options: &'a [String], name: &str) -> Option<&'a str> {
    options.iter().enumerate().find_map(|(index, option)| {
        if option == name {
            options.get(index + 1).map(String::as_str)
        } else {
            option
                .strip_prefix(name)
                .and_then(|suffix| suffix.strip_prefix('='))
        }
    })
}

fn run_detector(
    root: &Path,
    executable: Option<&Path>,
    inputs: &[OsString],
    options: &[String],
    stream_inputs: bool,
) -> anyhow::Result<Vec<u8>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .context("crate is outside workspace")?;
    let default = workspace.join("target/release/eidolon");
    let mut command = if let Some(path) = executable {
        Command::new(path)
    } else if default.is_file() {
        Command::new(default)
    } else {
        let mut cargo = Command::new("cargo");
        cargo
            .current_dir(workspace)
            .args(["run", "--release", "-p", "eidolon-cli", "--"]);
        cargo
    };
    command.args(options);
    let (output, input_error) = if stream_inputs {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting eidolon")?;
        let mut stdin = io::BufWriter::new(child.stdin.take().context("missing detector stdin")?);
        let write_result = inputs
            .iter()
            .try_for_each(|input| -> io::Result<()> {
                stdin.write_all(input.as_encoded_bytes())?;
                stdin.write_all(&[0])
            })
            .and_then(|()| stdin.flush());
        drop(stdin);
        (child.wait_with_output()?, write_result.err())
    } else {
        (
            command.args(inputs).output().context("starting eidolon")?,
            None,
        )
    };
    // Preserve detector output before parsing; errors remain inspectable offline.
    fs::write(root.join("eidolon-output.json"), &output.stdout)?;
    fs::write(root.join("eidolon-stderr.txt"), &output.stderr)?;
    ensure!(
        output.status.success(),
        "eidolon exited {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    if let Some(error) = input_error {
        return Err(error).context("writing detector inputs");
    }
    Ok(output.stdout)
}

fn transport_metrics(stdout: &mut impl io::Write, report: &Report) -> io::Result<()> {
    if let Some(options) = &report.eidolon_args {
        writeln!(stdout, "Eidolon args: {options:?}")?;
    }
    if let Some(traffic) = &report.url_traffic {
        writeln!(
            stdout,
            "URL body bytes: {} / {} full remote bytes; bytes-fetched ratio {:.4}; {} files",
            traffic.bytes_served,
            traffic.full_remote_bytes,
            traffic.bytes_fetched_ratio,
            traffic.files.len()
        )?;
    }
    for (name, metric) in &report.frame_of {
        writeln!(
            stdout,
            "FrameOf {name}: direct {}/{} ({:.3}), group {}/{} ({:.3})",
            metric.direct_found,
            metric.expected,
            metric.direct_recall,
            metric.group_found,
            metric.expected,
            metric.group_recall
        )?;
    }
    if let Some(traffic) = &report.served_traffic {
        writeln!(
            stdout,
            "HTTP body bytes served: {} / {}; served ratio {:.4}",
            traffic.bytes_served, traffic.full_remote_bytes, traffic.bytes_fetched_ratio
        )?;
    }
    Ok(())
}

fn deliver(report: &Report, json: bool, fail_under: Option<f64>) -> anyhow::Result<bool> {
    let mut stdout = io::stdout().lock();
    if json {
        serde_json::to_writer_pretty(&mut stdout, report)?;
        writeln!(stdout)?;
    } else {
        transport_metrics(&mut stdout, report)?;
        writeln!(
            stdout,
            "category                         precision direct-recall group-recall label-accuracy direction-accuracy"
        )?;
        for (name, metrics) in [
            ("DUPLICATES", &report.overall),
            ("VARIANTS", &report.variants),
        ]
        .into_iter()
        .chain(
            report
                .categories
                .iter()
                .map(|(name, metrics)| (name.as_str(), metrics)),
        ) {
            let label = metrics
                .relation_accuracy
                .map_or_else(|| "n/a".into(), |value| format!("{value:.3}"));
            let direction = metrics
                .direction_accuracy
                .map_or_else(|| "n/a".into(), |value| format!("{value:.3}"));
            let direct_recall = if metrics.expected == 0 {
                "n/a".into()
            } else {
                format!("{:.3}", metrics.direct_recall)
            };
            let group_recall = if metrics.group_expected == 0 {
                "n/a".into()
            } else {
                format!("{:.3}", metrics.group_recall)
            };
            writeln!(
                stdout,
                "{name:32} {:.3}     {direct_recall:13} {group_recall:12} {label:14} {direction}",
                metrics.direct_precision
            )?;
        }
        writeln!(
            stdout,
            "Group precision: {:.3}; hard-negative errors: {} direct / {} grouped; ignored sibling predictions: {}",
            report.overall.group_precision,
            report.hard_negative_false_positives,
            report.hard_negative_group_false_positives,
            report.ignored_sibling_predictions
        )?;
        for (title, diagnostics) in [
            ("Misses (after groups)", &report.misses),
            (
                "Direct misses (includes group-only finds)",
                &report.direct_misses,
            ),
            ("False positives", &report.false_positives),
            ("Group false positives", &report.group_false_positives),
            ("Wrong directions", &report.wrong_directions),
        ] {
            writeln!(stdout, "{title} (top 20):")?;
            for item in diagnostics {
                writeln!(
                    stdout,
                    "  {} <-> {} [{}] {:?}",
                    item.a.display(),
                    item.b.display(),
                    item.relation,
                    item.score
                )?;
            }
        }
    }
    let passes = fail_under.is_none_or(|threshold| report.overall.group_recall >= threshold);
    if !passes {
        eprintln!(
            "group recall {:.4} is below --fail-under {}",
            report.overall.group_recall,
            fail_under.unwrap_or(0.0)
        );
    }
    Ok(passes)
}

fn recall_threshold(value: &str) -> Result<f64, String> {
    let threshold: f64 = value
        .parse()
        .map_err(|_| "recall must be a fraction in 0..1".to_owned())?;
    if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
        Ok(threshold)
    } else {
        Err("recall must be a fraction in 0..1".into())
    }
}
