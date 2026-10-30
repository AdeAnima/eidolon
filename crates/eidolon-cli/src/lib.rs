//! Report-only CLI orchestration for exact and near-duplicate media detection.

mod args;
mod classify;
mod exact;
mod grouping;
mod inputs;
mod remote;
mod report;
mod settings;
mod skipped;

use std::collections::HashMap;
use std::io::{IsTerminal as _, Write as _};
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
use anyhow::Context as _;
use clap::Parser as _;
use eidolon_core::cache::FeatureCache;
use eidolon_core::{Config, Ctx, Embedder, FileId, FrameRefiner, Kind, Match};
use eidolon_image::embed::{ExecutionDevice, OnnxEmbedder};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// Parse arguments and report matches, or fetch models and prepare the macOS backend.
///
/// Scans never download models or change input files. Explicit URL inputs fetch
/// local analysis artifacts unless `--offline` is set. Diagnostics and
/// terminal-only progress go to stderr; stdout contains only the report.
///
/// # Errors
/// Returns an error if input discovery or exact hashing fails, the cache cannot
/// be accessed, a detector has a run-wide failure, or output cannot be written.
pub fn run() -> anyhow::Result<()> {
    let mut args = args::Args::parse();
    let model_dir = match args.model_dir.take() {
        Some(path) => path,
        None => args::default_path("models")?,
    };
    if matches!(args.command, Some(args::Command::FetchModels)) {
        anyhow::ensure!(!args.offline, "fetch-models cannot be used with --offline");
        eidolon_image::models::fetch(&model_dir)?;
        #[cfg(target_os = "macos")]
        prepare_auto_backend(&model_dir, &mut std::io::stderr())?;
        return Ok(());
    }
    let config = Config {
        timings: args.timings,
        remote_overlay: args.remote_overlay.enabled(),
        min_score: args.min_score,
        model_dir,
        no_embeddings: args.no_embeddings,
        no_keypoints: args.no_keypoints,
        jobs: args.jobs,
        fast: args.fast,
        video_fps: args.video_fps,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.jobs)
        .build()?;
    pool.install(|| scan(args, &config))
}

#[cfg(target_os = "macos")]
fn prepare_auto_backend(
    model_dir: &std::path::Path,
    writer: &mut impl std::io::Write,
) -> anyhow::Result<()> {
    writeln!(writer, "Preparing automatic embedding backend...")?;
    let mut embedder = OnnxEmbedder::load(model_dir)
        .context("prepare automatic embedding backend after model verification")?
        .context("fetched model is unavailable; automatic backend was not prepared")?;
    // Installation prepares the accelerator independent of any current scan.
    embedder.set_workload(usize::MAX)?;
    embedder.prepare()?;
    match embedder.device() {
        ExecutionDevice::CoreMl => writeln!(
            writer,
            "Prepared automatic embedding backend: CoreML (local compiled cache prepared)."
        )?,
        ExecutionDevice::Cpu => writeln!(
            writer,
            "Prepared automatic embedding backend: CPU fallback (CoreML preparation not confirmed)."
        )?,
        ExecutionDevice::Auto => {
            anyhow::bail!("embedding loader did not resolve the automatic execution device");
        }
    }
    Ok(())
}

fn scan(mut args: args::Args, config: &Config) -> anyhow::Result<()> {
    let progress = Progress::new()?;
    progress.stage("Resolving input files and explicit URLs");
    let mut roots = std::mem::take(&mut args.paths);
    if args.read_stdin {
        roots.extend(inputs::read_paths(std::io::stdin().lock(), args.nul)?);
    }
    anyhow::ensure!(
        args.read_stdin || !roots.is_empty(),
        "provide input paths, URLs or --stdin"
    );
    let remote::ResolvedInputs {
        mut files,
        mut sources,
        fetcher,
    } = remote::resolve(
        roots,
        args.hidden,
        args.offline,
        args.remote_video_mode,
        args.max_fetch_fraction,
        args.fetch_fresh,
        &mut std::io::stderr().lock(),
    )?;
    let cache = open_cache(&args)?;
    progress.stage("Classifying file contents");
    classify::classify(&mut files, &cache)?;
    progress.stage("Finding byte-identical files");
    let mut matches = exact::find(
        files
            .iter()
            .filter(|file| sources.eligible_for_exact(file.id)),
        &cache,
    )?;
    progress.stage("Loading optional embedding model");
    let mut embedder = load_embedder(config, args.embed_device)?;
    if let Some(model) = embedder.as_mut() {
        let thumbnails = args.thumbnails.enabled(sources.transfer.requested_urls > 0);
        let planning = Ctx {
            config,
            cache: &cache,
            embedder: Some(model),
            refiner: None,
        };
        let image_work = eidolon_image::uncached_embedding_work(
            &files,
            &planning,
            args.image_source_bank.enabled(),
        )?;
        let work = image_work.saturating_add(eidolon_video::uncached_embedding_work(
            &files, &planning, thumbnails,
        )?);
        model.set_workload(work)?;
        if work > 0
            && let Err(error) = model.prepare()
        {
            writeln!(
                std::io::stderr().lock(),
                "eidolon: embedding model unavailable ({error:#}); running without embeddings."
            )?;
            embedder = None;
        }
    }
    let settings = settings::Settings::new(
        &args,
        embedder.as_ref(),
        sources.transfer.requested_urls > 0,
        rayon::current_num_threads(),
    );
    settings.write_line(&mut std::io::stderr().lock())?;
    let ctx = Ctx {
        config,
        cache: &cache,
        embedder: embedder.as_ref().map(|embedder| embedder as &dyn Embedder),
        refiner: fetcher
            .as_ref()
            .filter(|fetcher| fetcher.has_sparse_videos())
            .map(|fetcher| fetcher as &dyn FrameRefiner),
    };
    let mut skipped = Vec::new();
    let near = find_near(
        &mut files,
        &ctx,
        &progress,
        settings.thumbnails,
        args.image_source_bank.enabled(),
        &mut skipped,
    )?;
    skipped.sort_unstable_by(|a, b| a.path.cmp(&b.path));
    sources.finish_transfers(&files, fetcher.as_ref(), &mut std::io::stderr().lock())?;
    merge_near(&mut matches, near, config.min_score);
    matches
        .sort_unstable_by_key(|matched| (matched.a, matched.b, relation_order(matched.relation)));
    progress.stage("Grouping matches");
    let groups = grouping::build(&files, &matches)?;
    drop(progress);
    report::write(
        args.json, &files, &matches, &groups, &sources, &settings, &skipped,
    )
}

fn open_cache(args: &args::Args) -> anyhow::Result<FeatureCache> {
    match (args.no_cache, args.cache.as_deref()) {
        (true, _) => FeatureCache::in_memory(),
        (false, Some(path)) => FeatureCache::open(path),
        (false, None) => FeatureCache::open(&args::default_path("features.sqlite")?),
    }
}

fn load_embedder(config: &Config, device: ExecutionDevice) -> anyhow::Result<Option<OnnxEmbedder>> {
    if config.no_embeddings {
        return Ok(None);
    }
    match OnnxEmbedder::load_with_device(&config.model_dir, device) {
        Ok(embedder) => {
            if embedder.is_none() {
                writeln!(
                    std::io::stderr().lock(),
                    "eidolon: embedding model absent; running without embeddings. Run `eidolon fetch-models` to install it."
                )?;
            }
            Ok(embedder)
        }
        Err(error) => {
            writeln!(
                std::io::stderr().lock(),
                "eidolon: embedding model unavailable ({error:#}); running without embeddings."
            )?;
            Ok(None)
        }
    }
}

fn find_near(
    files: &mut [eidolon_core::FileEntry],
    ctx: &Ctx<'_>,
    progress: &Progress,
    thumbnails: bool,
    image_source_bank: bool,
    skipped: &mut Vec<skipped::Skipped>,
) -> anyhow::Result<Vec<Match>> {
    // Temporarily partition owned entries, avoiding path clones for detector slices.
    // Restore FileId order before grouping and serialization.
    files.sort_unstable_by_key(|file| kind_order(file.kind));
    let image_end = files.partition_point(|file| kind_order(file.kind) == 0);
    let video_end = files.partition_point(|file| kind_order(file.kind) <= 1);
    let text_end = files.partition_point(|file| kind_order(file.kind) <= 2);
    let mut near = Vec::new();
    let mut valid_images = image_end;
    let frame_embeddings = if image_source_bank
        && image_end > 0
        && !ctx.config.no_embeddings
        && ctx.embedder.is_some()
    {
        progress.stage("Loading source-aspect center candidates");
        let started = Instant::now();
        let embeddings = eidolon_image::source_frame_embeddings(&files[..image_end], ctx)?;
        writeln!(
            std::io::stderr().lock(),
            "eidolon: source-aspect center candidates for {} images: {:.3}s",
            embeddings.len(),
            started.elapsed().as_secs_f64()
        )?;
        Some(embeddings)
    } else {
        None
    };
    if image_end > 0 {
        progress.stage("Comparing images");
        let started = Instant::now();
        let (matches, valid) = skipped::find(
            &mut files[..image_end],
            skipped,
            &mut std::io::stderr().lock(),
            |files| eidolon_image::find(files, ctx, frame_embeddings.as_ref()),
        )?;
        near.extend(matches);
        valid_images = valid;
        if frame_embeddings.is_some() {
            writeln!(
                std::io::stderr().lock(),
                "eidolon: image candidate generation and proof: {:.3}s",
                started.elapsed().as_secs_f64()
            )?;
        }
    }
    let mut valid_videos = video_end - image_end;
    if video_end > image_end {
        progress.stage("Comparing video timelines");
        let (matches, errors) =
            eidolon_video::find_with_skipped(&files[image_end..video_end], ctx)?;
        near.extend(matches);
        for input in errors {
            valid_videos = skipped::exclude(
                &mut files[image_end..video_end],
                valid_videos,
                input,
                skipped,
                &mut std::io::stderr().lock(),
            )?;
        }
    }
    if text_end > video_end {
        progress.stage("Comparing text and excerpts");
        let (matches, _) = skipped::find(
            &mut files[video_end..text_end],
            skipped,
            &mut std::io::stderr().lock(),
            |files| eidolon_text::find(files, ctx),
        )?;
        near.extend(matches);
    }
    if thumbnails && valid_images > 0 && valid_videos > 0 {
        progress.stage("Comparing images with video frames");
        let (matches, errors) = eidolon_video::find_frames_with_skipped(
            &files[..valid_images],
            &files[image_end..image_end + valid_videos],
            ctx,
        )?;
        near.extend(matches);
        for input in errors {
            near.retain(|matched| matched.a != input.file && matched.b != input.file);
            let (start, valid) = if files[..valid_images]
                .iter()
                .any(|file| file.id == input.file)
            {
                (0, &mut valid_images)
            } else {
                (image_end, &mut valid_videos)
            };
            *valid = skipped::exclude(
                &mut files[start..start + *valid],
                *valid,
                input,
                skipped,
                &mut std::io::stderr().lock(),
            )?;
        }
    }
    files.sort_unstable_by_key(|file| file.id);
    Ok(near)
}

fn merge_near(exact: &mut Vec<Match>, near: Vec<Match>, min_score: f64) {
    let pairs: HashMap<(FileId, FileId), usize> = exact
        .iter()
        .enumerate()
        .map(|(index, matched)| ((matched.a, matched.b), index))
        .collect();
    for mut matched in near
        .into_iter()
        .filter(|matched| matched.score >= min_score)
    {
        let pair = (matched.a.min(matched.b), matched.a.max(matched.b));
        if let Some(index) = pairs
            .get(&pair)
            .filter(|_| grouping::is_duplicate(&matched))
        {
            exact[*index].evidence.extend(
                matched
                    .evidence
                    .drain(..)
                    .filter(|evidence| evidence.signal == "quality"),
            );
        } else {
            exact.push(matched);
        }
    }
}

fn kind_order(kind: Kind) -> u8 {
    match kind {
        Kind::Image => 0,
        Kind::Video => 1,
        Kind::Text => 2,
        Kind::Other => 3,
    }
}

fn relation_order(relation: eidolon_core::Relation) -> u8 {
    use eidolon_core::Relation;
    match relation {
        Relation::Identical => 0,
        Relation::SameContent => 1,
        Relation::Reencoded => 2,
        Relation::Edited => 3,
        Relation::CropOf => 4,
        Relation::ClipOf => 5,
        Relation::ExcerptOf => 6,
        Relation::Similar => 7,
        Relation::Variant => 8,
        Relation::FrameOf => 9,
    }
}

struct Progress(ProgressBar);

impl Progress {
    fn new() -> anyhow::Result<Self> {
        let bar = ProgressBar::new_spinner();
        if std::io::stderr().is_terminal() {
            bar.set_style(ProgressStyle::with_template("{spinner} {wide_msg}")?);
            bar.enable_steady_tick(Duration::from_millis(100));
        } else {
            bar.set_draw_target(ProgressDrawTarget::hidden());
        }
        Ok(Self(bar))
    }

    fn stage(&self, message: &'static str) {
        self.0.set_message(message);
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.0.finish_and_clear();
    }
}

#[cfg(test)]
mod tests {
    use eidolon_core::{Evidence, Relation};

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Model {
        whole: AtomicUsize,
        center: AtomicUsize,
    }
    impl Embedder for Model {
        fn model_id(&self) -> &'static str {
            "image-only-source-center-contract"
        }
        fn embed(&self, _: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
            self.whole.fetch_add(1, Ordering::Relaxed);
            Ok(vec![1.0, 0.0])
        }
        fn embed_frame(&self, _: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
            self.center.fetch_add(1, Ordering::Relaxed);
            Ok(vec![0.0, 1.0])
        }
    }

    #[test]
    fn image_source_bank_option_controls_inference_independently_of_thumbnails()
    -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("source.png");
        image::RgbImage::from_fn(96, 64, |x, y| {
            image::Rgb([(x * 2) as u8, (y * 3) as u8, 71])
        })
        .save(&path)?;
        let metadata = std::fs::metadata(&path)?;
        let file = eidolon_core::FileEntry {
            id: FileId(0),
            path,
            size: metadata.len(),
            mtime_ns: i128::try_from(
                metadata
                    .modified()?
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos(),
            )?,
            kind: Kind::Image,
        };
        for (thumbnails, disabled, mode) in [
            (false, false, None),
            (true, false, Some("on")),
            (false, false, Some("off")),
            (true, false, Some("off")),
            (false, true, Some("on")),
        ] {
            let mut command = vec!["eidolon", "/tmp/source.png"];
            if let Some(mode) = mode {
                command.extend(["--image-source-bank", mode]);
            }
            if disabled {
                command.push("--no-embeddings");
            }
            let args = args::Args::try_parse_from(command)?;
            let image_source_bank = args.image_source_bank.enabled();
            let cache = FeatureCache::in_memory()?;
            let model = Model::default();
            let config = Config {
                timings: false,
                remote_overlay: true,
                min_score: 0.75,
                model_dir: temp.path().to_path_buf(),
                no_embeddings: disabled,
                no_keypoints: true,
                jobs: 1,
                fast: false,
                video_fps: 2.0,
            };
            let ctx = Ctx {
                config: &config,
                cache: &cache,
                embedder: Some(&model),
                refiner: None,
            };
            let mut files = vec![file.clone()];
            let mut skipped = Vec::new();
            let progress = Progress::new()?;
            assert_eq!(
                eidolon_image::uncached_embedding_work(&files, &ctx, image_source_bank)?,
                if disabled {
                    0
                } else {
                    1 + usize::from(image_source_bank)
                },
                "disabled source-bank work must not influence backend planning"
            );
            let settings = serde_json::to_value(settings::Settings::new(&args, None, false, 1))?;
            assert_eq!(settings["image_source_bank"], image_source_bank);
            for _ in 0..2 {
                assert!(
                    find_near(
                        &mut files,
                        &ctx,
                        &progress,
                        thumbnails,
                        image_source_bank,
                        &mut skipped
                    )?
                    .is_empty()
                );
                assert!(skipped.is_empty());
                assert_eq!(model.whole.load(Ordering::Relaxed), usize::from(!disabled));
                assert_eq!(
                    model.center.load(Ordering::Relaxed),
                    usize::from(!disabled && image_source_bank),
                    "source-center inference follows its option, not thumbnail matching"
                );
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn backend_preparation_never_claims_success_without_a_model() -> anyhow::Result<()> {
        let directory = tempfile::Builder::new()
            .prefix("eidolon-cli-model-preparation-")
            .tempdir_in("/tmp")?;
        let mut output = Vec::new();
        let result = prepare_auto_backend(directory.path(), &mut output);
        assert!(result.is_err(), "a missing model cannot prepare a backend");
        let output = String::from_utf8(output)?;
        assert!(!output.contains("Prepared automatic embedding backend:"));
        assert!(std::fs::read_dir(directory.path())?.next().is_none());
        Ok(())
    }

    #[test]
    fn exact_merge_keeps_variant_matches_instead_of_suppressing_them() {
        let mut exact = vec![Match {
            a: FileId(0),
            b: FileId(1),
            relation: Relation::Identical,
            score: 1.0,
            evidence: vec![Evidence {
                signal: "blake3",
                value: 1.0,
                note: "full-file hashes agree".into(),
            }],
        }];
        let variant = Match {
            a: FileId(0),
            b: FileId(1),
            relation: Relation::Variant,
            score: 0.9,
            evidence: vec![Evidence {
                signal: "composition",
                value: 0.9,
                note: "another version of the same design".into(),
            }],
        };
        merge_near(&mut exact, vec![variant], 0.75);
        assert_eq!(exact.len(), 2);
        assert_eq!(exact[0].relation, Relation::Identical);
        assert_eq!(exact[1].relation, Relation::Variant);
    }

    #[test]
    fn exact_merge_preserves_unverified_candidates_without_copying_quality() {
        let identical = Match {
            a: FileId(0),
            b: FileId(1),
            relation: Relation::Identical,
            score: 1.0,
            evidence: vec![Evidence {
                signal: "blake3",
                value: 1.0,
                note: "full-file hashes agree".into(),
            }],
        };
        let candidate = Match {
            relation: Relation::Reencoded,
            evidence: vec![
                Evidence {
                    signal: "unverified",
                    value: 1.0,
                    note: "pixel-level confirmation skipped".into(),
                },
                Evidence {
                    signal: "quality",
                    value: 0.0,
                    note: "file=0 pixels=1000000000".into(),
                },
            ],
            ..identical.clone()
        };
        let mut matches = vec![identical];
        merge_near(&mut matches, vec![candidate], 0.75);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].evidence.len(), 1);
        assert_eq!(matches[0].evidence[0].signal, "blake3");
        assert!(grouping::is_unverified(&matches[1]));
    }
}
