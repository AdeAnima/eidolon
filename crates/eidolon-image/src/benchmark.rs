//! Explicit, offline benchmark of a generated image corpus and its verification stages.

use std::path::PathBuf;
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{Context as _, ensure};
use eidolon_core::{Config, Ctx, Embedder, FileEntry, FileId, Kind, cache::FeatureCache};
use rayon::prelude::*;

#[test]
#[ignore = "requires an existing generated corpus and the fetched model; prints stage timings"]
fn profiles_generated_image_corpus() -> anyhow::Result<()> {
    let input = PathBuf::from(
        std::env::var_os("EIDOLON_IMAGE_BENCH_INPUT")
            .context("set EIDOLON_IMAGE_BENCH_INPUT to the generated media/image directory")?,
    );
    let mut paths = std::fs::read_dir(&input)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    let files = paths
        .into_iter()
        .enumerate()
        .filter(|(_, path)| path.is_file())
        .map(|(index, path)| {
            let metadata = path.metadata()?;
            Ok(FileEntry {
                id: FileId(index),
                path,
                size: metadata.len(),
                mtime_ns: i128::try_from(
                    metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos(),
                )?,
                kind: Kind::Image,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    ensure!(files.len() > 1, "benchmark needs at least two images");
    let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
        .join("Library/Caches/eidolon/models");
    let started = Instant::now();
    let model = crate::embed::OnnxEmbedder::load(&model_dir)?.context("fetch the model first")?;
    eprintln!("model load: {:.3}s", started.elapsed().as_secs_f64());
    let cache = std::env::var_os("EIDOLON_IMAGE_BENCH_CACHE")
        .map(PathBuf::from)
        .map_or_else(FeatureCache::in_memory, |path| FeatureCache::open(&path))?;
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.75,
        model_dir,
        no_embeddings: false,
        no_keypoints: false,
        jobs: std::env::var("EIDOLON_IMAGE_BENCH_JOBS")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from)),
        fast: false,
        video_fps: 2.0,
    };
    let ctx = Ctx {
        refiner: None,
        config: &config,
        cache: &cache,
        embedder: Some(&model as &dyn Embedder),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.jobs)
        .build()?;
    pool.install(|| -> anyhow::Result<()> {
        let started = Instant::now();
        let features = files
            .par_iter()
            .map(|file| crate::features::extract(file, &ctx))
            .collect::<anyhow::Result<Vec<_>>>()?;
        eprintln!(
            "{} images: extraction {:.3}s",
            files.len(),
            started.elapsed().as_secs_f64()
        );
        let started = Instant::now();
        let mut candidates = crate::detect::candidates(&features, &[]);
        eprintln!(
            "{} candidates: generation {:.3}s",
            candidates.len(),
            started.elapsed().as_secs_f64()
        );
        if let Some(path) = std::env::var_os("EIDOLON_IMAGE_BENCH_CANDIDATES") {
            serde_json::to_writer(
                std::io::BufWriter::new(std::fs::File::create_new(path)?),
                &candidates,
            )?;
        }
        if std::env::var_os("EIDOLON_IMAGE_BENCH_RETRIEVAL_ONLY").is_some() {
            return Ok(());
        }
        let started = Instant::now();
        let (files, features) = crate::features::prepare(&files, features, &mut candidates, &ctx)?;
        let matches = crate::detect::verify(&files, &features, &candidates, &ctx)?;
        eprintln!(
            "{} verified matches: verification {:.3}s",
            matches.len(),
            started.elapsed().as_secs_f64()
        );
        Ok(())
    })
}
