//! Remote transfer stops at independently proven native-frame containment.

mod support;

use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

use anyhow::{Context as _, ensure};
use eidolon_core::cache::FeatureCache;
use eidolon_core::{
    Config, Ctx, Embedder, FileEntry, FileId, FrameRefiner, Kind, RefinedWindow, Relation,
};

struct CandidateEmbedder;
impl Embedder for CandidateEmbedder {
    fn model_id(&self) -> &'static str {
        "remote-prefix-nomination/v1"
    }
    fn embed(&self, _: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        Ok(vec![1.0, 0.0])
    }
}

struct PrefixProvider {
    window: RefinedWindow,
    ends: Mutex<Vec<f64>>,
}
impl FrameRefiner for PrefixProvider {
    fn refine(
        &self,
        _: &FileEntry,
        center: f64,
        radius: f64,
    ) -> anyhow::Result<Option<RefinedWindow>> {
        self.ends.lock().expect("requests").push(center + radius);
        ensure!(
            center + radius <= 1.5 + 1e-7,
            "request fetched beyond the proven thumbnail time"
        );
        Ok(Some(self.window.clone()))
    }
}

fn ffmpeg(arguments: &[&str], output: &Path) -> anyhow::Result<()> {
    let result = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-threads", "1"])
        .args(arguments)
        .arg(output)
        .output()?;
    ensure!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

fn entry(path: &Path, id: usize, kind: Kind) -> anyhow::Result<FileEntry> {
    Ok(FileEntry {
        id: FileId(id),
        path: path.canonicalize()?,
        size: path.metadata()?.len(),
        mtime_ns: 0,
        kind,
    })
}

fn settings() -> Config {
    Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.75,
        model_dir: std::path::PathBuf::new(),
        no_embeddings: false,
        no_keypoints: true,
        jobs: 1,
        fast: false,
        video_fps: 2.0,
    }
}

#[test]
fn remote_frame_proof_stops_before_fetching_the_rest_of_the_candidate_window() -> anyhow::Result<()>
{
    let directory = tempfile::tempdir()?;
    let native = directory.path().join("native.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=160x90:rate=8:seed=713841:ratio=0.45:death_color=0x102030:life_color=0xe8c070,scale=640:360:flags=neighbor",
            "-t",
            "8",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-g",
            "64",
            "-keyint_min",
            "64",
            "-sc_threshold",
            "0",
        ],
        &native,
    )?;
    let native_path = native.to_str().context("native path")?;
    let proxy = directory.path().join("sparse.mp4");
    ffmpeg(
        &[
            "-i",
            native_path,
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
        ],
        &proxy,
    )?;
    std::fs::write(
        directory.path().join("sparse.mp4.remote.json"),
        r#"{"version":1,"duration_seconds":8.0,"source_fps":8.0,"sample_pts_seconds":[0.0]}"#,
    )?;
    let image = directory.path().join("thumbnail.png");
    ffmpeg(
        &[
            "-ss",
            "1.25",
            "-i",
            native_path,
            "-vf",
            "scale=320:180",
            "-frames:v",
            "1",
        ],
        &image,
    )?;
    let images = [entry(&image, 1, Kind::Image)?];
    let videos = [entry(&proxy, 0, Kind::Video)?];
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let mut ctx = Ctx {
        config: &config,
        cache: &cache,
        embedder: Some(&CandidateEmbedder),
        refiner: None,
    };
    assert!(
        eidolon_video::find_frames(&images, &videos, &ctx)?.is_empty(),
        "retained frame cannot prove the target"
    );
    let provider = PrefixProvider {
        window: RefinedWindow {
            path: native,
            source_offset_s: 0.0,
        },
        ends: Mutex::new(Vec::new()),
    };
    ctx.refiner = Some(&provider);
    let matches = eidolon_video::find_frames(&images, &videos, &ctx)?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::FrameOf);
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|evidence| evidence.signal == "frame_refinement")
    );
    assert_eq!(
        *provider.ends.lock().expect("requests"),
        vec![0.5, 1.0, 1.5],
        "proof stops before any later prefix"
    );
    Ok(())
}

struct RecordingFetcher {
    fetcher: eidolon_fetch::Fetcher,
    outcomes: Mutex<Vec<(f64, bool, u64)>>,
}

impl FrameRefiner for RecordingFetcher {
    fn refine(
        &self,
        video: &FileEntry,
        center: f64,
        radius: f64,
    ) -> anyhow::Result<Option<RefinedWindow>> {
        // Forward unchanged to production budget and range-cache semantics.
        let result = self.fetcher.refine(video, center, radius)?;
        self.outcomes.lock().expect("requests").push((
            center - radius,
            result.is_some(),
            self.fetcher.bytes_fetched(),
        ));
        Ok(result)
    }
}

struct IndependentSeeds(Vec<Vec<u8>>);

impl Embedder for IndependentSeeds {
    fn model_id(&self) -> &'static str {
        "independent-remote-window-candidates/v1"
    }

    fn embed(&self, image: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        // Only nomination is controlled; all proof uses independently decoded pixels.
        let selected = self.0.iter().any(|pixels| pixels == image.as_raw());
        Ok(vec![if selected { 1.0 } else { -1.0 }, 0.0])
    }
}

fn recovery_media(directory: &Path) -> anyhow::Result<(std::path::PathBuf, std::path::PathBuf)> {
    let native = directory.join("native.mp4");
    // Sixteen keyframes over 46 seconds give a 2.875-second sampling step.
    // Adding the 0.125-second frame period starts the second window at GOP 9s.
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=40x24:rate=8:seed=713841:ratio=0.45:death_color=0x102030:life_color=0xe8c070,scale=320:180:flags=neighbor,noise=alls=100:allf=t:all_seed=713841:enable='lt(t,3)'",
            "-t",
            "46",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-bf",
            "0",
            "-g",
            "24",
            "-keyint_min",
            "24",
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
        ],
        &native,
    )?;
    let image = directory.join("ordinary.png");
    ffmpeg(
        &[
            "-ss",
            "9.5",
            "-i",
            native.to_str().context("native path")?,
            "-frames:v",
            "1",
        ],
        &image,
    )?;
    Ok((native, image))
}

fn independent_seeds(proxy: &Path, image: &Path) -> anyhow::Result<IndependentSeeds> {
    // Four separated seeds prevent ordinary uncertainty windows from merging.
    let selected = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-threads", "1", "-i"])
        .arg(proxy)
        .args([
            "-vf",
            "select='eq(n,0)+eq(n,4)+eq(n,8)+eq(n,12)',scale=96:96:flags=area",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .output()?;
    ensure!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let mut selected: Vec<_> = selected
        .stdout
        .as_chunks::<{ 96 * 96 * 3 }>()
        .0
        .iter()
        .map(|pixels| pixels.to_vec())
        .collect();
    assert_eq!(selected.len(), 4);
    selected.push(
        image::open(image)?
            .resize_exact(96, 96, image::imageops::FilterType::Triangle)
            .to_rgb8()
            .into_raw(),
    );
    Ok(IndependentSeeds(selected))
}

fn fitting_allowance(cache_dir: &Path, url: &str) -> anyhow::Result<u64> {
    // Measure the cheap prefixes in a separate cache so calibration cannot make
    // the actual regression's native requests free cache hits.
    let client = eidolon_fetch::Fetcher::with_options(eidolon_fetch::FetchOptions {
        cache_dir: cache_dir.to_owned(),
        max_fetch_fraction: 1.0,
        ..eidolon_fetch::FetchOptions::default()
    })?;
    let resolved = client.resolve(url)?;
    let video = entry(&resolved.artifacts[0].local_path, 0, Kind::Video)?;
    let before = client.bytes_fetched();
    for center in [9.25, 9.75] {
        ensure!(client.refine(&video, center, 0.25)?.is_some());
    }
    let fitting_bytes = client.bytes_fetched() - before;
    ensure!(fitting_bytes > 0);
    // Leave 1/64 headroom for fractional-cap rounding, not an x264-specific size.
    Ok(fitting_bytes + fitting_bytes.div_ceil(64))
}

#[test]
fn ordinary_remote_query_continues_after_refused_window_to_fitting_match() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let (native, image) = recovery_media(directory.path())?;
    let source = std::fs::read(&native)?;
    let source_size = source.len() as u64;
    let server = support::VideoServer::new(source)?;
    let url = server.url();
    let cache_dir = directory.path().join("remote");
    let warm = eidolon_fetch::Fetcher::with_options(eidolon_fetch::FetchOptions {
        cache_dir: cache_dir.clone(),
        ..eidolon_fetch::FetchOptions::default()
    })?
    .resolve(&url)?;
    assert_eq!(warm.artifacts[0].mode, eidolon_fetch::FetchMode::Sparse);
    let videos = [entry(&warm.artifacts[0].local_path, 0, Kind::Video)?];
    let images = [entry(&image, 1, Kind::Image)?];
    let embedder = independent_seeds(&videos[0].path, &image)?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let mut ctx = Ctx {
        config: &config,
        cache: &cache,
        embedder: Some(&embedder),
        refiner: None,
    };
    assert!(
        eidolon_video::find_frames(&images, &videos, &ctx)?.is_empty(),
        "retained keyframes must not prove the ordinary query"
    );
    let allowance = fitting_allowance(&directory.path().join("calibration"), &url)?;
    let provider = RecordingFetcher {
        fetcher: eidolon_fetch::Fetcher::with_options(eidolon_fetch::FetchOptions {
            cache_dir,
            max_fetch_fraction: allowance as f64 / source_size as f64,
            ..eidolon_fetch::FetchOptions::default()
        })?,
        outcomes: Mutex::new(Vec::new()),
    };
    assert_eq!(provider.fetcher.resolve(&url)?.bytes_fetched, 0);
    let before = server.bytes();
    ctx.refiner = Some(&provider);
    let matches = eidolon_video::find_frames(&images, &videos, &ctx)?;
    let outcomes = provider.outcomes.lock().expect("requests");
    assert_eq!(outcomes.first(), Some(&(0.0, false, 0)), "{outcomes:?}");
    assert_eq!(matches.len(), 1, "{matches:#?}; requests={outcomes:?}");
    assert!(
        outcomes
            .iter()
            .any(|(start, served, _)| *start >= 9.0 && *served),
        "a later independent window must fit: {outcomes:?}"
    );
    assert_eq!(matches[0].a, images[0].id);
    assert_eq!(matches[0].b, videos[0].id);
    assert_eq!(matches[0].relation, Relation::FrameOf);
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|e| e.signal == "frame_refinement")
    );
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|e| { e.signal == "frame_timestamp" && (e.value - 9.5).abs() < 0.13 })
    );
    let transferred = server.bytes() - before;
    assert!(transferred > 0 && transferred <= allowance);
    assert_eq!(provider.fetcher.bytes_fetched(), transferred);
    assert_eq!(
        provider.fetcher.refinement_bytes(&videos[0].path)?,
        transferred
    );
    assert!(provider.fetcher.budget_exhausted(&videos[0].path)?);
    assert!(provider.fetcher.resolve(&url)?.artifacts[0].budget_exhausted);
    eprintln!(
        "MEASURE ordinary continuation: requests={outcomes:?}; bytes={transferred}/{allowance}; matches={}",
        matches.len()
    );
    Ok(())
}

fn threshold_media(
    directory: &Path,
) -> anyhow::Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
    let base = image::RgbImage::from_fn(320, 180, |x, y| {
        image::Rgb([
            (40 + (x / 11 * 31 + y / 13 * 17) % 170) as u8,
            (40 + (x / 17 * 23 + y / 9 * 29) % 170) as u8,
            (40 + (x / 7 * 13 + y / 19 * 37) % 170) as u8,
        ])
    });
    let mut videos = Vec::new();
    for (name, perturb) in [("retained", true), ("sufficient", false)] {
        let mut pixels = base.clone();
        if perturb {
            for (x, y, pixel) in pixels.enumerate_pixels_mut() {
                let delta = if (x / 8 + y / 8) % 2 == 0 { 18 } else { -18 };
                for channel in &mut pixel.0 {
                    *channel = (i16::from(*channel) + delta) as u8;
                }
            }
        }
        let image = directory.join(format!("{name}.png"));
        pixels.save(&image)?;
        let video = directory.join(format!("{name}.mp4"));
        ffmpeg(
            &[
                "-loop",
                "1",
                "-i",
                image.to_str().context("image path")?,
                "-t",
                "1",
                "-r",
                "8",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ],
            &video,
        )?;
        videos.push(video);
    }
    let proxy = directory.join("proxy.mp4");
    ffmpeg(
        &[
            "-i",
            videos[0].to_str().context("retained path")?,
            "-frames:v",
            "1",
            "-c:v",
            "copy",
        ],
        &proxy,
    )?;
    std::fs::write(
        proxy.with_extension("mp4.remote.json"),
        r#"{"version":1,"duration_seconds":8.0,"source_fps":8.0,"sample_pts_seconds":[0.0]}"#,
    )?;
    let query = directory.join("query.png");
    ffmpeg(
        &[
            "-i",
            videos[1].to_str().context("native path")?,
            "-frames:v",
            "1",
        ],
        &query,
    )?;
    Ok((proxy, query, videos.remove(1)))
}

#[test]
fn below_threshold_retained_match_still_refines_to_sufficient_native_frame() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let (proxy, query, native) = threshold_media(directory.path())?;
    let images = [entry(&query, 1, Kind::Image)?];
    let videos = [entry(&proxy, 0, Kind::Video)?];
    let mut config = settings();
    let cache = FeatureCache::in_memory()?;
    let retained = eidolon_video::find_frames(
        &images,
        &videos,
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: Some(&CandidateEmbedder),
            refiner: None,
        },
    )?;
    assert_eq!(retained.len(), 1, "{retained:#?}");
    assert!(
        retained[0].score > 0.75 && retained[0].score < 1.0,
        "{retained:#?}"
    );
    config.min_score = retained[0].score.midpoint(1.0);
    let provider = PrefixProvider {
        window: RefinedWindow {
            path: native,
            source_offset_s: 0.0,
        },
        ends: Mutex::new(Vec::new()),
    };
    let mut ctx = Ctx {
        config: &config,
        cache: &cache,
        embedder: Some(&CandidateEmbedder),
        refiner: None,
    };
    assert!(eidolon_video::find_frames(&images, &videos, &ctx)?.is_empty());
    ctx.refiner = Some(&provider);
    let matches = eidolon_video::find_frames(&images, &videos, &ctx)?;
    assert_eq!(
        matches.len(),
        1,
        "{matches:#?}; retained={retained:#?}; min_score={}",
        config.min_score
    );
    assert!(matches[0].score >= config.min_score);
    assert_eq!(matches[0].relation, Relation::FrameOf);
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|e| e.signal == "frame_refinement")
    );
    assert!(!provider.ends.lock().expect("requests").is_empty());
    eprintln!(
        "MEASURE threshold continuation: retained_score={}; min_score={}; native_score={}",
        retained[0].score, config.min_score, matches[0].score
    );
    Ok(())
}
