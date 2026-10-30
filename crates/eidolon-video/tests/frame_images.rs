//! Image-to-video containment checked through the public detector on real media.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::UNIX_EPOCH;

use anyhow::{Context as _, ensure};
use eidolon_core::cache::FeatureCache;
use eidolon_core::{
    Config, Ctx, Embedder, FileEntry, FileId, FrameRefiner, Kind, RefinedWindow, Relation,
};

struct ConstantEmbedder;

impl Embedder for ConstantEmbedder {
    fn model_id(&self) -> &'static str {
        "uninformative-frame-candidate/v1"
    }
    fn embed(&self, _: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        Ok(vec![1.0, 0.0])
    }
}

struct CountingEmbedder(AtomicUsize);

impl Embedder for CountingEmbedder {
    fn model_id(&self) -> &'static str {
        "image-policy-must-not-serve-frames/v1"
    }
    fn frame_model_id(&self) -> &'static str {
        "counted-frame-candidate/v1"
    }
    fn frame_cache_model_ids(&self) -> [Option<&str>; 2] {
        [Some(self.frame_model_id()), None]
    }
    fn embed(&self, _: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("image preprocessing must not serve frame embeddings")
    }
    fn embed_frame(&self, _: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(vec![1.0, 0.0])
    }
}

struct WindowRefiner {
    window: RefinedWindow,
    calls: AtomicUsize,
}

impl FrameRefiner for WindowRefiner {
    fn refine(&self, _: &FileEntry, _: f64, _: f64) -> anyhow::Result<Option<RefinedWindow>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(Some(self.window.clone()))
    }
}

struct SelectedFrameEmbedder {
    responses: Vec<(Vec<u8>, Vec<f32>)>,
    fallback: Vec<f32>,
}

impl Embedder for SelectedFrameEmbedder {
    fn model_id(&self) -> &'static str {
        "selected-native-frame-candidates/v1"
    }

    fn embed(&self, image: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        Ok(self
            .responses
            .iter()
            .find(|(pixels, _)| pixels == image.as_raw())
            .map_or_else(|| self.fallback.clone(), |(_, vector)| vector.clone()))
    }
}

fn normalized(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn raw_frame(video: &Path, time: &str) -> anyhow::Result<Vec<u8>> {
    let output = Command::new("ffmpeg")
        .args([
            "-v", "error", "-nostdin", "-threads", "1", "-ss", time, "-i",
        ])
        .arg(video)
        .args([
            "-frames:v",
            "1",
            "-vf",
            "fps=fps=2:start_time=0,scale=96:96:flags=area",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .output()?;
    ensure!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output.stdout)
}

fn ffmpeg(arguments: &[&str], output: &Path) -> anyhow::Result<()> {
    let result = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-threads",
            "1",
            "-filter_threads",
            "1",
        ])
        .args(arguments)
        .arg(output)
        .output()?;
    ensure!(
        result.status.success(),
        "FFmpeg: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

fn entry(path: &Path, id: usize, kind: Kind) -> anyhow::Result<FileEntry> {
    let metadata = path.metadata()?;
    Ok(FileEntry {
        id: FileId(id),
        path: path.to_path_buf(),
        size: metadata.len(),
        mtime_ns: i128::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
        kind,
    })
}

fn config() -> Config {
    Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.75,
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: true,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    }
}

fn source(directory: &Path) -> anyhow::Result<PathBuf> {
    let path = directory.join("source.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=12:duration=8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ],
        &path,
    )?;
    Ok(path)
}

#[test]
fn finds_scaled_frame_between_samples_with_direction_and_timestamp() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = source(temporary.path())?;
    let image = temporary.path().join("thumbnail.png");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
            "-vf",
            "scale=200:112",
        ],
        &image,
    )?;
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::FrameOf, "{matches:#?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(9), FileId(2)));
    let timestamp = matches[0]
        .evidence
        .iter()
        .find(|item| item.signal == "frame_timestamp")
        .context("source timestamp evidence")?;
    assert!(
        (timestamp.value - 3.25).abs() <= 1.0 / 12.0 + 0.025,
        "{matches:#?}"
    );
    Ok(())
}

#[test]
fn finds_cropped_thumbnail_with_local_overlay() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = source(temporary.path())?;
    let image = temporary.path().join("cropped-poster.png");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
            "-vf",
            "crop=iw*0.72:ih*0.8:iw*0.14:ih*0.1,scale=240:150,drawbox=x=iw*0.7:y=ih*0.8:w=iw*0.3:h=ih*0.2:color=white:t=fill",
        ],
        &image,
    )?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::FrameOf, "{matches:#?}");
    Ok(())
}

#[test]
fn finds_video_embedded_in_a_poster_with_branding() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = source(temporary.path())?;
    let image = temporary.path().join("poster.png");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
            "-vf",
            "scale=240:135,pad=300:210:30:35:color=0x404040,drawbox=x=220:y=145:w=40:h=20:color=white:t=fill",
        ],
        &image,
    )?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::FrameOf, "{matches:#?}");
    Ok(())
}

#[test]
fn fast_mode_stops_at_the_first_independently_proven_sample() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let image = temporary.path().join("texture.png");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "nullsrc=size=320x180:rate=8,geq=r='128+70*sin(X/15+Y/28)':g='128+70*cos(X/31-Y/17)':b='128+70*sin(X/23+Y/19)'",
            "-frames:v",
            "1",
        ],
        &image,
    )?;
    let video = temporary.path().join("samples.mp4");
    ffmpeg(
        &[
            "-loop",
            "1",
            "-i",
            image.to_str().context("UTF-8 image path")?,
            "-t",
            "4",
            "-r",
            "8",
            "-vf",
            "eq=brightness=0.04:enable='lt(t,0.5)',drawbox=x=0:y=0:w=24:h=18:color=white:t=fill:enable='gte(t,0.5)'",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
        ],
        &video,
    )?;
    let second_image = temporary.path().join("second-texture.png");
    std::fs::copy(&image, &second_image)?;
    let video_file = entry(&video, 2, Kind::Video)?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.fast = true;
    settings.no_embeddings = false;
    let matches = eidolon_video::find_frames(
        &[
            entry(&image, 9, Kind::Image)?,
            entry(&second_image, 10, Kind::Image)?,
        ],
        std::slice::from_ref(&video_file),
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert_eq!(matches.len(), 2, "{matches:#?}");
    assert!(
        matches
            .iter()
            .all(|found| found.evidence.iter().any(|item| item.signal == "thumbnail")),
        "{matches:#?}"
    );
    let timestamp = matches[0]
        .evidence
        .iter()
        .find(|item| item.signal == "frame_timestamp")
        .context("frame timestamp")?;
    assert!(
        timestamp.value < 0.5,
        "the first proven sample must stop verification before later, higher-quality samples: {matches:#?}"
    );
    assert!(
        cache
            .get(&video_file, "video.native-clock/v1;cap=1200")?
            .is_none(),
        "independently proven queries must not trigger native timeline decoding"
    );
    Ok(())
}

#[test]
fn rejects_unrelated_pixels_despite_identical_embeddings() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = temporary.path().join("life.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=96x54:rate=8:random_seed=41935:ratio=0.45:death_color=0x102030:life_color=0xF8C070,scale=320:180",
            "-t",
            "8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
        ],
        &video,
    )?;
    let image = temporary.path().join("different-life.png");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=96x54:rate=8:random_seed=28517:ratio=0.45:death_color=0x102030:life_color=0xF8C070,scale=200:112",
            "-frames:v",
            "1",
        ],
        &image,
    )?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert!(matches.is_empty(), "{matches:#?}");
    Ok(())
}

#[test]
fn sparse_proxy_reports_an_actual_source_sample_timestamp() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = source(temporary.path())?;
    let image = temporary.path().join("keyframe.png");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
        ],
        &image,
    )?;
    let proxy = temporary.path().join("proxy.mp4");
    ffmpeg(
        &[
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-vf",
            "select='eq(n,0)+eq(n,39)'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-g",
            "1",
            "-bf",
            "0",
        ],
        &proxy,
    )?;
    let mut sidecar = proxy.as_os_str().to_os_string();
    sidecar.push(".remote.json");
    std::fs::write(
        sidecar,
        r#"{"version":1,"duration_seconds":8.0,"source_fps":12.0,"sample_pts_seconds":[0.0,3.25]}"#,
    )?;
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&proxy, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    let timestamp = matches[0]
        .evidence
        .iter()
        .find(|item| item.signal == "frame_timestamp")
        .context("source timestamp evidence")?;
    assert!((timestamp.value - 3.25).abs() < 0.001, "{matches:#?}");
    assert!(timestamp.note.contains("sparse_proxy=true"), "{matches:#?}");
    Ok(())
}

#[test]
fn reuses_cached_features_without_reusing_input_ids() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = source(temporary.path())?;
    let image = temporary.path().join("thumbnail.png");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
        ],
        &image,
    )?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let model = CountingEmbedder(AtomicUsize::new(0));
    let ctx = Ctx {
        refiner: None,
        config: &settings,
        cache: &cache,
        embedder: Some(&model),
    };
    let first = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &ctx,
    )?;
    assert_eq!(first.len(), 1, "{first:#?}");
    let calls = model.0.load(Ordering::Relaxed);
    assert!(calls > 0, "retrieval should exercise the injected embedder");
    let cached = eidolon_video::find_frames(
        &[entry(&image, 19, Kind::Image)?],
        &[entry(&video, 12, Kind::Video)?],
        &ctx,
    )?;
    assert_eq!(
        model.0.load(Ordering::Relaxed),
        calls,
        "cached extraction must not infer again"
    );
    assert_eq!(cached.len(), 1, "{cached:#?}");
    assert_eq!((cached[0].a, cached[0].b), (FileId(19), FileId(12)));
    settings.no_embeddings = true;
    let disabled = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&model),
        },
    )?;
    assert_eq!(disabled.len(), 1, "{disabled:#?}");
    assert_eq!(
        model.0.load(Ordering::Relaxed),
        calls,
        "disabled embeddings must not infer"
    );
    Ok(())
}

#[test]
fn does_not_promote_recolored_designs_to_frame_containment() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = temporary.path().join("life.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=160x90:rate=8:seed=319813:ratio=0.45:death_color=0x102030:life_color=0xe8c070,scale=960:540:flags=neighbor",
            "-t",
            "8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
        ],
        &video,
    )?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    for (index, filter) in [
        "hue=110",
        "hue=110,crop=iw*0.72:ih*0.8:iw*0.14:ih*0.1,scale=240:150",
    ]
    .into_iter()
    .enumerate()
    {
        let image = temporary
            .path()
            .join(format!("recolored-design-{index}.png"));
        ffmpeg(
            &[
                "-ss",
                "3.25",
                "-i",
                video.to_str().context("UTF-8 video path")?,
                "-vf",
                filter,
                "-frames:v",
                "1",
            ],
            &image,
        )?;
        for fast in [false, true] {
            settings.fast = fast;
            let matches = eidolon_video::find_frames(
                &[entry(&image, 9, Kind::Image)?],
                &[entry(&video, 2, Kind::Video)?],
                &Ctx {
                    refiner: None,
                    config: &settings,
                    cache: &cache,
                    embedder: Some(&ConstantEmbedder),
                },
            )?;
            assert!(matches.is_empty(), "{filter}, fast={fast}: {matches:#?}");
        }
    }
    Ok(())
}

#[test]
fn finds_square_cropped_fine_detail_with_title_bar() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = temporary.path().join("fine-detail.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=160x90:rate=8:seed=491831:ratio=0.45:death_color=0x102030:life_color=0xe8c070,scale=960:540:flags=neighbor",
            "-t",
            "18",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
        ],
        &video,
    )?;
    let image = temporary.path().join("square.jpg");
    ffmpeg(
        &[
            "-ss",
            "10.5",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-vf",
            "crop=ih:ih,scale=320:320,drawbox=x=0:y=ih*0.88:w=iw:h=ih*0.12:color=0x101018:t=fill",
            "-frames:v",
            "1",
            "-q:v",
            "5",
        ],
        &image,
    )?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::FrameOf);
    let timestamp = matches[0]
        .evidence
        .iter()
        .find(|item| item.signal == "frame_timestamp")
        .context("source timestamp evidence")?;
    assert!((timestamp.value - 10.5).abs() < 0.126, "{matches:#?}");
    Ok(())
}

fn native_life_source(directory: &Path) -> anyhow::Result<PathBuf> {
    let video = directory.join("native.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=160x90:rate=8:seed=713841:ratio=0.45:death_color=0x102030:life_color=0xe8c070,scale=960:540:flags=neighbor",
            "-t",
            "8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-g",
            "16",
            "-keyint_min",
            "16",
            "-sc_threshold",
            "0",
        ],
        &video,
    )?;
    Ok(video)
}

#[test]
fn refines_a_sparse_proxy_with_real_native_frame_pixels() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = native_life_source(temporary.path())?;
    let proxy = temporary.path().join("sparse.mp4");
    ffmpeg(
        &[
            "-i",
            video.to_str().context("UTF-8 source path")?,
            "-vf",
            "select=not(mod(n\\,16))",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-g",
            "1",
        ],
        &proxy,
    )?;
    std::fs::write(
        temporary.path().join("sparse.mp4.remote.json"),
        r#"{"version":1,"duration_seconds":8.0,"source_fps":8.0,"sample_pts_seconds":[0.0,2.0,4.0,6.0]}"#,
    )?;
    let image = temporary.path().join("off-keyframe.jpg");
    ffmpeg(
        &[
            "-ss",
            "1.25",
            "-i",
            video.to_str().context("UTF-8 source path")?,
            "-vf",
            "scale=320:180",
            "-frames:v",
            "1",
            "-q:v",
            "4",
        ],
        &image,
    )?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let files = [entry(&image, 9, Kind::Image)?];
    let videos = [entry(&proxy, 2, Kind::Video)?];
    let without = eidolon_video::find_frames(
        &files,
        &videos,
        &Ctx {
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
            refiner: None,
        },
    )?;
    assert!(without.is_empty(), "{without:#?}");
    let refiner = WindowRefiner {
        window: RefinedWindow {
            path: video,
            source_offset_s: 0.0,
        },
        calls: AtomicUsize::new(0),
    };
    let matches = eidolon_video::find_frames(
        &files,
        &videos,
        &Ctx {
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
            refiner: Some(&refiner),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert!(refiner.calls.load(Ordering::Relaxed) > 0);
    let timestamp = matches[0]
        .evidence
        .iter()
        .find(|item| item.signal == "frame_timestamp")
        .context("source timestamp")?;
    assert!((timestamp.value - 1.25).abs() < 0.126, "{matches:#?}");
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|item| item.signal == "frame_refinement"),
        "{matches:#?}"
    );
    Ok(())
}

#[test]
fn refines_a_best_timestamp_outside_the_colliding_frame_buckets() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = native_life_source(temporary.path())?;
    let image = temporary.path().join("square-overlay.jpg");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 source path")?,
            "-vf",
            "crop=ih:ih,scale=320:320,drawbox=x=0:y=ih*0.88:w=iw:h=ih*0.12:color=0x101018:t=fill",
            "-frames:v",
            "1",
            "-q:v",
            "4",
        ],
        &image,
    )?;
    let query = normalized(
        (0..384)
            .map(|index| f64::from(index).sin() as f32)
            .collect(),
    );
    let mut noise: Vec<_> = (0..384)
        .map(|index| (f64::from(index) * std::f64::consts::SQRT_2).cos() as f32)
        .collect();
    let projection: f32 = query.iter().zip(&noise).map(|(a, b)| a * b).sum();
    for (value, axis) in noise.iter_mut().zip(&query) {
        *value -= projection * axis;
    }
    let relevant = query
        .iter()
        .zip(normalized(noise))
        .map(|(axis, noise)| 0.8 * axis + 0.6 * noise)
        .collect();
    let query_pixels = eidolon_image::decode(&image)?
        .resize_exact(96, 96, image::imageops::FilterType::Triangle)
        .to_rgb8()
        .into_raw();
    // Only the misleading frame at 0s collides with the query's semantic bucket.
    // The better source neighborhood must be found by ranking the nominated video.
    let model = SelectedFrameEmbedder {
        responses: vec![
            (raw_frame(&video, "0")?, query.clone()),
            (raw_frame(&video, "3")?, relevant),
            (query_pixels, query.clone()),
        ],
        fallback: query.into_iter().map(|value| -value).collect(),
    };
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&model),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    let timestamp = matches[0]
        .evidence
        .iter()
        .find(|item| item.signal == "frame_timestamp")
        .context("source timestamp")?;
    assert!((timestamp.value - 3.25).abs() < 0.126, "{matches:#?}");
    Ok(())
}

#[test]
fn shares_native_pixels_across_queries_without_mixing_timestamps() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = source(temporary.path())?;
    let times = ["3.25", "5.25", "7.25"];
    let filters = [
        "scale=200:112",
        "crop=iw*0.72:ih*0.8:iw*0.14:ih*0.1,scale=240:150",
        "scale=200:112,pad=320:180:60:34:color=white",
    ];
    let images = times
        .iter()
        .enumerate()
        .map(|(index, time)| {
            let image = temporary.path().join(format!("thumbnail-{index}.png"));
            ffmpeg(
                &[
                    "-ss",
                    time,
                    "-i",
                    video.to_str().context("UTF-8 video path")?,
                    "-frames:v",
                    "1",
                    "-vf",
                    filters[index],
                ],
                &image,
            )?;
            entry(&image, index + 9, Kind::Image)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let matches = eidolon_video::find_frames(
        &images,
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert_eq!(matches.len(), times.len(), "{matches:#?}");
    for (index, found) in matches.iter().enumerate() {
        assert_eq!((found.a, found.b), (FileId(index + 9), FileId(2)));
        assert_eq!(found.relation, Relation::FrameOf);
        let timestamp = found
            .evidence
            .iter()
            .find(|item| item.signal == "frame_timestamp")
            .context("source timestamp evidence")?;
        assert!(
            (timestamp.value - times[index].parse::<f64>()?).abs() < 0.126,
            "{found:#?}"
        );
    }
    let mut reordered = images.clone();
    reordered.reverse();
    let repeated = eidolon_video::find_frames(
        &reordered,
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert_eq!(
        serde_json::to_value(&matches)?,
        serde_json::to_value(&repeated)?,
        "shared native projections must not depend on query order"
    );
    Ok(())
}

#[test]
fn rejects_partial_scene_agreement_in_a_different_version() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = source(temporary.path())?;
    let image = temporary.path().join("version.png");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
        ],
        &image,
    )?;
    let mut version = image::open(&image)?.to_rgb8();
    let width = version.width();
    for (x, _, pixel) in version.enumerate_pixels_mut() {
        if x >= width / 2 {
            pixel.0 = pixel.0.map(|channel| channel / 3);
        }
    }
    version.save(&image)?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert!(
        matches.is_empty(),
        "a shared half-scene does not prove full-frame containment: {matches:#?}"
    );
    settings.fast = true;
    let fast = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert!(
        fast.is_empty(),
        "fast mode must reject a different composition with only half the pixels shared: {fast:#?}"
    );
    Ok(())
}

#[test]
fn finds_high_frequency_crop_with_local_caption() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = temporary.path().join("fine-texture.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=s=160x90:rate=8:ratio=0.45:seed=314159:life_color=0xE8C070:death_color=0x102030,scale=960:540:flags=neighbor",
            "-t",
            "8",
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
    let image = temporary.path().join("captioned-crop.jpg");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
            "-vf",
            "crop=540:540:210:0,scale=320:320,drawbox=x=0:y=284:w=320:h=36:color=black:t=fill",
        ],
        &image,
    )?;
    let mut settings = config();
    settings.no_embeddings = false;
    let cache = FeatureCache::in_memory()?;
    for fast in [false, true] {
        settings.fast = fast;
        let matches = eidolon_video::find_frames(
            &[entry(&image, 9, Kind::Image)?],
            &[entry(&video, 2, Kind::Video)?],
            &Ctx {
                refiner: None,
                config: &settings,
                cache: &cache,
                embedder: Some(&ConstantEmbedder),
            },
        )?;
        assert_eq!(matches.len(), 1, "fast={fast}: {matches:#?}");
        assert_eq!(matches[0].relation, Relation::FrameOf);
        assert!(
            matches[0]
                .evidence
                .iter()
                .any(|item| item.signal == "frame_spatial"),
            "unresolved fine detail must retain native pixel verification: {matches:#?}"
        );
    }
    Ok(())
}

#[test]
fn fast_smooth_crop_has_low_resolution_pixel_evidence() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-frame-")
        .tempdir_in("/tmp")?;
    let video = temporary.path().join("smooth.mp4");
    // A cached thumbnail can prove smooth detail; lost high-frequency pixels need refinement.
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "nullsrc=size=320x180:rate=8:duration=8,geq=r='128+70*sin(X/15+Y/28+T)':g='128+70*cos(X/31-Y/17-T)':b='128+70*sin(X/23+Y/19-T)'",
        ],
        &video,
    )?;
    let image = temporary.path().join("sampled-crop.png");
    ffmpeg(
        &[
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
            "-vf",
            "fps=2,select=eq(n\\,6),crop=iw*0.75:ih*0.75:iw*0.125:ih*0.125,scale=240:134",
        ],
        &image,
    )?;
    let cache = FeatureCache::in_memory()?;
    let settings = Config {
        timings: false,
        remote_overlay: true,
        fast: true,
        no_embeddings: false,
        ..config()
    };
    let matches = eidolon_video::find_frames(
        &[entry(&image, 9, Kind::Image)?],
        &[entry(&video, 2, Kind::Video)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: Some(&ConstantEmbedder),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|item| item.signal == "thumbnail"),
        "fast crops must prove their pixels rather than fall back to native decoding: {matches:#?}"
    );
    assert!(
        matches[0]
            .evidence
            .iter()
            .all(|item| item.signal != "unverified")
    );
    Ok(())
}

fn caption(image: &mut image::RgbImage, top: u32, height: u32) {
    for y in top..top + height {
        for x in 0..image.width() {
            image.put_pixel(x, y, image::Rgb([18, 22, 30]));
        }
    }
    // Repeated stems, bowls and gaps emulate high-contrast bitmap text, not a blank border.
    for glyph in 0..14 {
        for y in 0..7 {
            for x in 0..5 {
                if x == 0 || y == 0 || y == 3 || (x == 4 && y < 3) {
                    for dy in 0..3 {
                        for dx in 0..3 {
                            image.put_pixel(
                                7 + glyph * 21 + x * 3 + dx,
                                top + 7 + y * 3 + dy,
                                image::Rgb([242, 238, 225]),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn finds_caption_bands_at_top_middle_and_bottom_without_trusting_the_caption() -> anyhow::Result<()>
{
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-caption-")
        .tempdir_in("/tmp")?;
    let video = native_life_source(temporary.path())?;
    let clean = temporary.path().join("clean.png");
    ffmpeg(
        &[
            "-ss",
            "3.25",
            "-i",
            video.to_str().context("UTF-8 video path")?,
            "-frames:v",
            "1",
            "-vf",
            "crop=ih:ih,scale=320:320",
        ],
        &clean,
    )?;
    let decoded = image::open(&clean)?.to_rgb8();
    let mut images = Vec::new();
    for (index, (top, height)) in [(0, 36), (142, 36), (284, 36), (110, 70)]
        .into_iter()
        .enumerate()
    {
        let path = temporary.path().join(format!("caption-{top}.jpg"));
        let mut pixels = decoded.clone();
        caption(&mut pixels, top, height);
        pixels.save(&path)?;
        images.push(entry(&path, index + 9, Kind::Image)?);
    }
    // The same title cannot prove unrelated pixels or hide a large disagreement
    // outside the title. Embeddings deliberately nominate everything.
    for partial in [false, true] {
        let mut pixels = decoded.clone();
        for y in 60..250 {
            for x in 0..320 {
                if !partial || (120..180).contains(&x) {
                    pixels.put_pixel(x, y, image::Rgb([((x * 17 + y * 31) % 256) as u8, 40, 190]));
                }
            }
        }
        caption(&mut pixels, 284, 36);
        let path = temporary.path().join(format!("different-{partial}.jpg"));
        pixels.save(&path)?;
        images.push(entry(&path, if partial { 41 } else { 40 }, Kind::Image)?);
    }
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    for fast in [false, true] {
        settings.fast = fast;
        let matches = eidolon_video::find_frames(
            &images,
            &[entry(&video, 2, Kind::Video)?],
            &Ctx {
                refiner: None,
                config: &settings,
                cache: &cache,
                embedder: Some(&ConstantEmbedder),
            },
        )?;
        assert_eq!(matches.len(), 4, "fast={fast}: {matches:#?}");
        for found in matches {
            assert_eq!(found.relation, Relation::FrameOf);
            assert!((9..13).contains(&found.a.0), "{found:#?}");
            let time = found
                .evidence
                .iter()
                .find(|item| item.signal == "frame_timestamp")
                .context("source timestamp")?;
            assert!((time.value - 3.25).abs() < 0.126, "{found:#?}");
        }
    }
    Ok(())
}
