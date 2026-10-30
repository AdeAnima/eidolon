//! Remote captions retain strict native-pixel proof across sparse crop proposals.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use anyhow::{Context as _, ensure};
use eidolon_core::cache::FeatureCache;
use eidolon_core::{Config, Ctx, FileEntry, FileId, FrameRefiner, Kind, RefinedWindow, Relation};

struct RecordingRefiner {
    requests: Mutex<Vec<(f64, f64)>>,
}

impl FrameRefiner for RecordingRefiner {
    fn refine(
        &self,
        _: &FileEntry,
        center: f64,
        radius: f64,
    ) -> anyhow::Result<Option<RefinedWindow>> {
        self.requests
            .lock()
            .expect("requests")
            .push((center, radius));
        Ok(None)
    }
}

struct StrongSeedRefiner {
    native: PathBuf,
    requests: Mutex<Vec<(f64, f64)>>,
}

impl FrameRefiner for StrongSeedRefiner {
    fn refine(
        &self,
        _: &FileEntry,
        center: f64,
        radius: f64,
    ) -> anyhow::Result<Option<RefinedWindow>> {
        self.requests
            .lock()
            .expect("requests")
            .push((center, radius));
        ensure!(
            center - radius >= 60.0,
            "weaker early proposals exhausted the transfer budget"
        );
        ensure!(radius <= 0.25 + 1e-7, "native prefixes must stay bounded");
        Ok(Some(RefinedWindow {
            path: self.native.clone(),
            source_offset_s: center - radius,
        }))
    }
}

struct QuotaRefiner {
    native: PathBuf,
    skip_first: bool,
    requests: Mutex<Vec<(f64, f64)>>,
}

impl FrameRefiner for QuotaRefiner {
    fn refine(
        &self,
        _: &FileEntry,
        center: f64,
        radius: f64,
    ) -> anyhow::Result<Option<RefinedWindow>> {
        let mut requests = self.requests.lock().expect("requests");
        requests.push((center, radius));
        let available = if self.skip_first {
            requests.len() == 2
        } else {
            requests.len() == 1
        };
        Ok(available.then(|| RefinedWindow {
            path: self.native.clone(),
            source_offset_s: center - radius,
        }))
    }
}

struct KeyframeBoundaryRefiner {
    native: PathBuf,
    starts: Mutex<Vec<f64>>,
}

impl FrameRefiner for KeyframeBoundaryRefiner {
    fn refine(
        &self,
        _: &FileEntry,
        center: f64,
        radius: f64,
    ) -> anyhow::Result<Option<RefinedWindow>> {
        let start = center - radius;
        self.starts.lock().expect("requests").push(start);
        ensure!(start >= 31.25, "the request crosses the preceding long GOP");
        Ok(Some(RefinedWindow {
            path: self.native.clone(),
            source_offset_s: start,
        }))
    }
}

struct BoundaryRefiner {
    matching: PathBuf,
    unrelated: PathBuf,
    starts: Mutex<Vec<f64>>,
}

impl FrameRefiner for BoundaryRefiner {
    fn refine(
        &self,
        _: &FileEntry,
        center: f64,
        radius: f64,
    ) -> anyhow::Result<Option<RefinedWindow>> {
        let start = center - radius;
        self.starts.lock().expect("requests").push(start);
        ensure!(start >= 0.0 && center + radius <= 3.0 && radius <= 0.25 + 1e-7);
        let path = if start >= 2.0 {
            &self.matching
        } else {
            &self.unrelated
        };
        Ok(Some(RefinedWindow {
            path: path.clone(),
            source_offset_s: start,
        }))
    }
}

fn ffmpeg(arguments: &[&str], output: &Path) -> anyhow::Result<()> {
    let result = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
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
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

fn entry(path: &Path, id: usize, kind: Kind) -> anyhow::Result<FileEntry> {
    Ok(FileEntry {
        id: FileId(id),
        path: path.to_owned(),
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
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: true,
        jobs: 1,
        fast: false,
        video_fps: 2.0,
    }
}

fn caption(image: &mut image::RgbImage) {
    let top = image.height() - 36;
    for y in top..image.height() {
        for x in 0..image.width() {
            image.put_pixel(x, y, image::Rgb([18, 22, 30]));
        }
    }
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

fn cropped_fixture(directory: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
    let background = directory.join("source.png");
    let pixels = image::RgbImage::from_fn(640, 360, |x, y| {
        let cell = (x / 28 * 17 + y / 24 * 29) % 190;
        image::Rgb([
            (30 + cell) as u8,
            (35 + (x / 36 * 31 + y / 32 * 13) % 180) as u8,
            (40 + (x / 24 * 7 + y / 28 * 37) % 170) as u8,
        ])
    });
    pixels.save(&background)?;
    let native = directory.join("native.mp4");
    ffmpeg(
        &[
            "-loop",
            "1",
            "-i",
            background.to_str().context("background path")?,
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
        &native,
    )?;
    let proxy = directory.join("sparse.mp4");
    ffmpeg(
        &[
            "-i",
            native.to_str().context("native path")?,
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
        directory.join("sparse.mp4.remote.json"),
        r#"{"version":1,"duration_seconds":8.0,"source_fps":8.0,"sample_pts_seconds":[0.0]}"#,
    )?;
    let query = directory.join("crop.png");
    ffmpeg(
        &[
            "-i",
            native.to_str().context("native path")?,
            "-frames:v",
            "1",
            "-vf",
            "crop=400:270:160:45,scale=320:216",
        ],
        &query,
    )?;
    let mut pixels = image::open(&query)?.to_rgb8();
    caption(&mut pixels);
    pixels.save(&query)?;
    Ok((proxy, query))
}

#[test]
fn sparse_caption_crop_is_nominated_without_fetching_a_native_window() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let (proxy, query) = cropped_fixture(directory.path())?;
    let images = [entry(&query, 1, Kind::Image)?];
    let videos = [entry(&proxy, 0, Kind::Video)?];
    let cache = FeatureCache::in_memory()?;
    let provider = RecordingRefiner {
        requests: Mutex::new(Vec::new()),
    };
    let mut config = settings();
    for fast in [false, true] {
        config.fast = fast;
        let matches = eidolon_video::find_frames(
            &images,
            &videos,
            &Ctx {
                config: &config,
                cache: &cache,
                embedder: None,
                refiner: Some(&provider),
            },
        )?;
        assert_eq!(matches.len(), 1, "fast={fast}: {matches:#?}");
        assert_eq!(matches[0].relation, Relation::FrameOf);
        assert!(
            matches[0]
                .evidence
                .iter()
                .all(|item| item.signal != "frame_refinement")
        );
    }
    assert!(provider.requests.lock().expect("requests").is_empty());
    Ok(())
}

#[test]
fn decoded_native_prefix_is_reused_with_independent_query_proof() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    cropped_fixture(directory.path())?;
    let query = directory.path().join("centered-caption.png");
    ffmpeg(
        &[
            "-i",
            directory
                .path()
                .join("native.mp4")
                .to_str()
                .context("native path")?,
            "-frames:v",
            "1",
            "-vf",
            "crop=360:360:140:0,scale=320:320",
        ],
        &query,
    )?;
    let mut pixels = image::open(&query)?.to_rgb8();
    caption(&mut pixels);
    pixels.save(&query)?;
    let proxy = ranked_proxy(directory.path(), [30, 16, 30])?;
    let second = directory.path().join("second-caption.png");
    std::fs::copy(&query, &second)?;
    let unrelated = directory.path().join("unrelated-caption.png");
    let mut pixels = image::open(&query)?.to_rgb8();
    for pixel in pixels.pixels_mut() {
        for channel in &mut pixel.0 {
            *channel = channel.saturating_add(65);
        }
    }
    pixels.save(&unrelated)?;
    let images = [
        entry(&query, 1, Kind::Image)?,
        entry(&second, 2, Kind::Image)?,
        entry(&unrelated, 3, Kind::Image)?,
    ];
    let videos = [entry(&proxy, 0, Kind::Video)?];
    for remote_overlay in [false, true] {
        for jobs in [1, 2] {
            let cache = FeatureCache::in_memory()?;
            let provider = QuotaRefiner {
                native: directory.path().join("native.mp4"),
                skip_first: false,
                requests: Mutex::new(Vec::new()),
            };
            let mut config = settings();
            config.remote_overlay = remote_overlay;
            config.jobs = jobs;
            let matches = eidolon_video::find_frames(
                &images,
                &videos,
                &Ctx {
                    config: &config,
                    cache: &cache,
                    embedder: None,
                    refiner: Some(&provider),
                },
            )?;
            assert_eq!(
                matches.len(),
                2,
                "overlay={remote_overlay}; jobs={jobs}: {matches:#?}"
            );
            assert!(
                matches.iter().all(|matched| {
                    [FileId(1), FileId(2)].contains(&matched.a)
                        && matched.b == FileId(0)
                        && matched.relation == Relation::FrameOf
                        && matched
                            .evidence
                            .iter()
                            .any(|item| item.signal == "frame_refinement")
                }),
                "{matches:#?}"
            );
            let requests = provider.requests.lock().expect("requests");
            let first = requests.first().expect("native prefix");
            assert_eq!(
                requests
                    .iter()
                    .filter(|request| {
                        request.0.to_bits() == first.0.to_bits()
                            && request.1.to_bits() == first.1.to_bits()
                    })
                    .count(),
                1,
                "decode reuse must not ask the provider again: {requests:?}"
            );
        }
    }
    Ok(())
}

fn ranked_proxy(directory: &Path, biases: [u8; 3]) -> anyhow::Result<PathBuf> {
    let native = directory.join("native.mp4");
    for (index, bias) in biases.into_iter().enumerate() {
        let filter = format!("lutrgb=r=val+{bias}:g=val+{bias}:b=val+{bias}");
        ffmpeg(
            &[
                "-i",
                native.to_str().context("native path")?,
                "-frames:v",
                "1",
                "-vf",
                &filter,
            ],
            &directory.join(format!("coarse{index}.png")),
        )?;
    }
    let proxy = directory.join("ranked.mp4");
    ffmpeg(
        &[
            "-framerate",
            "1",
            "-i",
            directory
                .join("coarse%d.png")
                .to_str()
                .context("frame path")?,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ],
        &proxy,
    )?;
    std::fs::write(
        directory.join("ranked.mp4.remote.json"),
        r#"{"version":1,"duration_seconds":80.0,"source_fps":8.0,"sample_pts_seconds":[0.0,31.25,62.5]}"#,
    )?;
    Ok(proxy)
}

#[test]
fn sparse_caption_refines_the_strongest_seed_before_weaker_early_windows() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let (_, query) = cropped_fixture(directory.path())?;
    let proxy = ranked_proxy(directory.path(), [30, 30, 16])?;
    let cache = FeatureCache::in_memory()?;
    let provider = StrongSeedRefiner {
        native: directory.path().join("native.mp4"),
        requests: Mutex::new(Vec::new()),
    };
    let config = settings();
    let matches = eidolon_video::find_frames(
        &[entry(&query, 1, Kind::Image)?],
        &[entry(&proxy, 0, Kind::Video)?],
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: Some(&provider),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::FrameOf);
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|item| item.signal == "frame_refinement")
    );
    assert_eq!(provider.requests.lock().expect("requests").len(), 1);
    Ok(())
}

#[test]
fn overlapping_caption_proposals_preserve_the_outer_window_without_duplicate_requests()
-> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let (_, query) = cropped_fixture(directory.path())?;
    let proxy = ranked_proxy(directory.path(), [16, 30, 30])?;
    std::fs::write(
        directory.path().join("ranked.mp4.remote.json"),
        r#"{"version":1,"duration_seconds":3.0,"source_fps":8.0,"sample_pts_seconds":[0.0,0.125,0.375]}"#,
    )?;
    let native = directory.path().join("native.mp4");
    let unrelated = directory.path().join("unrelated.mp4");
    ffmpeg(
        &[
            "-i",
            native.to_str().context("native path")?,
            "-vf",
            "lutrgb=r=val+40:g=val+40:b=val+40",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
        ],
        &unrelated,
    )?;
    let provider = BoundaryRefiner {
        matching: native,
        unrelated,
        starts: Mutex::new(Vec::new()),
    };
    let cache = FeatureCache::in_memory()?;
    let config = settings();
    let matches = eidolon_video::find_frames(
        &[entry(&query, 1, Kind::Image)?],
        &[entry(&proxy, 0, Kind::Video)?],
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: Some(&provider),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    let mut starts = provider.starts.lock().expect("requests").clone();
    starts.sort_unstable_by(f64::total_cmp);
    assert!(
        starts
            .windows(2)
            .all(|pair| pair[0].to_bits() != pair[1].to_bits()),
        "{starts:?}"
    );
    assert!(
        starts.contains(&2.0),
        "the weaker seed extends the valid neighborhood: {starts:?}"
    );
    Ok(())
}

#[test]
fn an_unavailable_caption_prefix_does_not_rule_out_another_seed() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let (_, query) = cropped_fixture(directory.path())?;
    let proxy = ranked_proxy(directory.path(), [30, 30, 16])?;
    let cache = FeatureCache::in_memory()?;
    let provider = QuotaRefiner {
        native: directory.path().join("native.mp4"),
        skip_first: true,
        requests: Mutex::new(Vec::new()),
    };
    let config = settings();
    let matches = eidolon_video::find_frames(
        &[entry(&query, 1, Kind::Image)?],
        &[entry(&proxy, 0, Kind::Video)?],
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: Some(&provider),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(provider.requests.lock().expect("requests").len(), 2);
    Ok(())
}

#[test]
fn caption_prefixes_do_not_cross_the_strongest_keyframes_left_boundary() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let (_, query) = cropped_fixture(directory.path())?;
    let proxy = ranked_proxy(directory.path(), [30, 16, 30])?;
    let cache = FeatureCache::in_memory()?;
    let provider = KeyframeBoundaryRefiner {
        native: directory.path().join("native.mp4"),
        starts: Mutex::new(Vec::new()),
    };
    let config = settings();
    let matches = eidolon_video::find_frames(
        &[entry(&query, 1, Kind::Image)?],
        &[entry(&proxy, 0, Kind::Video)?],
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: Some(&provider),
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    let starts = provider.starts.lock().expect("requests");
    assert_eq!(starts.len(), 1, "{starts:?}");
    assert!(starts[0] >= 31.25, "{starts:?}");
    Ok(())
}

#[test]
fn stronger_caption_queries_receive_native_budget_before_weaker_inputs() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let (_, query) = cropped_fixture(directory.path())?;
    let proxy = ranked_proxy(directory.path(), [30, 30, 16])?;
    let negative = directory.path().join("earlier-unrelated.png");
    let mut image = image::open(&query)?.to_rgb8();
    for y in 0..image.height() - 36 {
        for x in 0..image.width() {
            for channel in &mut image.get_pixel_mut(x, y).0 {
                *channel = channel.saturating_add(56);
            }
        }
    }
    image.save(&negative)?;
    for jobs in [1, 2] {
        let cache = FeatureCache::in_memory()?;
        let provider = QuotaRefiner {
            native: directory.path().join("native.mp4"),
            skip_first: false,
            requests: Mutex::new(Vec::new()),
        };
        let mut config = settings();
        config.jobs = jobs;
        let matches = eidolon_video::find_frames(
            &[
                entry(&negative, 1, Kind::Image)?,
                entry(&query, 2, Kind::Image)?,
            ],
            &[entry(&proxy, 0, Kind::Video)?],
            &Ctx {
                config: &config,
                cache: &cache,
                embedder: None,
                refiner: Some(&provider),
            },
        )?;
        assert_eq!(matches.len(), 1, "jobs={jobs}: {matches:#?}");
        assert_eq!(matches[0].a, FileId(2), "jobs={jobs}: {matches:#?}");
        assert!(
            matches[0]
                .evidence
                .iter()
                .any(|item| item.signal == "frame_refinement")
        );
    }
    Ok(())
}
