//! Trait-injected embeddings tested against real cropped video frames.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{Context as _, ensure};
use eidolon_core::cache::FeatureCache;
use eidolon_core::{Config, Ctx, Embedder, FileEntry, FileId, Kind, Match, Relation};
use image::RgbImage;

struct ColorEmbedder {
    calls: AtomicUsize,
    id: &'static str,
    constant: bool,
}

impl Embedder for ColorEmbedder {
    fn model_id(&self) -> &str {
        self.id
    }

    fn embed(&self, image: &RgbImage) -> anyhow::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.constant {
            return Ok(vec![1.0, 0.0, 0.0]);
        }
        let mut color = [0.0_f32; 3];
        for pixel in image.pixels() {
            for channel in 0..3 {
                color[channel] += f32::from(pixel[channel]);
            }
        }
        let norm = color.iter().map(|value| value * value).sum::<f32>().sqrt();
        Ok(color.map(|value| value / norm).to_vec())
    }
}

fn encode(arguments: &[&str], path: &Path) -> anyhow::Result<()> {
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-filter_threads",
            "1",
        ])
        .args(arguments)
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
        ])
        .arg(path)
        .output()?;
    ensure!(
        output.status.success(),
        "FFmpeg: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

fn file(path: &Path, id: usize) -> anyhow::Result<FileEntry> {
    let metadata = path.metadata()?;
    Ok(FileEntry {
        id: FileId(id),
        path: path.to_owned(),
        size: metadata.len(),
        mtime_ns: i128::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
        kind: Kind::Video,
    })
}

fn settings() -> Config {
    Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.75,
        model_dir: PathBuf::new(),
        no_embeddings: false,
        no_keypoints: true,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    }
}

#[test]
fn uncached_embedding_work_reuses_qualified_video_and_image_caches() -> anyhow::Result<()> {
    struct CachedOnly;
    impl Embedder for CachedOnly {
        fn model_id(&self) -> &'static str {
            "qualified-cpu"
        }
        fn cache_model_ids(&self) -> [Option<&str>; 2] {
            [Some("qualified-cpu"), Some("qualified-coreml")]
        }
        fn frame_model_id(&self) -> &'static str {
            "frame-qualified-cpu"
        }
        fn frame_cache_model_ids(&self) -> [Option<&str>; 2] {
            [Some("frame-qualified-cpu"), Some("frame-qualified-coreml")]
        }
        fn prepare(&self) -> anyhow::Result<()> {
            anyhow::bail!("a cache-only workload must not initialize inference")
        }
        fn embed(&self, _: &RgbImage) -> anyhow::Result<Vec<f32>> {
            anyhow::bail!("a cache-only workload must not run inference")
        }
    }
    let temporary = tempfile::tempdir()?;
    let mut files = Vec::new();
    for id in 0..3 {
        let path = temporary.path().join(format!("{id}.media"));
        std::fs::write(&path, b"cache-only metadata")?;
        files.push(file(&path, id)?);
    }
    files[2].kind = Kind::Image;
    let cache = FeatureCache::in_memory()?;
    let config = settings();
    let ctx = Ctx {
        config: &config,
        cache: &cache,
        embedder: Some(&CachedOnly),
        refiner: None,
    };
    assert_eq!(crate::uncached_embedding_work(&files, &ctx, true)?, 2401);
    for (index, model) in [(0, "frame-qualified-coreml"), (1, "frame-qualified-cpu")] {
        let key = crate::frames::feature_key(config.video_fps, model, None);
        cache.put(&files[index], &key, b"null")?;
    }
    assert_eq!(crate::uncached_embedding_work(&files, &ctx, true)?, 1);
    let frame = crate::frames::describe(&RgbImage::new(96, 96), 0.0);
    let key = crate::frame_images::feature_key("frame-qualified-coreml");
    cache.put(&files[2], &key, &serde_json::to_vec(&frame)?)?;
    assert_eq!(crate::uncached_embedding_work(&files, &ctx, true)?, 0);
    assert!(crate::frames::load(&files[0], &ctx)?.is_none());
    files[0].mtime_ns += 1;
    assert_eq!(crate::uncached_embedding_work(&files, &ctx, true)?, 1200);
    assert_eq!(crate::uncached_embedding_work(&files[2..], &ctx, true)?, 0);
    Ok(())
}

fn crop(source: &Path, destination: &Path, start: &str) -> anyhow::Result<()> {
    encode(
        &[
            "-ss",
            start,
            "-i",
            source.to_str().context("UTF-8 test path")?,
            "-t",
            "10",
            "-vf",
            "crop=100:80:35:20,scale=160:120",
        ],
        destination,
    )
}

fn check_clip(matches: &[Match], expected: f64) -> anyhow::Result<()> {
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::ClipOf, "{matches:#?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(7), FileId(1)));
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|evidence| evidence.signal == "frame_embedding"),
        "{matches:#?}"
    );
    let alignment = matches[0]
        .evidence
        .iter()
        .find(|evidence| evidence.signal == "temporal_alignment")
        .context("alignment evidence")?;
    let offset: f64 = alignment
        .note
        .split(';')
        .next()
        .context("offset")?
        .strip_prefix("offset_seconds=")
        .context("offset value")?
        .parse()?;
    assert!((offset - expected).abs() <= 1.0, "{alignment:?}");
    Ok(())
}

#[test]
fn embedding_aligns_real_crop_and_reuses_versioned_cache() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("colors.mp4");
    let cropped = temporary.path().join("crop.mp4");
    // Spatial texture changes under cropping; time-varying channel ratios do not.
    let source = "nullsrc=size=160x120:rate=6:duration=40,geq=r='(128+110*sin(T*0.53))*(0.7+0.3*sin(X*0.17)*cos(Y*0.13))':g='(128+110*sin(T*0.79))*(0.7+0.3*sin(X*0.17)*cos(Y*0.13))':b='(128+110*sin(T*1.13))*(0.7+0.3*sin(X*0.17)*cos(Y*0.13))'";
    encode(&["-f", "lavfi", "-i", source], &original)?;
    crop(&original, &cropped, "12")?;
    let mut files = [file(&original, 1)?, file(&cropped, 7)?];
    let cache = FeatureCache::in_memory()?;
    let config = settings();
    let plain = crate::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert!(
        plain.is_empty(),
        "crop must require the injected signal: {plain:#?}"
    );
    let model = ColorEmbedder {
        calls: AtomicUsize::new(0),
        id: "test-color/v1",
        constant: false,
    };
    let ctx = Ctx {
        refiner: None,
        config: &config,
        cache: &cache,
        embedder: Some(&model),
    };
    check_clip(&crate::find(&files, &ctx)?, 12.0)?;
    let cold_calls = model.calls.load(Ordering::Relaxed);
    assert!(cold_calls >= 16);
    check_clip(&crate::find(&files, &ctx)?, 12.0)?;
    assert_eq!(
        model.calls.load(Ordering::Relaxed),
        cold_calls,
        "warm cache must not invoke inference"
    );
    crop(&original, &cropped, "24")?;
    files[1] = file(&cropped, 7)?;
    check_clip(&crate::find(&files, &ctx)?, 24.0)?;
    assert_eq!(
        model.calls.load(Ordering::Relaxed) - cold_calls,
        4,
        "only the changed 10-second clip is recomputed"
    );
    let new_model = ColorEmbedder {
        calls: AtomicUsize::new(0),
        id: "test-color/v2",
        constant: false,
    };
    check_clip(
        &crate::find(
            &files,
            &Ctx {
                refiner: None,
                config: &config,
                cache: &cache,
                embedder: Some(&new_model),
            },
        )?,
        24.0,
    )?;
    assert_eq!(
        new_model.calls.load(Ordering::Relaxed),
        cold_calls,
        "model identity invalidates all embeddings"
    );
    check_embedding_gates(&files, &ctx, &model)?;
    Ok(())
}

fn check_embedding_gates(
    files: &[FileEntry],
    ctx: &Ctx<'_>,
    model: &ColorEmbedder,
) -> anyhow::Result<()> {
    let uninformative = ColorEmbedder {
        calls: AtomicUsize::new(0),
        id: "test-constant/v1",
        constant: true,
    };
    assert!(
        crate::find(
            files,
            &Ctx {
                embedder: Some(&uninformative),
                ..*ctx
            }
        )?
        .is_empty(),
        "a shared scene vector alone must not establish temporal alignment"
    );
    let mut disabled = ctx.config.clone();
    disabled.no_embeddings = true;
    let before = model.calls.load(Ordering::Relaxed);
    assert!(
        crate::find(
            files,
            &Ctx {
                config: &disabled,
                ..*ctx
            }
        )?
        .is_empty()
    );
    assert_eq!(model.calls.load(Ordering::Relaxed), before);
    Ok(())
}

#[test]
fn rejects_shared_embedding_trajectory_without_shared_spatial_content() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let first = temporary.path().join("first.mp4");
    let second = temporary.path().join("second.mp4");
    for (path, texture) in [
        (&first, "sin(X*0.17)*cos(Y*0.13)"),
        (&second, "cos(X*0.31+Y*0.23)*sin(Y*0.29-X*0.11)"),
    ] {
        let source = format!(
            "nullsrc=size=160x120:rate=6:duration=20,geq=r='(128+110*sin(T*0.53))*(0.7+0.3*{texture})':g='(128+110*sin(T*0.79))*(0.7+0.3*{texture})':b='(128+110*sin(T*1.13))*(0.7+0.3*{texture})'"
        );
        encode(&["-f", "lavfi", "-i", &source], path)?;
    }
    let model = ColorEmbedder {
        calls: AtomicUsize::new(0),
        id: "test-color/spatial-negative",
        constant: false,
    };
    let cache = FeatureCache::in_memory()?;
    let config = settings();
    let matches = crate::find(
        &[file(&first, 0)?, file(&second, 1)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: Some(&model),
        },
    )?;
    assert!(
        matches.is_empty(),
        "aligned semantic changes must not substitute for spatial identity: {matches:#?}"
    );
    let fast_config = Config {
        timings: false,
        remote_overlay: true,
        fast: true,
        ..config
    };
    let fast = crate::find(
        &[file(&first, 0)?, file(&second, 1)?],
        &Ctx {
            refiner: None,
            config: &fast_config,
            cache: &cache,
            embedder: Some(&model),
        },
    )?;
    assert!(
        fast.is_empty(),
        "fast mode must corroborate semantic alignment with cached pixels: {fast:#?}"
    );
    Ok(())
}

#[test]
fn fast_nominates_embeddings_before_fitting_global_color_models() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("colors.mp4");
    let recolored = temporary.path().join("recolored.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=8:duration=18",
        ],
        &original,
    )?;
    encode(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "negate",
        ],
        &recolored,
    )?;
    let mut config = settings();
    config.fast = true;
    let cache = FeatureCache::in_memory()?;
    let model = ColorEmbedder {
        calls: AtomicUsize::new(0),
        id: "fast-color-candidate/v1",
        constant: true,
    };
    let context = Ctx {
        refiner: None,
        config: &config,
        cache: &cache,
        embedder: Some(&model),
    };
    let original_file = file(&original, 1)?;
    let recolored_file = file(&recolored, 7)?;
    let original_video =
        crate::frames::load(&original_file, &context)?.context("decoded original")?;
    let recolored_video =
        crate::frames::load(&recolored_file, &context)?.context("decoded inverse colors")?;
    let (first, second) = original_video
        .frames
        .iter()
        .zip(&recolored_video.frames)
        .find(|(a, b)| {
            a.embedding.is_some()
                && b.embedding.is_some()
                && crate::frames::hash_similarity(a, b, true).is_none()
                && crate::appearance::aligned(a, b).is_none()
                && matches!(
                    crate::appearance::color_change(a, b),
                    Some(crate::appearance::ColorChange::Variant(_))
                )
        })
        .context("decoded inverse colors require a color model or embedding nomination")?;
    let times = (first.time, second.time);
    let candidates = crate::candidates::generate(&[original_video, recolored_video], true);
    let nomination = candidates
        .get(&(0, 1))
        .into_iter()
        .flatten()
        .find(|anchor| (anchor.a_time, anchor.b_time) == times)
        .context("matching embeddings must nominate the indexed frame pair")?;
    assert!(
        nomination.embedded,
        "fast nomination must defer color fitting when embeddings already nominate the pair"
    );
    Ok(())
}

#[test]
fn finds_global_hue_shift_as_variant_without_embeddings() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("colors.mp4");
    let recolored = temporary.path().join("recolored.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=8:duration=18",
        ],
        &original,
    )?;
    encode(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "hue=h=45:s=1",
        ],
        &recolored,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let matches = crate::find(
        &[file(&recolored, 7)?, file(&original, 1)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Variant, "{matches:#?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(1), FileId(7)));
    let clip = temporary.path().join("recolored-clip.mp4");
    encode(
        &[
            "-ss",
            "5",
            "-i",
            recolored.to_str().context("UTF-8 test path")?,
            "-t",
            "8",
        ],
        &clip,
    )?;
    let matches = crate::find(
        &[file(&original, 1)?, file(&clip, 7)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Variant, "{matches:#?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(1), FileId(7)));
    Ok(())
}

#[test]
fn finds_global_luma_shift_as_variant_without_embeddings() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("source.mp4");
    let relit = temporary.path().join("relit.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=8:duration=18",
        ],
        &original,
    )?;
    encode(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "eq=brightness=0.12",
        ],
        &relit,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let matches = crate::find(
        &[file(&original, 1)?, file(&relit, 2)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Variant, "{matches:#?}");
    Ok(())
}

#[test]
fn keeps_global_color_variant_separate_despite_local_overlay() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("source.mp4");
    let edited_variant = temporary.path().join("edited-variant.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=8:duration=18",
        ],
        &original,
    )?;
    encode(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "hue=h=45,drawbox=x=iw*0.1:y=ih*0.72:w=iw*0.8:h=ih*0.18:color=magenta@0.9:t=fill",
        ],
        &edited_variant,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let matches = crate::find(
        &[file(&original, 1)?, file(&edited_variant, 2)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Variant, "{matches:#?}");
    Ok(())
}

#[test]
fn preserves_fine_color_changes_that_cancel_in_regional_means() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("fine-texture.mp4");
    let recolored = temporary.path().join("recolored.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=160x90:rate=8:seed=319813:ratio=0.45:death_color=0x102030:life_color=0xe8c070,scale=960:540:flags=neighbor",
            "-t",
            "18",
        ],
        &original,
    )?;
    encode(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "hue=h=110",
        ],
        &recolored,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let matches = crate::find(
        &[file(&original, 1)?, file(&recolored, 2)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Variant, "{matches:#?}");
    Ok(())
}

#[test]
fn detects_global_variant_when_compression_obscures_some_frames() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("source.mp4");
    let recolored = temporary.path().join("recolored.mp4");
    let compressed = temporary.path().join("compressed.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=160x90:rate=8:seed=91831:ratio=0.45:death_color=0x102030:life_color=0xe8c070,scale=960:540:flags=neighbor",
            "-t",
            "40",
        ],
        &original,
    )?;
    encode(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "hue=h=110",
        ],
        &recolored,
    )?;
    encode(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "scale=640:360",
            "-b:v",
            "90k",
            "-maxrate",
            "90k",
            "-bufsize",
            "180k",
        ],
        &compressed,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let matches = crate::find(
        &[file(&compressed, 1)?, file(&recolored, 2)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Variant, "{matches:#?}");
    Ok(())
}

#[test]
fn detects_global_hue_and_nonlinear_tone_without_embeddings() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("source.mp4");
    let version = temporary.path().join("version.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "nullsrc=size=320x180:rate=8:duration=12,geq=r='50+45*sin(X*0.037+Y*0.021+T*0.013)':g='70+60*sin(X*0.041-Y*0.033+T*0.017)':b='90+80*cos(X*0.023+Y*0.051+T*0.019)'",
        ],
        &original,
    )?;
    encode(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "hue=h=120,eq=gamma=0.3",
        ],
        &version,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let ctx = Ctx {
        refiner: None,
        config: &config,
        cache: &cache,
        embedder: None,
    };
    let matches = crate::find(&[file(&original, 1)?, file(&version, 2)?], &ctx)?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Variant, "{matches:#?}");
    Ok(())
}

#[test]
fn rejects_unrelated_dark_scenes_with_the_same_color_palette() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let first = temporary.path().join("first.mp4");
    let second = temporary.path().join("second.mp4");
    for (path, source) in [
        (
            &first,
            "nullsrc=size=320x180:rate=8:duration=12,geq=r='50+45*sin(X*0.037+Y*0.021+T*0.013)':g='70+60*sin(X*0.041-Y*0.033+T*0.017)':b='90+80*cos(X*0.023+Y*0.051+T*0.019)'",
        ),
        (
            &second,
            "nullsrc=size=320x180:rate=8:duration=12,geq=r='50+45*sin(X*0.061-Y*0.047+T*0.029)':g='70+60*cos(X*0.029+Y*0.071+T*0.037)':b='90+80*sin(X*0.043-Y*0.031+T*0.011)',hue=h=120,eq=gamma=0.3",
        ),
    ] {
        encode(&["-f", "lavfi", "-i", source], path)?;
    }
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let matches = crate::find(
        &[file(&first, 1)?, file(&second, 2)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert!(matches.is_empty(), "{matches:#?}");
    Ok(())
}

#[test]
fn unresolved_broad_recoloring_does_not_become_a_duplicate_clip() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("source.mp4");
    let version = temporary.path().join("version.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x180:rate=8,lutrgb=r='80+0.4*val':g='80+0.4*val':b='80+0.4*val',split[a][b];[a][b]hstack=inputs=2",
            "-t",
            "16",
        ],
        &original,
    )?;
    encode(
        &[
            "-ss",
            "4",
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-t",
            "8",
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            "[0:v]split=2[a][b];[a]crop=iw/2:ih:0:0,hue=h=120[l];[b]crop=iw/2:ih:iw/2:0,hue=h=-120[r];[l][r]hstack=inputs=2",
        ],
        &version,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let matches = crate::find(
        &[file(&original, 1)?, file(&version, 2)?],
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert!(
        matches
            .iter()
            .all(|found| found.relation != Relation::ClipOf),
        "unresolved recoloring must not bridge duplicate groups: {matches:#?}"
    );
    Ok(())
}

#[test]
fn preserves_dark_scene_gradients_instead_of_cropping_them() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("dark-gradient.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "nullsrc=size=96x96:rate=8:duration=4,geq=r='2+60*Y/H':g='100*(Y/H)^2':b='120*(Y/H)^2'",
        ],
        &original,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let video = crate::frames::load(
        &file(&original, 1)?,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
    )?
    .context("decoded dark gradient")?;
    assert!(
        video
            .frames
            .iter()
            .all(|frame| frame.rgb_thumbnail[0][0] <= 8),
        "dark detail at the scene edge must remain in the descriptor"
    );
    Ok(())
}

#[test]
fn sampling_rate_is_applied_and_cached_independently() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("sampling.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=8:duration=4",
        ],
        &original,
    )?;
    let entry = file(&original, 0)?;
    let cache = FeatureCache::in_memory()?;
    let model = ColorEmbedder {
        calls: AtomicUsize::new(0),
        id: "sampling-cache/v1",
        constant: true,
    };
    let mut config = settings();
    for (fps, expected_frames, expected_calls) in [(2.0, 8, 2), (1.0, 4, 4), (2.0, 8, 4)] {
        config.video_fps = fps;
        let video = crate::frames::load(
            &entry,
            &Ctx {
                config: &config,
                cache: &cache,
                embedder: Some(&model),
                refiner: None,
            },
        )?
        .context("sampled video")?;
        assert_eq!(video.frames.len(), expected_frames, "sampling at {fps} fps");
        assert!((video.step - 1.0 / fps).abs() < 1e-9);
        assert_eq!(model.calls.load(Ordering::Relaxed), expected_calls);
    }
    Ok(())
}

#[test]
fn measures_hash_extraction_on_real_full_hd_video() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("full-hd.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=40",
        ],
        &original,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let ctx = Ctx {
        refiner: None,
        config: &config,
        cache: &cache,
        embedder: None,
    };
    let started = Instant::now();
    let video = crate::frames::load(&file(&original, 0)?, &ctx)?.context("video stream")?;
    let elapsed = started.elapsed().as_secs_f64();
    assert_eq!(video.frames.len(), 80);
    assert!((video.frames[79].time - 39.5).abs() < 0.001);
    eprintln!(
        "FULL_HD_HASH_EXTRACTION: seconds_per_minute={:.3}; elapsed_seconds={elapsed:.3}; duration_seconds={}; resolution=1920x1080; fps=30; sampled_frames={}; threads=1; embeddings=none; cold_cache=true",
        elapsed * 60.0 / video.duration,
        video.duration,
        video.frames.len()
    );
    Ok(())
}

#[test]
fn caps_samples_across_a_real_twenty_minute_video() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let original = temporary.path().join("long.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=1:duration=1210",
        ],
        &original,
    )?;
    let config = settings();
    let cache = FeatureCache::in_memory()?;
    let ctx = Ctx {
        refiner: None,
        config: &config,
        cache: &cache,
        embedder: None,
    };
    let video = crate::frames::load(&file(&original, 0)?, &ctx)?.context("video stream")?;
    assert_eq!(video.frames.len(), 1200);
    assert!(video.step > 1.0);
    let last = video.frames.last().context("last sampled frame")?;
    assert!(last.time < video.duration && last.time > video.duration - video.step * 2.0);
    assert!(
        video
            .frames
            .windows(2)
            .any(|frames| frames[0].phash != frames[1].phash),
        "sampled frames must retain temporal changes"
    );
    let one_fps = Config {
        timings: false,
        remote_overlay: true,
        video_fps: 1.0,
        ..config
    };
    let sparse = crate::frames::load(
        &file(&original, 0)?,
        &Ctx {
            config: &one_fps,
            cache: &cache,
            embedder: None,
            refiner: None,
        },
    )?
    .context("one-fps bounded video")?;
    assert_eq!(sparse.frames.len(), 600);
    assert!((sparse.step - video.step * 2.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn maps_native_window_pixels_to_the_original_source_clock() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let video = temporary.path().join("window.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=8:duration=4",
        ],
        &video,
    )?;
    let window = eidolon_core::RefinedWindow {
        path: video,
        source_offset_s: 37.0,
    };
    let frames = crate::frames::refined_window(&window, 38.25, 0.375)?;
    assert_eq!(frames.len(), 6);
    assert!((frames[0].time - 37.875).abs() < 1e-6);
    assert!(frames.iter().all(|frame| frame.pixels.is_some()));
    assert!(
        frames
            .windows(2)
            .all(|pair| { (pair[1].time - pair[0].time - 0.125).abs() < 1e-6 })
    );
    Ok(())
}

#[test]
fn shared_native_timeline_retains_seeked_window_pixels() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    for (rate, fps) in [("8", 8.0), ("12", 12.0), ("30000/1001", 30_000.0 / 1001.0)] {
        let path = temporary.path().join(format!("native-{fps}.mp4"));
        let source = format!("testsrc2=size=160x90:rate={rate}:duration=4");
        encode(&["-f", "lavfi", "-i", &source], &path)?;
        let video = crate::frames::Video {
            duration: 4.0,
            width: 160,
            height: 90,
            fps,
            bitrate: None,
            step: 0.5,
            max_frames: 1200,
            frames: Vec::new(),
        };
        let entry = file(&path, 0)?;
        let cache = FeatureCache::in_memory()?;
        let native =
            crate::frames::native(&entry, &video, &cache)?.context("bounded native timeline")?;
        assert!(
            serde_json::to_value(&native[0])?["color_hashes"].is_null(),
            "unindexed native frames must not extract color-hash nomination planes"
        );
        let rgb = RgbImage::from_raw(
            crate::frames::WIDTH,
            crate::frames::HEIGHT,
            native[0]
                .pixels
                .as_ref()
                .context("native pixels")?
                .rgb
                .clone(),
        )
        .context("decoded native image")?;
        let indexed = crate::frames::describe(&rgb, native[0].time);
        let serialized = serde_json::to_value(&indexed)?;
        assert!(
            serialized["color_hashes"].is_array(),
            "indexed cache format changed"
        );
        let cached: crate::frames::Frame = serde_json::from_value(serialized)?;
        assert_eq!(cached.color_hashes, indexed.color_hashes);
        assert_eq!(cached.thumbnail, native[0].thumbnail);
        assert_eq!(cached.rgb_thumbnail, native[0].rgb_thumbnail);
        assert_eq!(cached.tiles, native[0].tiles);
        for center in [0.0_f64, 2.0, 2.5, 3.5] {
            let radius = 0.5 + 1.0 / fps;
            let window = crate::frames::refine(&entry, &video, center, radius)?;
            let start = (center - radius).max(0.0);
            let end = (center + radius).min(video.duration);
            let first = native.partition_point(|frame| frame.time + 1e-7 < start);
            let last = native.partition_point(|frame| frame.time + 1e-7 < end);
            assert_eq!(last - first, window.len(), "{rate} fps, center={center}");
            for (actual, expected) in native[first..last].iter().zip(window) {
                assert_eq!(
                    actual.pixels.as_ref().context("shared native pixels")?.rgb,
                    expected.pixels.context("seeked native pixels")?.rgb,
                    "{rate} fps, center={center}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn native_rgb_envelopes_preserve_decoded_rectangle_extrema() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let path = temporary.path().join("envelopes.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=8:duration=1",
        ],
        &path,
    )?;
    let frames = crate::frames::refined_window(
        &eidolon_core::RefinedWindow {
            path,
            source_offset_s: 0.0,
        },
        0.0,
        0.125,
    )?;
    let pixels = frames[0].pixels.as_ref().context("decoded native pixels")?;
    for height in 1..=4 {
        for width in 1..=4 {
            for top in (0..=32 - height).step_by(5) {
                for left in (0..=32 - width).step_by(5) {
                    let mut expected = [[255; 3], [0; 3]];
                    for y in top * 3..(top + height) * 3 {
                        for x in left * 3..(left + width) * 3 {
                            for (channel, &value) in
                                pixels.rgb[(y * 96 + x) * 3..][..3].iter().enumerate()
                            {
                                expected[0][channel] = expected[0][channel].min(value);
                                expected[1][channel] = expected[1][channel].max(value);
                            }
                        }
                    }
                    assert_eq!(
                        pixels.range(top..top + height, left..left + width),
                        expected,
                        "{left},{top} {width}×{height}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn variable_cadence_keeps_window_specific_refinement() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?;
    let path = temporary.path().join("variable.mp4");
    encode(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=24:duration=4",
            "-vf",
            "select='if(lt(t,2),not(mod(n,2)),1)'",
            "-fps_mode",
            "vfr",
        ],
        &path,
    )?;
    let entry = file(&path, 0)?;
    let cache = FeatureCache::in_memory()?;
    let config = Config {
        timings: false,
        remote_overlay: true,
        no_embeddings: true,
        ..settings()
    };
    let video = crate::frames::load(
        &entry,
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: None,
        },
    )?
    .context("variable-cadence video")?;
    assert!(crate::frames::native(&entry, &video, &cache)?.is_none());
    assert!(crate::frames::native(&entry, &video, &cache)?.is_none());
    assert!(!crate::frames::refine(&entry, &video, 2.0, 0.5)?.is_empty());
    Ok(())
}

#[test]
fn invalid_video_identifies_its_input_but_invalid_cache_is_fatal() -> anyhow::Result<()> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("invalid.mp4");
    std::fs::write(&path, b"\0\0\0\x18ftypisom\0\0\0\0isom")?;
    let files = [file(&path, 7)?];
    let cache = FeatureCache::in_memory()?;
    let config = settings();
    let ctx = Ctx {
        config: &config,
        cache: &cache,
        embedder: None,
        refiner: None,
    };
    let (matches, skipped) = crate::find_with_skipped(&files, &ctx)?;
    assert_eq!(matches.len(), 0);
    assert_eq!(skipped.len(), 1);
    let input = &skipped[0];
    assert_eq!(input.file, FileId(7));
    assert!(input.reason.contains("ffprobe failed"));
    let key = crate::frames::feature_key(config.video_fps, "none", None);
    cache.put(&files[0], &key, b"invalid cache JSON")?;
    let error = crate::find(&files, &ctx).expect_err("invalid cache must stay fatal");
    assert!(error.downcast_ref::<eidolon_core::InputError>().is_none());
    let error = crate::find_frames(&files, &files, &ctx)
        .expect_err("image-to-video cache failures must stay fatal too");
    assert!(error.downcast_ref::<eidolon_core::InputError>().is_none());
    Ok(())
}

#[test]
fn global_hue_can_change_luminance_structure_outside_the_fitted_tiles() -> anyhow::Result<()> {
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-hue-support-")
        .tempdir_in("/tmp")?;
    let original = RgbImage::from_fn(96, 96, |x, y| {
        let value = (40 + (x / 3 * 7 + y / 3 * 11) % 80) as u8;
        let green = if y < 48 { value + 80 } else { 240 - value };
        image::Rgb([value, green, (40 + (x / 3 * 13 + y / 3 * 5) % 80) as u8])
    });
    let hue = RgbImage::from_fn(96, 96, |x, y| {
        let [red, green, blue] = original.get_pixel(x, y).0;
        image::Rgb([green, red, blue])
    });
    let first = temporary.path().join("original.png");
    let second = temporary.path().join("hue.png");
    original.save(&first)?;
    hue.save(&second)?;
    let mut first = crate::frames::describe(&eidolon_image::decode(&first)?.to_rgb8(), 0.0);
    let mut second = crate::frames::describe(&eidolon_image::decode(&second)?.to_rgb8(), 0.0);
    // Native frames omit nomination-only color hashes; RGB affine proof must
    // still recognize the channel transform without relying on tone matching.
    first.color_hashes = None;
    second.color_hashes = None;
    assert!(matches!(
        crate::appearance::color_change(&first, &second),
        Some(crate::appearance::ColorChange::Variant(_))
    ));
    Ok(())
}

#[test]
fn a_single_band_edit_is_not_a_global_variant_even_when_an_rgb_transform_fits() -> anyhow::Result<()>
{
    let temporary = tempfile::Builder::new()
        .prefix("eidolon-video-band-edit-")
        .tempdir_in("/tmp")?;
    for vertical in [false, true] {
        let gray = RgbImage::from_fn(96, 96, |x, y| {
            let level = (40 + (x * 17 + y * 29) % 100) as u8;
            image::Rgb([level; 3])
        });
        let mut colored = gray.clone();
        for (x, y, pixel) in colored.enumerate_pixels_mut() {
            if (if vertical { x } else { y }) >= 72 {
                pixel[0] += 70;
            }
        }
        let mut paths = Vec::new();
        for (name, pixels) in [("original", colored), ("band-edit", gray)] {
            let still = temporary.path().join(format!("{name}-{vertical}.png"));
            let video = still.with_extension("mp4");
            pixels.save(&still)?;
            encode(
                &[
                    "-loop",
                    "1",
                    "-framerate",
                    "8",
                    "-i",
                    still.to_str().context("UTF-8 still path")?,
                    "-t",
                    "6",
                ],
                &video,
            )?;
            paths.push(video);
        }
        let config = settings();
        let cache = FeatureCache::in_memory()?;
        let matches = crate::find(
            &[file(&paths[0], 1)?, file(&paths[1], 2)?],
            &Ctx {
                refiner: None,
                config: &config,
                cache: &cache,
                embedder: None,
            },
        )?;
        assert_eq!(matches.len(), 1, "vertical={vertical}: {matches:#?}");
        assert_eq!(
            matches[0].relation,
            Relation::Edited,
            "a transform can fit a color that exists only in one band: {matches:#?}"
        );
    }
    Ok(())
}
