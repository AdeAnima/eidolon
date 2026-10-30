//! End-to-end checks on videos encoded and decoded by the installed `FFmpeg`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{Context as _, ensure};
use eidolon_core::cache::FeatureCache;
use eidolon_core::{Config, Ctx, FileEntry, FileId, Kind, Relation};
use tempfile::TempDir;

fn scratch() -> anyhow::Result<TempDir> {
    Ok(tempfile::Builder::new()
        .prefix("eidolon-video-")
        .tempdir_in("/tmp")?)
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

fn source(directory: &Path) -> anyhow::Result<PathBuf> {
    let path = directory.join("source.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=12:duration=40",
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

fn entry(path: &Path, id: usize) -> anyhow::Result<FileEntry> {
    let metadata = path.metadata()?;
    Ok(FileEntry {
        id: FileId(id),
        path: path.to_path_buf(),
        size: metadata.len(),
        mtime_ns: i128::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
        kind: Kind::Video,
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

#[test]
fn finds_lower_resolution_reencoding_without_embeddings() -> anyhow::Result<()> {
    let temporary = scratch()?;
    let original = source(temporary.path())?;
    let smaller = temporary.path().join("small.mp4");
    ffmpeg(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "scale=160:90",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "38",
        ],
        &smaller,
    )?;
    let files = [entry(&original, 8)?, entry(&smaller, 3)?];
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let started = Instant::now();
    let matches = eidolon_video::find(
        &files,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    eprintln!(
        "hash-only cold extraction+matching: {:.3} seconds/minute (80 video seconds)",
        started.elapsed().as_secs_f64() * 60.0 / 80.0
    );
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Reencoded, "{matches:#?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(3), FileId(8)));
    assert!(!matches[0].evidence.is_empty());
    let mut strict_settings = settings.clone();
    strict_settings.min_score = 1.0;
    assert!(
        eidolon_video::find(
            &files,
            &Ctx {
                refiner: None,
                config: &strict_settings,
                cache: &cache,
                embedder: None
            }
        )?
        .is_empty()
    );
    Ok(())
}

#[test]
fn labels_lossy_compression_without_requiring_perfect_frame_coverage() -> anyhow::Result<()> {
    let temporary = scratch()?;
    let original = temporary.path().join("source.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=128x72:rate=8:seed=57431:ratio=0.45:death_color=0x183048:life_color=0xe8c080,scale=768:432:flags=neighbor",
            "-t",
            "24",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ],
        &original,
    )?;
    let compressed = temporary.path().join("compressed.mp4");
    ffmpeg(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "scale=640:360",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-b:v",
            "90k",
            "-maxrate",
            "90k",
            "-bufsize",
            "180k",
        ],
        &compressed,
    )?;
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find(
        &[entry(&original, 0)?, entry(&compressed, 1)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Reencoded, "{matches:#?}");
    Ok(())
}

#[test]
fn finds_fractional_middle_clip_with_direction_and_offset() -> anyhow::Result<()> {
    let temporary = scratch()?;
    let original = source(temporary.path())?;
    let clip = temporary.path().join("clip.mp4");
    ffmpeg(
        &[
            "-ss",
            "12.4",
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-t",
            "10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "25",
        ],
        &clip,
    )?;
    let files = [entry(&original, 1)?, entry(&clip, 9)?];
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find(
        &files,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::ClipOf, "{matches:#?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(9), FileId(1)));
    let alignment = matches[0]
        .evidence
        .iter()
        .find(|evidence| evidence.signal == "temporal_alignment")
        .context("alignment evidence")?;
    let offset = alignment
        .note
        .split(';')
        .next()
        .context("offset field")?
        .strip_prefix("offset_seconds=")
        .context("offset value")?
        .parse::<f64>()?;
    assert!((offset - 12.4).abs() <= 1.0, "{alignment:?}");
    ffmpeg(
        &[
            "-ss",
            "2",
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-t",
            "37",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "25",
        ],
        &clip,
    )?;
    let almost_full = [entry(&clip, 9)?, entry(&original, 1)?];
    let matches = eidolon_video::find(
        &almost_full,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::ClipOf, "{matches:#?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(9), FileId(1)));
    Ok(())
}

#[test]
fn finds_watermarked_version_without_embeddings() -> anyhow::Result<()> {
    let temporary = scratch()?;
    let original = source(temporary.path())?;
    let marked = temporary.path().join("watermark.mp4");
    ffmpeg(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "drawbox=x=235:y=8:w=70:h=28:color=white@0.8:t=fill",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
        ],
        &marked,
    )?;
    let files = [entry(&original, 0)?, entry(&marked, 1)?];
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find(
        &files,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Edited, "{matches:#?}");
    ffmpeg(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,18,20)'",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
        ],
        &marked,
    )?;
    let edited = [entry(&original, 0)?, entry(&marked, 1)?];
    let matches = eidolon_video::find(
        &edited,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Edited, "{matches:#?}");
    Ok(())
}

#[test]
fn finds_large_local_overlay_without_embeddings() -> anyhow::Result<()> {
    let temporary = scratch()?;
    let original = temporary.path().join("source.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=192x128:rate=8:ratio=0.4:seed=98213:death_color=0x204060:life_color=0xffd080",
            "-t",
            "20",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ],
        &original,
    )?;
    let marked = temporary.path().join("overlay.mp4");
    ffmpeg(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "drawbox=x=iw*0.1:y=ih*0.72:w=iw*0.8:h=ih*0.18:color=magenta@0.9:t=fill",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "24",
        ],
        &marked,
    )?;
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find(
        &[entry(&original, 0)?, entry(&marked, 1)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Edited, "{matches:#?}");
    Ok(())
}

#[test]
fn finds_letterboxed_reencoding_without_embeddings() -> anyhow::Result<()> {
    let temporary = scratch()?;
    let original = source(temporary.path())?;
    let boxed = temporary.path().join("boxed.mp4");
    ffmpeg(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "scale=240:108,pad=300:180:30:36:black",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "24",
        ],
        &boxed,
    )?;
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find(
        &[entry(&original, 0)?, entry(&boxed, 1)?],
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!(matches[0].relation, Relation::Reencoded, "{matches:#?}");
    Ok(())
}

#[test]
fn rejects_unrelated_sources_and_reversed_temporal_order() -> anyhow::Result<()> {
    let temporary = scratch()?;
    let original = source(temporary.path())?;
    let unrelated = temporary.path().join("different.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "life=size=320x180:rate=12:ratio=0.3:seed=12345",
            "-t",
            "40",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ],
        &unrelated,
    )?;
    let reversed = temporary.path().join("reversed.mp4");
    ffmpeg(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "reverse",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
        ],
        &reversed,
    )?;
    let files = [
        entry(&original, 0)?,
        entry(&unrelated, 1)?,
        entry(&reversed, 2)?,
    ];
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find(
        &files,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert!(matches.is_empty(), "{matches:#?}");
    Ok(())
}

#[test]
fn skips_audio_and_corruption_without_losing_valid_matches() -> anyhow::Result<()> {
    let temporary = scratch()?;
    let original = source(temporary.path())?;
    let smaller = temporary.path().join("small.mp4");
    ffmpeg(
        &[
            "-i",
            original.to_str().context("UTF-8 test path")?,
            "-vf",
            "scale=160:90",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
        ],
        &smaller,
    )?;
    let audio = temporary.path().join("audio.wav");
    ffmpeg(
        &["-f", "lavfi", "-i", "sine=frequency=440:duration=2"],
        &audio,
    )?;
    let corrupt = temporary.path().join("corrupt.mp4");
    std::fs::write(&corrupt, b"not a video container")?;
    let files = [
        entry(&audio, 3)?,
        entry(&original, 8)?,
        entry(&corrupt, 5)?,
        entry(&smaller, 2)?,
    ];
    let cache = FeatureCache::in_memory()?;
    let settings = config();
    let matches = eidolon_video::find(
        &files,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
    )?;
    assert_eq!(matches.len(), 1, "{matches:#?}");
    assert_eq!((matches[0].a, matches[0].b), (FileId(2), FileId(8)));
    assert_eq!(matches[0].relation, Relation::Reencoded);
    Ok(())
}
