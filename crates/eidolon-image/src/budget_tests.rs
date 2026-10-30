//! Resource-budget checks use an actual decoded crop cohort rather than fabricated hashes.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use eidolon_core::{Config, Ctx, FileEntry, FileId, Kind, cache::FeatureCache};

#[test]
#[cfg(target_os = "macos")]
fn bounds_rgb_registration_candidates_for_real_crop_cohort() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-budget-")
        .tempdir_in("/tmp")?;
    let original = crate::decode(Path::new(
        "/System/Library/Desktop Pictures/.thumbnails/Sonoma.heic",
    ))?
    .to_rgb8();
    let cache = FeatureCache::in_memory()?;
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.80,
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: false,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    };
    let ctx = Ctx {
        refiner: None,
        config: &config,
        cache: &cache,
        embedder: None,
    };
    let mut features = Vec::new();
    let mut files = Vec::new();
    for index in 0..144_u32 {
        let percent = 50 + index % 50;
        let width = original.width() * percent / 100;
        let height = original.height() * percent / 100;
        let crop = image::imageops::crop_imm(
            &original,
            (original.width() - width) / 3,
            (original.height() - height) * 2 / 3,
            width,
            height,
        )
        .to_image();
        let path = temp.path().join(format!("crop-{index}.png"));
        crop.save(&path)?;
        let metadata = path.metadata()?;
        let file = FileEntry {
            id: FileId(index as usize),
            path,
            size: metadata.len(),
            mtime_ns: i128::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
            kind: Kind::Image,
        };
        features.push(crate::features::extract(&file, &ctx)?);
        files.push(file);
    }
    let mut candidates = super::candidates(&features, &[]);
    assert!(
        candidates.len() <= features.len() * 160,
        "hash-only geometry verification must have a linear neighbourhood budget"
    );
    let (_, features) = crate::features::prepare(&files, features, &mut candidates, &ctx)?;
    let dense = super::dense_candidates(&features, &candidates);
    assert!(
        !dense.is_empty(),
        "real crop cohort should retain RGB registration candidates"
    );
    assert!(
        dense.len() < candidates.len(),
        "dense registration must not inherit all hash candidates"
    );
    assert!(
        dense.len() <= features.len() * 8,
        "RGB work must have a linear per-image budget"
    );
    Ok(())
}
