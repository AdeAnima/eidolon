//! Recoverable per-input failures, distinct from cache/model/output failures.

use std::io::Write;
use std::path::PathBuf;

use eidolon_core::{FileEntry, InputError, Match};
use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct Skipped {
    pub path: PathBuf,
    pub reason: String,
}

pub(crate) fn find(
    files: &mut [FileEntry],
    skipped: &mut Vec<Skipped>,
    writer: &mut impl Write,
    detector: impl Fn(&[FileEntry]) -> anyhow::Result<Vec<Match>>,
) -> anyhow::Result<(Vec<Match>, usize)> {
    let mut active = files.len();
    loop {
        match detector(&files[..active]) {
            Ok(matches) => return Ok((matches, active)),
            Err(error) => {
                let input = error.downcast::<InputError>()?;
                // Already-extracted features remain cached across retries.
                active = exclude(files, active, input, skipped, writer)?;
            }
        }
    }
}

pub(crate) fn exclude(
    files: &mut [FileEntry],
    active: usize,
    input: InputError,
    skipped: &mut Vec<Skipped>,
    writer: &mut impl Write,
) -> anyhow::Result<usize> {
    let Some(index) = files[..active]
        .iter()
        .position(|file| file.id == input.file)
    else {
        return Err(input.into());
    };
    // Partition only this detector's slice; keep original kinds and IDs.
    let record = Skipped {
        path: files[index].path.clone(),
        reason: input.reason.replace(['\r', '\n'], " "),
    };
    writeln!(
        writer,
        "eidolon: skipped {}: {}",
        record.path.display(),
        record.reason
    )?;
    skipped.push(record);
    files.swap(index, active - 1);
    Ok(active - 1)
}

#[cfg(test)]
mod tests {
    use eidolon_core::{Config, Ctx, FileId, Kind, Relation, cache::FeatureCache};

    use super::*;

    #[test]
    fn unreadable_text_is_excluded_without_losing_valid_text_matches() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let mut files = Vec::new();
        for (id, text) in ["A short document.", "A SHORT DOCUMENT.", "Removed text."]
            .iter()
            .enumerate()
        {
            let path = dir.path().join(format!("{id}.txt"));
            std::fs::write(&path, text)?;
            files.push(FileEntry {
                id: FileId(id),
                path,
                size: text.len() as u64,
                mtime_ns: 0,
                kind: Kind::Text,
            });
        }
        std::fs::remove_file(&files[2].path)?;
        let config = Config {
            timings: false,
            remote_overlay: true,
            min_score: 0.5,
            model_dir: dir.path().into(),
            no_embeddings: true,
            no_keypoints: true,
            jobs: 1,
            fast: false,
            video_fps: 2.0,
        };
        let cache = FeatureCache::in_memory()?;
        let ctx = Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: None,
        };
        let mut skipped = Vec::new();
        let mut stderr = Vec::new();
        let (matches, valid) = find(&mut files, &mut skipped, &mut stderr, |files| {
            eidolon_text::find(files, &ctx)
        })?;
        assert_eq!(valid, 2);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].relation, Relation::SameContent);
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].path.ends_with("2.txt"));
        assert_eq!(std::str::from_utf8(&stderr)?.lines().count(), 1);
        Ok(())
    }

    #[test]
    fn run_wide_failure_is_not_skipped() {
        let mut skipped = Vec::new();
        let mut stderr = Vec::new();
        let error = find(&mut [], &mut skipped, &mut stderr, |_| {
            anyhow::bail!("cache database unavailable")
        })
        .expect_err("run-wide failure must propagate");
        assert_eq!(error.to_string(), "cache database unavailable");
        assert_eq!(skipped.len(), 0);
        assert_eq!(stderr, [] as [u8; 0]);
    }
}
