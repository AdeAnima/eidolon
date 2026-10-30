//! Size/sample/full-hash exact duplicate detection across every media kind.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};

use anyhow::{Context as _, ensure};
use eidolon_core::cache::FeatureCache;
use eidolon_core::{Evidence, FileEntry, Match, Relation};
use rayon::prelude::*;

const SAMPLE_BYTES: usize = 16 * 1024;
const FULL_HASH_FEATURE: &str = "cli.blake3/v1";

pub(crate) fn find<'a>(
    files: impl IntoIterator<Item = &'a FileEntry>,
    cache: &FeatureCache,
) -> anyhow::Result<Vec<Match>> {
    let mut sizes: BTreeMap<u64, Vec<&FileEntry>> = BTreeMap::new();
    for file in files {
        sizes.entry(file.size).or_default().push(file);
    }
    let candidates: Vec<_> = sizes
        .into_values()
        .filter(|group| group.len() > 1)
        .flatten()
        .collect();
    let probes: Vec<_> = candidates
        .par_iter()
        .map(|file| Ok((file.size, sampled_hash(file)?, *file)))
        .collect::<anyhow::Result<_>>()?;
    let mut sampled: BTreeMap<(u64, [u8; 32]), Vec<&FileEntry>> = BTreeMap::new();
    for (size, hash, file) in probes {
        sampled.entry((size, hash)).or_default().push(file);
    }
    let collisions: Vec<_> = sampled
        .into_values()
        .filter(|group| group.len() > 1)
        .flatten()
        .collect();
    let hashes: Vec<_> = collisions
        .par_iter()
        .map(|file| Ok((full_hash(file, cache)?, *file)))
        .collect::<anyhow::Result<_>>()?;
    let mut identical: BTreeMap<[u8; 32], Vec<&FileEntry>> = BTreeMap::new();
    for (hash, file) in hashes {
        identical.entry(hash).or_default().push(file);
    }
    let mut matches = Vec::new();
    for (hash, mut group) in identical {
        group.sort_unstable_by_key(|file| file.id);
        for (index, a) in group.iter().enumerate() {
            for b in &group[index + 1..] {
                matches.push(Match {
                    a: a.id,
                    b: b.id,
                    relation: Relation::Identical,
                    score: 1.0,
                    evidence: vec![Evidence {
                        signal: "blake3",
                        value: 1.0,
                        note: format!("full-file BLAKE3 {}", blake3::Hash::from_bytes(hash)),
                    }],
                });
            }
        }
    }
    Ok(matches)
}

fn sampled_hash(file: &FileEntry) -> anyhow::Result<[u8; 32]> {
    let mut source =
        File::open(&file.path).with_context(|| format!("sample {}", file.path.display()))?;
    let mut buffer = [0; SAMPLE_BYTES];
    let first_len = usize::try_from(file.size.min(SAMPLE_BYTES as u64))?;
    source.read_exact(&mut buffer[..first_len])?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&buffer[..first_len]);
    if file.size > SAMPLE_BYTES as u64 {
        source.seek(SeekFrom::Start(file.size - SAMPLE_BYTES as u64))?;
        source.read_exact(&mut buffer)?;
        hasher.update(&buffer);
    }
    ensure_unchanged(file)?;
    Ok(*hasher.finalize().as_bytes())
}

fn full_hash(file: &FileEntry, cache: &FeatureCache) -> anyhow::Result<[u8; 32]> {
    let bytes = cache.get_or_compute(file, FULL_HASH_FEATURE, || {
        let mut source =
            File::open(&file.path).with_context(|| format!("hash {}", file.path.display()))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update_reader(&mut source)?;
        ensure_unchanged(file)?;
        Ok(hasher.finalize().as_bytes().to_vec())
    })?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid cached BLAKE3 for {}", file.path.display()))
}

fn ensure_unchanged(file: &FileEntry) -> anyhow::Result<()> {
    let metadata = std::fs::metadata(&file.path)?;
    ensure!(
        metadata.len() == file.size && crate::inputs::mtime_ns(&metadata)? == file.mtime_ns,
        "input changed during hashing: {}",
        file.path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use eidolon_core::{FileId, Kind};

    use super::*;

    fn file(path: &std::path::Path, id: usize, kind: Kind) -> anyhow::Result<FileEntry> {
        Ok(FileEntry {
            id: FileId(id),
            path: path.canonicalize()?,
            size: std::fs::metadata(path)?.len(),
            mtime_ns: crate::inputs::mtime_ns(&std::fs::metadata(path)?)?,
            kind,
        })
    }

    #[test]
    fn renamed_copies_match_but_equal_size_different_files_do_not() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let mut original = vec![b'a'; 100_000];
        original[50_000] = b'b';
        let mut different = original.clone();
        different[50_000] = b'c';
        let a = dir.path().join("one.txt");
        let b = dir.path().join("renamed.bin");
        let c = dir.path().join("different.txt");
        std::fs::write(&a, original)?;
        std::fs::copy(&a, &b)?;
        std::fs::write(&c, different)?;
        let files = vec![
            file(&a, 0, Kind::Text)?,
            file(&b, 1, Kind::Other)?,
            file(&c, 2, Kind::Text)?,
        ];
        let cache = FeatureCache::in_memory()?;
        let matches = find(&files, &cache)?;
        assert_eq!(matches.len(), 1);
        assert_eq!((matches[0].a, matches[0].b), (FileId(0), FileId(1)));
        assert_eq!(matches[0].relation, Relation::Identical);
        assert_eq!(matches[0].score, 1.0);
        assert!(!matches[0].evidence.is_empty());
        assert!(cache.get(&files[0], FULL_HASH_FEATURE)?.is_some());
        assert!(cache.get(&files[2], FULL_HASH_FEATURE)?.is_some());
        assert_eq!(find(&files, &cache)?.len(), 1);
        Ok(())
    }

    #[test]
    fn unique_sizes_do_not_need_full_hashes_and_empty_files_match() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let a = dir.path().join("empty-a");
        let b = dir.path().join("empty-b");
        let c = dir.path().join("unique");
        std::fs::write(&a, [])?;
        std::fs::write(&b, [])?;
        std::fs::write(&c, "a unique size")?;
        let files = vec![
            file(&a, 0, Kind::Text)?,
            file(&b, 1, Kind::Text)?,
            file(&c, 2, Kind::Text)?,
        ];
        let cache = FeatureCache::in_memory()?;
        assert_eq!(find(&files, &cache)?.len(), 1);
        assert!(cache.get(&files[2], FULL_HASH_FEATURE)?.is_none());
        Ok(())
    }
}
