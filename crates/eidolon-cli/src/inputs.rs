//! Filesystem and delimited-stdin input collection.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use walkdir::WalkDir;

pub(crate) fn read_paths(mut reader: impl BufRead, nul: bool) -> anyhow::Result<Vec<PathBuf>> {
    let delimiter = if nul { 0 } else { b'\n' };
    let mut paths = Vec::new();
    let mut record = Vec::new();
    while reader.read_until(delimiter, &mut record)? != 0 {
        if record.last() == Some(&delimiter) {
            record.pop();
        }
        if !record.is_empty() {
            let bytes = std::mem::take(&mut record);
            #[cfg(unix)]
            let name = {
                use std::os::unix::ffi::OsStringExt as _;
                OsString::from_vec(bytes)
            };
            #[cfg(not(unix))]
            let name = OsString::from(String::from_utf8(bytes)?);
            paths.push(PathBuf::from(name));
        }
        record.clear();
    }
    Ok(paths)
}

pub(crate) fn collect(roots: &[PathBuf], hidden: bool) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = BTreeSet::new();
    for root in roots {
        for entry in WalkDir::new(root)
            .follow_links(false)
            .follow_root_links(false)
            .into_iter()
            .filter_entry(|entry| hidden || !is_hidden(entry.path()))
        {
            let entry = entry.with_context(|| format!("walk {}", root.display()))?;
            if entry.file_type().is_file() {
                paths.insert(
                    entry.path().canonicalize().with_context(|| {
                        format!("resolve input path {}", entry.path().display())
                    })?,
                );
            }
        }
    }
    Ok(paths.into_iter().collect())
}

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name.as_encoded_bytes().starts_with(b"."))
}

pub(crate) fn entries(
    paths: impl IntoIterator<Item = PathBuf>,
) -> anyhow::Result<Vec<eidolon_core::FileEntry>> {
    paths
        .into_iter()
        .enumerate()
        .map(|(id, path)| {
            let metadata =
                std::fs::metadata(&path).with_context(|| format!("stat {}", path.display()))?;
            Ok(eidolon_core::FileEntry {
                id: eidolon_core::FileId(id),
                path,
                size: metadata.len(),
                mtime_ns: mtime_ns(&metadata)?,
                kind: eidolon_core::Kind::Other,
            })
        })
        .collect()
}

pub(crate) fn mtime_ns(metadata: &std::fs::Metadata) -> anyhow::Result<i128> {
    let modified = metadata.modified()?;
    match modified.duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => Ok(i128::try_from(duration.as_nanos())?),
        Err(error) => Ok(-i128::try_from(error.duration().as_nanos())?),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn folder_lines_and_nul_inputs_find_the_same_files() -> anyhow::Result<()> {
        let dir = tempfile::Builder::new().prefix("eidolon-").tempdir()?;
        std::fs::create_dir(dir.path().join("nested"))?;
        let first = dir.path().join("first.txt");
        let second = dir.path().join("nested/second.txt");
        std::fs::write(&first, "a full sentence to discover")?;
        std::fs::write(&second, "another sentence in a subdirectory")?;
        std::fs::write(dir.path().join(".hidden"), "hidden")?;
        std::fs::create_dir(dir.path().join(".secret"))?;
        std::fs::write(dir.path().join(".secret/inside"), "hidden too")?;
        let walked = collect(&[dir.path().to_path_buf()], false)?;
        let line_paths = read_paths(
            Cursor::new(format!("{}\n{}\n", first.display(), second.display())),
            false,
        )?;
        let nul_paths = read_paths(
            Cursor::new(format!("{}\0{}\0", first.display(), second.display())),
            true,
        )?;
        assert_eq!(walked, collect(&line_paths, false)?);
        assert_eq!(walked, collect(&nul_paths, false)?);
        assert_eq!(walked.len(), 2);
        assert_eq!(collect(&[dir.path().to_path_buf()], true)?.len(), 4);
        Ok(())
    }

    #[test]
    fn overlapping_roots_are_deduplicated() -> anyhow::Result<()> {
        let dir = tempfile::Builder::new().prefix("eidolon-").tempdir()?;
        let path = dir.path().join("one");
        std::fs::write(&path, "one file")?;
        assert_eq!(
            collect(&[path.clone(), dir.path().to_path_buf(), path], false)?.len(),
            1
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn skips_symlinks_and_preserves_newlines_in_nul_paths() -> anyhow::Result<()> {
        use std::os::unix::fs::symlink;

        let dir = tempfile::Builder::new().prefix("eidolon-").tempdir()?;
        let path = dir.path().join("odd😀\nname");
        std::fs::write(&path, "a filename containing a newline")?;
        symlink(&path, dir.path().join("alias"))?;
        let mut encoded = path.as_os_str().as_encoded_bytes().to_vec();
        encoded.push(0);
        let paths = read_paths(Cursor::new(encoded), true)?;
        assert_eq!(paths, vec![path.clone()]);
        assert_eq!(
            collect(&[dir.path().to_path_buf()], false)?,
            vec![path.canonicalize()?]
        );
        assert_eq!(
            collect(&[dir.path().join("alias")], false)?,
            Vec::<PathBuf>::new()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn nul_input_preserves_non_utf8_path_bytes() -> anyhow::Result<()> {
        use std::os::unix::ffi::OsStrExt as _;

        // APFS rejects invalid UTF-8 names; exercise the byte-preserving parser without creating one.
        let paths = read_paths(Cursor::new(b"/tmp/eidolon-odd\xff\0".as_slice()), true)?;
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].as_os_str().as_bytes(), b"/tmp/eidolon-odd\xff");
        Ok(())
    }
}
