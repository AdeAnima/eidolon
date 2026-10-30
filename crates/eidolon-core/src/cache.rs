//! Persistent feature cache keyed by file identity and feature version.
//!
//! An entry is valid while the file's path, size and mtime are unchanged.
//! Values are opaque bytes; each detector owns its serialization and bumps
//! its feature name (e.g. `"phash/v2"`) when the format changes.

use std::path::Path;
use std::sync::Mutex;

use anyhow::Context as _;
use rusqlite::{Connection, OptionalExtension, params};

use crate::FileEntry;

/// SQLite-backed cache, safe to share across threads.
pub struct FeatureCache {
    conn: Mutex<Connection>,
}

impl FeatureCache {
    /// Open or create the cache database at `path`.
    ///
    /// # Errors
    /// Fails if the database cannot be opened or its schema created.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create cache dir {}", parent.display()))?;
        }
        let conn =
            Connection::open(path).with_context(|| format!("open cache {}", path.display()))?;
        Self::init(conn)
    }

    /// In-memory cache for tests and `--no-cache` runs.
    ///
    /// # Errors
    /// Fails if `SQLite` cannot create the schema.
    pub fn in_memory() -> anyhow::Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> anyhow::Result<Self> {
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS features (
                 path TEXT NOT NULL,
                 size INTEGER NOT NULL,
                 mtime_ns TEXT NOT NULL,
                 feature TEXT NOT NULL,
                 value BLOB NOT NULL,
                 PRIMARY KEY (path, feature)
             );",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Cached value of `feature` for `file`, if present and still fresh.
    ///
    /// # Errors
    /// Fails on database errors.
    pub fn get(&self, file: &FileEntry, feature: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let conn = self.lock();
        let value = conn
            .query_row(
                "SELECT value FROM features
                 WHERE path = ?1 AND feature = ?2 AND size = ?3 AND mtime_ns = ?4",
                params![
                    path_key(&file.path),
                    feature,
                    file.size.cast_signed(),
                    file.mtime_ns.to_string()
                ],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value)
    }

    /// Check whether `feature` is fresh for `file`, without copying its value.
    ///
    /// # Errors
    /// Fails on database errors.
    pub fn contains(&self, file: &FileEntry, feature: &str) -> anyhow::Result<bool> {
        Ok(self.lock().query_row(
            "SELECT EXISTS(SELECT 1 FROM features
             WHERE path = ?1 AND feature = ?2 AND size = ?3 AND mtime_ns = ?4)",
            params![
                path_key(&file.path),
                feature,
                file.size.cast_signed(),
                file.mtime_ns.to_string()
            ],
            |row| row.get(0),
        )?)
    }

    /// Store `value` as `feature` for `file`, replacing any older entry.
    ///
    /// # Errors
    /// Fails on database errors.
    pub fn put(&self, file: &FileEntry, feature: &str, value: &[u8]) -> anyhow::Result<()> {
        self.lock().execute(
            "INSERT OR REPLACE INTO features (path, size, mtime_ns, feature, value)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                path_key(&file.path),
                file.size.cast_signed(),
                file.mtime_ns.to_string(),
                feature,
                value
            ],
        )?;
        Ok(())
    }

    /// Return the cached value or compute, store and return it.
    ///
    /// # Errors
    /// Propagates errors from `compute` and the database.
    pub fn get_or_compute(
        &self,
        file: &FileEntry,
        feature: &str,
        compute: impl FnOnce() -> anyhow::Result<Vec<u8>>,
    ) -> anyhow::Result<Vec<u8>> {
        if let Some(hit) = self.get(file, feature)? {
            return Ok(hit);
        }
        let value = compute()?;
        self.put(file, feature, &value)?;
        Ok(value)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        // A panic while holding the lock leaves SQLite consistent; keep going.
        self.conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn path_key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FileId, Kind};

    fn entry(path: &Path, size: u64, mtime_ns: i128) -> FileEntry {
        FileEntry {
            id: FileId(0),
            path: path.to_path_buf(),
            size,
            mtime_ns,
            kind: Kind::Other,
        }
    }

    #[test]
    fn contains_only_fresh_requested_features() -> anyhow::Result<()> {
        let cache = FeatureCache::in_memory()?;
        let original = entry(Path::new("/x"), 10, 1);
        assert!(!cache.contains(&original, "f/v1")?);
        cache.put(&original, "f/v1", b"old")?;
        assert!(cache.contains(&original, "f/v1")?);
        assert!(!cache.contains(&entry(Path::new("/x"), 10, 2), "f/v1")?);
        assert!(!cache.contains(&entry(Path::new("/x"), 11, 1), "f/v1")?);
        assert!(!cache.contains(&entry(Path::new("/y"), 10, 1), "f/v1")?);
        assert!(!cache.contains(&original, "f/v2")?);
        Ok(())
    }

    #[test]
    fn changed_mtime_or_size_invalidates_entry() -> anyhow::Result<()> {
        let cache = FeatureCache::in_memory()?;
        let original = entry(Path::new("/x"), 10, 1);
        cache.put(&original, "f/v1", b"old")?;

        assert_eq!(cache.get(&original, "f/v1")?.as_deref(), Some(&b"old"[..]));
        assert_eq!(cache.get(&entry(Path::new("/x"), 10, 2), "f/v1")?, None);
        assert_eq!(cache.get(&entry(Path::new("/x"), 11, 1), "f/v1")?, None);
        assert_eq!(cache.get(&original, "f/v2")?, None);
        Ok(())
    }

    #[test]
    fn get_or_compute_computes_once() -> anyhow::Result<()> {
        let cache = FeatureCache::in_memory()?;
        let file = entry(Path::new("/y"), 1, 1);
        let mut calls = 0;
        for _ in 0..2 {
            cache.get_or_compute(&file, "f/v1", || {
                calls += 1;
                Ok(vec![7])
            })?;
        }
        assert_eq!(calls, 1);
        Ok(())
    }
}
