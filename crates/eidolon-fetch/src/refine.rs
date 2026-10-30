use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, ensure};
use eidolon_core::{FileEntry, FrameRefiner, Kind, RefinedWindow};

use crate::http::{self, Session};
use crate::{CACHE_VERSION, Cached, FetchMode, Fetcher, save, windows};

#[derive(Default)]
struct Account {
    initial: u64,
    refined: u64,
    budget_exhausted: bool,
}

#[derive(Default)]
pub(crate) struct Transfers {
    sparse: AtomicBool,
    accounts: Mutex<HashMap<PathBuf, Arc<Mutex<Account>>>>,
}

impl Transfers {
    pub(crate) fn register(&self, path: &Path, bytes: u64) -> anyhow::Result<()> {
        let path = std::fs::canonicalize(path).context("canonicalizing remote artifact")?;
        let account = {
            let mut accounts = self
                .accounts
                .lock()
                .map_err(|_| anyhow::anyhow!("refinement registry poisoned"))?;
            if let Some(account) = accounts.get(&path) {
                Arc::clone(account)
            } else {
                let account = Arc::new(Mutex::new(Account::default()));
                accounts.insert(path, Arc::clone(&account));
                account
            }
        };
        let mut account = account
            .lock()
            .map_err(|_| anyhow::anyhow!("artifact transfer lock poisoned"))?;
        account.initial += bytes;
        self.sparse.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn account(&self, path: &Path) -> anyhow::Result<Option<Arc<Mutex<Account>>>> {
        let path = std::fs::canonicalize(path).context("canonicalizing remote artifact")?;
        Ok(self
            .accounts
            .lock()
            .map_err(|_| anyhow::anyhow!("refinement registry poisoned"))?
            .get(&path)
            .cloned())
    }
}

impl Fetcher {
    /// Whether this client has resolved at least one sparse remote video.
    #[must_use]
    pub fn has_sparse_videos(&self) -> bool {
        self.refinements.sparse.load(Ordering::Relaxed)
    }

    /// Additional response-body bytes fetched to refine this artifact, including failed attempts.
    ///
    /// Add this delta once to the corresponding [`crate::Remote::bytes_fetched`]
    /// after detector calls. [`Self::bytes_fetched`] already includes the delta.
    /// Canonical and symlink-alias paths identify the same artifact; other files return zero.
    ///
    /// # Errors
    /// Returns an error for an unreadable artifact path or poisoned transfer lock.
    pub fn refinement_bytes(&self, path: &Path) -> anyhow::Result<u64> {
        let Some(account) = self.refinements.account(path)? else {
            return Ok(0);
        };
        let account = account
            .lock()
            .map_err(|_| anyhow::anyhow!("artifact transfer lock poisoned"))?;
        Ok(account.refined)
    }

    /// Whether native refinement has been refused by this client's fractional byte budget.
    ///
    /// Once true, the flag remains true even when a later request fits the remaining
    /// allowance or uses cached bytes. Canonical and symlink-alias paths share the flag.
    /// Other files return false. This state resets for a new client.
    ///
    /// # Errors
    /// Returns an error for an unreadable artifact path or poisoned transfer lock.
    pub fn budget_exhausted(&self, path: &Path) -> anyhow::Result<bool> {
        let Some(account) = self.refinements.account(path)? else {
            return Ok(false);
        };
        let account = account
            .lock()
            .map_err(|_| anyhow::anyhow!("artifact transfer lock poisoned"))?;
        Ok(account.budget_exhausted)
    }
}

impl FrameRefiner for Fetcher {
    fn refine(
        &self,
        video: &FileEntry,
        center_s: f64,
        radius_s: f64,
    ) -> anyhow::Result<Option<RefinedWindow>> {
        ensure!(
            center_s.is_finite() && radius_s.is_finite() && radius_s >= 0.0,
            "refinement center/radius must be finite and radius nonnegative"
        );
        if video.kind != Kind::Video {
            return Ok(None);
        }
        let Some(account) = self.refinements.account(&video.path)? else {
            return Ok(None);
        };
        // Serialize this artifact's windows, sharing its byte budget and retained range blocks.
        let mut account = account
            .lock()
            .map_err(|_| anyhow::anyhow!("artifact transfer lock poisoned"))?;
        let path = std::fs::canonicalize(&video.path)?;
        let metadata_path = path.with_extension("json");
        let mut cached: Cached = serde_json::from_slice(&std::fs::read(&metadata_path)?)
            .context("reading sparse source provenance")?;
        if cached.version != CACHE_VERSION
            || cached.mode != FetchMode::Sparse
            || (cached.headers.etag.is_none() && cached.headers.last_modified.is_none())
        {
            return Ok(None);
        }
        let url = http::validate(&cached.headers.url)?;
        let spent = account.initial.saturating_add(account.refined);
        let fraction_budget = if self.options.max_fetch_fraction < 1.0 {
            let size = cached
                .headers
                .total_size
                .context("fractional refinement budget needs Content-Length")?;
            Some(((size as f64 * self.options.max_fetch_fraction) as u64).saturating_sub(spent))
        } else {
            None
        };
        // A previous refusal is provenance, not a terminal state: cached reads cost
        // nothing, and every new range still checks this remaining cumulative cap.
        let session = Session::new(&self.client, &self.options, &self.bytes)
            .with_transfer_budget(fraction_budget);
        let result = (|| {
            if !self.options.fresh || !cached.headers.is_fresh() {
                let (headers, unchanged) = session.head(&url, Some(&cached.headers))?;
                ensure!(
                    unchanged,
                    "remote source changed since sparse proxy was fetched"
                );
                cached.headers = headers;
                save(&metadata_path, &serde_json::to_vec(&cached)?)?;
            }
            let remaining = self
                .options
                .max_video_bytes
                .saturating_sub(spent.saturating_add(session.bytes.get()));
            if crate::is_matroska(&cached.headers.content_type, &url) {
                crate::matroska::materialize(
                    &session,
                    &url,
                    &cached.headers,
                    &path,
                    center_s,
                    radius_s,
                    remaining,
                )
            } else {
                windows::materialize(
                    &session,
                    &url,
                    &cached.headers,
                    &path,
                    center_s,
                    radius_s,
                    remaining,
                )
            }
            .map(Some)
        })();
        account.refined += session.bytes.get();
        if session.budget_exhausted.get() {
            account.budget_exhausted = true;
            return Ok(None);
        }
        result
    }
}

#[cfg(all(test, unix))]
mod tests {
    #[test]
    fn registration_through_a_symlink_shares_the_canonical_transfer_account() {
        let directory = tempfile::tempdir().expect("artifact directory");
        let source = directory.path().join("source");
        std::fs::create_dir(&source).expect("source directory");
        let artifact = source.join("proxy.mp4");
        std::fs::write(&artifact, b"proxy").expect("artifact");
        let alias = directory.path().join("alias");
        std::os::unix::fs::symlink(&source, &alias).expect("directory alias");
        let aliased_artifact = alias.join("proxy.mp4");
        let canonical = std::fs::canonicalize(&artifact).expect("canonical artifact");
        let transfers = super::Transfers::default();
        transfers
            .register(&aliased_artifact, 13)
            .expect("alias registration");
        let account = transfers
            .account(&canonical)
            .expect("canonical lookup")
            .expect("registered artifact");
        assert_eq!(account.lock().expect("transfer account").initial, 13);
        transfers
            .register(&canonical, 7)
            .expect("canonical registration");
        let aliased_account = transfers
            .account(&aliased_artifact)
            .expect("alias lookup")
            .expect("registered alias");
        assert_eq!(
            aliased_account.lock().expect("transfer account").initial,
            20
        );
        assert_eq!(account.lock().expect("shared transfer account").initial, 20);
    }
}
