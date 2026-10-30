//! Explicit URL resolution, provenance and transfer accounting for input files.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use eidolon_core::{FileEntry, FileId};
use eidolon_fetch::{FetchMode, FetchOptions, Fetcher, VideoMode};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct UrlInfo {
    pub(crate) url: String,
    pub(crate) page_url: Option<String>,
    pub(crate) role: String,
    pub(crate) mode: FetchMode,
    pub(crate) mode_reason: String,
    pub(crate) initial_bytes_fetched: u64,
    pub(crate) refinement_bytes_fetched: u64,
    pub(crate) bytes_fetched: u64,
    pub(crate) total_size: Option<u64>,
    pub(crate) budget_exhausted: bool,
    #[serde(skip)]
    pub(crate) complete_bytes: bool,
}

impl UrlInfo {
    fn finish_refinement(&mut self, bytes: u64) -> anyhow::Result<()> {
        self.bytes_fetched = self
            .initial_bytes_fetched
            .checked_add(bytes)
            .context("remote artifact byte count exceeds u64")?;
        self.refinement_bytes_fetched = bytes;
        if bytes > 0
            && !self
                .mode_reason
                .split("; ")
                .any(|reason| reason == "native_prefix_refinement")
        {
            self.mode_reason.push_str(if self.mode_reason.is_empty() {
                "native_prefix_refinement"
            } else {
                "; native_prefix_refinement"
            });
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub(crate) struct Sources {
    pub(crate) by_id: HashMap<FileId, UrlInfo>,
    pub(crate) transfer: TransferSummary,
}

#[derive(Debug)]
pub(crate) struct TransferSummary {
    pub(crate) requested_urls: usize,
    pub(crate) failed_urls: usize,
    pub(crate) bytes_fetched: u64,
    pub(crate) total_size: Option<u64>,
}

impl Default for TransferSummary {
    fn default() -> Self {
        Self {
            requested_urls: 0,
            failed_urls: 0,
            bytes_fetched: 0,
            total_size: Some(0),
        }
    }
}

pub(crate) struct ResolvedInputs {
    pub(crate) files: Vec<FileEntry>,
    pub(crate) sources: Sources,
    pub(crate) fetcher: Option<Fetcher>,
}

pub(crate) struct FileLabel<'a> {
    file: &'a FileEntry,
    origin: Option<&'a UrlInfo>,
}

impl fmt::Display for FileLabel<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(origin) = self.origin {
            write!(formatter, "{} [{}]", origin.url, origin.role)
        } else {
            write!(formatter, "{}", self.file.path.display())
        }
    }
}

impl Sources {
    pub(crate) fn eligible_for_exact(&self, id: FileId) -> bool {
        self.by_id
            .get(&id)
            .is_none_or(|origin| origin.complete_bytes)
    }

    pub(crate) fn label<'a>(&'a self, file: &'a FileEntry) -> FileLabel<'a> {
        FileLabel {
            file,
            origin: self.by_id.get(&file.id),
        }
    }

    /// Finalize counters and emit per-video transport notes after detector refinement.
    pub(crate) fn finish_transfers(
        &mut self,
        files: &[FileEntry],
        fetcher: Option<&Fetcher>,
        writer: &mut impl Write,
    ) -> anyhow::Result<()> {
        if let Some(fetcher) = fetcher {
            self.transfer.bytes_fetched = fetcher.bytes_fetched();
            for file in files
                .iter()
                .filter(|file| file.kind == eidolon_core::Kind::Video)
            {
                if let Some(origin) = self.by_id.get_mut(&file.id) {
                    origin.finish_refinement(fetcher.refinement_bytes(&file.path)?)?;
                    origin.budget_exhausted = fetcher.budget_exhausted(&file.path)?;
                    let mode = match origin.mode {
                        FetchMode::Sparse => "sparse",
                        FetchMode::Full => "full",
                        FetchMode::Derived => "derived",
                    };
                    writeln!(
                        writer,
                        "Remote video: {}; mode={mode}; reason={}; initial={} bytes; refinement={} bytes; total={} bytes",
                        origin.url,
                        origin.mode_reason,
                        origin.initial_bytes_fetched,
                        origin.refinement_bytes_fetched,
                        origin.bytes_fetched,
                    )?;
                    if origin.budget_exhausted {
                        writeln!(
                            writer,
                            "eidolon: remote video refinement budget exhausted for {}; native windows skipped; recall may be limited",
                            origin.url,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn write_summary(&self, writer: &mut impl Write) -> anyhow::Result<()> {
        let summary = &self.transfer;
        if summary.requested_urls == 0 {
            return Ok(());
        }
        write!(
            writer,
            "Remote transfer: {} bytes fetched / ",
            summary.bytes_fetched
        )?;
        if let Some(total) = summary.total_size {
            write!(writer, "{total} bytes total remote size")?;
        } else {
            write!(writer, "unknown total remote size")?;
        }
        writeln!(
            writer,
            "; {} failed URLs ({} URL inputs)",
            summary.failed_urls, summary.requested_urls
        )?;
        Ok(())
    }
}

pub(crate) fn is_url(path: &Path) -> bool {
    path.to_str().is_some_and(|value| {
        value
            .get(..7)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
            || value
                .get(..8)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"))
    })
}

fn partition_inputs(
    roots: Vec<PathBuf>,
    offline: bool,
) -> anyhow::Result<(Vec<PathBuf>, BTreeSet<String>)> {
    if offline {
        anyhow::ensure!(
            !roots.iter().any(|path| is_url(path)),
            "URL inputs are disabled by --offline"
        );
    }
    let mut local = Vec::new();
    let mut urls = BTreeSet::new();
    for root in roots {
        if is_url(&root) {
            urls.insert(
                root.into_os_string()
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("URL input is not valid UTF-8"))?,
            );
        } else {
            local.push(root);
        }
    }
    Ok((local, urls))
}

pub(crate) fn resolve(
    roots: Vec<PathBuf>,
    hidden: bool,
    offline: bool,
    video_mode: VideoMode,
    max_fetch_fraction: f64,
    fresh: bool,
    warnings: &mut impl Write,
) -> anyhow::Result<ResolvedInputs> {
    let (local, urls) = partition_inputs(roots, offline)?;
    let mut successful_input = !local.is_empty();
    let mut paths: BTreeMap<_, Option<UrlInfo>> = crate::inputs::collect(&local, hidden)?
        .into_iter()
        .map(|path| (path, None))
        .collect();
    let mut sources = Sources::default();
    sources.transfer.requested_urls = urls.len();
    let mut retained_fetcher = None;
    if !urls.is_empty() {
        let fetcher = Fetcher::with_options(FetchOptions {
            video_mode,
            max_fetch_fraction,
            fresh,
            ..FetchOptions::default()
        });
        for url in urls {
            let result = match &fetcher {
                Ok(fetcher) => fetcher.resolve(&url),
                Err(error) => Err(anyhow::anyhow!("{error:#}")),
            };
            let result = match result {
                Ok(result) => result,
                Err(error) => {
                    sources.transfer.failed_urls += 1;
                    sources.transfer.total_size = None;
                    writeln!(warnings, "eidolon: warning: failed URL {url}: {error:#}")?;
                    continue;
                }
            };
            for warning in result.warnings {
                writeln!(warnings, "eidolon: warning: {warning}")?;
            }
            if result.artifacts.is_empty() {
                sources.transfer.failed_urls += 1;
                sources.transfer.total_size = None;
                writeln!(
                    warnings,
                    "eidolon: warning: URL {url} produced no supported artifacts"
                )?;
                continue;
            }
            successful_input = true;
            sources.transfer.total_size = match (sources.transfer.total_size, result.total_size) {
                (Some(known), Some(additional)) => Some(
                    known
                        .checked_add(additional)
                        .context("total remote size exceeds u64")?,
                ),
                _ => None,
            };
            for artifact in result.artifacts {
                let path = artifact
                    .local_path
                    .canonicalize()
                    .with_context(|| format!("resolve fetched artifact for {}", artifact.url))?;
                let slot = paths.entry(path).or_default();
                if slot.is_none() {
                    *slot = Some(UrlInfo {
                        url: artifact.url,
                        page_url: artifact.page_url,
                        role: artifact.role,
                        mode: artifact.mode,
                        mode_reason: artifact.mode_reason,
                        initial_bytes_fetched: artifact.bytes_fetched,
                        refinement_bytes_fetched: 0,
                        bytes_fetched: artifact.bytes_fetched,
                        total_size: artifact.total_size,
                        budget_exhausted: artifact.budget_exhausted,
                        complete_bytes: artifact.complete_bytes,
                    });
                }
            }
        }
        if let Ok(fetcher) = fetcher {
            sources.transfer.bytes_fetched = fetcher.bytes_fetched();
            retained_fetcher = Some(fetcher);
        }
    }
    if !successful_input && sources.transfer.failed_urls > 0 {
        sources.write_summary(warnings)?;
        anyhow::bail!("all {} URL inputs failed", sources.transfer.failed_urls);
    }
    let files =
        crate::inputs::entries(paths.into_iter().enumerate().map(|(id, (path, origin))| {
            if let Some(origin) = origin {
                sources.by_id.insert(FileId(id), origin);
            }
            path
        }))?;
    Ok(ResolvedInputs {
        files,
        sources,
        fetcher: retained_fetcher,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refinement_counter_finalization_is_idempotent_and_preserves_acquisition()
    -> anyhow::Result<()> {
        let mut origin = UrlInfo {
            url: "https://example.test/video.mp4".into(),
            page_url: None,
            role: "direct".into(),
            mode: FetchMode::Sparse,
            mode_reason: "indexed_keyframes".into(),
            initial_bytes_fetched: 120,
            refinement_bytes_fetched: 0,
            bytes_fetched: 120,
            total_size: Some(1000),
            budget_exhausted: false,
            complete_bytes: false,
        };
        let initial = serde_json::to_value(&origin)?;
        origin.finish_refinement(0)?;
        assert_eq!(serde_json::to_value(&origin)?, initial);
        origin.finish_refinement(80)?;
        let once = serde_json::to_value(&origin)?;
        origin.finish_refinement(80)?;
        assert_eq!(serde_json::to_value(&origin)?, once);
        assert_eq!(origin.mode, FetchMode::Sparse);
        assert_eq!(
            origin.bytes_fetched,
            origin.initial_bytes_fetched + origin.refinement_bytes_fetched
        );
        assert_eq!(
            origin
                .mode_reason
                .matches("native_prefix_refinement")
                .count(),
            1
        );
        Ok(())
    }

    #[test]
    fn offline_rejects_any_url_before_local_input_processing() -> anyhow::Result<()> {
        let mut warnings = Vec::new();
        let error = resolve(
            vec![
                "/tmp/eidolon-nonexistent-local-input".into(),
                "https://example.invalid/image.png".into(),
            ],
            false,
            true,
            VideoMode::Keyframes,
            0.25,
            false,
            &mut warnings,
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("offline URL unexpectedly accepted"))?;
        assert!(
            error
                .to_string()
                .contains("URL inputs are disabled by --offline")
        );
        assert_eq!(warnings, Vec::<u8>::new());
        Ok(())
    }

    #[test]
    fn offline_still_collects_local_files_without_remote_metadata() -> anyhow::Result<()> {
        let dir = tempfile::Builder::new()
            .prefix("eidolon-local-")
            .tempdir_in("/tmp")?;
        let path = dir.path().join("document.txt");
        std::fs::write(&path, "A local document never needs a network request.")?;
        let inputs = resolve(
            vec![path.clone()],
            false,
            true,
            VideoMode::Keyframes,
            0.25,
            false,
            &mut Vec::new(),
        )?;
        assert_eq!(inputs.files.len(), 1);
        assert_eq!(inputs.files[0].path, path.canonicalize()?);
        assert_eq!(inputs.sources.by_id.len(), 0);
        assert_eq!(inputs.sources.transfer.requested_urls, 0);
        assert!(inputs.fetcher.is_none());
        Ok(())
    }

    #[test]
    fn exact_stage_excludes_derived_artifacts_but_keeps_complete_downloads() -> anyhow::Result<()> {
        let dir = tempfile::Builder::new()
            .prefix("eidolon-exact-remote-")
            .tempdir_in("/tmp")?;
        let mut paths = Vec::new();
        for name in ["local.txt", "derived.txt", "complete.txt"] {
            let path = dir.path().join(name);
            std::fs::write(
                &path,
                "The same local artifact bytes do not prove equal remote resources.",
            )?;
            paths.push(path.canonicalize()?);
        }
        let files = crate::inputs::entries(paths)?;
        let mut sources = Sources::default();
        for id in [1, 2] {
            sources.by_id.insert(
                FileId(id),
                UrlInfo {
                    url: format!("https://example.test/resource-{id}"),
                    page_url: None,
                    role: "download".into(),
                    mode: if id == 2 {
                        FetchMode::Full
                    } else {
                        FetchMode::Derived
                    },
                    mode_reason: if id == 2 {
                        "complete_resource"
                    } else {
                        "article_text_extraction"
                    }
                    .into(),
                    initial_bytes_fetched: files[id].size,
                    refinement_bytes_fetched: 0,
                    bytes_fetched: files[id].size,
                    total_size: Some(files[id].size),
                    budget_exhausted: false,
                    complete_bytes: id == 2,
                },
            );
        }
        let cache = eidolon_core::cache::FeatureCache::in_memory()?;
        let matches = crate::exact::find(
            files
                .iter()
                .filter(|file| sources.eligible_for_exact(file.id)),
            &cache,
        )?;
        assert_eq!(matches.len(), 1);
        assert_eq!((matches[0].a, matches[0].b), (FileId(0), FileId(2)));
        Ok(())
    }

    #[test]
    fn url_detection_does_not_reinterpret_local_paths() {
        assert!(is_url(Path::new("https://example.test/media.png")));
        assert!(is_url(Path::new("HTTP://example.test/video.mp4")));
        assert!(!is_url(Path::new("./https://local/file")));
        assert!(!is_url(Path::new("/tmp/eidolon-https-not-a-url")));
    }
}
