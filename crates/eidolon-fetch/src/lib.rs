//! Bounded, revalidated HTTP artifacts for report-only duplicate detection.
//!
//! Constructing a [`Fetcher`] never contacts the network. [`Fetcher::resolve`]
//! and its [`eidolon_core::FrameRefiner`] implementation fetch only explicit URL inputs.
//! The caller classifies returned local files and handles each input error.

mod fragment;
mod fragment_video;
mod html;
mod http;
mod matroska;
mod platform;
mod refine;
mod transport;
mod video;
mod windows;

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context as _, ensure};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use url::Url;

use http::{Headers, Session};

const CACHE_VERSION: u8 = 4;

/// Remote-video transfer policy; neither mode ever fetches the audio track of indexed MP4s.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum VideoMode {
    /// Copy at most 1,200 independent sync samples, preserving original source timestamps.
    #[default]
    Keyframes,
    /// Copy every video packet so the detector can sample its full two-fps cadence.
    ///
    /// Long GOP dependencies often require nearly the whole video track; this is
    /// the explicit accuracy/transfer comparison, not a promise of 25% transfer.
    FullCadence,
}

/// How an artifact was acquired, before optional native-cadence refinement.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FetchMode {
    /// An indexed keyframe proxy; subsequent refinement may fetch additional source bytes.
    Sparse,
    /// A complete resource download or an explicitly requested full-cadence video-track proxy.
    Full,
    /// Article text extracted from a downloaded HTML page.
    Derived,
}

/// Limits and cache location used by a [`Fetcher`].
#[derive(Clone, Debug)]
pub struct FetchOptions {
    /// Persistent source bytes, derived artifacts and validators. Defaults to the user's cache directory.
    pub cache_dir: PathBuf,
    /// Maximum complete image response, in bytes. Default: 32 MiB.
    pub max_image_bytes: u64,
    /// Maximum HTML, text or platform-discovery response, in bytes. Default: 4 MiB.
    pub max_page_bytes: u64,
    /// Maximum bytes fetched for one video, including unsuccessful range attempts. Default: 64 MiB.
    pub max_video_bytes: u64,
    /// Maximum initial-plus-refinement body bytes as a fraction of the source size.
    ///
    /// Native refinement stops before exceeding this allowance. Initial acquisition
    /// is unchanged and can already exceed it. Default: 0.25; 1.0 disables this
    /// fractional limit, but [`Self::max_video_bytes`] always applies.
    pub max_fetch_fraction: f64,
    /// Per-request connection/body timeout. Default: 30 seconds.
    pub timeout: Duration,
    /// Total URL-resolution deadline, checked between requests and body reads. Default: 180 seconds.
    pub resolve_timeout: Duration,
    /// Maximum artifacts discovered in one page. Default: 32.
    pub max_artifacts: usize,
    /// Sparse keyframe proxy or full-cadence video-track transfer.
    pub video_mode: VideoMode,
    /// Reuse cached sources while their HTTP freshness window remains valid.
    /// Off by default; without Cache-Control, enabled reuse lasts 30 seconds.
    pub fresh: bool,
}

impl Default for FetchOptions {
    fn default() -> Self {
        let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
        let cache = if cfg!(target_os = "macos") {
            home.join("Library/Caches")
        } else {
            std::env::var_os("XDG_CACHE_HOME").map_or_else(|| home.join(".cache"), PathBuf::from)
        };
        Self {
            cache_dir: cache.join("eidolon/remote"),
            max_image_bytes: 32 * 1024 * 1024,
            max_page_bytes: 4 * 1024 * 1024,
            max_video_bytes: 64 * 1024 * 1024,
            max_fetch_fraction: 0.25,
            timeout: Duration::from_secs(30),
            resolve_timeout: Duration::from_secs(180),
            max_artifacts: 32,
            video_mode: VideoMode::Keyframes,
            fresh: false,
        }
    }
}

/// One normal local detector input, together with its remote provenance.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Remote {
    /// Actual media URL after redirects, or the article's page URL for text.
    pub url: String,
    /// Enclosing page URL, absent for direct media inputs.
    pub page_url: Option<String>,
    /// Discovery role, such as `direct`, `og:image`, `video thumbnail` or `article text`.
    pub role: String,
    /// Complete local image/text, or an ordinary timeline-preserving video proxy.
    pub local_path: PathBuf,
    /// Response-body bytes fetched for this artifact during this resolution; cached artifacts use zero.
    pub bytes_fetched: u64,
    /// Original remote representation size, if the origin supplies it.
    pub total_size: Option<u64>,
    /// True only for an unmodified complete resource with its known original byte length.
    /// Derived video proxies and extracted article text are always false.
    pub complete_bytes: bool,
    /// Acquisition mode; sparse proxies remain sparse even if native refinement fetches the source.
    pub mode: FetchMode,
    /// Why this acquisition mode was selected, retained across cache hits.
    pub mode_reason: String,
    /// Whether native refinement was refused by this client's fractional transfer budget.
    ///
    /// Refresh after detection through [`Fetcher::budget_exhausted`]. The flag is
    /// run-local: retained source bytes are not charged to a new client.
    pub budget_exhausted: bool,
}

/// Artifacts, nonfatal discovery failures and non-double-counted transfer totals.
#[derive(Debug)]
pub struct FetchResult {
    /// Unique media artifacts and optional extracted article text.
    pub artifacts: Vec<Remote>,
    /// Actual HTTP response-body bytes, including page/probe overhead and failed discoveries.
    pub bytes_fetched: u64,
    /// Sum of distinct source sizes, including HTML and failed discoveries; absent if any size is unknown.
    pub total_size: Option<u64>,
    /// Subresource failures, bounded-full-download notices and platform-accounting limitations.
    pub warnings: Vec<String>,
}

/// A reusable blocking HTTP client with a persistent, URL-keyed artifact cache.
pub struct Fetcher {
    client: Client,
    options: FetchOptions,
    bytes: Arc<AtomicU64>,
    refinements: refine::Transfers,
}

#[derive(Deserialize, Serialize)]
struct Cached {
    version: u8,
    headers: Headers,
    kind: StoredKind,
    extension: String,
    mode: FetchMode,
    mode_reason: String,
    complete_bytes: bool,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
enum StoredKind {
    Image,
    Video,
    Html,
    Text,
    Other,
}

struct Download {
    headers: Headers,
    path: PathBuf,
    kind: StoredKind,
    bytes: u64,
    complete_bytes: bool,
    mode: FetchMode,
    mode_reason: String,
}

impl Fetcher {
    /// Open the default cache and create an idle client; no network request is made.
    ///
    /// # Errors
    /// Returns an error if the cache directory or TLS client cannot be created.
    pub fn new() -> anyhow::Result<Self> {
        Self::with_options(FetchOptions::default())
    }

    /// Create an idle client with explicit limits and cache location.
    ///
    /// # Errors
    /// Rejects zero limits/timeouts, nonfinite or out-of-range fractions, and
    /// reports cache or HTTP-client initialization failures.
    pub fn with_options(mut options: FetchOptions) -> anyhow::Result<Self> {
        ensure!(
            options.max_image_bytes > 0
                && options.max_page_bytes > 0
                && options.max_video_bytes > 0
                && options.max_artifacts > 0,
            "fetch limits must be positive"
        );
        ensure!(
            !options.timeout.is_zero() && !options.resolve_timeout.is_zero(),
            "fetch timeouts must be positive"
        );
        ensure!(
            options.max_fetch_fraction.is_finite()
                && options.max_fetch_fraction > 0.0
                && options.max_fetch_fraction <= 1.0,
            "max_fetch_fraction must be finite and greater than 0 and at most 1"
        );
        std::fs::create_dir_all(&options.cache_dir).context("creating remote cache")?;
        options.cache_dir =
            std::fs::canonicalize(&options.cache_dir).context("canonicalizing remote cache")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&options.cache_dir, std::fs::Permissions::from_mode(0o700))?;
        }
        let client = Client::builder()
            .user_agent(format!("eidolon/{}", env!("CARGO_PKG_VERSION")))
            .connect_timeout(options.timeout.min(Duration::from_secs(10)))
            .timeout(options.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            client,
            options,
            bytes: Arc::new(AtomicU64::new(0)),
            refinements: refine::Transfers::default(),
        })
    }

    /// Cumulative HTTP response-body bytes, including partial transfers from failed resolutions.
    ///
    /// `yt-dlp` discovery traffic is not observable by this client; resolutions using
    /// that external tool explicitly report this limitation in their warnings.
    #[must_use]
    pub fn bytes_fetched(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Resolve one public HTTP(S) URL into local detector inputs, revalidating cached sources.
    ///
    /// Failed page discoveries are returned as warnings while other artifacts continue.
    /// A direct-input failure is an error; callers should print it and continue other inputs.
    /// No cookies, login credentials or browser state are read or sent.
    ///
    /// # Errors
    /// Reports unsupported URL schemes, HTTP failures, limits, invalid media indexes,
    /// missing external tools, or pages with no usable media or article text.
    pub fn resolve(&self, value: &str) -> anyhow::Result<FetchResult> {
        let url = http::validate(value)?;
        let session = Session::new(&self.client, &self.options, &self.bytes);
        let mut warnings = Vec::new();
        let mut artifacts = if platform::is_platform(&url) {
            self.platform(&session, &url, &mut warnings)?
        } else {
            let download = self.download(&session, &url, &mut warnings)?;
            if download.kind == StoredKind::Html {
                self.page(&session, &download, &mut warnings)?
            } else {
                vec![remote(download, None, "direct")]
            }
        };
        let mut seen = HashSet::new();
        artifacts.retain(|artifact| seen.insert(artifact.url.clone()));
        for artifact in &mut artifacts {
            if artifact.mode == FetchMode::Sparse {
                artifact.budget_exhausted = self.budget_exhausted(&artifact.local_path)?;
            }
        }
        ensure!(
            !artifacts.is_empty(),
            "URL produced no usable media or article text"
        );
        let total_size = session
            .sizes
            .borrow()
            .values()
            .try_fold(0_u64, |sum, size| sum.checked_add((*size)?));
        Ok(FetchResult {
            artifacts,
            bytes_fetched: session.bytes.get(),
            total_size,
            warnings,
        })
    }

    fn load_cached(&self, key: &str, metadata_path: &Path) -> Option<Cached> {
        std::fs::read(metadata_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .filter(|cached: &Cached| {
                cached.version == CACHE_VERSION
                    && !cached.extension.is_empty()
                    && cached.extension.chars().all(|c| c.is_ascii_alphanumeric())
                    && self
                        .options
                        .cache_dir
                        .join(format!("{key}.{}", cached.extension))
                        .is_file()
                    && (cached.mode != FetchMode::Sparse
                        || self
                            .options
                            .cache_dir
                            .join(format!("{key}.{}.remote.json", cached.extension))
                            .is_file())
            })
    }

    fn reuse(
        &self,
        session: &Session<'_>,
        key: &str,
        mut cached: Cached,
        headers: Headers,
        bytes: u64,
    ) -> anyhow::Result<Download> {
        if cached.headers.fresh_until != headers.fresh_until
            || cached.headers.fresh_for != headers.fresh_for
        {
            cached.headers = headers.clone();
            save(
                &self.options.cache_dir.join(format!("{key}.json")),
                &serde_json::to_vec(&cached)?,
            )?;
        }
        session.note_size(&headers.url, headers.total_size);
        let path = self
            .options
            .cache_dir
            .join(format!("{key}.{}", cached.extension));
        if cached.mode == FetchMode::Sparse {
            self.refinements.register(&path, bytes)?;
        }
        Ok(Download {
            path,
            headers,
            kind: cached.kind,
            bytes,
            complete_bytes: cached.complete_bytes,
            mode: cached.mode,
            mode_reason: cached.mode_reason,
        })
    }

    fn complete_limit(&self, kind: StoredKind, warnings: &mut Vec<String>) -> u64 {
        match kind {
            StoredKind::Video => {
                warnings
                    .push("non-indexed video: downloading in full under the size cap".to_owned());
                self.options.max_video_bytes
            }
            StoredKind::Html | StoredKind::Text => self.options.max_page_bytes,
            StoredKind::Image | StoredKind::Other => self.options.max_image_bytes,
        }
    }

    fn download(
        &self,
        session: &Session<'_>,
        url: &Url,
        warnings: &mut Vec<String>,
    ) -> anyhow::Result<Download> {
        session.note_size(url.as_str(), None);
        let before = session.bytes.get();
        let key = digest(&format!("{}:{:?}", url, self.options.video_mode));
        let metadata_path = self.options.cache_dir.join(format!("{key}.json"));
        let cached = self.load_cached(&key, &metadata_path);
        let (headers, unchanged) = if let Some(cached) = &cached
            && self.options.fresh
            && cached.headers.is_fresh()
        {
            (cached.headers.clone(), true)
        } else {
            session.head(url, cached.as_ref().map(|cached| &cached.headers))?
        };
        session.note_headers(url, &headers);
        if unchanged && let Some(cached) = cached {
            return self.reuse(session, &key, cached, headers, session.bytes.get() - before);
        }
        let (mut headers, kind) = sniff_headers(session, url, headers)?;
        let mut mode = FetchMode::Full;
        let mut complete_bytes = false;
        let extension;
        let path;
        let mode_reason;
        if kind == StoredKind::Video
            && (is_mp4(&headers.content_type, url) || is_matroska(&headers.content_type, url))
        {
            let matroska = is_matroska(&headers.content_type, url);
            extension = if matroska { "mkv" } else { "mp4" }.to_owned();
            path = self.options.cache_dir.join(format!("{key}.{extension}"));
            let remaining = self
                .options
                .max_video_bytes
                .saturating_sub(session.bytes.get() - before);
            let proxy = if matroska {
                matroska::proxy(session, url, &headers, &path, remaining)
            } else {
                video::proxy(session, url, &headers, &path, remaining)
            };
            match proxy {
                Ok(()) => {
                    if self.options.video_mode == VideoMode::Keyframes {
                        mode = FetchMode::Sparse;
                        mode_reason = "indexed_keyframes".to_owned();
                    } else {
                        mode_reason = "requested_full_cadence".to_owned();
                    }
                }
                Err(error) => {
                    mode_reason = format!("range proxy unavailable ({error:#})");
                    warnings.push(format!(
                        "{mode_reason}; downloading video in full under the size cap"
                    ));
                    let remaining = self
                        .options
                        .max_video_bytes
                        .saturating_sub(session.bytes.get() - before);
                    (headers, complete_bytes) =
                        complete_download(session, url, &headers, &path, remaining)?;
                    let _ = std::fs::remove_file(video::sidecar(&path));
                }
            }
        } else {
            mode_reason = if kind == StoredKind::Video {
                "non_indexed_video"
            } else {
                "complete_resource"
            }
            .to_owned();
            extension = extension_for(kind, url, &headers.content_type).to_owned();
            path = self.options.cache_dir.join(format!("{key}.{extension}"));
            let limit = self.complete_limit(kind, warnings);
            (headers, complete_bytes) = complete_download(session, url, &headers, &path, limit)?;
        }
        session.note_size(&headers.url, headers.total_size);
        let cached = Cached {
            version: CACHE_VERSION,
            headers: headers.clone(),
            kind,
            extension,
            mode,
            mode_reason,
            complete_bytes,
        };
        save(&metadata_path, &serde_json::to_vec(&cached)?)?;
        if mode == FetchMode::Sparse {
            self.refinements
                .register(&path, session.bytes.get() - before)?;
        }
        Ok(Download {
            headers,
            path,
            kind,
            bytes: session.bytes.get() - before,
            complete_bytes,
            mode,
            mode_reason: cached.mode_reason,
        })
    }

    fn page(
        &self,
        session: &Session<'_>,
        download: &Download,
        warnings: &mut Vec<String>,
    ) -> anyhow::Result<Vec<Remote>> {
        let page = http::validate(&download.headers.url)?;
        let bytes = std::fs::read(&download.path)?;
        let source = String::from_utf8_lossy(&bytes);
        let extracted = html::extract(&source, &page);
        let mut artifacts = Vec::new();
        if !extracted.text.is_empty() {
            let path = download.path.with_extension("article.txt");
            save(&path, extracted.text.as_bytes())?;
            artifacts.push(Remote {
                url: page.to_string(),
                page_url: Some(page.to_string()),
                role: "article text".to_owned(),
                local_path: path,
                bytes_fetched: download.bytes,
                total_size: download.headers.total_size,
                complete_bytes: false,
                mode: FetchMode::Derived,
                mode_reason: "article_text_extraction".to_owned(),
                budget_exhausted: false,
            });
        }
        if extracted.media.len() + artifacts.len() > self.options.max_artifacts {
            warnings.push(format!(
                "page discovery limited to {} artifacts",
                self.options.max_artifacts
            ));
        }
        for media in extracted
            .media
            .into_iter()
            .take(self.options.max_artifacts.saturating_sub(artifacts.len()))
        {
            match self.download(session, &media.url, warnings) {
                Ok(download) if download.kind != StoredKind::Html => {
                    artifacts.push(remote(download, Some(page.to_string()), &media.role));
                }
                Ok(_) => {
                    warnings.push(format!("{}: discovered media URL returned HTML", media.url));
                }
                Err(error) => warnings.push(format!("{}: {error:#}", media.url)),
            }
        }
        Ok(artifacts)
    }

    fn platform(
        &self,
        session: &Session<'_>,
        page: &Url,
        warnings: &mut Vec<String>,
    ) -> anyhow::Result<Vec<Remote>> {
        let media = platform::discover(page, &self.options)?;
        warnings.push("yt-dlp discovery traffic is not included in HTTP media-byte accounting; no cookies or login used".to_owned());
        let mut artifacts = Vec::new();
        for (url, role) in media {
            match self.download(session, &url, warnings) {
                Ok(download) if download.kind != StoredKind::Html => {
                    artifacts.push(remote(download, Some(page.to_string()), role));
                }
                Ok(_) => warnings.push(format!("{url}: platform media URL returned HTML")),
                Err(error) => warnings.push(format!("{url}: {error:#}")),
            }
        }
        Ok(artifacts)
    }
}

fn remote(download: Download, page_url: Option<String>, role: &str) -> Remote {
    Remote {
        url: download.headers.url,
        page_url,
        role: role.to_owned(),
        local_path: download.path,
        bytes_fetched: download.bytes,
        total_size: download.headers.total_size,
        complete_bytes: download.complete_bytes,
        mode: download.mode,
        mode_reason: download.mode_reason,
        budget_exhausted: false,
    }
}

fn identify(content_type: &str, url: &Url) -> StoredKind {
    let content_type = content_type.to_ascii_lowercase();
    let extension = url
        .path()
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if content_type.starts_with("image/") {
        StoredKind::Image
    } else if content_type.starts_with("video/")
        || matches!(
            extension.as_str(),
            "mp4" | "m4v" | "mov" | "webm" | "ogv" | "ogg" | "mkv" | "avi" | "mpeg" | "mpg"
        )
    {
        StoredKind::Video
    } else if content_type.contains("html") || matches!(extension.as_str(), "html" | "htm") {
        StoredKind::Html
    } else if content_type.starts_with("text/") || matches!(extension.as_str(), "txt" | "md") {
        StoredKind::Text
    } else if matches!(
        extension.as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "avif" | "bmp" | "tif" | "tiff" | "heic"
    ) {
        StoredKind::Image
    } else {
        StoredKind::Other
    }
}

fn is_mp4(content_type: &str, url: &Url) -> bool {
    content_type.starts_with("video/mp4")
        || content_type.starts_with("video/quicktime")
        || matches!(url.path().rsplit('.').next(), Some("mp4" | "m4v" | "mov"))
}

fn is_matroska(content_type: &str, url: &Url) -> bool {
    matches!(
        content_type.split(';').next(),
        Some("video/webm" | "video/x-matroska")
    ) || matches!(url.path().rsplit('.').next(), Some("webm" | "mkv"))
}

fn extension_for<'a>(kind: StoredKind, url: &'a Url, content_type: &str) -> &'a str {
    match kind {
        StoredKind::Html => "html",
        StoredKind::Text => "txt",
        StoredKind::Video => match content_type.split(';').next().unwrap_or_default() {
            "video/webm" => "webm",
            "video/ogg" => "ogv",
            _ => match url.path().rsplit('.').next() {
                Some("webm") => "webm",
                Some("ogv" | "ogg") => "ogv",
                Some("mkv") => "mkv",
                Some("avi") => "avi",
                _ => "video",
            },
        },
        StoredKind::Image => match content_type.split(';').next().unwrap_or_default() {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/gif" => "gif",
            "image/webp" => "webp",
            "image/avif" => "avif",
            "image/tiff" => "tiff",
            "image/bmp" => "bmp",
            "image/heic" | "image/heif" => "heic",
            _ => url
                .path()
                .rsplit('.')
                .next()
                .filter(|extension| {
                    matches!(
                        *extension,
                        "png"
                            | "jpg"
                            | "jpeg"
                            | "gif"
                            | "webp"
                            | "avif"
                            | "bmp"
                            | "tif"
                            | "tiff"
                            | "heic"
                    )
                })
                .unwrap_or("image"),
        },
        StoredKind::Other => "bin",
    }
}

fn sniff(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF8") {
        Some("image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        Some("image/webp")
    } else if bytes.get(4..8) == Some(b"ftyp") {
        Some("video/mp4")
    } else if bytes.starts_with(b"\x1a\x45\xdf\xa3") {
        Some("video/webm")
    } else if bytes.starts_with(b"OggS") {
        Some("video/ogg")
    } else if let Ok(text) = std::str::from_utf8(bytes) {
        let prefix = text.trim_start().to_ascii_lowercase();
        if prefix.starts_with("<!doctype html") || prefix.starts_with("<html") {
            Some("text/html")
        } else if !text.contains('\0') {
            Some("text/plain")
        } else {
            None
        }
    } else {
        None
    }
}

pub(crate) fn digest(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

pub(crate) fn save(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path.parent().context("cache path has no parent")?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(bytes)?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

fn sniff_headers(
    session: &Session<'_>,
    url: &Url,
    mut headers: Headers,
) -> anyhow::Result<(Headers, StoredKind)> {
    let mut kind = identify(&headers.content_type, url);
    if kind == StoredKind::Other {
        let (probed, bytes) = session.peek(url)?;
        headers = probed;
        if let Some(content_type) = sniff(&bytes) {
            content_type.clone_into(&mut headers.content_type);
        }
        kind = identify(&headers.content_type, url);
    }
    Ok((headers, kind))
}

fn complete_download(
    session: &Session<'_>,
    url: &Url,
    headers: &Headers,
    path: &Path,
    limit: u64,
) -> anyhow::Result<(Headers, bool)> {
    ensure!(
        headers.total_size.is_none_or(|size| size <= limit),
        "response exceeds {limit}-byte size cap"
    );
    let (headers, bytes) = session.get(url, limit)?;
    let complete = headers.total_size == Some(bytes.len() as u64);
    // Persist original bytes before parsing HTML or returning a detector input.
    save(path, &bytes)?;
    Ok((headers, complete))
}
