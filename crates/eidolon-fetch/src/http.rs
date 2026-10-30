use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, bail, ensure};
use reqwest::StatusCode;
use reqwest::blocking::{Client, RequestBuilder, Response};
use reqwest::header::{
    ACCEPT_ENCODING, AGE, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, ETAG,
    IF_MODIFIED_SINCE, IF_NONE_MATCH, IF_RANGE, LAST_MODIFIED, RANGE,
};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::transport::LimitedResponse;
use crate::{FetchOptions, digest, save};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct Headers {
    pub(crate) url: String,
    pub(crate) content_type: String,
    pub(crate) total_size: Option<u64>,
    pub(crate) etag: Option<String>,
    pub(crate) last_modified: Option<String>,
    pub(crate) fresh_for: u64,
    pub(crate) fresh_until: u64,
}

impl Headers {
    pub(crate) fn is_fresh(&self) -> bool {
        unix_seconds() < self.fresh_until
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn refresh_freshness(headers: &mut Headers, response: &Response) {
    if let Some(control) = response
        .headers()
        .get(CACHE_CONTROL)
        .and_then(|value| value.to_str().ok())
    {
        let mut directives = control.split(',').map(str::trim);
        headers.fresh_for = if directives.clone().any(|directive| {
            let name = directive.split('=').next().unwrap_or_default().trim();
            name.eq_ignore_ascii_case("no-cache") || name.eq_ignore_ascii_case("no-store")
        }) {
            0
        } else {
            directives
                .find_map(|directive| {
                    let (name, value) = directive.split_once('=')?;
                    name.trim()
                        .eq_ignore_ascii_case("max-age")
                        .then(|| value.trim().trim_matches('"').parse().ok())
                        .flatten()
                })
                .unwrap_or(30)
        };
    }
    let age = response
        .headers()
        .get(AGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    headers.fresh_until = unix_seconds().saturating_add(headers.fresh_for.saturating_sub(age));
}

pub(crate) struct Session<'a> {
    pub(crate) client: &'a Client,
    pub(crate) options: &'a FetchOptions,
    cumulative: &'a Arc<AtomicU64>,
    pub(crate) bytes: Cell<u64>,
    transfer_budget: Option<u64>,
    pub(crate) budget_exhausted: Cell<bool>,
    pub(crate) sizes: RefCell<HashMap<String, Option<u64>>>,
    started: Instant,
}

pub(crate) fn validate(value: &str) -> anyhow::Result<Url> {
    let mut url = Url::parse(value).context("parsing URL")?;
    ensure!(
        matches!(url.scheme(), "http" | "https"),
        "only HTTP(S) URLs are allowed"
    );
    ensure!(url.host_str().is_some(), "URL has no host");
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "URL credentials are not allowed"
    );
    url.set_fragment(None);
    Ok(url)
}

impl<'a> Session<'a> {
    pub(crate) fn new(
        client: &'a Client,
        options: &'a FetchOptions,
        cumulative: &'a Arc<AtomicU64>,
    ) -> Self {
        Self {
            client,
            options,
            cumulative,
            bytes: Cell::new(0),
            transfer_budget: None,
            budget_exhausted: Cell::new(false),
            sizes: RefCell::new(HashMap::new()),
            started: Instant::now(),
        }
    }

    pub(crate) fn with_transfer_budget(mut self, bytes: Option<u64>) -> Self {
        self.transfer_budget = bytes;
        self
    }

    fn check_transfer_budget(&self, bytes: u64) -> anyhow::Result<()> {
        if self
            .transfer_budget
            .is_some_and(|limit| bytes > limit.saturating_sub(self.bytes.get()))
        {
            self.budget_exhausted.set(true);
            bail!("fractional native-refinement transfer budget exhausted");
        }
        Ok(())
    }

    pub(crate) fn check_deadline(&self) -> anyhow::Result<()> {
        ensure!(
            self.started.elapsed() < self.options.resolve_timeout,
            "URL resolution timed out"
        );
        Ok(())
    }

    fn send(&self, request: RequestBuilder) -> anyhow::Result<LimitedResponse> {
        self.check_deadline()?;
        let deadline = self
            .started
            .checked_add(self.options.resolve_timeout)
            .context("resolution timeout exceeds monotonic clock")?;
        crate::transport::send(self.client, request, self.options.timeout, deadline)
    }

    pub(crate) fn head(
        &self,
        url: &Url,
        cached: Option<&Headers>,
    ) -> anyhow::Result<(Headers, bool)> {
        self.check_deadline()?;
        let mut request = self
            .client
            .head(url.clone())
            .header(ACCEPT_ENCODING, "identity");
        if let Some(cached) = cached {
            if let Some(etag) = &cached.etag {
                request = request.header(IF_NONE_MATCH, etag);
            } else if let Some(modified) = &cached.last_modified {
                request = request.header(IF_MODIFIED_SINCE, modified);
            }
        }
        let response = self.send(request).context("HTTP HEAD")?;
        if response.status() == StatusCode::NOT_MODIFIED {
            let mut headers = cached.context("304 without cached response")?.clone();
            refresh_freshness(&mut headers, &response);
            return Ok((headers, true));
        }
        if matches!(
            response.status(),
            StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_IMPLEMENTED
        ) {
            drop(response);
            let (headers, _) = self.peek(url)?;
            return Ok((headers, false));
        }
        ensure!(
            response.status().is_success(),
            "HTTP HEAD returned {}",
            response.status()
        );
        let headers = response_headers(&response);
        self.note_headers(url, &headers);
        let unchanged = cached.is_some_and(|cached| {
            headers
                .etag
                .as_ref()
                .is_some_and(|etag| cached.etag.as_ref() == Some(etag))
                || (headers.etag.is_none()
                    && headers.last_modified.is_some()
                    && headers.last_modified == cached.last_modified
                    && headers.total_size == cached.total_size)
        });
        Ok((headers, unchanged))
    }

    pub(crate) fn note_size(&self, url: &str, size: Option<u64>) {
        let mut sizes = self.sizes.borrow_mut();
        let known = sizes.entry(url.to_owned()).or_insert(None);
        *known = size.or(*known);
    }

    pub(crate) fn note_headers(&self, requested: &Url, headers: &Headers) {
        if requested.as_str() != headers.url {
            self.sizes.borrow_mut().remove(requested.as_str());
        }
        self.note_size(&headers.url, headers.total_size);
    }

    pub(crate) fn peek(&self, url: &Url) -> anyhow::Result<(Headers, Vec<u8>)> {
        self.check_deadline()?;
        let limit = self
            .options
            .max_page_bytes
            .min(self.options.max_image_bytes)
            .min(self.options.max_video_bytes)
            .min(4096);
        self.check_transfer_budget(limit)?;
        let request = self
            .client
            .get(url.clone())
            .header(ACCEPT_ENCODING, "identity")
            .header(RANGE, format!("bytes=0-{}", limit - 1));
        let response = self.send(request).context("HTTP probe")?;
        ensure!(
            response.status().is_success(),
            "HTTP probe returned {}",
            response.status()
        );
        let headers = response_headers(&response);
        self.note_headers(url, &headers);
        let bytes = self.read_prefix(response, limit as usize)?;
        save(
            &self
                .options
                .cache_dir
                .join(format!("{}.probe", digest(url.as_str()))),
            &bytes,
        )?;
        Ok((headers, bytes))
    }

    fn count(&self, count: usize) {
        let count = count as u64;
        self.bytes.set(self.bytes.get() + count);
        self.cumulative.fetch_add(count, Ordering::Relaxed);
    }

    fn read_prefix(&self, mut response: LimitedResponse, length: usize) -> anyhow::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 8192];
        while bytes.len() < length {
            self.check_deadline()?;
            let mut length = buffer.len().min(length - bytes.len());
            if let Some(limit) = self.transfer_budget {
                length = length.min(
                    usize::try_from(limit.saturating_sub(self.bytes.get())).unwrap_or(usize::MAX),
                );
            }
            if length == 0 {
                break;
            }
            let count = response
                .read(&mut buffer[..length])
                .context("reading HTTP body")?;
            if count == 0 {
                break;
            }
            self.count(count);
            bytes.extend_from_slice(&buffer[..count]);
        }
        Ok(bytes)
    }

    pub(crate) fn get(&self, url: &Url, limit: u64) -> anyhow::Result<(Headers, Vec<u8>)> {
        self.check_deadline()?;
        let request = self
            .client
            .get(url.clone())
            .header(ACCEPT_ENCODING, "identity");
        let response = self.send(request).context("HTTP GET")?;
        ensure!(
            response.status().is_success(),
            "HTTP GET returned {}",
            response.status()
        );
        let headers = response_headers(&response);
        self.note_headers(url, &headers);
        ensure!(
            headers.total_size.is_none_or(|length| length <= limit),
            "response exceeds {limit}-byte size cap"
        );
        let maximum =
            usize::try_from(limit.saturating_add(1)).context("size cap exceeds address space")?;
        let bytes = self.read_prefix(response, maximum)?;
        ensure!(
            bytes.len() as u64 <= limit,
            "response exceeds {limit}-byte size cap"
        );
        if let Some(size) = headers.total_size {
            ensure!(bytes.len() as u64 == size, "truncated HTTP body");
        }
        Ok((headers, bytes))
    }

    pub(crate) fn range(
        &self,
        url: &Url,
        headers: &Headers,
        start: u64,
        end: u64,
    ) -> anyhow::Result<Vec<u8>> {
        self.check_deadline()?;
        let mut request = self
            .client
            .get(url.clone())
            .header(ACCEPT_ENCODING, "identity")
            .header(RANGE, format!("bytes={start}-{end}"));
        if let Some(validator) = headers
            .etag
            .as_ref()
            .filter(|etag| !etag.starts_with("W/"))
            .or(headers.last_modified.as_ref())
        {
            request = request.header(IF_RANGE, validator);
        }
        let response = self.send(request).context("HTTP range GET")?;
        ensure!(
            response.status() == StatusCode::PARTIAL_CONTENT,
            "origin did not honor HTTP Range ({}); bounded full download required",
            response.status()
        );
        let expected = format!(
            "bytes {start}-{end}/{}",
            headers.total_size.context("range source has no size")?
        );
        ensure!(
            response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                == Some(expected.as_str()),
            "invalid or changed HTTP Content-Range"
        );
        let length = usize::try_from(end - start + 1)?;
        let bytes = self.read_prefix(response, length + 1)?;
        ensure!(bytes.len() == length, "truncated or oversized HTTP range");
        Ok(bytes)
    }
}

fn response_headers(response: &Response) -> Headers {
    let value = |name| {
        response
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    };
    let total_size = response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.rsplit_once('/'))
        .and_then(|(_, size)| size.parse().ok())
        .or_else(|| {
            response
                .headers()
                .get(CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse().ok())
        });
    let mut headers = Headers {
        url: response.url().to_string(),
        content_type: value(CONTENT_TYPE).unwrap_or_default(),
        total_size,
        etag: value(ETAG),
        last_modified: value(LAST_MODIFIED),
        fresh_for: 30,
        fresh_until: 0,
    };
    refresh_freshness(&mut headers, response);
    headers
}

const BLOCK: u64 = 16 * 1024;

/// Seekable HTTP source. Persist every fetched byte block before decoding it.
pub(crate) struct RangeSource<'a, 'b> {
    session: &'a Session<'b>,
    url: Url,
    headers: Headers,
    size: u64,
    position: u64,
    directory: PathBuf,
    blocks: BTreeMap<u64, Vec<u8>>,
    fetched: u64,
    cap: u64,
}

impl<'a, 'b> RangeSource<'a, 'b> {
    pub(crate) fn new(
        session: &'a Session<'b>,
        url: &Url,
        headers: &Headers,
        cap: u64,
    ) -> anyhow::Result<Self> {
        let size = headers
            .total_size
            .context("range video needs Content-Length")?;
        ensure!(size > 0, "empty video");
        let revision = if headers.etag.is_some() || headers.last_modified.is_some() {
            format!("{:?}:{:?}:{size}", headers.etag, headers.last_modified)
        } else {
            format!("{:?}:{size}", std::time::SystemTime::now())
        };
        let directory = session
            .options
            .cache_dir
            .join(format!("{}.ranges", digest(url.as_str())))
            .join(digest(&revision));
        std::fs::create_dir_all(&directory)?;
        Ok(Self {
            session,
            url: url.clone(),
            headers: headers.clone(),
            size,
            position: 0,
            directory,
            blocks: BTreeMap::new(),
            fetched: 0,
            cap,
        })
    }

    fn fill(&mut self, index: u64, wanted: usize) -> anyhow::Result<()> {
        if self.blocks.contains_key(&index) {
            return Ok(());
        }
        let cached = self.directory.join(index.to_string());
        let start = index * BLOCK;
        let expected = (self.size - start).min(BLOCK) as usize;
        if let Ok(bytes) = std::fs::read(&cached)
            && bytes.len() == expected
        {
            self.blocks.insert(index, bytes);
            return Ok(());
        }
        // Coalesce large sample reads; table scalar reads still share one cached block.
        let blocks = ((self.position % BLOCK + wanted as u64).div_ceil(BLOCK)).clamp(1, 16);
        let mut end = ((index + blocks) * BLOCK).min(self.size) - 1;
        for next in index + 1..index + blocks {
            if self.blocks.contains_key(&next) || self.directory.join(next.to_string()).is_file() {
                end = (next * BLOCK).min(end + 1) - 1;
                break;
            }
        }
        let length = end - start + 1;
        self.session.check_transfer_budget(length)?;
        ensure!(
            self.fetched.saturating_add(length) <= self.cap,
            "video range transfer exceeds {}-byte remaining cap",
            self.cap
        );
        let bytes = self.session.range(&self.url, &self.headers, start, end)?;
        self.fetched += bytes.len() as u64;
        for (offset, block) in bytes.chunks(BLOCK as usize).enumerate() {
            let index = index + offset as u64;
            save(&self.directory.join(index.to_string()), block)?;
            self.blocks.insert(index, block.to_vec());
        }
        Ok(())
    }
}

impl Read for RangeSource<'_, '_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if buffer.is_empty() || self.position >= self.size {
            return Ok(0);
        }
        let index = self.position / BLOCK;
        self.fill(index, buffer.len())
            .map_err(std::io::Error::other)?;
        let bytes = self
            .blocks
            .get(&index)
            .ok_or_else(|| std::io::Error::other("missing range block"))?;
        let offset = (self.position % BLOCK) as usize;
        let count = buffer.len().min(bytes.len() - offset);
        buffer[..count].copy_from_slice(&bytes[offset..offset + count]);
        self.position += count as u64;
        Ok(count)
    }
}

impl Seek for RangeSource<'_, '_> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let position = match position {
            SeekFrom::Start(position) => i128::from(position),
            SeekFrom::End(offset) => i128::from(self.size) + i128::from(offset),
            SeekFrom::Current(offset) => i128::from(self.position) + i128::from(offset),
        };
        if position < 0 || position > i128::from(self.size) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "range seek outside source",
            ));
        }
        self.position = position as u64;
        Ok(self.position)
    }
}

pub(crate) fn inspect_boxes(source: &mut (impl Read + Seek), size: u64) -> anyhow::Result<()> {
    let mut position = 0;
    for _ in 0..4096 {
        if position == size {
            source.rewind()?;
            return Ok(());
        }
        ensure!(
            size.saturating_sub(position) >= 8,
            "truncated MP4 box header"
        );
        source.seek(SeekFrom::Start(position))?;
        let mut header = [0_u8; 8];
        source.read_exact(&mut header)?;
        let mut length = u64::from(u32::from_be_bytes(header[..4].try_into()?));
        let mut header_size = 8;
        if length == 1 {
            let mut extended = [0_u8; 8];
            source.read_exact(&mut extended)?;
            length = u64::from_be_bytes(extended);
            header_size = 16;
        } else if length == 0 {
            length = size - position;
        }
        ensure!(
            length >= header_size && length <= size - position,
            "invalid MP4 box length"
        );
        if &header[4..] == b"moov" {
            ensure!(length <= 16 * 1024 * 1024, "MP4 metadata exceeds 16 MiB");
        }
        ensure!(
            &header[4..] != b"moof",
            "fragmented MP4 needs bounded full download"
        );
        position += length;
    }
    bail!("too many MP4 boxes")
}
