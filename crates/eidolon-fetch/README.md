# eidolon-fetch

Resolve explicit HTTP(S) inputs to ordinary local files for eidolon's image,
video and text detectors. Local-path-only scans never construct a network request.
The fetch crate does not classify detector inputs or change local originals.

```rust
use eidolon_fetch::Fetcher;

# fn example() -> anyhow::Result<()> {
let fetcher = Fetcher::new()?;
for input in ["https://example.com/photo.jpg", "https://example.com/article"] {
    match fetcher.resolve(input) {
        Ok(result) => {
            for artifact in result.artifacts {
                // Classify artifact.local_path by its content in the CLI.
                // Keep artifact.url/page_url/role in the report.
                let _ = artifact;
            }
            for warning in result.warnings {
                eprintln!("warning: {warning}");
            }
        }
        Err(error) => eprintln!("warning: {input}: {error:#}"),
    }
}
# Ok(())
# }
```

## Contract

`Fetcher::new()` and `Fetcher::with_options(FetchOptions)` create an idle blocking
client. `resolve(&self, url: &str) -> anyhow::Result<FetchResult>` performs network
access only for that URL. `FetchResult` contains `artifacts`, `warnings`,
`bytes_fetched` and `total_size`. One failed input must not abort other CLI inputs.

Each `Remote` contains the final media `url`, optional enclosing `page_url`,
discovery `role`, `local_path`, current-resolution `bytes_fetched`, original
`total_size`, `complete_bytes`, `mode`, `mode_reason` and `budget_exhausted`.
`FetchMode` serializes as
`sparse`, `full` or `derived`. The reason records indexed keyframes, explicit full
cadence, non-indexed video, complete resource, article extraction, or the actual
range-proxy failure that required a complete download. Both fields survive cache
revalidation; cache schema 4 rejects older metadata rather than inventing a reason.
`complete_bytes` is true only for a complete, unmodified response of a known byte
length. Remuxed videos and extracted article text are **not byte-exact originals**,
even in full-cadence mode. Exclude them from byte-identical-original claims;
near/frame detection still applies.

`Fetcher::bytes_fetched()` counts cumulative HTTP response-body bytes even when
resolution fails. The per-resolution aggregate includes HTML, probes and failed
or partial subresource work. Source representation sizes are summed once per
distinct final URL, including HTML and attempted discovered media. The aggregate
`FetchResult::total_size` is `None` if any size is unknown, including a discovered
resource that fails before its original size is learned. A successful HEAD or
Content-Range probe can preserve a known size even if the later transfer fails.
Consumed bytes remain counted either way; callers must not print a numeric
bytes-fetched ratio for an unknown total. HEAD/304 bodies cost zero.
These are bytes read from HTTP response bodies, not TCP/TLS/header overhead.
`yt-dlp`'s external discovery traffic cannot be observed and is explicitly noted
in warnings; fetched stream/thumbnail bodies are counted normally.

## Video proxies and honest accuracy

For indexed MP4/MOV and fragmented MP4 with supported H.264/VP9 sample entries,
HTTP Range reads fetch the sample index and independent sync samples. The local MP4
copies encoded video samples without scaling or recompression, preserves the
original source timeline/edit list/duration, and removes audio. It retains at
most 1,200 keyframes, no more densely than two fps; long inputs use the detector's
adaptive `max(0.5, duration/1200)` spacing. The source's GOP spacing may be much
coarser. No held/repeated frame counts as a newly observed source frame.

There is no keyframe-density fallback: even a one-keyframe video stays in sparse
mode. Ordinary MP4 metadata after `mdat` is reached by seeking over the media
payload; a trailing `moov` does not itself trigger a complete download.

Fragmented inputs validate `sidx` v0/v1 when present and follow bounded
top-level headers to read `tfhd`/`tfdt`/`trun` metadata. `mdat` bodies are
seek-skipped except for selected samples. Fragment metadata is capped at 16 MiB,
one million samples and 100,000 top-level boxes. Non-contiguous decode timelines
are rejected rather than silently remapped.

WebM/Matroska with a finite `Info` duration, supported VP8/VP9/H.264 video track,
and `Cues` uses the same sparse policy. A bounded EBML reader follows `SeekHead`
to front or trailing metadata, then reads each selected cue's independent block.
`CueRelativePosition` is used when present; otherwise block headers locate the
cue within its cluster without downloading intervening packet payloads.
The local Matroska proxy preserves the selected track's codec configuration and
encoded blocks, including lacing and `BlockGroup` metadata, and removes audio.
Source PTS include the original cluster/block timestamp and laced frame cadence.
Native windows can span non-keyframe continuation clusters. Unsupported clocks,
encryption, codecs, unknown-sized clusters, or missing indexes use the explicit
bounded full-download fallback. Matroska full-cadence mode also uses that fallback.

A sparse proxy `example.mp4` has `example.mp4.remote.json`:

```json
{
  "version": 1,
  "duration_seconds": 64.0,
  "source_fps": 24.0,
  "sample_pts_seconds": [0.0, 4.0, 8.0]
}
```

The timestamp list contains **every retained sample**, in decode/output order,
and accounts for supported original MP4 edit lists. `eidolon-video` uses these
source timestamps instead of inventing observations by fps resampling.

`FetchOptions { video_mode: VideoMode::FullCadence, ..Default::default() }`
transfers all original video packets, still without audio or recompression.
It preserves the full decoder dependency chain and therefore the detector's
normal two-fps/adaptive accuracy. For long-GOP video this can require nearly
100% of the original video-track bytes: arbitrary P/B frames cannot be decoded
from isolated byte ranges. The sparse/default mode offers a measured transfer
trade-off, **not a claim that missing frames or arbitrary clips remain exact**.
A local video can still be densely refined when matching a remote thumbnail.
Sparse remote videos can supply native windows through the provider below;
without one, only retained keyframes count as verified observations.

Unsupported containers/indexes/codecs, origins without Range, and unsupported
edit timelines fall back to a complete download **under the remaining video
transfer cap**, with an explicit warning for stderr. HEVC uses this fallback
because the MP4 dependency does not retain its complete decoder configuration.
A known-oversize fallback is rejected before starting a full GET.

### Native-cadence refinement

`Fetcher` implements `eidolon_core::FrameRefiner`. Keep the original client
alive through detection and supply it as `Ctx::refiner`. Only sparse video
artifacts resolved by that client are eligible; local files and complete
fallbacks never trigger a refinement request.

A request retrieves the dependency prefix from the preceding keyframe through
the requested interval's final presentation sample, not the remainder of its GOP.
MP4 reordering retains preceding decode-order reference pictures needed by that
sample. The artifact is an ordinary native-cadence MP4 or Matroska window with
`RefinedWindow::source_offset_s`; encoded packets, decoded pixels, and source
presentation times are preserved. The video detector requests successive windows
of at most 0.5 seconds (or one frame for slower sources), proves native pixels,
and stops before requesting another prefix once a match passes its score gate.
The supplied candidate-time window bounds the search; a known time can be requested
directly through `FrameRefiner::refine`, including a zero-radius point request.
Stable ETag/Last-Modified validators are required; changed origins are rejected.
Per-artifact locking avoids duplicate concurrent work. Cached windows/range blocks
add no body bytes. Registration and lookup canonicalize artifact paths, so symlinked
cache directories share the canonical artifact's transfer budget and lock.
HTTP range blocks are 16 KiB: metadata and bounded block read-ahead still count.

`FetchOptions::max_fetch_fraction` defaults to **0.25** and accepts finite values
in `(0, 1]`. It limits **additional native-prefix transfer**, charging both initial
acquisition and every refinement attempt to the same artifact. Before a range
would exceed `floor(max_fetch_fraction × total_size)`, refinement returns
`Ok(None)` rather than fetching beyond the allowance. Already received partial
window bytes still count. A refusal applies only to that request: later prefixes
can use the remaining allowance, and cached windows/ranges remain available even
with zero allowance, subject to the usual source validation.
`1.0` disables the fractional restriction; the absolute `max_video_bytes` limit
still applies. Explicit full-cadence acquisition and complete-download fallbacks
are not native refinement and remain governed by their existing absolute caps.

**Initial sparse acquisition is unchanged and can already exceed 25%.** In that
case a native-window request needing new body bytes is refused, while cached
windows/ranges can still be served. This is not a hard fractional limit on initial
acquisition or a guarantee of equal recall. A new client starts a new transfer
account; retained cache bytes are not charged again.

After detection, refresh `Remote::budget_exhausted` with
`Fetcher::budget_exhausted(path)`. A true flag means at least one refinement was
refused by this client's fractional budget. It stays true even if later requests
succeed; callers must emit a stderr note that some windows could not be fetched
and recall can be lower. `resolve` also reports the current same-client flag when
returning an already resolved artifact.

Initial and refinement transfers share `max_video_bytes`, including failed
attempts. After detection, add `Fetcher::refinement_bytes(path)` once to that
artifact's initial `Remote::bytes_fetched`. `Fetcher::bytes_fetched()` already
includes it: do not add the delta again to the aggregate.

`Remote::mode` describes initial acquisition, not total transfer after detection.
A sparse artifact remains `sparse` after refinement, even when requested prefixes
collectively cover the entire source. Report initial and additional native-prefix
bytes separately; 100% transfer in that case is not a full-download fallback.

## Pages, platforms and safety

HTML discovery covers `og:image`, `og:video`, `twitter:image`, video/source URLs,
posters, and recursively nested JSON-LD `thumbnailUrl`/`contentUrl`. Relative URLs
resolve against the final page URL. Duplicate media URLs are fetched once per
page, with the first role retained. An article/main-first extraction strips
navigation, scripts, forms, footers and hidden content; it emits UTF-8 text.
Subresource failures become warnings without losing the article or other media.

YouTube, Vimeo, Dailymotion, Twitch, TikTok and Bilibili URL inputs use installed
`yt-dlp` for anonymous stream and thumbnail discovery. Sparse mode prefers direct
HTTP(S) H.264/VP9 renditions at or below 360p, then a supported higher resolution.
It never selects AV1; an AV1-only platform video fails discovery rather than
silently downloading an unsupported sparse codec. Explicit full-cadence mode keeps
the broader rendition selector. No cookie jars, browser
profiles, login, netrc, user configuration or plugins are loaded. Update checks,
remote components and SponsorBlock are disabled; `DO_NOT_TRACK=1` and `DNT=1`
are set. The maintainers explicitly state ["We will not add telemetry"](
https://github.com/yt-dlp/yt-dlp/issues/14198); removed `--no-call-home` is not
used as a fake privacy switch. No tool installation was necessary on the
implementation host (`yt-dlp 2026.08.19`, `ffmpeg 9.0.2`). Other hosts can install
with `brew install yt-dlp`; no auto-install occurs.

Only HTTP/HTTPS URLs without embedded credentials are accepted, including all
redirect destinations. User-Agent is `eidolon/<package-version>`. Defaults:
32 MiB image, 4 MiB HTML/text/discovery, 64 MiB transferred video, 30-second
request timeout, 180-second resolution deadline, ten redirects, 32 artifacts.
Chunked bodies are bounded too. External discovery has the same resolution
limit and bounded stdout/stderr. The library returns warnings instead of
printing into structured CLI output.

HTTP requests share a process-wide two-request limit per origin, including
redirects and streamed bodies. HTTP 429/503 honors `Retry-After` delta-seconds
and HTTP dates, with at most two retries and a 30-second maximum delay, always
within the resolution deadline.

Cache root on macOS: `~/Library/Caches/eidolon/remote/`; Linux uses
`$XDG_CACHE_HOME/eidolon/remote` or `~/.cache/eidolon/remote`. URL/mode hashes avoid
unsafe filenames. By default, `FetchOptions::fresh` is false: cached sources
require ETag/Last-Modified revalidation on each resolution and native-refinement
request, even when the origin sends `max-age`. Set it to true (CLI:
`--fetch-fresh`) to reuse fresh sources without a conditional HEAD.
The fallback lifetime without Cache-Control is 30 seconds; server `max-age`
overrides it, `Age` reduces it, and `no-cache`/`no-store` force revalidation.
A 304 extends the stored window using its freshness headers or the previous lifetime.
Sources without ETag/Last-Modified are refetched after the window expires (by default,
on every resolution): without `--fetch-fresh`, they are fetched again every run
rather than returning a conditional 304.
The original metadata schema remains version 4, with required `fresh_for` and
`fresh_until` fields; the freshness write path runs regardless of the opt-in flag.
Loopback small4 measured 72–81% fewer requests with identical body bytes and matches,
not a general speedup: the warm run was 60% slower (cause unassigned).
Freshness reuse and the CLI's `--timings` native-refinement diagnostic are opt-in
and disabled by default. Timing clocks and counters still run; only their stderr
line is gated.
Full source bodies, original HTML, anonymous platform JSON and fetched video
range blocks remain available locally; extracted artifacts do not replace the
source. Range blocks are keyed by URL and source validator, never reused across
unknown revisions. Cache directories are private and writes are atomic.

## Verification and retained smoke sources

```sh
cargo clippy -p eidolon-fetch --all-targets -- -D warnings
cargo nextest run -p eidolon-fetch --success-output immediate
cargo nextest run -p eidolon-fetch --run-ignored only --success-output immediate
```

The in-process `tiny_http` origin supports HEAD, open/closed/suffix ranges,
redirects and validators, and counts the actual body bytes it serves. Tests
decode real images/videos and compare proxy keyframe pixels and source PTS,
not only metadata. H.264 fixtures cover ordinary/fragmented MP4 prefix refinement
and bounded long GOPs. VP9 WebM with audio and reordered H.264 Matroska cover Cues,
native prefixes, continuation clusters, unchanged pixels, source clocks, and
audio removal. Offline yt-dlp fixtures cover low-resolution H.264/VP9 selection
and AV1 rejection. `MEASURE` lines expose fetched/served/original sizes and coverage.
Frame availability is not the same metric as duplicate-relation recall.

Ignored real-web tests and the manual example use these NASA public-domain
Commons sources (license statements are on their linked file pages):

- [Blue Marble 2002](https://commons.wikimedia.org/wiki/File:Blue_Marble_2002.png),
  [1280px PNG thumbnail](https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Blue_Marble_2002.png/1280px-Blue_Marble_2002.png).
  The 529 MiB original intentionally exceeds the image cap; the thumbnail is a
  complete remote image resource and can match a local original perceptually.
- [NASA: 60 Years in 60 Seconds](https://commons.wikimedia.org/wiki/File:NASA-_60_Years_in_60_Seconds.webm),
  [480p VP9 WebM transcode](https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f1/NASA-_60_Years_in_60_Seconds.webm/NASA-_60_Years_in_60_Seconds.webm.480p.vp9.webm).
  Retained opt-in source for indexed WebM acquisition. The current integration
  smoke uses the separately authorized NASA YouTube `1UaBgr_sq9A`; this Commons
  test has not been rerun.

Manual run (sources and measurement JSON stay in the specified private cache):

```sh
EIDOLON_FETCH_CACHE="$HOME/Library/Caches/eidolon/remote/manual-smoke" \
  cargo run -p eidolon-fetch --example measure -- \
  'https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Blue_Marble_2002.png/1280px-Blue_Marble_2002.png' \
  'https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f1/NASA-_60_Years_in_60_Seconds.webm/NASA-_60_Years_in_60_Seconds.webm.480p.vp9.webm'
```

Final holdout results and practical limits are in the [root README](../../README.md#known-limits).
