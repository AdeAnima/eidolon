# Remote media: identify without downloading it all

Research date: 2026-10-29. Scope: raise remote-video `FrameOf` recall (holdout7: 14/63 direct)
and cut transferred bytes (holdout7 URL run: 135,857,173 / 338,674,709 = 0.401) without giving
up precision 1.000. Research only: no eidolon code was built, run or changed.

Claims marked **[MEASURED]** come from commands run for this note (method in the appendix).
Claims marked **[INFERENCE]** are my reasoning, not observed results.

## 1. Diagnosis first: where the holdout7 remote bytes and misses come from

Source: `eval-results/holdout7/url-output.json`, `eidolon-score.json`, and `ffprobe` on
`evaluation/eidolon-eval-holdout7/media/video/*` **[MEASURED]**.

| Consumed bytes (112 remote files) | Bytes | Share of 135.86 MB |
|---|---:|---:|
| One remote `000_exact.webm`, full download (`non_indexed_video`) | 59,850,769 | 44.1% |
| Native-GOP refinement, 24 sparse MP4s | 57,065,472 | 42.0% |
| 84 remote images, complete | 14,413,225 | 10.6% |
| Initial sparse keyframe proxies, 27 MP4s | 4,527,707 | 3.3% |

- **24 of 27 sparse MP4s ended with `budget_exhausted = true`.**
- **The eval H.264 derivatives use very long GOPs.** They are Constrained Baseline, 8 fps,
  with no B-frames, and have one keyframe every 21-30 s. A 10-60 s video has 1-2 keyframes in
  total. The sparse proxy therefore holds 1-6 frames per video.
- **Thumbnail times are uniform random** (`generate.rs:1187`), so no keyframe is aligned with
  them: 0 of the 21 remote thumbnail timestamps falls on a keyframe.
- **A whole GOP rarely fits the 25% budget.** For the 21 remote `thumb_frame` pairs, the GOP
  that contains the thumbnail time is a median **44.8%** of the file. Only **1 of 21** fits in
  `0.25 × size − initial bytes`. As configured, whole-GOP refinement cannot succeed on most of
  this corpus, however good the time localisation is.
- **A GOP prefix often does fit.** Decoding only from the keyframe up to the thumbnail time
  costs a median **12.7%** of the file. **12 of 21** fit within the 25% budget when the time is
  known (oracle). This is the upper bound for any approach that is limited to bytes.
- **Without time localisation, the gain mostly disappears.** A blind sequential scan with early
  stop (keyframe forward, stop at the match) fits 5/20 MP4 pairs at 0.25, 11/20 at 0.5 and 20/20
  at 1.0. Ranking GOPs by the MSE between the keyframe and the thumbnail was *worse* than
  sequential order (7/20 at 0.5). Localisation is therefore the second lever.
- **The WebM original has much shorter GOPs.** `000_exact.webm` is VP9, with 44 keyframes in
  164 s (average GOP 3.7 s). Its keyframes are 3.42 MB of 57.2 MB of video packets (6.0%). The
  file has a front `SeekHead` and a **927-byte `Cues` element at the end of the file**. Reaching
  the thumbnail frame from its keyframe would cost 2.7% of the file.

Per-variant remote recall (`eidolon-score.json`): `thumb_frame` 6/21, `thumb_poster` 6/21,
`thumb_overlay` 2/21. Locally the same categories reach 84/89, 84/89 and 43/89. The remote gap
is therefore mostly frame **availability**, not matching quality. The exception is overlay,
which is weak everywhere.

## 2. Ranked candidates

### 1. GOP-prefix streaming refinement with early stop (decode while fetching)

- **What:** For a refinement window, issue one `Range: bytes=<keyframe offset>-<GOP end>`
  request. Feed the samples to a decoder as they arrive (H.264 Annex B or VP9 IVF into an
  `ffmpeg -f h264 -i pipe:0` / `-f ivf` pipe). Hash each decoded frame at the detector cadence
  and compare it with the thumbnail. Stop reading the body when the existing native-frame proof
  passes, or at the target time. Today eidolon fetches "the original GOP(s) covering the
  source-time window" (fetch README). The decoder dependency only requires the bytes *before*
  the target in decode order, not the rest of the GOP.
- **Sources:**
  - ITU-T H.264 (08/2021) §7.4.1, `nal_ref_idc`: non-reference pictures can be dropped without
    affecting other pictures. https://www.itu.int/rec/T-REC-H.264
  - Apple QuickTime File Format, "Sample dependency flags atom ('sdtp')": per-sample
    `sample_is_depended_on = 2` marks a sample as disposable.
    https://developer.apple.com/documentation/quicktime-file-format/sample_dependency_flags_atom
  - RFC 9110 §14 (Range requests). https://www.rfc-editor.org/rfc/rfc9110#section-14
- **Licence / maturity:** These are standards. No new dependency is needed, because the ffmpeg
  CLI is already used. Alternative in-process decoder: `openh264` crate 0.9.9 (BSD-2-Clause,
  updated 2026-10-15, 130 stars).
- **Measured numbers [MEASURED, holdout7]:**
  - Median bytes to reach the thumbnail frame: whole GOP 44.8% of file, prefix 12.7%.
  - Fit within the 25% budget: 1/21 whole GOP, 12/21 prefix with oracle time, 5/20 blind
    sequential scan.
- **Disposable-frame skipping (small, real-web only):**
  - Holdout7 MP4s have no B-frames and no non-reference frames, so there is zero gain here.
  - YouTube avc1: non-reference frames are 37-47% of samples and **14-22% of bytes** (144p 21.5%,
    360p 18.0%, 1080p 14.1%; first three fragments) **[MEASURED]**.
  - YouTube's `trun` flags all say "unknown" for dependency. Skipping therefore needs the first
    5 bytes of each sample, which means multi-range requests. Treat it as a later refinement.
- **Weak spot:** Remote `FrameOf` recall and refinement bytes (42% of consumed bytes).
- **Upper bound [INFERENCE]:** At the current 0.25 cap, 12 decodable timestamps × 3 thumbnail
  variants gives at most 36/63 direct. Applying the local per-variant proof rates
  (0.94/0.94/0.48) gives about 28/63, versus 14/63 today. That needs good GOP choice;
  sequential order alone gives about 5 timestamps.
- **Budget decision:** Recall depends directly on the cap on this corpus:
  - 0.25: 5/20 decodable timestamps (blind sequential scan)
  - 0.5: 11/20
  - 1.0: 20/20
  A per-proof budget, or a `--remote-video-budget` default above 0.25 only for `FrameOf`
  candidates, is a product decision, not a research one.
- **Effort:** 3-5 days. Work items: streaming decode pipe, early-stop hook into the existing
  proof, transfer accounting for aborted bodies. Eidolon already counts consumed body bytes,
  which fits early stop.
- **Telemetry:** None.

### 2. Matroska/WebM `Cues` sparse proxy

- **What:** Read EBML header + `SeekHead` (first ~1 KB), then a range for `Cues` (front or
  tail), then each cue point's keyframe block (`CueClusterPosition` + `CueRelativePosition`).
  Write either a sparse WebM or the existing sparse MP4 (VP9 is already a supported MP4 sample
  entry). Native-GOP refinement then works on WebM. With 3.7 s GOPs it is cheap.
- **Source:** S. Lhomme, M. Bunkus, D. Rice, "Matroska Media Container Format Specification",
  RFC 9559, Oct 2024. `Cues`/`CuePoint`/`CueTrackPositions`/`CueClusterPosition` §5.1.5;
  Cues guidance §22. https://www.rfc-editor.org/rfc/rfc9559
- **Licence / maturity / Rust:**
  - `matroska-demuxer` 0.8.1: MIT/Apache-2.0/Zlib, 2026-08-07, only 23 stars.
  - `webm-iterable` 0.7.1: MIT, 2026-08-28, 573 k downloads.
  - A minimal own EBML walker (~300-500 lines) matches the existing hand-written MP4 index
    code **[INFERENCE]**.
- **Measured numbers [MEASURED]:**
  - `000_exact.webm`: 59.85 MB full download, versus about 3.42 MB keyframes + 927 B Cues +
    ~1 KB header.
  - Recomputing the holdout7 URL run with that substitution: ≈ 79.4 MB / 338.7 MB =
    **≈ 0.235 instead of 0.401** **[INFERENCE: arithmetic on the measured parts]**.
- **Real-web relevance:**
  - Wikimedia Commons serves WebM/VP9 transcodes. The fetch README's own smoke file
    `…480p.vp9.webm` uses the full-download fallback.
  - YouTube offers VP9 WebM DASH formats (e.g. `278`/`243`).
- **Weak spot:** Bytes. Also remote `FrameOf` for WebM targets, through cheap refinement.
- **Precision note:** A Cues proxy is not byte-exact. A remote *exact copy* would then need
  either a full fetch or a weaker relation than `identical`. That is the same rule as sparse
  MP4 today.
- **Effort:** 2-4 days, including EBML lacing and `BlockGroup` cases, plus `CueRelativePosition`
  absent (then scan the cluster).
- **Telemetry:** None.

### 3. yt-dlp rendition choice: stop pulling 1080p AV1

- **What:**
  - The current selector (`platform.rs:46`) is
    `bestvideo[ext=mp4][protocol=https]/best[ext=mp4]…`. On the YouTube test video
    `5KLPxDtMqe8` it selects **format 399, AV1 1080p, 42,503,785 bytes** **[MEASURED via
    `yt-dlp --load-info-json`]**.
  - The fetch README lists only H.264/VP9 sample entries for sparse proxies. AV1 should
    therefore fall back to a complete download **[INFERENCE: not run through eidolon]**.
  - Selecting `bv*[vcodec^=avc1][height<=360][protocol=https]` chose **format 134, 640×360
    avc1, 7,777,104 bytes**. 144p avc1 (`160`) is 2,021,188 bytes.
  - Both are fragmented MP4 with `sidx`. The 144p `sidx` has **49 subsegments of about 5 s,
    each starting with a SAP** **[MEASURED]**, so eidolon's existing `sidx` path applies.
- **Source:** yt-dlp README "Format selection" / "Sorting formats"
  (https://github.com/yt-dlp/yt-dlp#format-selection); yt-dlp 2026.08.19 source.
- **Licence / maturity:** Unlicense, 203,692 stars, pushed 2026-10-29. Already a runtime
  dependency.
- **Numbers [MEASURED]:**
  - Total file: 360p avc1 is 5.5× smaller than the selected AV1; 144p is 21× smaller.
  - The sparse keyframe proxy is a fraction of that.
  - Proof resolution: eval thumbnails are 320×180 and posters 1280×720. 640×360 derivatives
    (`low_bitrate_360p`) are part of the 84/89 local `thumb_frame` recall, so 360p sources
    should still prove `FrameOf` **[INFERENCE: not split per derivative]**.
- **Weak spot:** Bytes and speed for platform URLs. **No effect on holdout7**, which is served
  over loopback with no platform.
- **Effort:** 2-4 hours, plus a fixture.
- **Telemetry:** Unchanged. The maintainers say "We will not add telemetry" (already
  documented in the fetch README).

### 4. Time localisation from platform previews: storyboards, server-side seek thumbnails, image playlists

- **What:** Use a cheap low-res timeline (sprite sheets) to find *which* GOP or subsegment holds
  the thumbnail. Then fetch only that prefix (#1). The storyboard pixels are derived, so they
  are used for **localisation only**. The proof stays on native frames, so precision is
  unchanged.
- **Sources and measured numbers:**
  - **YouTube storyboards via yt-dlp:** formats `sb0..sb3` (`mhtml`), built in
    `yt_dlp/extractor/youtube/_video.py::_extract_storyboard` from
    `playerStoryboardSpecRenderer.spec` (undocumented by Google).
    - For the 248 s test video, `sb0` is 159×90 tiles at 0.504 fps in 5×5 sheets: 5 sheets of
      800×450, **210,422 bytes in total** **[MEASURED]**.
    - Compare 2.02 MB for the 144p video and 62.8 MB for 1080p avc1.
    - Twitch VOD (`seekPreviewsURL`) and France.tv (`spritesheets`) storyboards are also
      exposed by yt-dlp (`twitch.py:445`, `francetv.py:218`).
  - **Wikimedia Commons / MediaWiki TimedMediaHandler:** `…/thumb/<f>/330px-seek%3D<s>-<f>.jpg`
    returns a server-decoded frame at second `<s>`.
    - Measured on `NASA-_60_Years_in_60_Seconds.webm`: 5-19 KB per 330 px frame.
    - Integer seconds worked; `seek=33.5` returned the default poster.
    - Only standard widths are allowed (`320px` → HTTP 400, "Use thumbnail sizes listed on
      https://w.wiki/GHai"). Commons rate-limits bots (HTTP 429 seen).
    - A 60 s video at 1 fps costs ≈ 0.9 MB versus 34.5 MB for the original
      **[INFERENCE: 60 × ~15 KB]**.
    - Doc: https://www.mediawiki.org/wiki/Extension:TimedMediaHandler (`thumbtime`/`seek`).
  - **DASH-IF IOP v4.3, thumbnail tiles:** an image `AdaptationSet` with
    `EssentialProperty schemeIdUri="http://dashif.org/thumbnail_tile" value="5x2"`, plus
    trick-mode sets `http://dashif.org/guidelines/trickmode`.
    https://dashif.org/docs/DASH-IF-IOP-v4.3.pdf
  - **HLS image playlists (Roku extension):** `EXT-X-IMAGES-ONLY`, `EXT-X-TILES`,
    `EXT-X-IMAGE-STREAM-INF`. https://developer.roku.com/dev/docs/hls-and-dash
  - **Platform poster/oEmbed:** oEmbed `thumbnail_url` (https://oembed.com/ §2.3.4). YouTube
    `i.ytimg.com/vi/<id>/maxresdefault.jpg` is already discovered via yt-dlp `thumbnail`.
- **Maturity:** Specs are stable. The YouTube storyboard spec is private and can change; yt-dlp
  tracks it.
- **Weak spot:** Remote `FrameOf` localisation (the 12/21 oracle versus 5/20 blind gap above)
  and bytes. Not testable on holdout7 as served.
- **Effort:** 2-3 days: sprite slicing, tile→time mapping, ranking subsegments by tile
  pHash/DINOv2 versus thumbnail. DASH/HLS tile parsing comes with #5.
- **Telemetry:** None for HTTP fetches. yt-dlp, as above.

### 5. Native HLS/DASH manifests: lowest rendition + I-frame playlists

- **What:**
  - Parse `.m3u8`/`.mpd` met on pages, or returned by yt-dlp `manifest_url`. Pick the lowest
    H.264/VP9 representation.
  - Use `EXT-X-I-FRAME-STREAM-INF` playlists. Each entry is a byte range of one I-frame.
  - This gives keyframe proxies without parsing `moov`.
- **Sources:**
  - R. Pantos, W. May, "HTTP Live Streaming", RFC 8216, Aug 2017: §4.3.3.6
    `EXT-X-I-FRAMES-ONLY`, §4.3.4.3 `EXT-X-I-FRAME-STREAM-INF`.
    https://www.rfc-editor.org/rfc/rfc8216
  - Apple HLS Authoring Specification: I-frame playlists required; IDR every 2 s recommended
    (1.13); dense 1 fps I-frame renditions recommended.
    https://developer.apple.com/documentation/http-live-streaming/hls-authoring-specification-for-apple-devices
  - DASH-IF IOP v4.3 trick-mode adaptation sets (above).
- **Rust:**
  - `m3u8-rs` 6.0.1 (MIT, 2026-07-30, 1.75 M downloads).
  - `dash-mpd` 0.21.0 (MIT, 2026-10-04). Use only its parser; its downloader shells out to
    ffmpeg/mp4decrypt.
- **Numbers:**
  - Apple-conformant streams give one keyframe per ≤2 s, versus 21-30 s in the holdout7
    derivatives.
  - Counter-evidence **[MEASURED]**: YouTube's HLS master playlist has **no**
    `EXT-X-I-FRAME-STREAM-INF` entries. For YouTube, use DASH `sidx` (#3) instead.
- **Weak spot:** Bytes and `FrameOf` on news/broadcaster sites that serve HLS **[INFERENCE:
  common for publishers, not measured]**.
- **Effort:** 3-5 days (TS segments need a demux path; fMP4/CMAF reuses the fragment code).
- **Telemetry:** None.

### 6. Image prefix fetch for candidate gating

- **What:** Fetch only an image prefix first. Run candidate hashes/embeddings on the preview.
  Fetch the rest only if the image becomes a candidate that needs proof. Preview sources:
  - progressive JPEG DC scan (1/8 scale)
  - EXIF IFD1 thumbnail
  - HEIF `thmb` item (libheif is already linked)
  - JPEG XL progressive LF
  - layered AVIF
- **Sources:**
  - ITU-T T.81 (09/1992) Annex G, progressive DCT. https://www.w3.org/Graphics/JPEG/itu-t81.pdf
  - libjpeg-turbo buffered-image mode (`jpeg_consume_input`/`jpeg_start_output`, `libjpeg.txt`).
  - CIPA DC-008-2012 (Exif 2.3) IFD1 `JPEGInterchangeFormat`, 160×120 recommended.
    https://www.cipa.jp/std/documents/e/DC-008-2012_E_Corrigendum.pdf
  - ISO/IEC 18181 JPEG XL; `jxl-oxide` 0.12.6 (MIT/Apache-2.0, 475 stars) incremental
    `feed`/`try_init`.
  - AVIF v1.1.0 `a1lx`/`lsel` progressive layers.
    https://aomediacodec.github.io/av1-avif/v1.1.0.html
- **Measured numbers [MEASURED]:**
  - Re-encoding 10 holdout7 originals as progressive JPEG with the libjpeg default script:
    the DC scan ends at **4.4-12.0%** of file bytes. That gives a ~120 px preview of a 960 px
    image, enough for 32×32 pHash/dHash.
  - 10 random Commons JPEGs over 500 KB: **4/10 had EXIF thumbnails** (160×107 to 504×376,
    ending within the first 51 KB). 3/10 were progressive. Small sample; Commons throttled
    further requests.
- **Weak spot:** Bytes only. Images are 10.6% of holdout7 bytes, and eval images are small
  (≤960 px). The gain is larger for multi-MB web photos.
- **Precision:** Unchanged if previews only gate and the proof runs on the full image.
- **Effort:** 2-3 days (JPEG/EXIF/HEIF). JXL/AVIF later; both are rare on the web.
- **Telemetry:** None.

### 7. Segment-size / sample-size fingerprints (zero media bytes)

- **What:** `stsz`, `trun`, `sidx` or `Cues` give per-sample or per-segment sizes before any
  media byte is fetched. Identical encodes have identical size sequences. This is a free
  candidate for same-encode copies (mirrors, CDN copies), and a reason to skip duplicate
  remote fetches.
- **Source:** R. Schuster, V. Shmatikov, E. Tromer, "Beauty and the Burst: Remote
  Identification of Encrypted Video Streams", USENIX Security 2017, pp. 1357-1374. Their
  YouTube detector: 0.98 recall with no false positives, from DASH segment sizes alone.
  https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/schuster
- **Limits:**
  - Useless for re-encodes, crops, watermarks or `FrameOf`.
  - It cannot prove byte identity, so `identical` still needs the full bytes or a keyframe
    hash proof. Precision-safe only as a candidate signal.
  - Holdout7 has one remote exact video.
- **Effort:** About 1 day.
- **Telemetry:** None.

### 8. Not recommended: server-side hashes, PhotoDNA, Content ID, PDQ for remote savings

- **PDQ / TMK+PDQF / vPDQ** (Meta ThreatExchange, BSD with listed exceptions, 1,405 stars,
  active). Paper: J. Dalins, C. Wilson, D. Boudry, "PDQ & TMK+PDQF — A Test Drive of
  Facebook's Perceptual Hashing Algorithms", arXiv:1912.07745 (2019). README thresholds:
  distance ≤31, quality ≤49 discarded.
  - Rust `pdqhash` 0.1.1 is stale (2022-07).
  - These are local hashes. They save no download bytes, and eidolon already has pHash/dHash +
    DINOv2.
- **PhotoDNA:** Proprietary and vetted access only. It uploads content to Microsoft: phone-home.
  Reject.
- **YouTube Content ID:** No public API for third-party matching **[INFERENCE: none found]**.
- **Platforms do not expose their perceptual hashes publicly** **[INFERENCE]**.

## 3. Top 3 and the eval experiment that proves each

1. **GOP-prefix streaming refinement with early stop (#1) + a GOP-ranking prior.**
   - Experiment: `eidolon-eval run <holdout7> --urls --remote-video-mode keyframes --json`,
     with `max_fetch_fraction` 0.25 and 0.5.
   - Pass: remote `frame_of` direct > 14/63 (target ≥25/63 at 0.25; ceiling ≈36/63), precision
     1.000.
   - Pass: `url_traffic` refinement bytes below 57.07 MB.
   - Report the `budget_exhausted` count (24/27 today).
2. **WebM `Cues` sparse proxy (#2).**
   - Same run.
   - Pass: `000_exact.webm` reports `mode = sparse`, about 3.5 MB instead of 59.85 MB.
   - Pass: total consumed fraction ≤ 0.25 (from 0.401).
   - Pass: its 3 `frame_of` pairs and the group recall do not regress; precision 1.000.
3. **yt-dlp rendition fix (#3), then storyboards (#4).** Holdout7 cannot measure either.
   - Add a fixture: 10 CC-licensed YouTube/Commons videos, each with its own platform
     thumbnail.
   - Compare current versus new selector: bytes per URL (expect ≥5×), resolution mode
     (`sparse` versus `full`), and `FrameOf` of thumbnail ↔ video.
   - For #4, also add an `eidolon-eval serve` mode that serves a DASH `sidx` 360p rendition
     plus a storyboard sprite, so localisation can be scored offline.

## Appendix: measurement method (all read-only)

- **Byte breakdown:** `url-output.json` `files[].url` (`mode`, `initial_bytes_fetched`,
  `refinement_bytes_fetched`, `total_size`, `budget_exhausted`).
- **GOP and keyframes:** `ffprobe -show_entries packet=pts_time,size,pos,flags` on the 28
  remote videos.
- **Reference flags:** `nal_ref_idc` read from the AVCC NAL headers at each packet `pos`.
- **Thumbnail time:** each remote `thumb_frame` matched against all native frames (64×36 gray,
  minimum MSE) of the full-frame derivative of the same family. Timestamps came out as
  multiples of 0.125 s, which is consistent with `generate.rs:1187`.
- **Byte costs:**
  - Whole GOP: packets from the keyframe to the next keyframe.
  - Prefix: packets from the keyframe to the target frame.
  - Fit test: `cost ≤ 0.25 × file size − initial sparse bytes`.
  - Blind scan: GOPs in file order, each fully decoded until the target GOP.
- **Ranked scan:** GOPs ordered by the MSE of the bounding keyframes against the thumbnail.
- **WebM structure:** top-level EBML walk of `000_exact.webm`.
- **YouTube data:** `yt-dlp --ignore-config --no-cache-dir -J` (DO_NOT_TRACK=1), followed by
  Range reads of the format `160`/`134`/`137` `sidx`, the first three fragments, and the `sb0`
  sheets.
- **Commons data:** TimedMediaHandler `seek` thumbnails and 10 random JPEG prefixes
  (≤128 KiB each).
