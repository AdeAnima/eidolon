# eidolon

Report exact and near-duplicate images, videos and text from local paths or explicit HTTP(S) URLs, including thumbnails that show frames of a video. **eidolon never moves, deletes or rewrites your input files.** Duplicate findings are a report, not an automatic cleanup decision.

## Install

### Homebrew (Apple Silicon macOS)

Once the first source release and tap are published:

```sh
brew install adeanima/tap/eidolon
```

Homebrew installs a prebuilt Apple Silicon bottle when one is available; otherwise it builds from source. To request a source build, add `--build-from-source`.

### Cargo from Git

Requires Rust 1.99.1 or newer, Xcode Command Line Tools and Homebrew:

```sh
brew install opencv onnxruntime ffmpeg libheif jpeg-turbo pkgconf llvm
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
export EIDOLON_ORT_DYLIB_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
cargo install --git https://github.com/AdeAnima/eidolon eidolon-cli --locked
```

The build records the ONNX Runtime library path; installed binaries do not need Cargo's environment. When `EIDOLON_ORT_DYLIB_PATH` is not supplied at build time, the build detects `brew --prefix onnxruntime` on macOS. Set `ORT_DYLIB_PATH` at runtime only to override that recorded path. A missing runtime produces an error naming this variable. `LIBCLANG_PATH` is build-only and can override the repository's development default.

`libheif` provides full HEIC decoding for pixel proofs; macOS ImageIO provides reduced-size HEIC decoding for retrieval. `jpeg-turbo` provides scaled JPEG decoding. `pkgconf` supplies `pkg-config` for native discovery. No build step downloads native binaries. `ffmpeg` and `ffprobe` must be on `PATH`.

### First run and platforms

```sh
eidolon fetch-models
eidolon ~/Pictures
```

`fetch-models` explicitly downloads and verifies the optional embedding model; local scans do not download models. Without it, scans still perform exact and non-neural matching.

Tested platform: **macOS arm64 (Apple Silicon)**. CI is configured for macOS 15 and 26 arm64. The Homebrew tap supports this platform only. ImageIO and CoreML are macOS-specific and conditionally compiled; other source-build platforms remain unverified.

Agents: copy [skills/eidolon](skills/eidolon/SKILL.md) into your agent's skills directory (for example `~/.claude/skills/eidolon/`).

Evaluation media and model weights are not redistributed. The source manifests in [eval-results](eval-results/README.md) retain attribution, licence URLs and source hashes. See [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) for Rust dependency terms/source access and [NOTICE](NOTICE) for optional model and native-library licensing.

### Source releases (maintainers)

When `Cargo.lock` changes, regenerate `THIRD_PARTY_LICENSES` with the command in [about.hbs](about.hbs) before releasing.

Commit the release tree, create and push its `v<version>` tag, then run `scripts/release.sh <version>`. The script archives only that tag into `dist/eidolon-<version>.tar.gz` and prints its SHA-256. Upload that exact tarball as a GitHub Release asset, then update the tap formula's asset URL and SHA-256. Open a tap pull request to build/test Apple Silicon bottles. Recompute the hash from the real release tag: a local preparation snapshot is not the release. All crates have `publish = false`; distribution uses GitHub and Homebrew, not the unrelated crates.io package named `eidolon`.

After all tap checks pass, review the complete PR diff (including workflows) and verify its current head SHA. Dispatch the tap's `publish.yml` on `main` with `pull_request` and the verified `head_sha` inputs. A changed head requires another review. The optional `pr-pull` label path requires an applicable Actions event policy explicitly allowing `pull_request_target`; GitHub's [default policy blocks that event for public repositories from December 2, 2026](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target#default-policy-for-pull_request_target).

## Scan

```sh
eidolon ~/Pictures ~/Movies notes.txt
find ~/Pictures -type f | eidolon --stdin
find ~/Pictures -type f -print0 | eidolon --stdin -0
eidolon --json --min-score 0.85 ~/Pictures > report.json
eidolon --thumbnails on ~/Pictures ~/Movies
eidolon --fast --video-fps 1 --thumbnails off ~/Pictures ~/Movies
eidolon --hidden --jobs 4 --no-embeddings --no-keypoints ~/Pictures
```

Directories are recursive. Dotfiles and dot-directories are skipped unless `--hidden` is set. Symlinks are not followed, including symlink input roots. Overlapping inputs are deduplicated by canonical absolute path. Files are classified by their contents, not their filename extensions. Text must be valid UTF-8, or valid BOM-prefixed UTF-16, without decoded NUL characters. Recognized documents and archives, including PDF (even ASCII-only), office formats, ZIP and RTF, are classified as `other`: they participate only in exact byte matching, not text extraction, near matching or the `skipped` list. Unknown binary containers are probed with `ffprobe`; unrecognized files still participate in exact matching.

`--stdin` reads one local path or URL per newline; `-0` / `--null` switches to NUL-separated inputs and preserves filenames containing newlines. Command-line and stdin inputs can be combined.

A local file classified as image, video or text that cannot be decoded or extracted—for example unsupported, truncated or corrupt media—is skipped for near-duplicate analysis, not fatal. Each skipped file prints one stderr line, `eidolon: skipped <path>: <reason>`, and remains eligible for exact byte-identical BLAKE3 matching with its original classified kind. Skipped inputs are excluded from unavailable near-duplicate and thumbnail comparisons. The human summary reports `<N> skipped files (near-duplicate analysis)`; JSON always includes the path-sorted `skipped` list of `{path, reason}` objects. Errors outside individual input decoding/extraction, including cache DB/serialization, model integrity, inference/native backend, worker setup and report-output I/O, remain fatal.

Individual URL failures produce stderr warnings and are counted in the transfer summary while other inputs continue; the exit status is nonzero if every input failed. Successful scans exit **0 even when duplicates are found**. Progress is stderr-only and hidden when stderr is not a terminal.

Every scan writes one `Effective settings: {...}` JSON line to stderr. JSON reports repeat the same resolved values in `settings`, including the actual worker count and selected embedding device (or `null` when unavailable). Fully cached scans do not initialize ONNX Runtime or create sessions; the device then describes the plan for new inference, not the provenance of every cached vector.

### URL inputs

```sh
eidolon ~/Pictures 'https://example.com/image.jpg' 'https://example.com/article'
printf '%s\n' '/path/to/local.mp4' 'https://example.com/video.mp4' | eidolon --json --stdin
eidolon --remote-video-mode full-cadence 'https://example.com/video.mp4' ~/Pictures
eidolon --offline ~/Pictures ~/Movies
```

Only explicitly supplied HTTP(S) URLs enable fetching. A webpage can yield discovered images (including `og:image`), posters, video analysis artifacts and extracted article text. The downloaded artifacts are classified by content like local files and cached under `~/Library/Caches/eidolon/remote`. Human reports display the actual resource URL and its role instead of the cache path; page provenance is retained in JSON.

Remote videos may be represented by bounded or remuxed analysis artifacts rather than a full original download. Only complete, unmodified resource bytes whose received length equals the known remote size enter the exact BLAKE3 stage. Video proxies, partial bodies and extracted article text still enter near-duplicate and frame matching, but cannot establish byte identity of the remote originals.

| Remote video mode | Accuracy / transfer tradeoff |
| --- | --- |
| `keyframes` (default) | Uses independent sync samples, bounded to 1,200 for indexed MP4s. Image-to-video candidates can fetch native-cadence windows around a sparse seed; frames without a usable seed and short clips between sparse samples can still be missed. |
| `full-cadence` | Preserves every video packet for the detector's normal sampling cadence. More accurate temporal coverage, but long GOP dependencies can require nearly the whole video track. |

Neither mode downloads the audio track for indexed MP4 sampling. There is no fixed transfer-percentage guarantee; compare measured bytes and results for your media. Mode-specific cache identities prevent a sparse proxy from being reused as full-cadence input.

`--max-fetch-fraction` defaults to `0.25`: native-window refinement stops before initial-plus-refinement body bytes would exceed one quarter of the known source size, and warns that recall may be limited. Initial acquisition can already exceed this allowance; this is not a total-transfer percentage guarantee. Set `--max-fetch-fraction 1` to disable the fractional refinement limit; the fetcher's absolute 64 MiB per-video cap still applies.

`--offline` rejects every URL input before any request, even if a cached artifact exists, and also rejects `fetch-models`. Local-only scans do not initialize the URL fetcher. After detection, each remote video gets a stderr note with its acquisition mode and reason, initial fetched bytes, additional native-GOP refinement bytes and combined total. The aggregate transfer summary includes response-body bytes from refinement, fetched HTML, probes and failed or partial discovered-resource transfers. It sums known representation sizes once per distinct source URL, including HTML. If any size is unavailable—including a discovered resource that fails before its size is known—the aggregate `total_size` is unknown (`None` internally): the summary prints `unknown total remote size`, with no numeric bytes-fetched ratio. A size learned from HEAD or Content-Range before a later failure remains known; consumed body bytes are counted either way. Probe traffic and revalidation can make transferred bytes differ from representation sizes. With `--json`, transport notes and the aggregate summary go to stderr; stdout remains one JSON object. The aggregate transfer summary is not serialized in that object.

### Optional models

```sh
eidolon fetch-models
eidolon fetch-models --model-dir /path/to/models
eidolon --model-dir /path/to/models ~/Pictures
```

`fetch-models` explicitly downloads the pinned embedding model and verifies its SHA-256. Scanning **never downloads a missing model**: it runs without embeddings and prints one hint to run `eidolon fetch-models`. `--no-embeddings` disables loading and the hint. Video comparisons use the same optional image embedder for sampled frames, or frame hashes when it is unavailable. `--no-keypoints` disables image keypoint verification. Disabling a signal can reduce near-duplicate recall, especially for local image edits and crops.

On macOS, `fetch-models` also initializes the Auto embedding backend after model verification, preparing the persistent local CoreML compilation cache when CoreML loads successfully. It reports the actual prepared backend: CoreML with its compiled cache, or a ready CPU fallback. CoreML success is not guaranteed. If local backend preparation fails entirely, the command returns an error; the verified downloaded model remains cached. Other platforms only download and verify the model.

### Cache and tuning

| Option | Default / behaviour |
| --- | --- |
| `--min-score <f>` | `0.75`; finite confidence threshold from 0 to 1. Exact matches always score 1. |
| `--fast` | Off; use cheaper aligned RGB thumbnail verification for images and cached RGB/texture corroboration with temporal alignment for videos. Proof-backed matches can join groups; any explicitly unverified candidates remain separate from groups and keep hints. |
| `--video-fps <f>` | `2`; positive, finite samples per second. The final default retains thumbnail recall: 1 fps lost 9.4 percentage points of thumbnail-overlay `frame_of` recall on `small4`, exceeding the allowed 1-point loss. Lower rates remain an explicit speed/recall tradeoff. |
| `--thumbnails <mode>` | `auto` enables image-to-video `frame_of` comparisons only when URL inputs were supplied. `on` also enables local comparisons; `off` never dispatches them. |
| `--remote-overlay <mode>` | `on` (default) or `off`; enables additional crop proposals and prioritized native refinement for captions on sparse remote videos. Adds URL work; local caption matching is unchanged. |
| `--image-source-bank <mode>` | `on` (default) or `off`; supplements whole-image candidates with source-center embeddings. `off` skips bank preparation and its inference-work estimate, trading some image recall for lower cold cost independently of `--thumbnails`. |
| `--embed-device <device>` | `auto`, `cpu` or `coreml`; default `auto`. Auto selects from uncached work: CPU through 2,193 estimated embeddings, otherwise CoreML on supported Macs, with CPU fallback on initialization failure. It reuses either current provider-qualified cache identity; explicit devices remain isolated. The selected device is reported separately from the requested policy. |
| `--max-fetch-fraction <f>` | `0.25`; finite value greater than 0 and at most 1. Limits native refinement; `1` disables the fractional limit. |
| `--fetch-fresh` | Off by default; reuses fresh cached URL sources, with a 30-second fallback without Cache-Control. Honors `max-age`/`Age` and `no-cache`/`no-store`. Without this flag, each native-refinement window sends a conditional HEAD (loopback: 1521 HEADs versus 18 enabled); real origins may rate-limit these requests. Measured 72–81% fewer requests and identical body bytes, not a speedup guarantee: the warm run was slower. |
| `--timings` | Off by default; prints `Video refinement timing:` JSON with provider/decode/proof clocks and counters to stderr when native refinement runs. |
| `--jobs <n>` | `0` = all available cores; limits per-file feature workers. |
| `--cache <path>` | `~/Library/Caches/eidolon/features.sqlite` |
| `--no-cache` | In-memory features only; does not disable the separate URL artifact cache. |
| `--model-dir <path>` | `~/Library/Caches/eidolon/models` |
| `--offline` | Reject URL inputs and explicit model downloads without making a request. |
| `--remote-video-mode <mode>` | `keyframes` or `full-cadence`; default `keyframes`. |

Features, classifications and full-file BLAKE3 hashes are cached by canonical path, file size, modification time and feature version. Exact detection first groups by size, then compares the first and last 16 KiB, then computes full-file hashes only for remaining collisions. This applies across all file kinds, with remote artifacts restricted to complete original bytes. Cache/model files and explicitly fetched URL artifacts are the only local data written by eidolon; redirecting stdout writes the report wherever you choose. The persistent feature database contains file paths and derived features; use `--no-cache` when those should not be retained. URL artifacts and their provenance remain in the separate remote cache.

## Relations

| Relation | Meaning |
| --- | --- |
| `identical` | Byte-for-byte identical, confirmed by full-file BLAKE3. |
| `same_content` | Same decoded or normalized content, different encoding or container. |
| `reencoded` | Same content at different resolution, quality, bitrate or format. |
| `edited` | Local overlays such as a watermark, logo or added text. |
| `variant` | Same design or composition in another version: recoloured, light vs dark, re-lit or re-framed. Related, not a duplicate. |
| `crop_of` | Image `a` is a cropped region of image `b`. |
| `clip_of` | Video `a` is a time segment of video `b`. |
| `excerpt_of` | Text `a` is contained in text `b`. |
| `frame_of` | Image `a` shows a frame of video `b`: a thumbnail, poster or screenshot, possibly resized, cropped or overlaid. |
| `similar` | Strong similarity without a more specific established relation. |

For `crop_of`, `clip_of` and `excerpt_of`, `a` is the part and `b` is the whole. For `frame_of`, `a` is the image and `b` is the video; it is the cross-kind relation and joins duplicate groups when verified. Other relations order endpoints by `FileId`. A confidence score is a detector score, not a guarantee of equivalence.

## Groups and keep hints

Verified duplicate matches are joined transitively into connected groups, ordered by member count descending. `variant` matches and matches carrying `unverified` evidence never connect groups or influence keep hints. A connection through another file does **not** establish a direct match; the human report labels these as “transitive to keep_hint (not directly verified)” and shows the actual link's relation, score and evidence. Direct reversed containment is printed as “contains crop/clip/excerpt of keep_hint”. Matching evidence prioritizes full hashes, thumbnail/geometric/containment verification, normalized content, embeddings, then perceptual or text hashes; raw values from unrelated signals are not treated as comparable confidence scores.

Related versions appear after the duplicate groups in a separate **Variants (related, not duplicates)** section, with each pair's score and strongest evidence. The summary counts duplicate matches and variants separately. A variant relationship is not a reason to discard either version.

Fast-mode image matches carry `thumbnail` evidence from aligned RGB agreement, normally using stride-4 samples of cached 128×128 RGB thumbnails (up to 32×32 samples). Borderline crop candidates are rechecked at full cached-thumbnail density before confirmation; the evidence identifies distributed-sample or adaptive full-preview proof. Its evidence value is normalized mean absolute error (lower is better), not a confidence score. Proof-backed matches participate in groups normally.

Any pair explicitly marked `unverified` appears under **Unverified candidates (not duplicate groups)**, and the summary counts it separately. It remains in JSON `matches` with its `unverified` evidence, but is not a reason to discard a file. Exact full-hash matches remain confirmed in fast mode.

`keep_hint` is only a deterministic quality heuristic, not a recommendation to delete the other files. It ranks known detector evidence lexicographically by:

1. Prefer a video over an image whenever the group contains a video; a thumbnail is never the keep hint instead of its video.
2. Resolution (pixel count), then bitrate, then content length (video seconds or normalized text characters).
3. Largest file size when the quality values tie or are unavailable.
4. Earliest modification time, then lowest file ID for a final deterministic tie-break.

An unknown quality value is ranked as zero. Compressed size, dimensions, duration and modification time cannot establish artistic quality, provenance or which edits you prefer. Inspect every candidate yourself.

## JSON contract

`--json` emits one object and a newline, with no progress text on stdout:

```json
{"version":1,"settings":{"fast":false,"video_fps":2.0,"embed_device_requested":"auto","embed_device":null,"thumbnails_requested":"auto","thumbnails":false,"remote_overlay":true,"image_source_bank":true,"max_fetch_fraction":0.25,"fetch_fresh":false,"timings":false,"min_score":0.75,"jobs":4,"no_embeddings":true,"no_keypoints":false,"remote_video_mode":"keyframes"},"files":[{"id":0,"path":"/absolute/file.txt","size":100,"mtime_ns":42,"kind":"text"}],"matches":[],"groups":[],"variants":[],"skipped":[]}
```

`FileId` is a bare number and indexes `files`. Each match has `a`, `b`, `relation`, `score` and a nonempty `evidence` list; evidence entries contain `signal`, `value` and `note`. Variant matches remain in `matches` with `"relation":"variant"` and are also projected into the top-level `variants` list as `{"a":0,"b":1,"score":0.95}` pairs. The contract remains `"version":1`; `settings`, `variants` and `skipped` are always present. Each duplicate group has `files` (file IDs) and `keep_hint` (one member ID). Files without a verified duplicate match remain in `files` but are not in a group, even if they participate in a variant pair or unverified candidate.

`skipped` is a path-sorted list of `{"path":"/canonical/input","reason":"decode/read diagnostic"}` objects, or `[]` when none. A listed file retains its classified kind and remains in `files` and exact BLAKE3 matching; only its unavailable near-duplicate or thumbnail analysis is skipped. Reasons are flattened to one line for stderr. Human output reports the count separately as `<N> skipped files (near-duplicate analysis)`.

Remote entries in `files` additionally contain an optional `url` object; local entries omit it. `path` remains the canonical local artifact path, and the file ID stays stable:

```json
{"url":"https://example.com/thumbnail.jpg","page_url":"https://example.com/article","role":"og:image","mode":"full","mode_reason":"complete_resource","initial_bytes_fetched":12500,"refinement_bytes_fetched":0,"bytes_fetched":12500,"total_size":12500,"budget_exhausted":false}
```

`page_url` and per-artifact `total_size` may be null. Unknown sizes never become zero; the separate aggregate stderr summary also omits a numeric transfer ratio when its total is unknown. `mode` describes initial artifact acquisition: `sparse` for indexed keyframe sampling (MP4 sample index, or Matroska/WebM Cues for VP8, VP9 and H.264), `full` for full-cadence or full-body acquisition, and `derived` for extracted text. `mode_reason` records the specific acquisition reason, including the actual cause when range-based video acquisition falls back (missing or unsupported index). Proof windows fetch only a bounded prefix from the preceding keyframe through the needed frame and its decode-order dependencies (plus small metadata read-ahead), not the whole GOP, and append `native_prefix_refinement` without relabeling a sparse acquisition as a full-download fallback. Very long GOPs can still make combined transfer approach the whole source size.

`initial_bytes_fetched` records artifact acquisition traffic and `refinement_bytes_fetched` records additional window traffic; their sum is `bytes_fetched`. The human/stderr aggregate also includes page and probe traffic across the run. URL provenance does not change match endpoint IDs or the meaning of the `variants` projection.

`budget_exhausted` becomes true when the native-refinement allowance prevents a requested window from being fetched; the corresponding stderr warning states that recall may be limited.

## Known limits

The final **holdout8** evaluation used 760 source-disjoint natural-source files from Wikimedia Commons and public-domain texts, with default settings plus `--thumbnails on` (default `auto` skips local thumbnail-to-video comparisons, so local `frame_of` rows below need that flag). It was scored once, with one cold local scan and one cold mixed-URL scan; these results are not a promise for other collections. The score predates three later fixes (remote window continuation, text template guard, docs), which were verified on tuning corpora only. [Scores, reports, timings and binary hashes](eval-results/holdout8/) retain the measured evidence; the earlier [holdout7](eval-results/holdout7/) evaluation of the pre-integration binary is kept for reference.

| Measured limitation | Result |
| --- | --- |
| Thumbnail with text overlay → video, direct recall | **0.81 local** (86/106), **0.50 remote video** (7/14). |
| Local overlay-thumbnail group recall | **0.99** (105/106), only when a clean frame of the same video is also in the input. A group link is not direct overlay verification. |
| All remote-video `frame_of` recall | **17/42 direct**, **21/42 grouped** (0.40 / 0.50). Sparse sampling and bounded refinement can miss the necessary frame. |
| Local video Variant recall | **0.33** (0.333); related versions can remain unreported. |
| Mixed-URL body transfer | **27%** (101,387,066 / 377,932,602 remote bytes; 0.2683), not a universal saving guarantee. |
| Cold scan wall time, 760 files | **541 s local / 505 s URL** (540.61 / 504.64 s). Cold refers to application feature/URL caches, not flushed OS caches. |
| First scan of an image-only folder | The default source-aspect embedding bank adds **33–41%** cold time (tuning corpora: 60.7 → 80.4 s for 1,320 images); warm rescans +2–3%. `--image-source-bank off` skips it at the cost of Variant/crop recall. |

Separately, `--fast` measured **1.34× image speedup** in an earlier tuning comparison and **6.02× video speedup** on warm small4 runs (6.041 s default versus 1.003 s with `--fast`, unchanged recall and precision 1.000), not this frozen holdout. These are observed speedups, not a 2× requirement or a guarantee. Image timing predates explicit CPU-provider isolation.

Approximate candidate retrieval, unsupported decoders and heavily altered or small overlaps limit recall. Report scores and deterministic keep hints are evidence to inspect, not safe automatic deletion decisions.

Remote thumbnail-to-frame results can vary with worker scheduling when parallel image queries share a native-refinement byte allowance: different native windows can become affordable in otherwise equivalent runs. Cached-window reuse reduces decoding work, but does not make shared-budget allocation deterministic.

## Privacy

Local-only scans and inference are offline and send no telemetry. **Network access happens only for explicit HTTP(S) URL inputs or the explicit `eidolon fetch-models` command.** `--offline` blocks both. URL resolution may request a supplied webpage and its discovered media; remote servers can observe those requests. Local file contents, hashes and embeddings are not uploaded. Model downloads are verified against pinned SHA-256 hashes. Scanning, URL fetching and model downloading never change your local input files.
