# eidolon-video

Report-only video duplicate detection and image → video-frame containment. Public entry points are `find` and `find_frames`; the latter returns directional `FrameOf` matches with source timestamps. Inputs are never modified. FFmpeg/FFprobe decoding is offline and restricted to local `file,pipe` protocols.

## Video matching

Versioned frame descriptors combine perceptual/difference hashes, regional hashes, color/texture tiles, and optional injected embeddings. Candidate offsets require ordered multi-frame agreement. Independent spatial verification distinguishes contained clips, re-encodings, localized edits, and broad color/tone variants; semantic similarity alone never proves a match.

Global affine and nonlinear color variants require visible changes across at least three rows **and** three columns of the spatial grid. This footprint is measured across all RGB pixels before selecting structurally aligned affine-fit tiles: a global hue transform can itself change luminance structure. The affine fit still needs its original spatial coverage, texture, correlation, and residual-error gates. A single wide or tall edited band does not become a global variant merely because its distinct colors admit a fitted transform.

## Thumbnail → frame matching

Ordinary queries nominate videos through the frame indexes, rank bounded source samples, and refine promising source windows at native cadence. RGB/texture verification supports resized frames, crops, and images containing a video viewport. Fast mode can finish on independently proven sampled RGB rectangles; unresolved queries still receive native-pixel refinement.

A caption-specific path detects repeated high-contrast, near-neutral strokes on a uniform full-width horizontal backing in the 96px image preview. It can mask one band at the top, middle, or bottom, including its resampling halo, while preserving at least 70% of the image. Blank padding, an oversized band, or texture without the required strokes does not establish a caption mask. Detection uses backing uniformity and horizontal contrast runs, not OCR, a caption string, a model, or an OS service (following the text-localization approach of Lienhart and Wernicke, 2002).

Caption-bearing queries do not use the obscured hashes or embeddings for nomination. Masked color/correlation comparisons rank bounded samples under full-frame and centered aspect-correct crop mappings. Their native search windows retain at least the 1.5-second uncertainty used for embedding nominations, since coarse cropped pixels can alias fine detail. Independent native-pixel proof uses masked normalized correlation (the exclusion principle described by Padfield, 2012), color error, textured spatial consensus, and a scattered-outlier bound. Excluded pixels never become replacement pixels or positive evidence. The visible support must cover at least 70% of the compared rectangle, have correlation ≥0.97, RGB RMS error <12/255, at least eight agreeing spatial tiles including four textured tiles, and at most 1/32 outliers. Caption detection or candidate rank alone cannot establish `FrameOf`.

When a caption-bearing query is compared with a separately branded source frame, an abrupt solid majority-width band can also be excluded on the source. That band is detected from the source's own color uniformity and boundaries, never from disagreement with the query. Detection is memoized for both cached nomination planes and native proof frames without changing cached feature bytes. The combined exclusions must still leave at least 70% visible proof support.

With `Config::remote_overlay` enabled (`--remote-overlay on`, the CLI default), sparse remote videos also use bounded translated/scaled crop proposals when the centered masked nomination fails. Stronger caption queries receive the source's shared native allowance first, in a deterministic per-source order. Their strongest retained-frame proposals are visited before expanding their neighborhoods; the first prefix starts on or after its keyframe to avoid pulling the preceding long GOP. Overlapping proposals share one prefix grid. An unavailable prefix does not rule out another cheaper or cached proposal. This scheduling never proves a match or lowers the pixel-proof thresholds. `--remote-overlay off` disables this additional remote search while retaining centered masked nomination and ordinary window order. Local caption matching is unaffected by the option.

Successful native-window decodes are shared across the image queries for one source during the current scan. Both retained window count and decoded-frame count are bounded by the source descriptor's configured frame-count cap; the cache is dropped when that source's queries finish. Refusals and errors are not cached, and every query still receives independent spatial proof against the reused pixels. No persistent feature bytes or cache identities change.

All remote queries, including ordinary non-caption images and `--remote-overlay off`, continue to the next independent candidate window when the provider refuses a prefix (`Ok(None)`). A refusal skips the rest of that window, not the whole query: another window may be cheaper or already cached. The fetcher's cumulative byte cap and sticky `budget_exhausted` provenance remain unchanged.

A retained sparse frame suppresses native refinement only if its actual score (`0.75 + quality * 0.25`) meets `Config::min_score`. A spatially verified retained frame below that threshold remains eligible for native refinement; mere presence of retained proof is not sufficient to stop the search.

With `Config::timings` / `--timings`, each source that requests native refinement emits a `Video refinement timing:` JSON line on stderr, keyed by `file_id`, with `provider_seconds`, `decode_seconds`, `proof_seconds`, their respective `_calls` counts, and `cache_hits`. The provider span includes cache validation, HTTP, prefix materialization, and source-account lock waits; it is not a pure HTTP-wait clock. Decode includes native-window probing, decoding, and proof-feature extraction. Proof measures independent native-frame spatial comparison. These are cumulative per-call elapsed times: concurrent calls overlap, so their sums are not a partition of total scan wall time. Initial acquisition, retained-frame proof, and video-to-video matching are outside these spans.

The mask is limited to a single full-width uniform-backed band; free-floating, multiple, very large, or low-contrast captions are not specially localized. They retain the ordinary proof path when no band is detected.

## Limits

Conservative caption proof can reject genuine direct pairs even when grouping recovers their relationship. The approved corrected5 capture still has 24 unproven overlay pairs, including the `034` overlay against its original, exact, letterboxed, watermarked, and quarter-length sources; the `041` overlay against its letterboxed source; and the `056` overlay against its watermarked source. Unannotated predictions are not established positives or negatives, so their disappearance must not be counted as a precision gain.

Remote captions are not exhaustively covered by a fractional byte allowance. A useful native prefix can remain unaffordable after other proposals use the allowance; cached/fitting request reuse does not eliminate every miss. When image queries share that allowance in parallel, worker scheduling can change which native windows fit, so remote `FrameOf` direct results can vary between otherwise equivalent runs. This nondeterminism is not fixed by decoded-window reuse. The current corrected5 on/off comparison gains nine annotated overlay edges but loses the `video_048_thumb_frame.jpg` and `video_048_thumb_poster.jpg` direct edges against `048_letterboxed.mp4`; both clean and poster group recall nevertheless increase.

Video-to-video direct retrieval can depend on lexicographic input/cache-path order. In the retained small4 URL comparison, `media/video/002_original.mp4` ↔ `media/video/002_watermark.mp4` was reported when the remote watermark preceded the local original, but disappeared after moving the URL cache changed their pair orientation. Watermark group recall remained 1.000. This orientation-sensitive direct-match defect is not fixed by caption scheduling.

## Feature cache

Video descriptors use `video.frames/v6`; still-image frame features use `video.image-frame/v6` and include the detected text-band mask. Keys include sampling settings and embedder identity. Native refinement frames omit nomination-only hash planes. No model downloads or network access occur in this crate.

Video frames and thumbnail-side retrieval both use `Embedder::embed_frame`, `frame_model_id`, and `frame_cache_model_ids`, including uncached-work estimates. Their policy and cache identities are independent of ordinary image ↔ image retrieval. A provider may therefore retain center-cropped frame embeddings while its ordinary image embeddings use whole-image preprocessing.

## Checks

```sh
cargo fmt -p eidolon-video -- --check
cargo clippy -p eidolon-video --all-targets --locked -- -D warnings
cargo nextest run -p eidolon-video --locked
```
