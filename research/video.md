# Video copy detection, segment localisation and thumbnail→frame matching

Research for `eidolon` (2026-10-29). Scope: video copy detection, segment localisation, thumbnail ↔ video-frame matching, video `Variant`, and decode or byte efficiency. No eidolon code was built, scanned or benchmarked. The only measurements below are small FFmpeg/NumPy micro-experiments on synthetic clips and macOS desktop-picture thumbnails in `/tmp`, and they are labelled as such. `[INFERENCE]` marks claims I derived and did not read in a source.

## Baseline facts that shape the ranking

These come from `crates/eidolon-eval/README.md`, `eval-results/holdout7/local-score.txt`, `crates/eidolon-eval/src/generate.rs` and `crates/eidolon-video/src/frames.rs`:

- **Thumbnail overlay.** `thumb_overlay` takes a centre square crop of the frame (540×540 out of 960×540 → 320×320). `title_bar()` then paints a dark `[16,16,24]` band over rows 284..320, about 11 % of the image, and writes bitmap glyphs "EIDOLON FRAME" on it. Local direct recall is 54/110. Clean `thumb_frame` and `thumb_poster` both reach 104/110. The overlay is the only difference, so the failure is the partial occlusion and not the frame lookup.
- **Existing margin exclusion.** eidolon already excludes uniform near-black margins. The title bar is not uniform because of the glyph pixels, so it is not excluded. `[INFERENCE]`
- **Video `Variant`.** The transform is `hue=h=135,eq=gamma=0.35` (`generate.rs:1099`). I recomputed the holdout7 local output against `truth.json`: 22/60 = 0.367. By partner: original 6/20, exact 6/20, `low_bitrate_360p` 10/20. Even full-length, same-clock pairs fail, so this is a per-frame descriptor problem and not a temporal-alignment problem. `[INFERENCE]`
- **Decode path.** Index decode is software, `-threads 1`, full decode followed by an `fps=` filter (`frames.rs:438-492`). No `-skip_frame` and no `-hwaccel` are used.
- **Local tools.** The installed FFmpeg is 9.0.2 with `videotoolbox` hwaccel, `scale_vt`, `scdet`, `signature` (MPEG-7), `-skip_frame` and `-skip_loop_filter` available (checked with `ffmpeg -hwaccels` / `-filters` / `-h full`).

## Ranked candidates

### 1. Occlusion-tolerant FrameOf proof: overlay localisation + masked correlation

**Weak spot:** overlay thumbnail → video, direct recall 0.49 local and 0.10 remote. It also helps `crop_watermark`.

**What it is.** Detect overlay regions on the *image* side (title bands, lower thirds, caption text). Exclude them from every stage that compares pixels: hash candidates, ORB keypoints and RGB proof. Then prove the match on the remaining area with a masked normalised cross-correlation. Precision is protected by requiring a minimum unmasked area, for example ≥ 70 % of the image `[INFERENCE: threshold to be set on a dev split]`, plus the existing geometric fit. This avoids inventing pixels (contrast with LaMa below).

**Components and sources**

- **Masked NCC/registration:** D. Padfield, "Masked Object Registration in the Fourier Domain", *IEEE Trans. Image Processing* 21(5), 2012 (volume from memory; DOI not checked). It computes NCC with per-pixel validity masks for both images in O(N log N), and the reference implementation is scikit-image `phase_cross_correlation(reference_mask=…)`. In eidolon's 32×32 / native-window proofs a direct masked sum is enough; no FFT is needed `[INFERENCE]`.
- **Overlay detection, cheap and deterministic:**
  - Caption and text localisation from edge density and colour uniformity: R. Lienhart, A. Wernicke, "Localizing and Segmenting Text in Images and Videos", *IEEE TCSVT* 12(4):256–268, 2002. The synthetic title bar, a low-variance band with sparse high-contrast glyphs, is exactly this case.
  - A band detector extending eidolon's existing margin logic would handle it in a few dozen lines `[INFERENCE]`.
- **Overlay detection, learned, for real YouTube thumbnails with stylised text:**
  - Apple Vision `VNDetectTextRectanglesRequest` / `VNRecognizeTextRequest` via the `objc2-vision` crate 0.3.2 (both requests are behind Cargo features; docs.rs `objc2_vision`). It runs on device and macOS only. Network use: none documented for these requests `[INFERENCE: verify with Little Snitch / nettop once]`.
  - Portable alternative: `ocrs` (robertknight/ocrs, about 1.9k stars, crate code MIT OR Apache-2.0, Rust + RTen ONNX runtime, "early preview", Latin only). Only its *detection* model is needed. Network: the CLI downloads models on first run into `~/.cache/ocrs`, so eidolon should vendor or pin the model files. The maintainer considers the model weights to inherit CC-BY-SA-4.0 from HierText (ocrs-models issue, July 2026), so check the licence before redistributing.

**Maturity.** Padfield's method is standard (scikit-image). Lienhart is a classic. Vision is a system framework.

**Measured numbers.** No source measures thumbnail↔frame recall under overlays. The closest literature is thumbnail *selection* (e.g. "To Click or Not To Click", arXiv 1609.01388), which is not a retrieval method. The evidence is eidolon's own clean-vs-overlay gap: 104/110 vs 54/110.

**Effort.** Band/caption mask plus masked RGB proof: 1–2 days. Vision text rectangles as an optional macOS mask source: +1 day.

**Telemetry.** None for the deterministic variant. ocrs fetches models only if not vendored.

### 2. Copy-detection-trained frame descriptor (SSCD or ISC21-1st) beside or instead of DINOv2

**Weak spots:** overlay thumbnails (candidate stage), `crop_40` / `crop_watermark`, remote FrameOf candidate recall.

**What it is.** DINOv2 is a generic SSL descriptor. SSCD is trained *for* copy detection, with augmentations that include `overlay_text` and `overlay_emoji` (files `sscd/transforms/overlay_text.py` and `overlay_emoji.py` in the repo), crops and mixup.

**Sources**

- E. Pizzi, S. Dutta Roy, S. N. Ravindra, P. Goyal, M. Douze, "A Self-Supervised Descriptor for Image Copy Detection", CVPR 2022, pp. 14532–14542, arXiv 2202.10261. Repo: facebookresearch/sscd-copy-detection.
- The VSC22 baselines use it per frame at 1 fps: E. Pizzi et al., "The 2023 Video Similarity Dataset and Challenge", arXiv 2306.09489; repo facebookresearch/vsc2022.
- Alternative: S. Yokoo, "Contrastive Learning with Large Memory Bank and Negative Embedding Subtraction for Accurate Copy Detection", arXiv 2112.04323. This is the ISC21 Descriptor Track 1st place: EfficientNetV2-M, 256-d. Repo lyakaap/ISC21-Descriptor-Track-1st, MIT, about 153 stars, weights in GitHub releases v1.0.0–v1.0.3 (last 2023-04-26).

**Measured numbers (from the sources)**

| Source | Benchmark | Result |
| --- | --- | --- |
| SSCD `sscd_disc_mixup` (ResNet-50, 512-d) | DISC2021 validation | µAP 61.4 % raw; 72.4 % with score normalisation (repo `docs/Evaluation.md`, reproducing paper Table 2: 61.5 / 72.5) |
| SSCD guidance | DISC pairs | cosine > 0.75 ⇒ copies at 90 % precision (README) |
| VSC22 baselines (training set) | Descriptor µAP / Matching µAP | SSCD 0.4754 / 0.3650 vs DINO 0.4402 / 0.3393 vs DnS 0.4129 / 0.3211 (vsc2022 README) |
| VSC22 winner | Descriptor track, SSCD-style training on an image-transformer backbone | Phase 2 AP 0.8717 (1st); the authors' paper text says "second prize". Ablation: basic 0.8580 → +frame-scene split 0.9075 → +edit detection 0.9492. T. Wang, F. Ma, Z. Liu, F. Rao, "A Dual-level Detection Method for Video Copy Detection", arXiv 2305.12361; repo FeipengMa6/VSC22-Submission, MIT. |

**Licence.** SSCD repo MIT. The weight files on dl.fbaipublicfiles.com carry no separate licence statement, so treat them as unconfirmed and check before shipping. The repo is archived (read-only) since 2023-10-31, about 418 stars.

**Rust availability.** No ONNX is published. Export with `torch.onnx.export` from the TorchScript or torchvision checkpoint; ResNet-50 exports trivially and runs through the existing `ort` path with CPU and CoreML providers. Cost is likely similar to DINOv2-S/14 at 224 px: ResNet-50 at 288 px is about 6.8 GFLOPs vs about 5.5 for ViT-S/14 `[INFERENCE]`.

**Effort.** Export plus a feature-key/model-id switch plus an A/B run: 1–2 days. Threshold calibration on a dev split: +½ day.

**Telemetry.** None. The model is a static file.

### 3. Hue-rotation-invariant chroma-magnitude hash for video `Variant`

**Weak spot:** video `Variant` recall 0.367, where the original↔variant pairs themselves fail. It also bears on the image `variant_hue` misses visible in the holdout7 top misses.

**What it is.** FFmpeg `hue=h=…` rotates the U/V chroma vector. `eq=gamma=…` changes only luma. The chroma magnitude `sqrt((U-128)²+(V-128)²)`, a saturation-like quantity, is therefore invariant to both. Luma-based hashes, including rank-normalised ones, cannot survive gamma 0.35: in my test 67 % of 32×32 luma pixels were crushed to ≤ 2. The theory is colour-invariant features: T. Gevers, A. W. M. Smeulders, "Color-based object recognition", *Pattern Recognition* 32(3):453–464, 1999, and J.-M. Geusebroek et al., "Color Invariance", *IEEE TPAMI* 23(12):1338–1350, 2001.

**Own micro-experiment** (`/tmp`, not eidolon). Setup: 40 macOS desktop-picture thumbnails, 32×32 area-scaled, DCT 8×8 pHash, transform `hue=h=135,eq=gamma=0.35`.

| Hash | Positive Hamming, median / max | Share of positives ≤ 14/64 | Negatives (780 pairs), min / 1st pct / median |
| --- | --- | --- | --- |
| Luma pHash | 12 / 26 | 0.675 | — |
| Rank-normalised luma pHash | 15 / 30 | 0.50 | — |
| Chroma-magnitude pHash | 1 / 4 | 1.00 | 8 / 17.6 / 32 |

In the negatives, the minimum of 8 is between light and dark versions of the same wallpaper, which are legitimate design variants.

**Caveat.** This invariance matches the *synthetic* transform exactly. Real variants (iMac colour series, relit photos) do not preserve chroma magnitude. Treat it as a candidate channel plus an explicit proof: per-frame chroma-magnitude correlation over an aligned run, plus the existing structural/edge agreement. It must not be treated as proof by itself. Reporting it as `Variant` (never duplicate) keeps duplicate precision untouched `[INFERENCE]`.

**Effort.** One extra 64-bit hash per frame (needs the 32×32 YUV thumbnail already decoded), LSH table and run verifier hookup: ½–1 day. It needs a feature-version bump (`video.frames/v7`).

**Telemetry.** None.

### 4. Keyframe-only and hardware decode for the index pass; targeted native windows

**Weak spots:** cold scan 418 s / 760 files; FrameOf native refinement about 590 s warm on 2,280 files; remote bytes.

**What it is**

- FFmpeg `-skip_frame nokey` makes the decoder discard non-keyframes. FFmpeg codec docs state that `nointra` is needed if strictly every I-frame is wanted.
- `-skip_loop_filter all` gives cheaper H.264/HEVC decode for candidate-only passes.
- `-hwaccel videotoolbox` plus `scale_vt` puts decode on Apple's media engine.
- `-lowres` exists but only for MPEG-1/2/4-part-2/MJPEG-class decoders, not H.264/HEVC.
- Compressed-domain DC-image matching (S. Bekhet, A. Ahmed, A. Altadmri, A. Hunter, "Compressed video matching: Frame-to-frame revisited", *Multimedia Tools and Applications* 75(23):15763–15778, 2016, DOI 10.1007/s11042-015-2887-8) reports 0.56 s vs 1.3 s per video pair, about 57 % less time. Keyframe-only decode gets the same benefit through FFmpeg without a custom bitstream parser `[INFERENCE]`.

**Own micro-measurement** (`/tmp`). Setup: 40 s 1080p30 `testsrc2`, x264 veryfast, GOP 60, decoded to 96×96 rgb24, M3 Max, single stream.

| Mode | Wall (s) | User CPU (s) | Frames decoded |
| --- | --- | --- | --- |
| Software, 1 thread | 3.14 | 3.54 | 1,200 |
| `-skip_frame nokey` | 0.12 | 0.13 | 20 keyframes |
| VideoToolbox | 3.55 | 0.99 | 1,200 |
| VideoToolbox + nokey | 0.21 | 0.08 | 20 keyframes |

- VideoToolbox does not make one stream faster, but it cuts CPU about 3.5×. That matters because eidolon runs `jobs` parallel decoders `[INFERENCE]`.
- A third-party application benchmark (M4 Pro, 1080p) reports 153 → 389 fps H.264 and 185 → 708 fps HEVC with VideoToolbox. It is not an FFmpeg-only measurement, so treat it as indicative.

**How to use it without losing recall or precision `[INFERENCE]`**

- **Coarse candidate index from keyframes.** Embeddings every 3 s are already sparse, and typical web GOPs are 1–5 s. Keep the 2 fps hashes for alignment but source embeddings from keyframes.
- **Hardware decode for the full-cadence passes.** Use VideoToolbox decode for native-window proofs.
- **No proof from skipped-filter output.** Proofs must never come from `-skip_loop_filter` output.
- **Determinism.** VideoToolbox output can differ by rounding from software decode. Cache keys must include the decoder identity, and proof thresholds need a check on both paths.

**Effort.** VideoToolbox for native windows: about 4 h plus validation. Keyframe-sourced embeddings: 1 day. Both need a feature-version bump.

**Telemetry.** None.

### 5. PDQ / vPDQ quality-gated frame hashes and pruning (Meta ThreatExchange)

**Weak spots:** speed (fewer embeddings) and a principled "featureless frame" gate.

**What it is.** PDQ is a 256-bit DCT hash with a 0–100 gradient **quality** score. vPDQ hashes frames (typically at 1 fps) and prunes consecutive frames within D_prune. Matching uses distance D (example 31), a quality filter F (example 50) and coverage thresholds P_c / P_q. TMK+PDQF builds a fixed-length whole-video signature of about 256 KB and does *not* support clip matching (vpdq/README.md).

**Sources.** facebook/ThreatExchange, `vpdq/` and `hashing/hashing.pdf`. Licence: BSD per the repo README, with listed exceptions (e.g. `pdq/cpp/CImg.h`). The repo is actively maintained. Benchmark: F. Morales, S. Sharma, C. Doan, B. Levine, "PHVSpec: A Benchmark-based Analysis of Perceptual Hash Systems for Videos", Tech Coalition, 2024-09-26.

**Measured numbers (PHVSpec Table 4, VCDB, best-F1 threshold)**

| System | Precision | Recall | Matching time (s per second of query video) | Hash time (s per second of video, PHVSpec Table 2) |
| --- | --- | --- | --- | --- |
| vPDQ | 0.98 | 0.41 | 0.0026 | 0.004–0.007 |
| TMK+PDQF | 0.88 | 0.24 | 0.0004 | — |
| PhotoDNA-for-Video (closed) | 0.99 | 0.70 | — | — |

**Rust availability.** `pdqhash` 0.1.1 on crates.io (pure Rust port, returns hash plus quality). `pdq-rs` 1.0.0 wraps Meta's C++.

**Relevance.** It is not a recall upgrade: eidolon's hash plus embedding plus alignment stack is stronger. The reusable ideas are the quality gate and D_prune pruning: skip embedding near-identical consecutive frames to cut DINOv2 calls on static content `[INFERENCE]`.

**Effort.** Pruning on existing hashes: ½ day. PDQ itself is not needed.

**Telemetry.** None (local library).

### 6. Learned segment localisation over frame-similarity matrices (VCSL, TransVCL, RTR)

**Weak spots:** future harder edits such as picture-in-picture, speed change and non-linear re-edits. These are not exercised by the current eval.

**Sources**

- **VCSL.** S. He et al., "A Large-scale Comprehensive Dataset and Copy-overlap Aware Evaluation Protocol for Segment-level Video Copy Detection", CVPR 2022, arXiv 2203.02654. Repo alipay/VCSL, MIT, about 146 stars. 160k copy pairs, more than 280k segment pairs. Ships baselines (DTW, TN, DP, SPD) and features.
- **TransVCL.** S. He et al., "TransVCL: Attention-enhanced Video Copy Localization Network with Flexible Supervision", AAAI 2023, 37(1):799–807, arXiv 2211.13090. Repo transvcl/TransVCL, MIT, about 61 stars.
- **RTR.** M. Lu, Y. Lu, S. Nie, X. Yang, X. Zhang, "Self-Supervised Video Copy Localization with Regional Token Representation", ECCV 2024. Repo eccv1818rtr/ECCV1818, about 4 stars, no licence file seen. It adds a regional token for picture-in-picture-style local copies.
- **VCDB**, the classic partial-copy benchmark: Y.-G. Jiang, Y. Jiang, J. Wang, "VCDB: A Large-Scale Database for Partial Copy Detection in Videos", ECCV 2014.

**Measured numbers.** RTR repo/paper, segment F1 / video F1:

| Model | VCSL | VCDB |
| --- | --- | --- |
| Self-supervised | 68.83 / 95.42 | 77.26 / 93.31 |
| Fine-tuned | 71.56 / 96.46 | 78.78 / 94.12 |
| TransVCL (as reported by RTR) | 66.51 / 90.06 | — |

**Rust availability.** None. The models are PyTorch: a transformer plus a YOLOX-style head over a similarity map. ONNX export is feasible but untested `[INFERENCE]`.

**Relevance.** eidolon's offset voting plus ordered-run verification already covers constant-offset clips (clip recall 0.90–0.95). Learned localisers mainly add value for speed-warped and non-linear edits. Their scores are not proofs, so a precision-1.0 system would still need eidolon's verifier.

**Effort.** 1–2 weeks including export and calibration. Low priority now.

### 7. ViSiL / DnS / S2VS: fine-grained video similarity with a selector

**Weak spot:** speed at thousands of files (architecture idea).

**Sources**

- **ViSiL.** G. Kordopatis-Zilos et al., "ViSiL: Fine-Grained Spatio-Temporal Video Similarity Learning", ICCV 2019, pp. 6351–6360. Repo MKLab-ITI/visil, Apache-2.0.
- **DnS.** G. Kordopatis-Zilos et al., "DnS: Distill-and-Select for Efficient and Accurate Video Indexing and Retrieval", IJCV 2022, arXiv 2106.13266. Repo mever-team/distill-and-select.
- **S2VS.** G. Kordopatis-Zilos et al., "Self-Supervised Video Similarity Learning", CVPRW 2023, arXiv 2304.03378. Repo gkordo/s2vs, MIT. Pretrained models are loaded via `torch.hub`.

**Measured numbers.** FIVR-200K DSVR, per query (DnS paper): ViSiL teacher mAP 0.899 at 451.9 s; `DnS5%B` 0.862 at 8.154 s; DnS 30 % re-rank 0.900–0.913 at about 45 s. PHVSpec dropped DnS from its benchmark for "unacceptable" speed and resource requirements.

**Relevance.** eidolon already follows the DnS shape: cheap hash/embedding candidates, then an expensive verifier on a small subset. The transferable lesson is to train or tune a **selector** that decides which candidate pairs deserve native-window refinement, instead of refining all seeds `[INFERENCE]`. The networks themselves are not worth importing.

**Effort.** A heuristic selector from existing evidence features: 1–2 days.

### 8. Scene-adaptive sampling (TransNetV2, FFmpeg `scdet`, PySceneDetect)

**Weak spots:** embedding count and speed. It can also help FrameOf when a thumbnail comes from a short shot.

**Sources.**

- **TransNetV2.** T. Souček, J. Lokoč, arXiv 2008.04838. Repo soCzech/TransNetV2, MIT. F1 77.9 ClipShots, 96.2 BBC Planet Earth, 93.9 RAI. No official ONNX.
- **FFmpeg `scdet`.** Built in.
- **PySceneDetect.** Python, BSD-3-Clause.

**Relevance.** Keep fixed 2 fps hashes, because the alignment and offset voting rely on a uniform clock. Add shot-boundary-triggered extra embeddings, plus vPDQ-style pruning of near-static spans, so that each shot gets at least one embedding even if it is shorter than 3 s `[INFERENCE]`.

**Effort.** `scdet` scores during the existing decode: ½–1 day.

**Telemetry.** None.

### Considered and not recommended

- **videohash** (akamhy, Python). A 64-bit hash of a frame collage; the docs say it is not meant for clip-in-video detection. Last PyPI release 3.0.1 on 2022-05-29. Weaker than eidolon's current stack.
- **TMK+PDQF.** Whole-video only; VCDB recall 0.24 (PHVSpec).
- **MPEG-7 Video Signature** (FFmpeg `signature` filter). The paper is S. Paschalakis et al., "The MPEG-7 Video Signature Tools for Content Identification", *IEEE TCSVT*, 2012: 95.49 % success at ≤ 5 ppm false alarms. My `/tmp` micro-test with FFmpeg 9.0.2 default thresholds reported "no matching" even for a 360p/90 kbit/s re-encode of a 40 s `testsrc2` clip. Synthetic content may be unrepresentative, but it is not drop-in.
- **Inpainting the overlay** (LaMa, arXiv 2109.07161, Apache-2.0; ONNX ports exist). About 1.6 s per 512² erase on a Ryzen 7 8845HS CPU (HF model card gumka/lama-onnx). It hallucinates pixels that a pixel proof must not trust. Masking (#1) is strictly better for a precision-1.0 system `[INFERENCE]`.
- **Hand-written compressed-domain parsers** (motion vectors, DC images). Superseded by `-skip_frame nokey` plus VideoToolbox (#4) at far lower maintenance cost `[INFERENCE]`.

### Remote FrameOf note (overlaps the remote-fetch research)

- **YouTube-class URLs.** yt-dlp exposes storyboard formats `sb0`–`sb3` (typically 320×180 down to 48×27 per tile, MHTML sprite sheets). They give a nearly free coarse timeline for thumbnail → timestamp seeding before any media range request. `[INFERENCE]`: interval and size vary per video and must be read from the metadata.
- **Generic HTTP.** The eval's loopback server falls in this case. Seed timestamps from the sparse keyframe proxy, then range-fetch only the GOP around the best seed. The win depends on the fetch planner; see the remote research file.

## Top-3 recommendation and the experiment that proves each

All tuning must use a freshly generated **dev** corpus (`eidolon-eval run <dir> --seed <new>`), never the frozen holdout7. Confirm once on a new frozen holdout. Gates for every experiment: duplicate direct/group precision = 1.000, hard-negative errors 0/0, and `Variant` precision ≥ 0.999.

1. **Masked overlay proof (#1).**
   - Experiment: report `thumb_overlay` direct recall, local and remote, before and after, plus a stage split of misses: candidate not nominated vs proof rejected.
   - Success: local direct ≥ 0.90 (clean thumbnails reach 0.945); `thumb_frame` / `thumb_poster` unchanged; zero new FrameOf false positives.
   - Also add a dev-only corpus variant with a top band and a centred caption, so the band detector is not fitted to the bottom bar.
2. **SSCD descriptor A/B (#2).**
   - Experiment: same corpus, run with `model=dinov2-small` and with `model=sscd_disc_mixup` (ONNX, same cosine LSH). Compare overlay, `crop_40`, `crop_watermark` and remote FrameOf candidate recall, i.e. the share of positives nominated before proof, and embeddings/s on CPU and CoreML.
   - Success: higher candidate recall on those categories at ≤ 1.2× embedding time, with precision gates unchanged.
3. **Chroma-magnitude Variant channel (#3).**
   - Experiment: `video/variant` direct recall split by partner (original, exact, 360p, plus clips if added), and image `variant_hue`.
   - Success: video Variant ≥ 0.80 with `Variant` precision ≥ 0.999. Also add a dev-only real-colour-series video negative and positive set, so the channel is not credited only for the synthetic `hue+gamma` transform.

Runner-up for pure speed: #4. Experiment: VideoToolbox decode for native windows plus keyframe-sourced embeddings, measuring cold-scan wall time and FrameOf stage time at `--jobs` default. Success: ≥ 25 % less wall time with identical relation output on the dev corpus.
