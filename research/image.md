# Image near-duplicate / copy detection: what eidolon could adopt

Research only, 2026-10-29. Scope: still-image candidate generation and proof (crop, watermark, overlay, Variant), plus the image side of thumbnail → video (`FrameOf`). Nothing here was run on eidolon; every eidolon impact claim is marked [INFERENCE] until the experiment at the end is done.

## eidolon facts this report relies on (read from source, not run)

- Embedding preprocessing crops the **central square** of the short side, resizes to 256 and centre-crops 224 (`crates/eidolon-image/src/embed.rs:543-553`, cache key `imagenet-center224`). For a 16:9 picture the model therefore sees about 49 % of the width and 87.5 % of the height, roughly 43 % of the area.
- The embedding is the DINOv2-small `pooler_output` (CLS), 384-d (`embed.rs:578-581`).
- Homography uses `geometry::RANSAC`, 3.5 px threshold, 2000 iterations (`src/geometry.rs:37-43`). ORB: 4000 points, scale 1.2, 8 levels (`src/features.rs:319-322`).
- There is no mirror/rotation handling apart from EXIF/HEIF orientation (no `flip`/`mirror`/`dihedral` in `eidolon-image/src`).
- Eval Variant transforms are strong: `huerotate(135)`, light/dark, gamma 0.35 / 2.8 (`crates/eidolon-eval/src/generate.rs:500-502`). Crops are 80/60/40 % of the area; `crop_watermark_q25` is a cropped, watermarked JPEG q25.

## Summary ranking

| # | Candidate | Weak spot it targets | Effort | New dependency |
|---|---|---|---|---|
| 1 | Full-frame (squashed) embedding input instead of centre crop | crop_40, crop_watermark, thumbnail ↔ frame candidates | 2-4 h | none |
| 2 | Copy-detection descriptor: CopyNCE ViT-S (or SSCD ResNet-50) instead of / beside DINOv2 CLS | thumbnail with text overlay, crops, watermark | 2-3 days | one ONNX model |
| 3 | D2LV "global-local" region embeddings | crop_40, overlays, thumbnail = region of a frame | 1-2 days | none |
| 4 | OpenCV `USAC_MAGSAC` + RootSIFT fallback for ORB-failed pairs | crop_40, crop_watermark | 4-8 h | none (OpenCV 5 already linked) |
| 5 | Variant candidates from ordinal (rank) luminance hash and grayscale embedding | image Variant 0.84, video Variant 0.37 | 1 day | none |
| 6 | Learned local features: XFeat + LighterGlue, or ALIKED/RaCo + LightGlue (ONNX via `ort`) | crop/watermark geometry when ORB fails | 3-5 days | one or two ONNX models |
| 7 | RDCD compact descriptor (EfficientNet-B0, 64-256-d) | embedding CPU cost (57 emb/s) | 2-3 days | one ONNX model |
| 8 | PDQ with dihedral hashes; survey of existing dedup tools | flips/rotations (not in eval); nothing else | 1 day | `pdqhash` crate |

Precision rule for all of them: they only change **candidate generation or geometric registration**. The existing pixel/structural proof stays mandatory, so precision 1.000 should hold by construction [INFERENCE]; every experiment must still check it.

---

## 1. Full-frame embedding input (no centre crop)

**What.** Feed the whole image to the embedder by resizing it directly to 224×224 (aspect not preserved) instead of cutting the central square. SSCD's authors recommend exactly this for copy detection: "Copy detection using square tensors benefits from directly resizing to the target tensor size. This skews the image, and does not preserve aspect ratio."

**Source.** SSCD README, "Preprocessing": https://github.com/facebookresearch/sscd-copy-detection. Paper: Pizzi, Roy, Ravindra, Goyal, Douze, *A Self-Supervised Descriptor for Image Copy Detection*, CVPR 2022, arXiv:2202.10261, Table 9.

**Numbers.** Copydays CD10K "strong" mAP for SSCD ResNet-50: 84.9 (224² squashed), 87.4 (320² squashed), 86.6 (short side 288, no crop). These are SSCD numbers; there is no published DINOv2 squash-vs-crop ablation.

**Why it matters for eidolon.** A `crop_40` that comes from a corner of the original shares little of the original's *central* square. A 16:9 thumbnail and a 16:9 frame lose the same outer 57 % of the width, so overlays or crops near the edges change the CLS vector more than needed [INFERENCE]. DINOv2 is a ViT, so non-square aspect is not a problem in principle; CoreML needs the static 1×3×224×224 shape, which squashing keeps.

- Licence / maturity: no code to adopt.
- Rust: change one function in `embed.rs`; bump the preprocessing part of the cache key.
- Effort: 2-4 h plus a cold re-embed of the eval corpus.
- Telemetry: none.
- Risk: squashed images of different aspect ratios move further apart. This affects only candidate ranking; proof is unchanged.

## 2. Copy-detection descriptor (CopyNCE ViT-S, SSCD ResNet-50)

**What.** DINOv2 is a general semantic model. Copy-detection (ICD) descriptors are trained so that edited copies (crops, text/emoji overlays, colour jitter, blur, JPEG) stay close and *different photos of the same scene* stay far. That is eidolon's definition of a duplicate.

**Candidates.**

- **CopyNCE** (best released descriptor). Lu, Nie, Lu, Yang, Zhang, Zhang, *Tracing Copied Pixels and Regularizing Patch Affinity in Copy Detection*, ICCV 2025, pp. 19248-19257, arXiv:2602.17484. Code: https://github.com/eddielyc/CopyNCE (Apache-2.0, 7 stars, released checkpoints on Google Drive).
  - Released descriptor ViT-S/16, 224×224, not fine-tuned: **70.5 µAP / 63.6 RP90 on DISC21**. Paper's best descriptor (fine-tuned, 256-d): 72.6 µAP / 68.4 RP90.
  - Released **matcher** (pairwise ViT-S, 8 encoder + 4 fusion layers): 83.5 µAP / 75.4 RP90; best (336²) 88.7 µAP / 83.9 RP90, above the ISC21 winner D2LV, which ensembled 33 models.
  - Backbone is DINO (v1) ViT-S/16 with a 512-d head (`core/configs/descriptor_default_config.yaml`). ViT-S/16 at 224 has 196 patches vs DINOv2-S/14's 256, so per-image cost should be equal or ~20 % lower [INFERENCE].
- **SSCD** (mature fallback). Same paper as item 1; repo https://github.com/facebookresearch/sscd-copy-detection (MIT, 418 stars, archived 2023-10-31; weights on `dl.fbaipublicfiles.com`).
  - DISC21 val, `sscd_disc_mixup` ResNet-50 512-d: **61.5 µAP, 72.5 µAP with score normalisation**; `sscd_disc_large` ResNeXt-101 1024-d: 63.7 / 75.3.
  - Same table, DINO ViT-B/16 (1536-d): **32.2 / 53.8**. That is the closest published proxy for "self-supervised general model used for copy detection".
  - README: for `sscd_disc_mixup`, cosine > 0.75 gives 90 % precision on DISC pairs.
  - Training augmentations include `OverlayText`, `OverlayEmoji`, `ColorJitter(0.8, 0.8, 0.8, hue 0.2)`, `RandomGrayscale(0.2)` and mixup (`sscd/transforms/settings.py`). Text overlay is in the training distribution; that is the eidolon thumbnail weak spot.
- **DINOv2 / DINOv3 / CLIP / SigLIP.** No published DISC21 µAP was found for DINOv2, DINOv3, CLIP or SigLIP. DINOv3 (arXiv:2508.10104) reports instance retrieval only for ViT-7B, and its weights use the custom "DINOv3 License", not Apache. CLIP/SigLIP are caption-trained, i.e. even more semantic [INFERENCE]. Immich uses CLIP for duplicates with a very tight distance (see item 8), which suggests CLIP needs a near-identity threshold for this job.

**Weak spots addressed.** Thumbnail with text overlay → video (0.49 local) through better thumbnail ↔ frame candidates. Also crop_40 and crop_watermark candidate recall [INFERENCE].

**Rust path.** Export to ONNX once, offline: SSCD from the `.torchvision.pt` weights plus GeM (`sscd/models/model.py`); CopyNCE from its PyTorch checkpoint with `torch.onnx.export`. Then run it through the existing `ort` session pool and CoreML path. Pin SHA-256 as `models::fetch` does today. Google Drive is not a stable host; re-host the converted file. Apache-2.0 (CopyNCE) and MIT (SSCD) allow redistribution.

**Effort.** 2-3 days: export with a parity check (PyTorch vs ORT cosine), embedder generalisation, cache key, Auto CPU/CoreML crossover re-measurement, threshold calibration (0.82 is DINOv2-specific).

**Telemetry.** None at inference; ORT telemetry is already disabled. A one-time export needs PyTorch; if `huggingface_hub` is used for downloads, set `HF_HUB_DISABLE_TELEMETRY=1`.

**Matcher use (optional, later).** The CopyNCE matcher is a pairwise scorer. Use it only to *rank* rejected candidates before the expensive 17-scale sweep and FrameOf native-frame proofs, never as proof.

## 3. Global-local region embeddings (D2LV)

**What.** Wang, Sun, Zhang, Yang, *D²LV: A Data-Driven and Local-Verification Approach for Image Copy Detection*, arXiv:2111.07090. First place in the ISC 2021 matching track (0.8329 µAP, 1,103 participants). Code: https://github.com/WangWenhao0716/ISC-Track1-Submission.

- *Global-local*: reference images are split into 2×2 and 3×3 tiles plus "exact centre" and "1/3 centre" crops; each tile gets a global descriptor; the query's global descriptor is compared with every tile (max score).
- *Local-global*: query patches (rotations, centre crops, selective-search proposals, YOLOv5-detected overlays) against the reference's global descriptor.

**Numbers (paper Table 2, DISC21 Phase 2).** Global-only 70.81 µAP → global-local **82.73** → both **83.72**. This is the largest single gain in their ablation, larger than the advanced augmentations.

**For eidolon.** This is the embedding version of the existing 27-region hash bank. `crop_40` and a thumbnail that shows only part of a frame should match one tile far better than the whole image [INFERENCE]. Cost: +4 (2×2) or +13 (2×2 + 3×3) embeddings per image. Restrict it to images whose global kNN did not produce a proven crop/frame match, or compute tiles only for the larger member. Store tile vectors in the same feature cache.

- Licence: method only (paper); code repo has no extra dependency worth taking.
- Effort: 1-2 days.
- Telemetry: none.
- Speed risk: tiles multiply embedding work; with CoreML at 251 emb/s, +4 tiles on 760 files is about 12 s [INFERENCE: arithmetic from the README throughput].

## 4. OpenCV USAC_MAGSAC and RootSIFT (no new dependency)

**What.** Two drop-in changes inside the already linked OpenCV 5:

1. Replace `geometry::RANSAC` with `USAC_MAGSAC` (or `USAC_ACCURATE`) in `find_homography`. OpenCV's USAC framework (since 4.5.0) adds LO-RANSAC/GC-RANSAC/MAGSAC++, SPRT and degeneracy tests.
2. For pairs where ORB fails, retry with SIFT, using the RootSIFT descriptor mapping (L1-normalise, then square root) and the same mutual ratio test.

**Sources.**

- Jin, Mishkin, Mishchuk, Matas, Fua, Yi, Trulls, *Image Matching across Wide Baselines: From Paper to Practice*, IJCV 129:517-547 (2021), arXiv:2003.01587. Stereo mAA@10°, 8k features, DEGENSAC: **RootSIFT 0.3125 vs ORB 0.0819**.
- Mishkin, "Evaluating OpenCV new RANSACs" (2021): https://ducha-aiki.github.io/wide-baseline-stereo-blog/posts/2021-05-17-opencv-new-ransacs.html. The legacy `RANSAC` flag was the worst method on all three IMC 2021 datasets. `USAC_ACCURATE` gave strong results within 0.01 s at 10k iterations. `USAC_MAGSAC` was the only method whose optimal threshold was the same across all datasets.
- ISC 2021 third place (matching track) used ~600 SIFT features per 300 px image. SIFT was weaker than the global descriptor in Phase 1 but "significantly boosted" the Phase 2 score (Sun et al., arXiv:2112.02373).

**Caveat.** Both benchmarks are wide-baseline 3-D scenes, not 2-D edits. The benefit for crop/JPEG/watermark is plausible but unmeasured [INFERENCE].

- Licence: OpenCV Apache-2.0; the SIFT patent expired in 2020 and SIFT is in the main `features` module.
- Rust: `opencv` crate 0.101.1 already exposes `SIFT::create` and the `USAC_*` flags.
- Effort: 4-8 h; SIFT descriptors are 128 floats (512 B/point), so cache them only for the fallback path or not at all.
- Telemetry: none.

## 5. Variant candidates: ordinal luminance hash and grayscale embedding

**Problem.** The eval's Variant transforms are outside the training invariances of the neural candidates. `huerotate(135)` exceeds DINOv2's colour-jitter hue range of 0.1 (±36°; `dinov2/data/augmentations.py`) and SSCD's 0.2 (±72°). Gamma 0.35 / 2.8 is far beyond brightness jitter 0.4. pHash/dHash on raw luminance also move under strong gamma. So a recall loss could come from **candidate generation**, not the proof [INFERENCE]. Find out first: see the experiment at the end.

**Options.**

- **Ordinal (rank) measures.** Bhat & Nayar, *Ordinal Measures for Image Correspondence*, IEEE TPAMI 20(4):415-423, 1998, doi:10.1109/34.677275. Rank order of intensities is invariant to any monotone tone curve (gamma, brightness, contrast). Cheap version: histogram-equalise (rank-normalise) the cached 128² luminance preview, then compute the existing pHash/dHash on it and add one more banded index. eidolon already rank-normalises channels in the structural proof, so the code exists.
- **Hue-invariant embedding input.** Embed the luminance, replicated to three channels. Both DINOv2 and SSCD trained with `RandomGrayscale(p=0.2)`, so grayscale input is in-distribution. A hue rotation that roughly preserves luma should then give a near-identical vector [INFERENCE]. Combine with histogram equalisation for relight. This costs a second embedding per image, or replaces the colour one if item 2's experiment shows no loss.
- **Light/dark (inversion).** The existing complemented hashes already handle it.

- Licence / maturity: methods only.
- Effort: 1 day for the ordinal hash; +0.5 day for the grayscale embedding.
- Telemetry: none.
- Video Variant (0.37) uses per-frame hashes, so the same ordinal hash applies to frames [INFERENCE].

## 6. Learned local features (XFeat, ALIKED/RaCo + LightGlue)

**What.** Learned keypoint detectors/descriptors and an attention matcher. They replace ORB only for pairs that ORB fails to register.

**Candidates and numbers.**

- **XFeat**: Potje, Cadar, Araujo, Martins, Nascimento, *XFeat: Accelerated Features for Lightweight Image Matching*, CVPR 2024, arXiv:2404.19174; https://github.com/verlab/accelerated_features (Apache-2.0, 1.76k stars). MegaDepth-1500 AUC@10° and FPS at VGA on an Intel i5-1135G7 CPU: **ORB 27.6 / 44.3 FPS; SuperPoint 50.1 / 3.0 FPS; XFeat 56.4 / 27.1 FPS; XFeat\* (semi-dense) 65.4 / 19.2 FPS**. 64-d descriptors. "LighterGlue" matcher is about 3× faster than LightGlue. ONNX: community export https://github.com/DavideCatto/XFeat-ONNX (Apache-2.0, 51 stars, last push 2024-06); the upstream README asks for an official ONNX export.
- **LightGlue**: Lindenberger, Sarlin, Pollefeys, ICCV 2023, arXiv:2306.13643; https://github.com/cvg/LightGlue (Apache-2.0, 4.85k stars). CPU (i7-10700K): 20 FPS at 512 keypoints; GPU RTX 3080: 150 FPS at 1024 keypoints. Pretrained for SuperPoint, DISK, ALIKED and SIFT.
  - **Licence trap:** SuperPoint weights and inference file use Magic Leap's restrictive non-commercial licence. DISK is Apache-2.0; **ALIKED** (arXiv:2304.03608, https://github.com/Shiaoming/ALIKED) is BSD-3-Clause.
- **LightGlue-ONNX**: https://github.com/fabio-sim/LightGlue-ONNX (Apache-2.0, 821 stars). Release v3.0 (2026-07-19) ships end-to-end `raco_aliked_lightglue_pipeline_k512 … k3584.onnx`; v2.0 (2024-07-17) ships DISK and SuperPoint pipelines. **RaCo** = Shenoi, Lindenberger, Sarlin, Pollefeys, *RaCo: Ranking and Covariance for Practical Learned Keypoints*, 3DV 2026, arXiv:2602.15755 (https://github.com/cvg/RaCo, Apache-2.0). It ranks keypoints for small budgets and reports robustness to large in-plane rotation.

**Rust.** No maintained Rust crate for any of them (crates.io and GitHub searches for `lightglue`, `xfeat`, `superpoint`, `aliked` found only toy repos). Path: ONNX through the existing `ort` pool. LightGlue's dynamic keypoint count is awkward for CoreML static shapes; the fixed-`k` v3.0 pipelines suit it better [INFERENCE].

**Caveat.** All are trained on MegaDepth-style 3-D viewpoint change, not JPEG q25/watermark/strong downscale. Gains on 2-D copy edits are unproven [INFERENCE]. Try item 4 first: it is free and covers the same failure class.

- Effort: 3-5 days, including keypoint-to-homography plumbing and keeping the RANSAC coverage gates.
- Telemetry: none in the ONNX files; model download via GitHub releases.

## 7. RDCD compact descriptor (speed)

**What.** Kim, Woo, Nang, *Relational Self-supervised Distillation with Compact Descriptors for Image Copy Detection*, WACV 2025, arXiv:2405.17928; https://github.com/juntae9926/RDCD (MIT, 7 stars). Distils SSCD into small students.

**Numbers (README, DISC21 µAPsn).** EfficientNet-B0 (4.7 M params): 64-d **53.5**, 128-d **61.1**, 256-d **65.7** (weights released on Google Drive). FastViT-T12 256-d: 67.4 (no weights link). For comparison: SSCD ResNet-50 512-d 72.5, DINO ViT-B/16 53.8. EfficientNet-B0 needs about 0.39 GFLOPs at 224² vs about 4-6 GFLOPs for ViT-S [INFERENCE: standard FLOP counts, not measured here].

**For eidolon.** About 10× less embedding compute and a 4-6× smaller vector, at copy-detection quality still above DINO [INFERENCE]. It matters if embeddings dominate the 418 s cold scan, and for video frames (embedding every 3 s). It is weaker than item 2, so use it only if item 2 is too slow.

- Effort: 2-3 days (same export path as item 2).
- Telemetry: none; re-host the Google Drive weights.

## 8. PDQ, and what existing dedup tools do

**PDQ.** Meta ThreatExchange, https://github.com/facebook/ThreatExchange/tree/main/pdq (BSD licence). It is a 256-bit DCT hash with a quality score. The README suggests a match distance ≤ 31 and discarding quality ≤ 49 as starting points. It can cheaply emit 8 **dihedral** hashes (rotations/flips), though rotation invariance is not exact. Rust: `pdqhash` 0.1.1 (Apache-2.0, 57k downloads, 2022, listed upstream) or `yume-pdq` 1.1.0 (Apache-2.0, SIMD; its NEON path needs nightly `portable-simd`).
- Value for eidolon: low for the measured weak spots. pHash/dHash already cover this ground. It would add mirror/90° rotation recall, which the eval does not test. Effort: about 1 day.

**Existing tools: nothing to adopt; eidolon is already ahead.** None of them proves matches, finds direction (crop vs whole), or separates Variant from duplicate.

| Tool | Method (primary source) | Licence / maturity |
|---|---|---|
| czkawka | `image_hasher` hashes (Gradient, VertGradient, DoubleGradient, Blockhash, Mean, Median; 8-64 bit) + BK-tree on Hamming distance ([Instruction_Core.md](https://github.com/qarmin/czkawka/blob/master/instructions/Instruction_Core.md)) | MIT core, 34.6k stars, v12.0.2 2026-09-09 |
| imagededup | PHash/DHash/WHash/AHash or MobileNetV3 CNN cosine ([README](https://github.com/idealo/imagededup)) | Apache-2.0, 5.7k stars, v0.3.3.post2 2025-08 |
| dupeGuru | 15×15 grid of average colours, exhaustive pairwise comparison ([scan docs](https://dupeguru.voltaicideas.net/help/en/scan.html)) | GPL-3.0, 7.9k stars |
| AllDup | aHash/bHash/dHash/mHash/pHash, optional 90°/180° rotations and flips ([docs](https://alldup.info/en/alldup/documentation/search-methods/similar-pictures.php)) | Freeware, closed source |
| Immich | CLIP embedding distance ≤ `maxDistance` 0.01 by default, range 0.001-0.1 ([config](https://docs.immich.app/install/config-file/), [duplicate.service.ts](https://github.com/immich-app/immich/blob/main/server/src/services/duplicate.service.ts)) | AGPL-3.0, 117.6k stars |
| PhotoPrism | Exact duplicates by SHA-1 and size; perceptual hash only for "similar" sort, never auto-grouped ([docs](https://docs.photoprism.app/developer-guide/metadata/perceptual-hashes/)) | AGPL-3.0 |

Two lessons: (1) every tool trades recall against precision with a single threshold and no proof. (2) Rotation/flip support (AllDup, PDQ dihedral) is the one capability those tools have and eidolon does not.

---

## Benchmarks and datasets for reference

- Douze et al., *The 2021 Image Similarity Dataset and Challenge*, arXiv:2106.09672. DISC21: 1 M reference, 50k dev + 50k test queries, about 350 GB (https://ai.meta.com/datasets/disc21-dataset/). Too large to pull whole; a few thousand dev queries with their references would be a useful external check.
- Papakipos et al., *Results and findings of the 2021 Image Similarity Challenge*, PMLR 176, https://proceedings.mlr.press/v176/papakipos22a.html.
- ISC21 descriptor track: Yokoo, arXiv:2112.04323 (open-source winner, 0.6354 µAP; "negative embedding subtraction"); Papadakis & Addicam, arXiv:2112.03415; Wang et al., arXiv:2111.08004. All three use database-aware post-processing (hard-negative subtraction / descriptor stretching). For eidolon it is relevant only as score normalisation of kNN candidates, because there is no fixed reference database.

## Top-3 recommendation

1. **Descriptor A/B including preprocessing (items 1 + 2).** Run four configurations: DINOv2 centre-crop (today), DINOv2 squashed, CopyNCE ViT-S squashed, SSCD ResNet-50 squashed. Expected: the largest gain on thumbnail-with-overlay and crop candidates, at equal or lower embedding cost.
2. **Global-local tile embeddings (item 3)** for images without a proven match. Expected: crop_40 and partial-frame thumbnails reach the candidate list. The published gain is +11.9 µAP on DISC21.
3. **`USAC_MAGSAC` + RootSIFT fallback (item 4).** Hours of work, no new dependency. It targets the ORB-registration failures behind crop_40 / crop_watermark.

Variant (item 5) needs the attribution experiment below before any code change.

## Experiment that would prove it on eidolon's eval

Use the final source-disjoint holdout (760 files) with a cold cache per configuration and the same machine.

1. **Instrument candidate recall separately from end-to-end recall.** For each ground-truth pair, record whether it reached the candidate set (hash bands, region hashes, kNN top-64 ≥ threshold) and whether proof accepted it. Report this per transform: crop_40, crop_watermark_q25, thumbnail_overlay, variant_hue, variant_relit, variant_light_dark. This tells which misses are retrieval and which are proof. Run it first; it decides between items 5 and 6.
2. **Descriptor matrix (items 1-2, plus 7 if speed is a problem).** For each embedder/preprocessing:
   - candidate recall@64 per transform;
   - end-to-end direct recall per transform;
   - **precision (must stay 1.000)**;
   - cold scan time, embeddings/s on CPU and CoreML, peak RSS.

   Calibrate each model's cosine threshold on the synthetic set, not the holdout. Success: thumbnail-overlay direct recall clearly above 0.49 and crop_40 ≥ 0.95, with precision 1.000 and cold time no worse than 418 s.
3. **Tiles (item 3).** Same metrics, plus the number of extra embeddings and their time share.
4. **Geometry (item 4).** Count the ORB-failed pairs today. For those pairs only, compare RANSAC vs `USAC_MAGSAC`, and ORB vs RootSIFT: proven matches gained, false geometric fits rejected by pixel proof, ms per pair.
