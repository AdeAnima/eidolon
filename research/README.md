# Research: what eidolon could integrate or learn from

Written 2026-10-29 by five research agents (one file per topic). Nothing in this folder has been tried in eidolon yet; every expected gain is from the cited sources or marked [INFERENCE] in the topic file.

| File | Topic |
|---|---|
| [image.md](image.md) | Image copy detection, local features, Variant detection |
| [video.md](video.md) | Video copy detection, thumbnail→frame matching, video Variant |
| [text.md](text.md) | Text near-duplicates, document formats |
| [scale.md](scale.md) | Candidate retrieval, decoding and inference speed |
| [remote.md](remote.md) | Identifying remote media from partial downloads |

## Cross-topic ranking (proposal)

Ranked by weak spot addressed per day of effort. Precision must stay 1.000, so every recall idea keeps a proof step.

| # | Idea | Weak spot | Effort | Source file |
|---|---|---|---|---|
| 1 | Exact Hamming candidate search (multi-index hashing) instead of 8 byte bands, which miss pairs more than 7 bits apart | crop_40 / watermark recall 0.80-0.95 | 0.5-1 day | scale.md |
| 2 | Embed the whole image (resize, no centre crop): `embed.rs` keeps only ~43% of a 16:9 frame | crops, thumbnails | hours | image.md |
| 3 | Mask the caption/overlay band before thumbnail→frame proof (masked correlation) | overlay FrameOf direct 0.49 local | 1-2 days | video.md |
| 4 | Remote FrameOf: fetch from the keyframe only up to the thumbnail time and stop when proven; WebM `Cues` keyframe index | remote FrameOf 14/63, 40% bytes | 3-5 + 2-4 days | remote.md |
| 5 | Reduced-size decode (`fast_image_resize`, turbojpeg scaled decode, ImageIO thumbnails for HEIC); full decode only for pixel proof | cold scan time | 1-2 days | scale.md |
| 6 | USAC_MAGSAC instead of RANSAC, RootSIFT fallback (both in linked OpenCV 5) | crop/watermark proof | hours | image.md |
| 7 | Copy-detection embedding (CopyNCE ViT-S / SSCD) vs DINOv2, plus tile embeddings for unmatched images | crops, overlays | 2-4 days | image.md, video.md |
| 8 | Densified one-permutation MinHash + binary cache | text speed | 0.5 day | text.md |
| 9 | Saturation-based frame hash for video Variant (needs proof step) | video Variant recall 0.37 | 0.5-1 day | video.md |
| 10 | PDF / DOCX / HTML text extraction (pdfium-render, zip + quick-xml, html2text; not AGPL mupdf) | new formats | 2-3 days | text.md |

Licence blockers found: MobileCLIP2 (research-only), DINOv3 (custom licence), mupdf (AGPL). Archived projects to avoid: instant-distance, jwalk, bincode (GitHub), SSCD repo (archived 2023, weights licence unconfirmed).

## Cheapest first experiment

Offline replay of cached holdout features through an exact Hamming search (scale.md §4 step 2): measures how many true pairs the current band index misses, without changing code.
