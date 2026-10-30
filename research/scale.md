# Scale research: candidate retrieval and inference efficiency at 10k–1M files

Researched 2026-10-29 for eidolon (Rust, macOS Apple Silicon M3 Max). Research only: no eidolon code was built, run or changed. Crate versions and GitHub statistics were read from the crates.io and GitHub APIs on 2026-10-29. Numbers marked "source" come from the linked primary source. Everything marked **[INFERENCE]** is my own arithmetic or judgement, not a measurement on eidolon.

## 1. Where eidolon stands (read from source, 2026-10-29)

| Stage | Current implementation | Scaling behaviour |
| --- | --- | --- |
| Image hash candidates | `eidolon-image/src/detect.rs::candidates`: 64-bit pHash, dHash, and 27 region pHashes. They are indexed in **8 exact-match byte bands** (`HashMap<(band, byte), Vec<…>>`). Radius ≤14 (pHash), ≤12 (dHash), ≤20 (regions). The search keeps the best 64 full-image and 48+48 region neighbours. | Exact byte match guarantees retrieval only up to distance 7 (pigeonhole). See §2. |
| Image embedding candidates | `embedding_neighbors`: for every image, a **linear scan over all other images** with a scalar f32 dot product (vectors are pre-normalised), threshold cosine ≥0.82, top 64. | O(n²·384) scalar. |
| Video candidates | `eidolon-video/src/candidates.rs`: 12+16 bit-sampling LSH tables with 16-bit keys, 8 sign-random-projection embedding tables, and one-bit probing. | Bucket occupancy grows linearly with frame count. |
| Decode | `image` 0.25 (its JPEG backend is zune-jpeg) plus `libheif-rs` 2.7 through its `image` hook. The full-resolution decode is followed by `image::imageops::resize` (Triangle/CatmullRom) to 800 px (ORB), 128 px (preview/hash) and 256→224 px (DINOv2). | Full-resolution decode for every file. Historical sample: a 6016×6016 Sonoma HEIC took **5.33 s** to extract versus 0.16 s for a 356×356 HEIC. An earlier 660-image stage run measured **82.0 s extraction vs 0.037 s candidate generation vs 18.2 s verification**. |
| Embedding inference | ORT `=2.0.0-rc.12`, DINOv2-small ONNX. CPU session pool (57 emb/s) or one CoreML MLProgram session with static `[1,3,224,224]` input, FP32 I/O, `MLComputeUnits=ALL` (251 emb/s). | Batch 1 and one CoreML session. |
| Exact duplicates | `eidolon-cli/src/exact.rs`: groups by size, then a 16 KiB head+tail BLAKE3 sample, then full BLAKE3. Same staging as fclones. | Already good. |
| Cache | `eidolon-core/src/cache.rs`: SQLite WAL with `synchronous=NORMAL`. Key is `(path, feature)` plus a size and mtime check. One `Mutex<Connection>`. Every `put` is its own autocommit statement, prepared ad hoc. | A move or rename recomputes every feature. Writes are serialised and committed one row at a time. |
| Directory walk | `walkdir` 2 (serial). | Serial `readdir`/`stat`. |

### Scale arithmetic [INFERENCE]

| Files n | Embedding pairs n²/2 | f32 multiply-adds (384-d) | 384-bit binary popcount words | f32 vectors in RAM | Binary codes in RAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 5.0e7 | 1.9e10 | 3.0e8 | 15 MB | 0.5 MB |
| 100k | 5.0e9 | 1.9e12 | 3.0e10 | 154 MB | 4.8 MB |
| 1M | 5.0e11 | 1.9e14 | 3.0e12 | 1.5 GB | 48 MB |

- **10k:** the current scalar loop is not the bottleneck. Extraction (decode + ORB + DINOv2) is.
- **100k:** the scalar loop costs minutes.
- **1M:** the scalar loop costs hours. A blocked matrix-multiply join (§4.3) or an ANN index (§4.4) is required.
- **Inference at 1M stills** at the measured 251 emb/s CoreML rate: ≈66 min of DINOv2 alone. Decode at the historical ≈124 ms/image/core (82 s / 660) over 16 cores: ≈2.2 h. Both must shrink.

## 2. A latent recall hole and a hard scale wall in the hash index

**Recall hole.** I simulated uniformly random bit flips [INFERENCE: real pHash flips are correlated, not uniform]. The probability that an 8-way exact byte-band index misses a pair at a given Hamming distance d:

| d | 8 | 10 | 12 | 14 | 16 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P(miss) | 0.4% | 4.4% | 14.3% | **28.5%** | 44.2% | **70.3%** |

- For full-image pHash, d ≤14 is accepted, but up to ~29% of pairs at the edge of that radius are never surfaced by the pHash bands.
- For region hashes, d ≤20 is accepted, but ~70% of region pairs at d=20 are not surfaced by the band lookup.
- dHash, complements and DINOv2 kNN rescue some of these pairs.
- Crops (crop_40 recall 0.80–0.95) are exactly the case where region hashes are supposed to help without embeddings.

**Scale wall.** Expected random collisions per query hash under a uniform 64-bit code model [INFERENCE: binomial tail]:

| Radius r | P(random code within r) | Random hits per query at 1M codes | Random hits at 28M codes (1M images × 28 hashes) |
| ---: | ---: | ---: | ---: |
| 12 | 2.3e-7 | 0.23 | 6.4 |
| 14 | 3.5e-6 | 3.5 | 99 |
| 20 | 1.8e-3 | 1,845 | **51,660** |

- Full-image pHash at ≤14 stays selective at 1M images.
- Region hashes at ≤20 do not. At 1M images every query returns tens of thousands of chance hits, and the "best 48" truncation then fills with noise.
- At ≳100k images, crop seeding must come from embeddings or local-feature retrieval, not 64-bit region pHashes.

## 3. Ranked candidates

### 1. Reduced-resolution decode: ImageIO thumbnails, JPEG DCT scaling, SIMD resize

**What it is.** Stop decoding every image at full resolution when only ≤800 px (ORB), 224 px (DINOv2) and 128 px (hash/preview) are needed. Three independent pieces:

**a. Apple ImageIO `CGImageSourceCreateThumbnailAtIndex`** with `kCGImageSourceThumbnailMaxPixelSize`, `…CreateThumbnailFromImageAlways` and `…WithTransform`.
- Decodes JPEG/HEIC/PNG directly to a bounded size. Uses the platform HEVC path for HEIC [INFERENCE: hardware use is not documented].
- Sources: Apple, *Image and Graphics Best Practices*, WWDC 2018 session 219, https://developer.apple.com/videos/play/wwdc2018/219/ ([slides PDF](https://devstreaming-cdn.apple.com/videos/wwdc/2018/219mybpx95zm9x/219/219_image_and_graphics_best_practices.pdf)); [`kCGImageSourceThumbnailMaxPixelSize` docs](https://developer.apple.com/documentation/imageio/kcgimagesourcethumbnailmaxpixelsize).
- Rust binding: `objc2-image-io` 0.3.2 (Zlib/Apache-2.0/MIT, part of madsmtm/objc2).

**b. libjpeg-turbo scaled IDCT** (scale factors M/8; the README says 1/2 and 1/4 are SIMD-accelerated, 1/8 is not).
- Sources: https://github.com/libjpeg-turbo/libjpeg-turbo (BSD-style IJG/Modified-BSD/zlib, 4.5k★); [TurboJPEG API](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h).
- Rust: `turbojpeg` 1.5.1 (Unlicense/MIT, 2026-07-25) has `Decompressor::set_scaling_factor(ScalingFactor::ONE_QUARTER)` ([docs](https://docs.rs/turbojpeg/latest/turbojpeg/struct.Decompressor.html)).
- Pure-Rust alternative: `jpeg-decoder` 0.3.2 `Decoder::scale(w,h)` uses 1/8, 1/4, 1/2 or 1 IDCT ([docs](https://docs.rs/jpeg-decoder/latest/jpeg_decoder/struct.Decoder.html#method.scale)).
- `zune-jpeg`, the current `image` backend, has **no** scaling API: `JpegDecoder` methods are decode/headers/info only (docs.rs, 2026-10-29). The `image` maintainers judged it "on par" with libjpeg-turbo ([image-rs/image#1845](https://github.com/image-rs/image/issues/1845)).

**c. `fast_image_resize` 6.1.0** (MIT/Apache-2.0, 465★, released 2026-07-21, actively maintained). NEON SIMD resize to replace `image::imageops::resize`.
- Source: [benchmarks-arm64.md](https://github.com/Cykooz/fast_image_resize/blob/main/benchmarks-arm64.md).

**d. libheif threading.** `heif_context_set_max_decoding_threads` controls tile-parallel decoding of grid HEICs ([libheif-sys docs](https://docs.rs/libheif-sys/latest/libheif_sys/fn.heif_context_set_max_decoding_threads.html)). Apple 36-MP HEICs are tile grids.
- Eidolon decodes through the libheif `image` hook inside a Rayon pool. The libheif default is 4 threads, so nested oversubscription is likely [INFERENCE].
- Note the 2026 deadlock advisory for parallel grid decoding in libheif 1.23.3: [GHSA-prgh-72vc-3xmc](https://github.com/strukturag/libheif/security/advisories/GHSA-prgh-72vc-3xmc).

**Measured numbers (source)**
- `fast_image_resize` on arm64 (Neoverse-N1 2 GHz), RGB8 4928×3279 → 852×567:
  - Bilinear: `image` 161.06 ms vs fir NEON 29.21 ms (5.5×).
  - Bicubic: 279.08 ms vs 53.59 ms (5.2×).
- ImageIO thumbnail of a 12 MP iPhone photo on an M4 Pro, macOS 26:
  - Warm: **16 ms JPEG, 43 ms HEIC, 145 ms PNG**.
  - Cold: 73 / 151 / 183 ms.
  - The full-size `NSImage` draw path took 628–675 ms.
  - Source is a third-party developer blog (one image, not controlled): M. Seelemann, *Fast Thumbnails with CGImageSource*, 2025-09-23, https://macguru.dev/fast-thumbnails-with-cgimagesource/.
- Apple WWDC18 shows the memory reduction from ImageIO downsampling, not a codec benchmark.
- libjpeg-turbo publishes no end-to-end table per scale factor. Entropy decoding does not shrink, so expect far less than the pixel ratio [INFERENCE].

**Addresses.** Cold-scan speed and memory for every image and thumbnail (the 418 s cold scan, the 5.3 s 36-MP HEIC). For URL inputs it changes nothing about bytes fetched.

**Precision guard (required).**
- ImageIO, libjpeg-turbo and zune produce different pixels. Keep one canonical full decoder for the `SameContent` decoded-pixel digest and for native-pixel-density registration. Run it lazily, only for candidate pairs that need it.
- Put the decoder identity and scale into every feature cache key.
- ImageIO returns a CGImage in the source colour space. Eidolon's ICC/P3 handling must be replicated by an explicit `CGColorSpace` conversion. This is the main risk.
- **Never** use EXIF/HEIC embedded thumbnails as evidence. Editors often leave them stale (see ExifTool docs on `ThumbnailImage`, https://exiftool.org/faq.html), which would create false `SameContent`/`Edited` evidence. `FromImageAlways` avoids them.

**Effort**
- `fast_image_resize` swap: 0.5 day plus a full eval rerun, because thresholds see slightly different previews.
- turbojpeg scaled JPEG path: 0.5–1 day.
- ImageIO path with colour management: 2–3 days.
- libheif thread setting: 1 hour.

**Telemetry.** None. All four are local libraries or OS frameworks without network code [INFERENCE for ImageIO internals].

### 2. Multi-index hashing (MIH) for exact Hamming range search

**What it is.** Split the 64-bit code into m substrings. Probe each substring table within radius ⌊r/m⌋ (some tables only ⌊r/m⌋−1). The pigeonhole principle then **guarantees** that every code within r is retrieved. Finally verify with the full popcount.

**Citations**
- M. Norouzi, A. Punjani, D. J. Fleet, *Fast Search in Hamming Space with Multi-Index Hashing*, CVPR 2012, pp. 3108–3115, doi:10.1109/CVPR.2012.6248043, https://www.cs.toronto.edu/~fleet/research/Papers/chunkingCVPR12.pdf.
- Extended version: *Fast Exact Search in Hamming Space with Multi-Index Hashing*, IEEE TPAMI 2014, https://www.cs.toronto.edu/~fleet/research/Papers/MIH-pami2014.pdf.
- Reference C++ code: https://github.com/norouzi/mih (289★, unmaintained since 2015, licence file not SPDX-detected).

**Measured numbers (source, TPAMI Table 1, AMD Opteron, single core)**
- Exact kNN on 1B 64-bit codes: **823× / 390× faster than linear scan** for 1-NN / 1000-NN (MLH codes).
- The 1000-NN query takes ≈50 ms.
- The paper recommends substring length ≈ log₂ n. Query cost is ≈ O(n^{H(r/q)}), which is sublinear only for small r/q.

**Fit for eidolon [INFERENCE]**
- Keep eidolon's 8 byte-bands but probe each band at radius 1–2 instead of 0:
  - r=14 needs 64 bucket lookups per query instead of 8.
  - r=20 needs 212.
- Retrieval within the stated radii becomes exact, which removes the §2 miss rates (28% at d=14, 70% at d=20).
- At 10k images × 28 hashes the 8-bit buckets hold ~1.1k entries. Total work ≈ 1e10 popcounts ≈ seconds on 16 cores.
- Above ~100k images, switch to 16-bit substrings (m=4) for full-image hashes. Abandon radius-20 region seeding (the §2 scale wall).
- The same analysis applies to video's 16-bit LSH keys: their bucket length should track log₂(frame count).

**Rust availability.** No crate. About 150 lines in `detect.rs`, reusing the existing `HashMap` buckets.

**Addresses.**
- Crop recall (crop_40 / crop_watermark, 0.80–0.95), because region candidates are currently missed silently.
- Thumbnail → video recall where hash seeding is the channel.
- Deterministic candidate sets at any n.

**Precision.** Unchanged, because the proof stages are untouched. More candidates means more verification time. The existing top-48/64 caps bound it.

**Effort.** 0.5–1 day plus eval.

**Telemetry.** None.

### 3. Exact blocked matrix-multiply threshold join for embeddings (≤~300k files)

**What it is.** Replace the per-image linear scan with a tiled GEMM: S = E·Eᵀ in blocks of e.g. 4096×4096. Filter S ≥ 0.82 inside each tile and keep the top 64 per row.
- On Apple Silicon, Accelerate's `cblas_sgemm` uses the matrix coprocessor [INFERENCE: Apple does not document AMX publicly].
- Rust options:
  - `accelerate-src` + `cblas-sys`.
  - The pure-Rust `gemm`/`matrixmultiply` crates.
  - SimSIMD 6.5.16 (`simsimd`, Apache-2.0, 1.9k★) for f16/i8/binary dot kernels with NEON (https://github.com/ashvardanian/SimSIMD).

**Measured numbers.** No primary benchmark for this exact join. The arithmetic: 1.9e12 multiply-adds at 100k. At an assumed sustained ≥1 TFLOP/s it takes seconds; at 1M it is ~400 s and still feasible offline [INFERENCE].

**Addresses.** The O(n²) scalar wall from 10k to ~300k files, with **zero recall or precision change**: results are identical to brute force up to float summation order. Any tie-sensitive ordering must keep the current `(similarity, index)` tie-break.

**Effort.** 0.5–1 day.

**Telemetry.** None.

### 4. USearch: HNSW index with i8/f16/binary storage, exact mode, multi-vector keys, disk views (>~300k files or a persistent index)

**What it is.** A single-header C++ HNSW with official Rust bindings.
- Crate `usearch` 2.26.6, Apache-2.0, 4.4k★, released 2026-10-25, 745k recent downloads; used by ClickHouse, DuckDB, ScyllaDB, GreptimeDB.
- Source: https://github.com/unum-cloud/USearch; [Rust docs](https://unum-cloud.github.io/USearch/rust/index.html).
- Rust API (read from `rust/lib.rs`):
  - `search`, `exact_search` (brute force with the same API), `filtered_search`.
  - `ScalarKind::{F32, F16, BF16, I8, B1, …}`.
  - `multi: true`, which allows many vectors per key. That fits one video with N frame embeddings.
  - `view(path)` serves an index from disk without loading it.

**Measured numbers (source, [BENCHMARKS.md](https://github.com/unum-cloud/usearch/blob/main/BENCHMARKS.md))**

AWS c7g.metal, 64 Graviton-3 cores, 256-d vectors, connectivity 16, ef_add 128, ef_search 64:

| Storage | Search QPS | Recall@1 |
| --- | ---: | ---: |
| f32 | 171,856 | 99.1% |
| f16 | 153,788 | 98.4% |
| i8 | 274,653 | 98.9% |

README: indexing 100M 96-d vectors took 0.3 / 0.2 / 0.2 h (f32/f16/i8) vs FAISS 2.6 h (Intel Sapphire Rapids).

**Alternatives considered**
- `hnsw_rs` 0.3.5 (MIT/Apache, 255★, maintained): sift1m 10-NN at 15,000 req/s with recall 0.9907 on an i9-13900HX ([README](https://github.com/jean-pierreBoth/hnswlib-rs)). SIMD on aarch64 requires **nightly** `std::simd`, so it is slower on stable M3 [INFERENCE].
- `instant-distance` 0.6.1: last release 2023-06, **repository archived**. Reject.
- `faiss` 0.13.0 (faiss-rs, 250★): needs a locally built Faiss C API (Faiss itself is MIT, 41k★). Heavy build, and only worth it for IVF-PQ at ≫10M.
- `lancedb` 0.41.0 (Apache-2.0, 11.8k★): a full Arrow/DataFusion embedded database. Too heavy for a report-only CLI.
- `arroy` 0.9.0 (Meilisearch, LMDB Annoy) and `diskann` 0.61.0 (Microsoft): built for disk/billion scale. Overkill.

**Addresses.**
- Embedding retrieval at 300k–1M+ files.
- Incremental runs: persist the index next to the SQLite cache so a warm 1M scan does not rebuild O(n²).
- Video frame embeddings (millions of vectors).

**Precision and recall trade-off.** HNSW is approximate (98–99% Recall@1 in the source). Eidolon's proof stages keep precision at 1.000, but recall can drop silently. Use `exact_search` below a size threshold and measure recall loss on the holdout above it.

**Effort.** 1–2 days. Needs a C++ toolchain for `cxx`; already present because OpenCV is a dependency.

**Telemetry.** None found in the source [INFERENCE: no network code in the library].

### 5. Binary and RaBitQ embedding compression with exact rescoring

**What it is**
- Sign-binarise the 384-d DINOv2 vector into 48 bytes. Search in Hamming space (popcount, or via §2/§4). Then rescore survivors with the f32 (or i8) vector at the true 0.82 threshold.
- RaBitQ adds a random rotation and an **unbiased inner-product estimator with an O(1/√D) error bound**. The bound lets a range query discard only candidates whose upper bound is below 0.82.

**Citations**
- J. Gao, C. Long, *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search*, Proc. ACM Manag. Data 2(3), Art. 167, SIGMOD 2024, doi:10.1145/3654970, https://arxiv.org/abs/2405.12497.
- Code: https://github.com/VectorDB-NTU/RaBitQ-Library (C++, Apache-2.0, 292★, active 2026-10).
- Practical binary/int8 study: Hugging Face, *Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval*, 2024-03-22, https://huggingface.co/blog/embedding-quantization.

**Measured numbers (source)**
- HF blog (text model mxbai-embed-large-v1, MTEB retrieval):
  - Binary keeps 92.53% of float performance, **96.45% with float rescoring**.
  - int8 with rescoring keeps ≈99%.
  - 32× / 4× smaller.
- None of these numbers are DINOv2 numbers.

**Addresses.**
- Memory: 1M vectors shrink from 1.5 GB to 48 MB.
- Cache bytes and read bandwidth on warm scans.
- Faster prefilter before §3/§4.

**Rust availability.** USearch `B1`/`I8`. `turbovec` 1.1.2 and `fastrq` 0.2.0 are young RaBitQ-style crates (not evaluated). Otherwise a ~200-line port.

**Precision.** Safe with exact rescoring. Recall must be measured: DINOv2 pairs near 0.82 are exactly where binarisation errs.

**Effort.** Sign-binary prefilter: 1 day. RaBitQ port: 2–3 days.

**Telemetry.** None.

### 6. Embedding inference throughput on Apple Silicon: Core ML batching, ANE routing, low precision; ORT int8 on CPU

**What it is.** Measure before changing models:
- **a.** ORT CoreML EP options documented at https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html:
  - `MLComputeUnits=CPUAndNeuralEngine` excludes the GPU. Eidolon uses `ALL`, which lets Core ML pick the GPU.
  - `ProfileComputePlan=1` logs the device each op is dispatched to.
  - `SpecializationStrategy=FastPrediction`.
  - A static **batch dimension N>1** via the same symbolic-dimension override eidolon already uses for 224×224.
- **b.** Apple's ANE guidance:
  - *Deploying Transformers on the Apple Neural Engine*, Apple ML Research, 2022-06-06, https://machinelearning.apple.com/research/neural-engine-transformers. The optimised DistilBERT ran **up to 10× faster with 14× less peak memory** than the stock Core ML conversion on iPhone 13. Code: https://github.com/apple/ml-ane-transformers (2.75k★).
  - For ViTs: *Deploying Attention-Based Vision Transformers to the Apple Neural Engine*, 2024-01-05, https://machinelearning.apple.com/research/vision-transformers. The paper gives figures only, no table.
- **c.** Core ML W8A8 via coremltools `linear_quantize_activations` + `linear_quantize_weights` uses the ANE int8-int8 compute path "from iPhone 15 Pro onward" ([coremltools guide](https://github.com/apple/coremltools/blob/main/docs-guides/source/opt-quantization-api.md); coremltools BSD-3, 5.4k★). The guide gives no general speedup figure. Whether M3 Max uses the int8 path is unverified [INFERENCE].
- **d.** ORT CPU dynamic int8 (S8S8 recommended for transformers; the U8S8 saturation caveat is x86-only): https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html.

**Measured numbers.** Only the Apple numbers in (b) (other model, other device). Apple states that `predictions(fromBatch:)` does not guarantee a fused batch ([docs](https://developer.apple.com/documentation/coreml/mlmodel/predictions%28frombatch%3A%29)). Batch size is therefore a measured knob.

**Addresses.** DINOv2 cost at scale: 1M stills ≈ 66 min at 251 emb/s, plus video frame embeddings.

**Precision.** Any precision change alters vectors. Eidolon already requires CPU/CoreML cosine ≥0.999 and versions cache identities by provider/precision. Keep that gate, then re-run the holdout. The 0.82 threshold may need re-validation.

**Effort**
- Batch-N static shape plus `CPUAndNeuralEngine` A/B: 1 day.
- W8A8 requires a native Core ML package outside ORT (`objc2-core-ml`): 3–5 days.
- ORT dynamic int8: 0.5 day plus eval.

**Telemetry.** ORT telemetry is already disabled by eidolon. coremltools is used only offline at conversion time.

**Not recommended: MLX** (https://github.com/ml-explore/mlx, MIT, 29k★; Rust `mlx-rs` 0.33.0, Apache-2.0). It targets CPU+GPU, not the ANE, and DINOv2 would need a port. It brings no advantage over Core ML for batch-1 ViT-S inference [INFERENCE].

### 7. Cache identity and I/O design (fclones / czkawka patterns)

**What it is**
- **Inode-keyed reuse.** fclones keys its hash cache by internal file id (inode), not path, so "cached hashes are not invalidated by file moves". Metadata (mtime, length) must still match exactly. It also uses device-aware thread pools (SSD vs HDD), physical-placement ordering on HDD, and prefix-compressed paths for millions of files.
  - Source: P. Kołaczkowski, fclones README, https://github.com/pkolaczk/fclones (MIT, 3.0k★; last push 2025-03).
  - Benchmark (source): 1,460,720 paths / 316 GB, NVMe, cold cache: **fclones 34.59 s / 266 MB vs rmlint 2:28 / 942 MB vs jdupes 5:02 vs fdupes 5:46** (old versions).
- **Parameter-specific caches + BK-tree.** czkawka stores per-parameter cache files (`cache_similar_image_<SIZE>_<HASH>_<FILTER>_<INVARIANCE>.bin`) and queries similar images through a BK-tree.
  - Source: https://github.com/qarmin/czkawka/blob/master/instructions/Instruction_Core.md (34.6k★).
  - Eidolon's versioned feature names already cover the parameter part. A BK-tree is inferior to MIH (§2) at these radii [INFERENCE].
- **SQLite write batching.** The SQLite FAQ says autocommit inserts are limited by transactions, not statements. Wrap batches in one transaction and reuse prepared statements (https://www.sqlite.org/faq.html, "INSERT is really slow"). In rusqlite: `prepare_cached` plus a writer that commits every ~1,000 rows.
- **Parallel walk.** `ignore::WalkParallel` (ripgrep's crate, Unlicense/MIT, 0.4.34, maintained). `jwalk` 0.9.0 is **archived** on GitHub, so reject it.

**Addresses.**
- Thousands-to-millions of files staying practical across folder reorganisations: a rename currently re-runs DINOv2/ORB/ffmpeg.
- Warm-scan latency.
- Write contention on the single connection mutex.

**Precision.** Unchanged if the key includes `(dev, ino, size, mtime_ns)` and the stored path is used only for reporting. APFS clones share no inode, so they simply miss [INFERENCE].

**Effort.** 1 day (cache schema v-bump plus batching).

**Telemetry.** None.

### 8. Smaller or alternative embedding models (licence check first)

| Model | Params / speed (source) | Licence | Verdict |
| --- | --- | --- | --- |
| DINOv2 ViT-S/14 distilled (current) | 21M params ([DINOv2 README](https://github.com/facebookresearch/dinov2)). Oquab et al., *DINOv2*, arXiv:2304.07193 (TMLR 2024). Oxford-5k M/H mAP 68.8/43.2 (Table 9). | Apache-2.0 | Keep as baseline |
| DINOv3 ViT-S/16 / ConvNeXt-Tiny (distilled from 7B) | 21M / 29M params ([model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)). Siméoni et al., *DINOv3*, arXiv:2508.10104. | **DINOv3 License** (custom, not OSI) | Legal review needed. Accuracy at copy detection unmeasured. |
| MobileCLIP2-S0 | 1.5 ms image encoder on iPhone 12 Pro Max, 71.5% IN-1k zero-shot. Faghri et al., *MobileCLIP2*, arXiv:2508.20691. MobileCLIP-S0: 1.5 ms, 67.8% (Vasu et al., CVPR 2024). | **apple-amlr: research-only, no commercial use** | Reject for a product |

**Addresses.** Inference cost. The ONNX export path exists for all three (timm/open_clip → `torch.onnx.export`).

**Effort.** 2–3 days per model including threshold re-tuning and the full holdout.

**Telemetry.** None at runtime. Model download only through eidolon's explicit `fetch-models`.

## 4. Top-3 recommendation and the experiment that proves each

1. **Reduced-resolution decode (§3.1): `fast_image_resize` first, then turbojpeg scaled JPEG, then the ImageIO HEIC path.**
   - Experiment: run the final 760-file holdout with stage timings, cold cache, CPU and CoreML. Before/after:
     - per-stage extraction seconds;
     - peak RSS;
     - Sonoma 6016×6016 HEIC single-file time;
     - every scored category.
   - Pass: precision 1.000 everywhere, no direct-recall drop >0 in any image category, cold scan time down (target ≥2× on extraction).
2. **Exact candidate retrieval: MIH for hashes (§3.2) plus a blocked GEMM embedding join (§3.3).**
   - Experiment 1, offline replay on cached holdout features: for every ground-truth positive pair, log the true min distance per channel (pHash, dHash, complement, region) and whether the band index surfaced it. Then swap in MIH and report:
     - candidate count;
     - verification seconds;
     - direct recall for crop_40, crop_watermark and thumbnail-overlay→video.
   - Experiment 2: assert that GEMM neighbour sets equal the current loop's sets on the holdout. Time it on 100k synthetic DINOv2-like vectors.
   - Pass: precision 1.000, recall ≥ baseline, candidate-generation time at 100k in seconds rather than minutes.
3. **Embedding throughput A/B on Core ML (§3.6): static batch N∈{1,4,8,16} × `MLComputeUnits∈{ALL, CPUAndNeuralEngine}`, with `ProfileComputePlan` on.**
   - Experiment: embeddings/s on 1,000 decoded holdout images. Cosine vs the CPU reference must stay ≥0.999 per vector. Then run a full holdout with the faster setting.
   - Pass: identical relation identities, precision 1.000, throughput >251 emb/s.

Follow-on once n >~300k or warm-scan persistence matters: USearch (§3.4) with an i8 or binary prefilter (§3.5) and exact rescoring.
