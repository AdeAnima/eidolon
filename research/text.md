# Text near-duplicate detection: research for eidolon-text

Scope: eidolon-text already scores 1.000 on the holdout. This note targets speed, memory, new input formats (PDF/DOCX/HTML), and paraphrase recall. Precision must stay 1.000, so every recall idea below stays behind an exact or structural proof step.

Verification, 2026-10-29:
- Star counts and push dates come from the GitHub REST API.
- Crate versions and licences come from the crates.io API (`max_stable_version` / newest version licence).
- Model licences and parameter counts come from the Hugging Face API.
- Paper numbers come from arXiv abstracts or the paper PDF text (pdftotext). Exact quotes are marked "PDF".
- Claims I could not verify were removed. Inferences are marked [INFERENCE].

## Baseline (read from `crates/eidolon-text/src/lib.rs`)
- Normalisation: NFKC, then casefold, then NFKC again.
- Shingles: word or character shingles, hashed with FNV-1a + splitmix64.
- Candidates: 128 independent MinHash permutations (`PERMUTATIONS = 128`). LSH bands keyed `(band, [u64; 4])`, so 32 bands × 4 rows.
- Excerpts: 64 containment anchors (`ANCHORS = 64`).
- Proof: exact verification by sorted shingle-set intersection.
- Cache: feature `text.shingles/v1`, serialized with serde_json.
- Formats: raw bytes plus BOM/UTF-16 detection only. No PDF, DOCX, or HTML.

## Ranked candidates

### 1. Densified one-permutation MinHash, or C-MinHash (replace 128 independent hashes)
- **What:**
  - Densified one-permutation hashing: hash each shingle once, then fill K bins. Cost is O(d + k) instead of O(dk).
  - C-MinHash: 2 permutations, the second reused K times by circulant shift.
- **Sources:**
  - Shrivastava, "Optimal Densification for Fast and Accurate Minwise Hashing", ICML 2017, https://arxiv.org/abs/1703.04664.
    - Abstract: "compute k minwise hashes … in mere (d + k) computations … over the classical O(dk)". The proposed scheme is "variance-optimal".
    - PDF, Table 2 / Conclusion 2: densification is "10-18x faster for computing 300 hashes" on RCV1, URL, and News20 (avg. non-zeros 73/115/402).
  - Li & Li, "C-MinHash: Rigorously Reducing K Permutations to Two", ICML 2022, https://arxiv.org/abs/2109.03337. Abstract: unbiased Jaccard estimate, and "the theoretical variance is uniformly smaller than" classical MinHash.
  - Ertl, "SuperMinHash – A New Minwise Hashing Algorithm for Jaccard Similarity Estimation", https://arxiv.org/abs/1706.05698. An alternative with lower variance.
- **Licence:** algorithms only.
- **Rust:** hand-roll it (~40 lines) in `Features::extract`.
  - `gaoya` 0.2.2 (MIT; github.com/serega/gaoya, 83★, pushed 2026-05-29) offers 64/32/16/8-bit MinHash and 64/128-bit SimHash with an LSH index (README).
  - Its README does not mention densification or OPH, so it does not deliver this speedup.
- **Weak spot:** CPU and speed.
  - [INFERENCE] With 128 hashes per shingle, signature building is probably the dominant text CPU cost.
  - The 10–18× figure is for k=300; at k=128, expect less.
- **Effort:** 4–6 hours, plus a cache bump to `text.shingles/v2`.
- **Telemetry:** none.

### 2. Compact signatures and a binary cache
- **What:** Store b-bit or u32 minima instead of u64, and replace serde_json with a binary codec.
- **Source:** Li & König, "b-Bit Minwise Hashing", WWW 2010, https://arxiv.org/abs/0910.3349. Abstract: "even in the least favorable scenario, using b=1 may reduce the storage space at least by a factor of 21.3 (or 10.7) compared to using b=64 (or b=32), if one is interested in resemblance > 0.5".
- **Rust:**
  - `postcard` 1.1.3 (MIT OR Apache-2.0).
  - Avoid `bincode`: crates.io shows 3.0.0, and GitHub `bincode-org/bincode` is **archived** (API: `archived: true`).
- **Weak spot:** memory and cache size.
- **Caveat:** b-bit only fits the LSH stage. Exact verification still needs the shingle set.
- **Effort:** 3–5 hours.
- **Telemetry:** none.

### 3. Document text extraction (new formats)
- **PDF:**
  - `pdfium-render` 0.9.4 (MIT OR Apache-2.0; github.com/ajrcarey/pdfium-render, 717★, pushed 2026-10-08). It wraps Google PDFium and needs a dynamic libpdfium.
  - Prebuilt libpdfium binaries: github.com/bblanchon/pdfium-binaries (MIT, 1,505★, pushed 2026-10-26).
  - Pure-Rust alternatives:
    - `pdf-extract` 0.12.1 (MIT; github.com/jrmuizel/pdf-extract, 604★, pushed 2026-09-16).
    - `lopdf` 0.46.0 (released 2026-10-21).
  - [INFERENCE] Pure-Rust extraction is weaker on CID/Type3 fonts. Measure the failure rate on the eval set before choosing.
  - Avoid the `mupdf` 0.8.0 crate: it is AGPL-3.0 (crates.io and GitHub messense/mupdf-rs).
- **DOCX:** `zip` 9.1.3 + `quick-xml` 0.42.0, reading the `<w:t>` runs in `word/document.xml` (~80 lines). Alternatively `docx-rs` 0.4.22 (MIT).
- **HTML:** `html2text` 0.17.4 (MIT). Useful for deduplicating article text behind URLs fetched by eidolon-fetch.
- **Measured:** no source benchmark found.
- **Weak spot:** new formats.
- **Effort:**
  - DOCX: 0.5 day.
  - HTML: 0.5 day.
  - PDF via pdfium: 1–2 days, including bundling libpdfium for macOS arm64.
- **Telemetry:** none in these crates. [INFERENCE] PDFium has no network calls in a text-extraction path.

### 4. Winnowing (MOSS) fingerprints for excerpts
- **What:** Select the minimum hash in each window of w consecutive k-gram hashes.
- **Source:** Schleimer, Wilkerson, Aiken, "Winnowing: Local Algorithms for Document Fingerprinting", SIGMOD 2003, doi:10.1145/872757.872770. PDF: https://www.cs.purdue.edu/homes/xyzhang/fall07/Papers/moss.pdf.
  - Guarantee threshold t = w + k − 1: every shared substring of length ≥ t is detected.
  - Expected density is 2/(w+1), assuming random hashes.
  - Lower bound for local algorithms: 1.5/(w+1).
- **Rust:** hand-roll it (~60 lines).
- **Weak spot:** robust, guaranteed excerpt detection with offsets. It replaces the 64 fixed anchors.
- **Effort:** 1 day, including cache v2.
- **Telemetry:** none.

### 5. Suffix-array exact substring proof (Lee et al. 2022)
- **Source:** Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini, "Deduplicating Training Data Makes Language Models Better", ACL 2022, https://arxiv.org/abs/2107.06499.
  - Code: https://github.com/google-research/deduplicate-text-datasets (Apache-2.0, 1,273★, **archived**, last push 2024-07-30; written in Rust).
  - PDF: NearDup found "between 3.04% (on C4) to 13.63% (on RealNews) near duplicates".
  - PDF: they chose k = 50 tokens as the minimum match; "manual inspection of length-25 matches found no false positives".
  - Abstract: dedup cuts memorised emissions "ten times"; train-test overlap affects "over 4% of the validation set".
- **Rust:** reuse ideas from the archived repo, or `suffix` 1.3.0 (Unlicense/MIT, BurntSushi, last release 2022-07-05).
- **Fit:** pairwise longest-common-span proof for candidate pairs only. A corpus-wide suffix array is unnecessary at eidolon scale [INFERENCE].
- **Effort:** 1 day.
- **Telemetry:** none.

### 6. SimHash, as an optional prefilter only
- **Sources:**
  - Charikar, "Similarity Estimation Techniques from Rounding Algorithms", STOC 2002.
  - Manku, Jain, Das Sarma, "Detecting Near-Duplicates for Web Crawling", WWW 2007, https://research.google/pubs/detecting-near-duplicates-for-web-crawling/. 64-bit fingerprints with Hamming distance k=3 are a reasonable choice for 8 B pages.
- **Rust:** `gaoya` (64/128-bit SimHash).
- **Fit:** weak for containment. Low value next to the existing MinHash.
- **Effort:** 3 hours. **Rank:** low.
- **Telemetry:** none.

### 7. Semantic dedup for paraphrases (opt-in only)
- **Source:** Abbas et al., "SemDeDup: Data-efficient learning at web-scale through semantic deduplication", https://arxiv.org/abs/2303.09540. Abstract: removes "50% of the data with minimal performance loss" on a LAION subset; also improves on C4.
- **Models (HF API):**
  - `sentence-transformers/all-MiniLM-L6-v2`: Apache-2.0, 22.7 M params.
  - `intfloat/multilingual-e5-small`: MIT, 117.7 M; multilingual, so it covers German.
  - `BAAI/bge-m3`: MIT.
  - `minishlab/potion-base-8M`: MIT, 7.6 M. Static embeddings; the Model2Vec README claims "up to 500 times faster on CPU than the original model" with "a small drop in performance".
- **Rust:**
  - `fastembed` 7.5.0 (Apache-2.0; 1,050★, pushed 2026-10-26).
  - `model2vec-rs` 0.3.0 (MIT per repo badge; crates.io licence field "non-standard"; 223★).
  - `ort` (already in eidolon) + `tokenizers` 0.23.2.
- **Precision risk:** HIGH. A high cosine on its own means "same topic", not "duplicate". Use it only to generate candidates, under a separate `Paraphrase` kind.
  - Proof options: order-preserving sentence alignment, or an NLI cross-encoder.
  - Neither is an exact proof, so keep it opt-in (`--semantic`).
- **Weak spot:** paraphrase recall. It is currently unmeasured, so a paraphrase holdout comes first.
- **Effort:** 2–3 days, +1 day for the eval set.
- **Telemetry:** default model download from the Hugging Face Hub. Pin local model paths and set `HF_HUB_OFFLINE=1`.

### 8. Reference tools and parameter evidence (learn, do not depend)
- **datasketch:** MIT, 2,977★, pushed 2026-10-26. Includes LSH Ensemble.
  - Zhu et al., "LSH Ensemble: Internet-Scale Domain Search", VLDB 2016, https://arxiv.org/abs/1603.07410.
  - Containment search, with partitioning because "Jaccard similarity is known to have poor performance over sets with large differences in their domain sizes" (abstract). This is relevant to excerpts of very different lengths.
- **text-dedup:** Apache-2.0, 764★, pushed 2026-03-09. README MinHash config: `num_perm = 240`, `threshold = 0.7`, `ngram_size = 5`.
- **RefinedWeb:** https://arxiv.org/abs/2306.01116, PDF. "MinHash deduplication using 9,000 hashes per document, calculated over 5-grams and divided into 20 buckets of 450 hashes". This is followed by exact substring dedup, so MinHash + exact substring is the industry pattern.
- **probabilistic-collections:** 0.7.0, last release 2020-05-11. Stale; not worth a dependency.

## Top 3 recommendations
1. **Densified OPH or C-MinHash, plus a `postcard` cache.**
   - Experiment: run v1 and v2 on the text holdout plus 10k synthetic docs.
   - Pass: identical match list, precision and recall 1.000, faster `extract` wall time, smaller cache bytes per doc.
2. **PDF/DOCX/HTML extraction** (pdfium-render, zip + quick-xml, html2text).
   - Experiment: a holdout of the same source exported to PDF, DOCX, and HTML, plus excerpted and revised versions.
   - Pass: cross-format recall ≥0.95 at precision 1.000. Also report extraction failure rate and time per page.
3. **Winnowing plus a pairwise suffix-array span proof.**
   - Experiment: synthetic excerpts of 15, 30, 60, and 120 words, with edits outside the spans.
   - Pass: every excerpt ≥ t detected, exact offsets reported, precision 1.000, at most 10% slowdown.

Paraphrase/semantic (candidate 7) comes only after a paraphrase holdout with same-topic negative pairs exists.
