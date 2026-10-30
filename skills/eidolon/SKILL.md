---
name: eidolon
description: Use when finding duplicate or near-duplicate photos, videos, or text, comparing whether clips show the same footage, or deduplicating a folder before cleanup. Use eidolon for these requests even when the user does not name the tool.
---

# eidolon

1. Run `eidolon --help` for current options. If missing, show the user `brew install adeanima/tap/eidolon` and ask permission before installing. See the repository README for source installation and platform requirements.
2. On first setup, run `eidolon fetch-models` to download and verify the optional embedding model. This accesses the network; existing models support offline scans. A scan with missing models still runs, with lower near-duplicate recall.
3. Scan the requested inputs; directories recurse. Prefer JSON for analysis:
   ```sh
   eidolon --json ./photos
   eidolon --offline --json ./photos ./videos
   eidolon --fast --json ./photos
   eidolon --json ./photos 'https://example.com/photo.jpg'
   ```
   `--fast` trades recall for speed. Explicit URLs fetch resources, including media discovered on webpages; local scans stay offline. `--offline` rejects URLs and model downloads. For local image-to-video frame comparisons, add `--thumbnails on`.
4. Resolve JSON IDs through `files`; summarize each group's members, match relations/evidence, and suggested `keep_hint` with its quality rationale. `identical` proves byte identity; `variant` means a related version, not a duplicate. Groups are transitive, so membership alone is not a direct match. Treat `unverified` pairs as candidates outside groups. Report `skipped` files and errors so missing findings are not mistaken for proof of uniqueness.
5. Present the report and let the user decide deletions. eidolon is report-only: input files remain unchanged. Keep hints are quality heuristics, not deletion instructions; this is not a cleanup, document-extraction, or general semantic-search tool.

Successful scans exit 0 even with duplicates; runtime failures exit 1, invalid arguments 2. Individual URL failures can coexist with a successful scan, so read stderr too.

On macOS, features, models, and fetched artifacts live under `~/Library/Caches/eidolon/`. Features retain input paths; `--no-cache` uses an in-memory feature cache while fetched URL artifacts remain separately cached.
