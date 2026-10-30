# eidolon

Report-only finder for exact and near-duplicate images, videos and text. It reads files and prints matches; it never moves, deletes or rewrites them.

## Layout

- `crates/eidolon-core`: shared contract (`FileEntry`, `Match`, `Relation`, `Ctx`, `Embedder`) and the SQLite `FeatureCache`. Changes here touch every crate: keep them additive.
- `crates/eidolon-{image,video,text}`: one detector each, entry point `pub fn find(files: &[FileEntry], ctx: &Ctx) -> anyhow::Result<Vec<Match>>`. `eidolon_image::find` adds a third argument: an optional source-aspect embedding bank (`eidolon_image::source_frame_embeddings`) that the CLI supplies for image scans with embeddings enabled and `--image-source-bank on` (default); `off` passes `None`. Independent of thumbnail matching.
- `crates/eidolon-cli`: binary `eidolon`: input walking, classification, exact matching, grouping, report.
- `crates/eidolon-eval`: synthetic ground-truth corpus and precision/recall scoring. Detector changes are judged by its numbers.

## Verify loop

Run per crate while iterating (`-p <crate>`), then the whole workspace once before handing back:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
cargo test --workspace --doc
```

Use rust-analyzer or your editor's language server for definitions, references and diagnostics; Cargo commands above are the final gate.

## Conventions

- Toolchain pinned in `rust-toolchain.toml`; lints in `[workspace.lints]`. Fix lints; a scoped `#[allow]` needs a one-line reason.
- Every public item has a doc comment describing current behaviour. Each item is reachable through one path; no re-exports of crate items.
- Tests check observable behaviour on real decoded inputs (thresholds, relation direction, cache invalidation), never restated constants.
- Native deps come from Homebrew (`opencv`, `onnxruntime`, `ffmpeg`). Set `LIBCLANG_PATH` for builds if needed. The build records `EIDOLON_ORT_DYLIB_PATH`, or detects Homebrew's ONNX Runtime prefix on macOS; runtime `ORT_DYLIB_PATH` overrides that recorded path. No build step downloads binaries.
- Network access happens only in the explicit `eidolon fetch-models` command (with pinned model SHA-256 hashes) and when HTTP(S) URL inputs are supplied. Local-only scans stay offline.
