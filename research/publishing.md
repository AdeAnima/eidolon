# Publication setup sources

Checked 2026-10-30. Local preparation does not publish either repository or any release.

- GitHub's [hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories) lists standard `macos-15` and `macos-26` as arm64. GitHub [retires macOS 14 runners on 2026-11-02](https://github.blog/changelog/2026-10-01-github-actions-macos-14-runner-image-retirement/); Homebrew's [support tiers](https://docs.brew.sh/Support-Tiers) put Sonoma 14 in Tier 3 as of September 2026. CI therefore uses 15/26 and explicitly checks `uname -m`; bottles build on 15. These primary pages were checked on 2026-10-30. Local acceptance does not establish that hosted CI has passed.
- The installed Homebrew `Library/Homebrew/dev-cmd/tap-new.rb` and [Homebrew workflow templates](https://github.com/Homebrew/brew/tree/HEAD/.github/workflows) provide the `test-bot` and `pr-pull` flow. The default publication path is manual dispatch with the reviewed PR head SHA. The optional maintainer-applied `pr-pull` label additionally requires an applicable Actions event policy explicitly allowing `pull_request_target`, which GitHub's [default public-repository policy blocks from December 2, 2026](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target#default-policy-for-pull_request_target).
- [`ort` dynamic loading](https://docs.rs/ort/2.0.0-rc.12/ort/fn.init_from.html) accepts an explicit shared-library path. The pinned crate's `src/lib.rs` loads it through `libloading` and checks API compatibility. eidolon resolves an explicit runtime `ORT_DYLIB_PATH` first, then its build-recorded `EIDOLON_ORT_DYLIB_PATH`; the build can detect Homebrew's stable `opt` prefix on macOS. Missing paths remain actionable errors, not silent inference disablement.
- [taiki-e/install-action](https://github.com/taiki-e/install-action#usage) supports `cargo-nextest@<version>` and `fallback: none`. Actions in these workflows use full commit SHA pins.

ImageIO and CoreML are conditionally compiled for macOS. Other source platforms have not been validated. The tap deliberately supports only the tested Apple Silicon macOS channel. Crates stay non-publishable because the crates.io name belongs to an unrelated package.

## Model and evaluation provenance

The [upstream DINOv2-small model card](https://huggingface.co/facebook/dinov2-small/blob/main/README.md) declares `license: apache-2.0`. The [pinned ONNX conversion card](https://huggingface.co/onnx-community/dinov2-small/blob/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/README.md) identifies `facebook/dinov2-small` as its base model and describes the ONNX conversion, but contains no licence metadata. These primary cards were read on 2026-10-30; conversion-specific terms are not independently verified. We do not redistribute model weights. `NOTICE` records this distinction rather than assuming the conversion has independently verified terms.

Evaluation media and raw source captures are not redistributed. Source manifests preserve per-source authors, licences, URLs and payload hashes. Construction-lineage hashes identify the original construction artifacts; source and payload hashes identify media bytes.

## Third-party Rust notices

[`cargo-about`'s generation documentation](https://embarkstudios.github.io/cargo-about/cli/generate/index.html) specifies that `--frozen` combines a locked dependency graph with fully offline operation, preventing all tool network access. `about.toml` and `about.hbs` generate `THIRD_PARTY_LICENSES` from the resolved workspace graph with all workspace features and target-platform dependencies, including build/test dependencies. The text identifies each exact source archive and explains MPL-covered source access. Workspace code is covered separately by `LICENSE`.

The locked `ring` 0.17.14 package declares `Apache-2.0 AND ISC`: its `LICENSE` points to packaged `LICENSE-BoringSSL` and `LICENSE-other-bits`. `about.toml` pins the SHA-256 of those full files instead of using cargo-about's workaround for an older ring licence layout. No guessed supplementary OpenSSL terms are added. Other unavailable standalone licence files use cargo-about's standard SPDX text; the inventory includes package authors and exact source archive URLs.

## Native runtime licences

Checked locally on 2026-10-31 with `brew info --json=v2 ffmpeg opencv onnxruntime`, the installed FFmpeg formula, and `ffmpeg -hide_banner -buildconf`. Homebrew declares [FFmpeg](https://formulae.brew.sh/formula/ffmpeg) as `GPL-3.0-or-later`, [OpenCV](https://formulae.brew.sh/formula/opencv) as `Apache-2.0`, and [ONNX Runtime](https://formulae.brew.sh/formula/onnxruntime) as `MIT`. The installed FFmpeg 9.0.2_1 formula explicitly enables `--enable-shared`, `--enable-gpl` and `--enable-version3`; the binary reports the same configuration. The corresponding [formula source](https://github.com/Homebrew/homebrew-core/blob/e35528c6d6a89b17a22ee31e6d95bddfff6c34e1/Formula/f/ffmpeg.rb) explains the resulting GPL v3-or-later terms. `NOTICE` names these dynamically linked runtime libraries and retains their own licence terms; it does not describe Homebrew's FFmpeg build as LGPL-only.

