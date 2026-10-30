#!/usr/bin/env bash
# After committing the release: git tag v<version>; scripts/release.sh <version>.
# Upload dist/eidolon-<version>.tar.gz as a GitHub Release asset, then update
# the Homebrew formula's release-asset URL and SHA-256 to the printed values.
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
  echo "Usage: scripts/release.sh <version> (for an existing local v<version> tag)" >&2
  exit 2
fi

version=$1
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
tag="refs/tags/v${version}"
git -C "$root" rev-parse --verify "${tag}^{commit}" >/dev/null
mkdir -p "$root/dist"
archive="$root/dist/eidolon-${version}.tar.gz"
temporary=$(mktemp "$root/dist/.eidolon-${version}.XXXXXX")
trap 'rm -f "$temporary"' EXIT

# git supplies stable file order, modes and commit timestamps; gzip adds no date.
git -C "$root" archive --format=tar --prefix="eidolon-${version}/" "$tag" | gzip -n >"$temporary"
mv "$temporary" "$archive"
shasum -a 256 "$archive"
