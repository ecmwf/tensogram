#!/usr/bin/env bash
#
# (C) Copyright 2026- ECMWF and individual contributors.
# Licensed under the Apache Licence Version 2.0.
#
# Pack a Tensogram FFI release tarball from a `cargo cinstall --destdir` tree.
#
# Single source of truth for the tarball layout, INSTALL.md content, and
# tar invocation. Called from .github/workflows/publish-ffi.yml by both
# the Linux and macOS matrix jobs.
#
# Usage: pack-ffi-tarball.sh <staging-dir> <version> <platform-label> <out-asset>
#   <staging-dir>     directory created by `cargo cinstall --destdir=...`,
#                     containing usr/local/{lib,include,...}
#   <version>         the release tag (e.g. 0.20.0)
#   <platform-label>  e.g. linux-x86_64, macos-aarch64
#   <out-asset>       output filename (e.g. tensogram-ffi-0.20.0-linux-x86_64.tar.gz)
#
# Output: <out-asset> contains
#   lib/                              libraries + pkgconfig
#   include/tensogram/                C header
#   share/doc/tensogram/{LICENSE,README.md,INSTALL.md}    metadata (FHS path)
# rooted such that the user extracts under /usr/local with
#     sudo tar -C /usr/local -xzf <out-asset>
# matching the prefix=/usr/local baked into the bundled tensogram.pc.

set -euo pipefail

STAGING="${1:?usage: pack-ffi-tarball.sh <staging-dir> <version> <label> <out-asset>}"
VERSION="${2:?missing version}"
LABEL="${3:?missing platform-label}"
ASSET="${4:?missing out-asset}"

if [ ! -d "$STAGING/usr/local" ]; then
    echo "error: $STAGING/usr/local missing — did cargo cinstall succeed?" >&2
    exit 1
fi

# Explicit template form: works on both GNU and BSD mktemp.  Some BSD
# variants reject `mktemp -d` without a template, and this script runs
# on the macOS publish-ffi.yml matrix.
ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tensogram-ffi-pack.XXXXXX")"
trap 'rm -rf "$ROOT"' EXIT

cp -a "$STAGING/usr/local/." "$ROOT/"

DOCDIR="$ROOT/share/doc/tensogram"
mkdir -p "$DOCDIR"
cp LICENSE "$DOCDIR/"
cp rust/tensogram-ffi/README.md "$DOCDIR/"

cat > "$DOCDIR/INSTALL.md" <<EOF
# Tensogram FFI binary distribution ($VERSION, $LABEL)

Default install (drop into \`/usr/local\`):

    sudo tar --no-same-owner -C /usr/local -xzf $ASSET
    pkg-config --cflags --libs tensogram

\`--no-same-owner\` is defence in depth: the archive is packed with
uid=0 / gid=0, so by default this is already what \`sudo tar\` would
do, but the flag also protects you against a self-rebuilt tarball
that did not normalise ownership.

Custom prefix: the bundled \`tensogram.pc\` hard-codes \`prefix=/usr/local\`.
For any other prefix, build from source with the desired prefix:

    cargo install cargo-c
    cargo cinstall --release -p tensogram-ffi \\
        --prefix="\$HOME/.local" --libdir=lib

Documentation: https://sites.ecmwf.int/docs/tensogram/main/guide/c-api.html
EOF

# Normalise ownership in the archive: extracting `sudo tar` would otherwise
# preserve whatever uid/gid the CI runner used (often a numeric uid with no
# matching name on the consumer's machine), leaving files under /usr/local
# owned by an arbitrary user.  Force uid=0 / gid=0 (root:root).
# GNU tar (Linux) and BSD/libarchive tar (macOS) spell the flag differently;
# distinguish by the `tar --version` banner.
if tar --version 2>&1 | grep -q '^bsdtar'; then
    TAR_OWN_FLAGS=(--uid 0 --gid 0)
else
    TAR_OWN_FLAGS=(--owner=0 --group=0)
fi

tar "${TAR_OWN_FLAGS[@]}" -czf "$ASSET" -C "$ROOT" .
echo "packed $ASSET ($(du -h "$ASSET" | cut -f1))"
