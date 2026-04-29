#!/usr/bin/env bash
#
# (C) Copyright 2026- ECMWF and individual contributors.
# Licensed under the Apache Licence Version 2.0.
#
# Smoke test for a `cargo cinstall`-produced Tensogram C FFI distribution.
#
# Verifies that, given an installed prefix containing
#   <prefix>/lib/libtensogram.{so,dylib,a}
#   <prefix>/lib/pkgconfig/tensogram.pc
#   <prefix>/include/tensogram/tensogram.h
# a C program can be compiled, linked, and run against it via pkg-config.
#
# The C program does a `tgm_encode` -> `tgm_decode` round trip and
# compares the decoded payload against the input with `memcmp`, so the
# linker resolves real FFI symbols and the runtime actually loads the
# library.
#
# Usage: test_cargo_c_smoke.sh <installed-prefix>
#
# The script resolves the repo's VERSION file relative to its own
# location, so it can be invoked from any working directory.
#
# Two callers:
# - CI cargo-c job (and release-preflight) installs to `mktemp -d`:
#       cargo cinstall --prefix=$PFX -p tensogram-ffi
#       bash test_cargo_c_smoke.sh $PFX
# - publish-ffi.yml stages with --destdir, sets PKG_CONFIG_SYSROOT_DIR:
#       cargo cinstall --prefix=/usr/local --destdir=$PWD/staging -p tensogram-ffi
#       PKG_CONFIG_SYSROOT_DIR=$PWD/staging \
#         bash test_cargo_c_smoke.sh $PWD/staging/usr/local

set -euo pipefail

PREFIX="${1:?usage: test_cargo_c_smoke.sh <installed-prefix>}"

if [ ! -d "$PREFIX" ]; then
    echo "error: $PREFIX is not a directory" >&2
    exit 1
fi

# Resolve the repo's VERSION file relative to this script, not the
# caller's CWD. This script lives at cpp/tests/test_cargo_c_smoke.sh,
# so the repo root is two levels up. Locating the file this way lets
# the script be run from any directory (e.g. a CI build dir, or a
# tarball-validation workflow that has its own CWD).
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if [ ! -f "$REPO_ROOT/VERSION" ]; then
    echo "error: $REPO_ROOT/VERSION not found (resolved from script path)" >&2
    exit 1
fi

export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

pkg-config --exists tensogram || {
    echo "error: pkg-config cannot find tensogram (PKG_CONFIG_PATH=$PKG_CONFIG_PATH)" >&2
    exit 1
}

EXPECTED_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/VERSION")"
ACTUAL_VERSION="$(pkg-config --modversion tensogram)"
if [ "$ACTUAL_VERSION" != "$EXPECTED_VERSION" ]; then
    echo "error: pkg-config --modversion = $ACTUAL_VERSION, expected $EXPECTED_VERSION" >&2
    exit 1
fi

CFLAGS_OUT="$(pkg-config --cflags tensogram)"
case "$CFLAGS_OUT" in
    *-I*) ;;
    *) echo "error: pkg-config --cflags missing -I: '$CFLAGS_OUT'" >&2; exit 1 ;;
esac

LIBS_OUT="$(pkg-config --libs tensogram)"
case "$LIBS_OUT" in
    *-ltensogram*) ;;
    *) echo "error: pkg-config --libs missing -ltensogram: '$LIBS_OUT'" >&2; exit 1 ;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/main.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <tensogram/tensogram.h>

static const char *err_or(const char *s) { return s ? s : "(null)"; }

int main(void) {
    const char *meta_json =
        "{\"descriptors\":[{\"type\":\"ntensor\",\"ndim\":1,"
        "\"shape\":[4],\"strides\":[4],\"dtype\":\"float32\","
        "\"byte_order\":\"little\",\"encoding\":\"none\","
        "\"filter\":\"none\",\"compression\":\"none\"}]}";

    float in[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const uint8_t *ptrs[1] = { (const uint8_t*)in };
    const size_t lens[1] = { sizeof(in) };

    tgm_bytes_t enc = {0};
    tgm_error rc = tgm_encode(meta_json, ptrs, lens, 1, "xxh3", 0, &enc);
    if (rc != TGM_ERROR_OK) {
        fprintf(stderr, "encode failed: rc=%d msg=%s\n", (int)rc, err_or(tgm_last_error()));
        return 1;
    }

    tgm_message_t *msg = NULL;
    rc = tgm_decode(enc.data, enc.len,
                    /*native_byte_order=*/1,
                    /*threads=*/0,
                    &msg);
    if (rc != TGM_ERROR_OK || msg == NULL) {
        fprintf(stderr, "decode failed: rc=%d msg=%s\n", (int)rc, err_or(tgm_last_error()));
        tgm_bytes_free(enc);
        return 1;
    }

    size_t n = tgm_message_num_objects(msg);
    if (n != 1) {
        fprintf(stderr, "expected 1 object, got %zu\n", n);
        tgm_message_free(msg);
        tgm_bytes_free(enc);
        return 1;
    }

    size_t out_len = 0;
    const uint8_t *out_data = tgm_object_data(msg, 0, &out_len);
    if (out_data == NULL || out_len != sizeof(in) ||
        memcmp(out_data, in, out_len) != 0) {
        fprintf(stderr, "round-trip data mismatch (out_len=%zu)\n", out_len);
        tgm_message_free(msg);
        tgm_bytes_free(enc);
        return 1;
    }

    tgm_message_free(msg);
    tgm_bytes_free(enc);

    printf("wire_version=%u round_trip=ok\n", (unsigned)TGM_WIRE_VERSION);
    return 0;
}
EOF

cc $CFLAGS_OUT "$TMP/main.c" $LIBS_OUT -o "$TMP/main"

LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}" \
DYLD_LIBRARY_PATH="$PREFIX/lib:${DYLD_LIBRARY_PATH:-}" \
"$TMP/main" | grep -q "wire_version=3 round_trip=ok"

echo "cargo-c smoke test OK (prefix=$PREFIX)"
