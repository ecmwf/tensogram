#!/usr/bin/env bash
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
#
# Single source of truth for element types (AGENTS.md): the C `tgm_dtype` enum
# in the checked-in tensogram.h must cover exactly the variants of the Rust core
# `Dtype` in rust/tensogram/src/dtype.rs — no more, no fewer.
#
# The Rust side already fails to compile when a core variant is added (the
# `From<Dtype> for TgmDtype` conversion and its unit test both match
# exhaustively).  This guard covers what a Rust test cannot: tensogram.h is a
# *checked-in generated artefact* that the C++ and Fortran bindings mirror by
# hand, so a stale or hand-edited header can disagree with the crate it was
# supposed to be generated from.
#
# Deliberately compares NAME SETS, not name=value pairs, unlike
# check_frame_type_enum.sh.  Frame-type numbers are wire-format values and must
# mirror the core discriminants; dtype numbers are not on the wire (the format
# stores dtype as text) — they are frozen C ABI values that must NOT follow a
# reordering of the core enum.  What the guard does check about the numbers is
# that they stay a dense, duplicate-free 0..N-1 range, which is the property
# the bindings' lookup tables rely on.
#
# Usage: check_dtype_enum.sh <dtype.rs> <tensogram.h>

set -euo pipefail

DTYPE_RS="$1"
HDR="$2"

# Header: enum members appear one per line as `  TGM_DTYPE_NAME = <n>,` inside
# the `typedef enum { ... } tgm_dtype;` block. Doc-comment references start
# with ` * ` so they never match `^[[:space:]]*TGM_...`.
extract_hdr_pairs() {
	grep -oE '^[[:space:]]*TGM_DTYPE_[A-Z0-9_]+[[:space:]]*=[[:space:]]*[0-9]+' "$1" |
		grep -oE 'TGM_DTYPE_[A-Z0-9_]+[[:space:]]*=[[:space:]]*[0-9]+' |
		sed -E 's/[[:space:]]*=[[:space:]]*/=/' |
		sort -u
}

extract_hdr_names() {
	extract_hdr_pairs "$1" | cut -d= -f1 | sort -u
}

# Rust core: the bare `Variant,` lines of `pub enum Dtype { ... }`. Digits are
# part of the name (`Float16` -> `FLOAT16`), so underscores are inserted only at
# lower->upper boundaries — the same spelling cbindgen produces.
extract_core_names() {
	sed -n '/^pub enum Dtype {/,/^}/p' "$1" |
		grep -oE '^[[:space:]]+[A-Za-z][A-Za-z0-9]*,' |
		tr -d ' ,' |
		sed -E 's/([a-z0-9])([A-Z])/\1_\2/g' |
		tr '[:lower:]' '[:upper:]' |
		sed -E 's/^/TGM_DTYPE_/' |
		sort -u
}

# `|| true` so a missing file or a moved declaration reaches the explicit
# diagnostic below instead of dying on a bare non-zero from the pipeline.
hdr_names=$(extract_hdr_names "$HDR") || true
core_names=$(extract_core_names "$DTYPE_RS") || true

# A vacuous pass (both sides empty because a path or a declaration moved) is
# worse than a failure — it looks green while guarding nothing.
if [ -z "$hdr_names" ] || [ -z "$core_names" ]; then
	echo "dtype enum guard extracted no variants — check both paths exist"
	echo "  C header  : $HDR"
	echo "  Rust core : $DTYPE_RS"
	exit 1
fi

if [ "$hdr_names" != "$core_names" ]; then
	echo "dtype enum mismatch (C header vs Rust core Dtype):"
	echo "  C header  : $HDR"
	echo "  Rust core : $DTYPE_RS"
	diff <(printf '%s\n' "$hdr_names") <(printf '%s\n' "$core_names") || true
	exit 1
fi

# The C values must stay a dense 0..N-1 range with no duplicates: the bindings
# index lookup tables by them.
n=$(printf '%s\n' "$hdr_names" | grep -c .)
values=$(extract_hdr_pairs "$HDR" | cut -d= -f2 | sort -n)
expected=$(seq 0 $((n - 1)))
if [ "$values" != "$expected" ]; then
	echo "dtype enum values are not a dense 0..$((n - 1)) range:"
	printf '%s\n' "$(extract_hdr_pairs "$HDR")"
	exit 1
fi

echo "dtype enum consistency OK (${n} dtypes match, values 0..$((n - 1)))"
