#!/usr/bin/env bash
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
#
# Single source of truth for frame-type codes (AGENTS.md): the C
# `tgm_frame_type` enum in tensogram.h must mirror the Rust core `FrameType`
# discriminants in rust/tensogram/src/wire.rs exactly. Those numbers ARE the
# wire format's frame-type field (plans/WIRE_FORMAT.md §2.2), so any skew
# silently mislabels frames in every C-family binding rather than failing
# loudly. Fails (non-zero) on any drift.
#
# Note the deliberate gap at 4 — the obsolete v2 data-object layout, reserved
# and unusable, so neither side declares it. The guard compares name=value
# pairs, so the gap is preserved by construction and re-checked explicitly.
#
# Usage: check_frame_type_enum.sh <wire.rs> <tensogram.h>

set -euo pipefail

WIRE="$1"
HDR="$2"

# Header: enum members appear one per line as `  TGM_FRAME_TYPE_NAME = <n>,`
# inside the `typedef enum { ... } tgm_frame_type;` block. Doc-comment
# references start with ` * ` so they never match `^[[:space:]]*TGM_...`.
extract_hdr() {
	grep -oE '^[[:space:]]*TGM_FRAME_TYPE_[A-Z_]+[[:space:]]*=[[:space:]]*[0-9]+' "$1" |
		grep -oE 'TGM_FRAME_TYPE_[A-Z_]+[[:space:]]*=[[:space:]]*[0-9]+' |
		sed -E 's/[[:space:]]*=[[:space:]]*/=/' |
		sort -u
}

# Rust variant name -> the C spelling cbindgen produces for the mirrored FFI
# enum: drop the redundant `Frame` suffix (`NTensorFrame` -> `NTensor`), then
# upper-case with `_` inserted at each lower->upper boundary
# (`NTensor` -> `NTENSOR`, `HeaderMetadata` -> `HEADER_METADATA`).
camel_to_screaming() {
	printf '%s' "${1%Frame}" |
		sed -E 's/([a-z0-9])([A-Z])/\1_\2/g' |
		tr '[:lower:]' '[:upper:]'
}

# Rust core: the `Variant = <n>,` lines of `pub enum FrameType { ... }`.
# Comment lines (`// Type 4 reserved …`, `/// …`) carry no `=` digit pair and
# are skipped.
extract_wire() {
	sed -n '/^pub enum FrameType {/,/^}/p' "$1" |
		grep -oE '^[[:space:]]+[A-Za-z0-9]+[[:space:]]*=[[:space:]]*[0-9]+' |
		sed -E 's/[[:space:]]//g' |
		while IFS='=' read -r name value; do
			printf 'TGM_FRAME_TYPE_%s=%s\n' "$(camel_to_screaming "$name")" "$value"
		done |
		sort -u
}

# `|| true` so a missing file or a moved declaration reaches the explicit
# diagnostic below instead of dying on a bare non-zero from the pipeline.
hdr=$(extract_hdr "$HDR") || true
wire=$(extract_wire "$WIRE") || true

# A vacuous pass (both sides empty because a path or a declaration moved) is
# worse than a failure — it looks green while guarding nothing.
if [ -z "$hdr" ] || [ -z "$wire" ]; then
	echo "frame-type enum guard extracted no variants — check both paths exist"
	echo "  C header  : $HDR"
	echo "  Rust core : $WIRE"
	exit 1
fi

if [ "$hdr" != "$wire" ]; then
	echo "frame-type enum mismatch:"
	echo "  C header  : $HDR"
	echo "  Rust core : $WIRE"
	diff <(printf '%s\n' "$hdr") <(printf '%s\n' "$wire") || true
	exit 1
fi

# Type 4 is reserved by the wire format and must never reappear.
if printf '%s\n' "$hdr" | grep -qE '=4$'; then
	echo "frame type 4 is reserved (obsolete v2 data-object layout) but is declared:"
	printf '%s\n' "$hdr" | grep -E '=4$'
	exit 1
fi

n=$(printf '%s\n' "$hdr" | grep -c .)
echo "frame-type enum consistency OK (${n} types match)"
