#!/usr/bin/env bash
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
#
# Single source of truth for the VALUED C enums the Fortran binding mirrors
# (AGENTS.md): a `TGM_<ENUM>_*` integer parameter block in tensogram.F90 must
# match the `typedef enum { ... }` of the same name in tensogram.h exactly —
# same member names, same numbers, no extras on either side. Fails (non-zero)
# on any drift, so a variant added to (or renumbered in) the header without
# updating the Fortran mirror is caught at CI time.
#
# Generalises check_error_enum.sh / check_value_type_enum.sh, which do the same
# job for one enum each, to every enum whose header members carry an explicit
# `= <n>`: TGM_FRAME_TYPE (wire-format frame-type codes),
# TGM_DTYPE / TGM_BYTE_ORDER (frozen C ABI codes for the typed object
# accessors), TGM_AGGREGATE_HASH_POLICY and TGM_COMPRESSION_BACKEND (the
# encode-option knobs). One script, one test per enum — the extraction is
# identical, only the prefix differs.
#
# Note what this does NOT need to re-check. The name=value comparison
# preserves structural quirks by construction: the frame-type sequence's
# deliberate GAP AT 4 (the obsolete v2 data-object layout, reserved and
# undeclared) cannot appear on the Fortran side unless it appears in the
# header, which check_frame_type_enum.sh separately forbids; and the dtype
# codes' dense 0..N-1 range is guarded against the Rust core by
# check_dtype_enum.sh. This guard covers the remaining edge those two do not:
# a hand-maintained Fortran mirror drifting from the generated header.
#
# Usage: check_enum_mirror.sh <tensogram.F90> <tensogram.h> <TGM_ENUM_PREFIX>

set -euo pipefail

F90="$1"
HDR="$2"
PREFIX="$3"

# Header: enum members appear one per line as `  TGM_<ENUM>_NAME = <n>,` inside
# the `typedef enum { ... }` block. Doc-comment references start with ` * ` so
# they never match `^[[:space:]]*TGM_...`, and they carry no `= <n>` anyway.
extract_hdr() {
	grep -oE "^[[:space:]]*${PREFIX}_[A-Z0-9_]+[[:space:]]*=[[:space:]]*[0-9]+" "$HDR" |
		grep -oE "${PREFIX}_[A-Z0-9_]+[[:space:]]*=[[:space:]]*[0-9]+" |
		sed -E 's/[[:space:]]*=[[:space:]]*/=/' |
		sort -u
}

# Fortran: `integer(c_int), parameter :: TGM_<ENUM>_NAME = <n>` declarations.
# Anchoring on `parameter ::` keeps prose in the doc comments (which name the
# parameters freely) from being mistaken for a declaration.
extract_f90() {
	grep -E 'parameter[[:space:]]*::' "$F90" |
		grep -oE "${PREFIX}_[A-Z0-9_]+[[:space:]]*=[[:space:]]*[0-9]+" |
		sed -E 's/[[:space:]]*=[[:space:]]*/=/' |
		sort -u
}

# `|| true` so a missing file or a moved declaration reaches the explicit
# diagnostic below instead of dying on a bare non-zero from the pipeline.
hdr=$(extract_hdr) || true
f90=$(extract_f90) || true

# A vacuous pass (both sides empty because a path, a prefix or a declaration
# moved) is worse than a failure — it looks green while guarding nothing.
if [ -z "$hdr" ] || [ -z "$f90" ]; then
	echo "${PREFIX} enum guard extracted no variants — check the paths and the prefix"
	echo "  C header : $HDR"
	echo "  Fortran  : $F90"
	echo "  prefix   : $PREFIX"
	exit 1
fi

if [ "$hdr" != "$f90" ]; then
	echo "${PREFIX} enum mismatch (Fortran mirror vs C header):"
	echo "  C header : $HDR"
	echo "  Fortran  : $F90"
	diff <(printf '%s\n' "$hdr") <(printf '%s\n' "$f90") || true
	exit 1
fi

n=$(printf '%s\n' "$hdr" | grep -c .)
echo "${PREFIX} enum consistency OK (${n} members match)"
