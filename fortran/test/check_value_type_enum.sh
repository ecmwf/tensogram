#!/usr/bin/env bash
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
#
# Single source of truth for value-type codes (AGENTS.md): the Fortran
# TGM_VALUE_TYPE_* parameter mirror in tensogram.F90 must match the C
# `tgm_value_type` enum in tensogram.h exactly. The header enum is unvalued
# (cbindgen emits bare variants), so members are numbered 0,1,2,... in
# declaration order and compared against the Fortran explicit `= <n>` mirror.
# Fails (non-zero) on any drift, so a variant reordered or added in the header
# without updating the Fortran mirror is caught at CI time.
#
# Usage: check_value_type_enum.sh <tensogram.F90> <tensogram.h>

set -euo pipefail

F90="$1"
HDR="$2"

# Header: enum members appear one per line as `  TGM_VALUE_TYPE_NAME,` inside
# the `typedef enum { ... } tgm_value_type;` block (leading whitespace, no `=`).
# Doc-comment references start with ` * ` so they do not match `^\s*TGM_...`.
# Number the members positionally (0-based) in declaration order.
extract_hdr() {
	grep -oE '^[[:space:]]*TGM_VALUE_TYPE_[A-Z]+' "$1" |
		grep -oE 'TGM_VALUE_TYPE_[A-Z]+' |
		awk '{ print $1 "=" NR - 1 }' |
		sort -u
}

# Fortran: explicit `TGM_VALUE_TYPE_NAME = <n>` integer parameters.
extract_f90() {
	grep -oE 'TGM_VALUE_TYPE_[A-Z]+[[:space:]]*=[[:space:]]*[0-9]+' "$1" |
		sed -E 's/[[:space:]]*=[[:space:]]*/=/' |
		sort -u
}

hdr=$(extract_hdr "$HDR")
f90=$(extract_f90 "$F90")

if [ "$hdr" != "$f90" ]; then
	echo "value-type enum mismatch:"
	echo "  C header : $HDR"
	echo "  Fortran  : $F90"
	diff <(printf '%s\n' "$hdr") <(printf '%s\n' "$f90") || true
	exit 1
fi

n=$(printf '%s\n' "$hdr" | grep -c .)
echo "value-type enum consistency OK (${n} kinds match)"
