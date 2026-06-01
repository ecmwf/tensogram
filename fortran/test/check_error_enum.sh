#!/usr/bin/env bash
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
#
# Single source of truth for error codes (AGENTS.md): the Fortran
# TGM_ERROR_* parameter mirror in tensogram.F90 must match the C `tgm_error`
# enum in tensogram.h exactly. Fails (non-zero) on any drift, so a new
# variant added to the header without updating the Fortran mirror is caught
# at CI time.
#
# Usage: check_error_enum.sh <tensogram.F90> <tensogram.h>

set -euo pipefail

F90="$1"
HDR="$2"

extract() {
	grep -oE 'TGM_ERROR_[A-Z_]+[[:space:]]*=[[:space:]]*[0-9]+' "$1" |
		sed -E 's/[[:space:]]*=[[:space:]]*/=/' |
		sort -u
}

hdr=$(extract "$HDR")
f90=$(extract "$F90")

if [ "$hdr" != "$f90" ]; then
	echo "error-enum mismatch:"
	echo "  C header : $HDR"
	echo "  Fortran  : $F90"
	diff <(printf '%s\n' "$hdr") <(printf '%s\n' "$f90") || true
	exit 1
fi

n=$(printf '%s\n' "$hdr" | grep -c .)
echo "error-enum consistency OK (${n} codes match)"
