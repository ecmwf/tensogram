#!/usr/bin/env bash
# Download ECMWF opendata GRIB test fixtures via byte-range requests.
#
# Source: IFS 0.25-degree operational forecast, 2026-04-04 00z, step 0h
# URL: https://data.ecmwf.int/forecasts/20260404/00z/ifs/0p25/oper/20260404000000-0h-oper-fc.grib2
#
# These byte offsets come from the companion .index file.

set -euo pipefail
cd "$(dirname "$0")"

BASE="https://data.ecmwf.int/forecasts/20260404/00z/ifs/0p25/oper/20260404000000-0h-oper-fc.grib2"

echo "Downloading lsm.grib2 (land-sea mask, sfc) ..."
curl -s -o lsm.grib2 -r 5434055-5621835 "$BASE"

echo "Downloading 2t.grib2 (2m temperature, sfc) ..."
curl -s -o 2t.grib2 -r 74573515-75234113 "$BASE"

echo "Downloading q_150.grib2 (specific humidity, 150 hPa) ..."
curl -s -o q_150.grib2 -r 0-477031 "$BASE"

echo "Downloading t_600.grib2 (temperature, 600 hPa) ..."
curl -s -o t_600.grib2 -r 25307135-25818452 "$BASE"

echo "Done."
ls -lh *.grib2
