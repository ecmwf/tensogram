#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""
17 — convert GRIB to Tensogram (pure Python)

Reads a real ECMWF opendata GRIB2 fixture (committed under
``rust/tensogram-grib/testdata/``) via the PyO3 wrapper around
``tensogram-grib``, demonstrates both entry points, and then decodes
the result with the ``tensogram`` Python bindings.

Two entry points are showcased:

1. **File path** — ``tensogram.convert_grib("path.grib2")``.
   Simplest case: hand ecCodes a filesystem path.
2. **In-memory buffer** — ``tensogram.convert_grib_buffer(bytes_obj)``.
   Useful when the GRIB bytes come from an HTTP byte-range fetch,
   a cache, or any other in-memory source — no need to stage through
   the filesystem.

Run::

    # from repo root — requires the grib Cargo feature:
    cd python/bindings && maturin develop --features grib
    python examples/python/17_convert_grib.py

Requires:
    - ``libeccodes`` installed at the OS level
      (``brew install eccodes`` / ``apt install libeccodes-dev``)
    - ``tensogram`` bindings built with the ``grib`` feature
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
    import tensogram
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print(
        "Install with: uv pip install numpy && "
        "(cd python/bindings && maturin develop --features grib)",
        file=sys.stderr,
    )
    sys.exit(1)


def repo_root() -> Path:
    """Walk up from this script to the tensogram repo root."""
    return Path(__file__).resolve().parents[2]


def grib_fixture(name: str) -> Path:
    """Resolve a GRIB fixture by name under ``rust/tensogram-grib/testdata/``."""
    return repo_root() / "rust" / "tensogram-grib" / "testdata" / name


def main() -> int:
    if not getattr(tensogram, "__has_grib__", False):
        print(
            "ERROR: tensogram was built without the 'grib' feature.\n"
            "Rebuild with: cd python/bindings && maturin develop --features grib",
            file=sys.stderr,
        )
        return 1

    path = grib_fixture("2t.grib2")
    if not path.exists():
        print(
            f"ERROR: GRIB fixture not found at {path}\n"
            "Run: bash rust/tensogram-grib/testdata/download.sh",
            file=sys.stderr,
        )
        return 1

    print(f"Source: {path}  ({path.stat().st_size} bytes)")
    print("(real ECMWF opendata, IFS 0.25 deg operational, 20260404 00z, 2m temperature)")

    # ── 1. File-path conversion ───────────────────────────────────────────────
    print()
    print("1. tensogram.convert_grib(path)")
    messages_file = tensogram.convert_grib(str(path))
    print(
        f"   -> {len(messages_file)} Tensogram message(s), first is {len(messages_file[0])} bytes"
    )

    # ── 2. In-memory buffer conversion ────────────────────────────────────────
    print()
    print("2. tensogram.convert_grib_buffer(bytes_obj)")
    with open(path, "rb") as fh:
        grib_bytes = fh.read()
    messages_buffer = tensogram.convert_grib_buffer(grib_bytes)
    print(
        f"   -> {len(messages_buffer)} Tensogram message(s), "
        f"first is {len(messages_buffer[0])} bytes"
    )

    # ── 3. Decode and inspect ────────────────────────────────────────────────
    print()
    print("3. Decoding message[0] with tensogram.decode()")
    meta, objects = tensogram.decode(messages_file[0])
    print(f"   metadata version = {meta.version}")
    print(f"   {len(meta.base)} base entry(ies)")
    mars_keys = sorted(meta.base[0].get("mars", {}).keys())
    print(f"   mars keys: {mars_keys}")

    for obj_idx, (desc, array) in enumerate(objects):
        print(
            f"   object[{obj_idx}] shape={list(desc.shape)} "
            f"dtype={desc.dtype} encoding={desc.encoding} "
            f"compression={desc.compression}"
        )
        print(f"     value range: [{float(array.min()):.2f}, {float(array.max()):.2f}] (Kelvin)")

    # ── 4. Verify file vs buffer round-trip agreement ────────────────────────
    #
    # Encoded bytes may differ (each encode call stamps a fresh uuid
    # and timestamp in `_reserved_`), but the *decoded payload* must be
    # bit-identical.
    _, objs_file = tensogram.decode(messages_file[0])
    _, objs_buf = tensogram.decode(messages_buffer[0])
    np.testing.assert_array_equal(objs_file[0][1], objs_buf[0][1])
    print()
    print("4. file and buffer paths produce bit-identical decoded payloads")

    # ── 5. Re-encode with a compression pipeline ─────────────────────────────
    print()
    print(
        "5. tensogram.convert_grib(path, encoding='simple_packing', bits=16, compression='zstd')"
    )
    compressed = tensogram.convert_grib(
        str(path),
        encoding="simple_packing",
        bits=16,
        compression="zstd",
    )
    ratio = len(compressed[0]) / len(messages_file[0]) * 100
    print(f"   compressed size: {len(compressed[0])} bytes ({ratio:.1f}% of raw f64)")

    print()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
