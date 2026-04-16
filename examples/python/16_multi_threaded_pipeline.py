#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 16 — Multi-threaded coding pipeline (v0.13.0).

Demonstrates the caller-controlled ``threads`` kwarg added in v0.13.0.

Key invariants shown:

1. ``threads=0`` (default) matches the sequential path byte-identically.
2. Transparent codecs (simple_packing, szip, ...) produce byte-identical
   encoded payloads across any ``threads`` value.
3. Opaque codecs (blosc2, zstd with workers) round-trip losslessly
   regardless of thread count.

Run with:
    python examples/python/16_multi_threaded_pipeline.py
"""

from __future__ import annotations

import time

import numpy as np
import tensogram


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0  # ms


def encoded_payload_hashes(msg: bytes) -> list[str]:
    """Return per-object payload hashes — identical => encoded bytes identical."""
    _, descriptors = tensogram.decode_descriptors(msg)
    return [d.hash for d in descriptors]


def main() -> None:
    # ── 1. Build a large single-object message ────────────────────────────
    n = 16_000_000
    rng = np.random.default_rng(42)
    values = 250.0 + rng.standard_normal(n).astype(np.float64) * 30.0
    meta = {"version": 2, "base": [{}]}
    desc = {"type": "ntensor", "shape": [n], "dtype": "float64"}

    print(f"Encode {n} f64 values (= {values.nbytes / (1024 * 1024):.1f} MiB):")

    # Warm up codec/init costs
    _ = tensogram.encode(meta, [(desc, values)])

    _, dur_seq = timed(tensogram.encode, meta, [(desc, values)], threads=0)
    _, dur_par = timed(tensogram.encode, meta, [(desc, values)], threads=8)
    print(f"  threads=0: {dur_seq:>7.1f} ms")
    print(
        f"  threads=8: {dur_par:>7.1f} ms   (x{dur_seq / max(dur_par, 1e-9):.2f} speedup)"
    )

    # Determinism invariant: encoded payload hashes equal
    msg_seq = tensogram.encode(meta, [(desc, values)], threads=0)
    msg_par = tensogram.encode(meta, [(desc, values)], threads=8)
    assert encoded_payload_hashes(msg_seq) == encoded_payload_hashes(msg_par), (
        "transparent pipeline must be byte-identical across thread counts"
    )
    print("  ✓ encoded payloads are byte-identical across threads")

    # ── 2. Decode ─────────────────────────────────────────────────────────
    _, dec_seq_ms = timed(tensogram.decode, msg_seq, threads=0)
    _, dec_par_ms = timed(tensogram.decode, msg_seq, threads=8)
    print()
    print(f"Decode (threads=0): {dec_seq_ms:>7.1f} ms")
    print(
        f"Decode (threads=8): {dec_par_ms:>7.1f} ms   "
        f"(x{dec_seq_ms / max(dec_par_ms, 1e-9):.2f} speedup)"
    )

    # ── 3. Opaque codec (blosc2) — differ byte-wise, same values ──────────
    blosc2_desc = {
        **desc,
        "compression": "blosc2",
        "blosc2_clevel": 5,
        "blosc2_codec": "lz4",
    }
    msg_b0 = tensogram.encode(meta, [(blosc2_desc, values)], threads=0)
    msg_b8 = tensogram.encode(meta, [(blosc2_desc, values)], threads=8)

    decoded_b0 = tensogram.decode(msg_b0).objects[0][1]
    decoded_b8 = tensogram.decode(msg_b8).objects[0][1]

    print()
    print("blosc2 opaque pipeline:")
    print(f"  encode bytes differ across threads: {msg_b0 != msg_b8}")
    print(f"  decoded data matches: {np.array_equal(decoded_b0, decoded_b8)}")
    np.testing.assert_array_equal(
        decoded_b0,
        decoded_b8,
        err_msg="blosc2 must round-trip losslessly regardless of threads",
    )

    # ── 4. Environment variable fallback ──────────────────────────────────
    print()
    print("Tip: set TENSOGRAM_THREADS=N at the process level to enable")
    print("     parallel coding for every call that uses threads=0 (the")
    print("     default).  Explicit threads=... arguments always win.")


if __name__ == "__main__":
    main()
