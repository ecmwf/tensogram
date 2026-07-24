# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""
Example 19 — Async streaming encoder + compute_common
=====================================================

Demonstrates two symmetry additions to the Python binding:

  1. ``tensogram.AsyncStreamingEncoder`` — the asyncio sibling of
     ``StreamingEncoder``.  Emit forecast fields frame-at-a-time from an
     async producer, with a ``PrecederMetadata`` frame ahead of each
     object, then ``await enc.finish()`` for the complete wire bytes.
     Introspect progress with ``object_count()`` / ``bytes_written()``.

  2. ``tensogram.compute_common`` — a direct binding of the Rust core
     utility that extracts the metadata keys shared (with identical
     values) across every ``base[i]`` entry.  Commonalities are a
     post-decode software convenience — they are never encoded on the
     wire.

Run:  python 19_async_streaming_and_common.py
"""

import asyncio

import numpy as np
import tensogram

# Four "forecast fields".  Every object shares the same top-level
# `source` / `experiment` annotations; only the nested `mars.param`
# varies.  `compute_common` will surface the shared top-level keys and
# leave `mars` per-object (its sub-map differs).
PARAMS = ["2t", "10u", "10v", "msl"]
SHARED_MARS = {"class": "od", "stream": "oper", "date": "20260404"}
SHARED_TOP_LEVEL = {"source": "synthetic", "experiment": "demo-001"}
SHAPE = (2, 3)


def _descriptor() -> dict:
    return {
        "type": "ntensor",
        "shape": list(SHAPE),
        "dtype": "float32",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


async def build_message() -> bytes:
    """Stream several fields through the async encoder → one message."""
    # `create` is an async factory: the preamble + header frame are written
    # to the in-memory buffer before the coroutine resolves.
    enc = await tensogram.AsyncStreamingEncoder.create({"base": []})

    for i, param in enumerate(PARAMS):
        # A PrecederMetadata frame lands ahead of the object so a
        # progressive reader learns its identity before the payload bytes.
        await enc.write_preceder({**SHARED_TOP_LEVEL, "mars": {**SHARED_MARS, "param": param}})

        data = np.random.default_rng(seed=i).random(SHAPE, dtype=np.float32)
        await enc.write_object(_descriptor(), data)

        print(
            f"  wrote {param:<4} → object_count={enc.object_count()}, "
            f"bytes_written={enc.bytes_written()}"
        )

    # `finish_backfilled` patches total_length so the message is
    # backward-locatable (O(1) scan from EOF); `finish` would leave
    # total_length=0 (forward-scan only).
    msg = await enc.finish_backfilled()
    print(f"  finished: {len(msg)} bytes")
    return msg


def show_common(msg: bytes) -> None:
    """Discover the metadata keys shared across every object."""
    # Full `decode` surfaces the enriched footer metadata (with the
    # per-object preceder payloads merged in); the fast `decode_metadata`
    # only reads the streaming header, whose base is still empty here.
    meta = tensogram.decode(msg).metadata
    print(f"\ndecoded {meta.num_objects} objects; per-object mars.param:")
    for i, entry in enumerate(meta.base):
        print(f"  base[{i}].mars.param = {entry['mars']['param']}")

    # `compute_common` accepts a Metadata object (or a list like meta.base)
    # and returns (common, remaining) — mirroring the Rust core tuple.
    common, remaining = tensogram.compute_common(meta)
    print(f"\ncommon across all objects: {common}")
    print(f"first object's remaining (non-common) keys: {remaining[0]}")

    # The identical top-level keys are common; `mars` is NOT (its nested
    # `param` differs per object), so it stays in each `remaining` entry.
    assert common == SHARED_TOP_LEVEL
    assert "mars" not in common
    assert all("mars" in entry for entry in remaining)


def show_inline_hashes(msg: bytes) -> None:
    """Read the per-object integrity digests from the wire bytes."""
    # In v3 the per-object hash lives in each frame's inline footer slot,
    # not the CBOR descriptor — `object_inline_hashes` is the accessor.
    digests = tensogram.object_inline_hashes(msg)
    print("\nper-object inline xxh3-64 digests:")
    for i, d in enumerate(digests):
        print(f"  object[{i}] = {d}")
    assert all(isinstance(d, str) and len(d) == 16 for d in digests)


async def main() -> None:
    print("1. AsyncStreamingEncoder — streaming forecast fields:")
    msg = await build_message()

    # Round-trip check (verify_hash recomputes the inline digests).
    decoded = tensogram.decode(msg, verify_hash=True)
    assert len(decoded.objects) == len(PARAMS)

    print("\n2. compute_common — discovering shared metadata:")
    show_common(msg)

    show_inline_hashes(msg)

    print("\nExample 19 complete.")


if __name__ == "__main__":
    asyncio.run(main())
