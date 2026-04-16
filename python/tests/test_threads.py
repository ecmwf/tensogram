# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the multi-threaded coding pipeline (v0.13.0).

These tests exercise the ``threads=N`` kwarg added to ``encode``,
``decode``, ``decode_object``, ``decode_range``, and the batch
variants.  The core invariants are:

1. Default ``threads=0`` preserves the pre-0.13.0 sequential behaviour
   (byte-identical output across repeated calls).
2. For *transparent* pipelines (encoding/filter = simple_packing/shuffle,
   or compression in {none, lz4, szip, zfp, sz3}) the **encoded payload
   bytes** must be byte-identical across thread counts.  The top-level
   ``_reserved_`` metadata (uuid, time) is intentionally different on
   each encode call and is deliberately excluded from the comparison.
3. For *opaque* pipelines (compression = blosc2 or zstd w/ workers) the
   encoded bytes may differ across thread counts, but the **decoded
   values** must round-trip losslessly.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensogram

THREAD_COUNTS = (0, 1, 2, 4, 8)


def _encoded_payloads(msg: bytes) -> list[bytes]:
    """Extract encoded payload bytes per object (ignoring top-level metadata).

    This is the structural equality used by the determinism contract.
    """
    # The Python bindings don't expose raw frame bytes, but descriptors
    # include the xxh3 hash computed over the payload.  Equal hashes
    # => equal payloads.  We compare hashes instead.
    #
    # Use decode_descriptors (metadata-only) so we don't need to compare
    # decoded ndarrays (which would also round-trip losslessly for
    # transparent pipelines, but hash equality is the stronger claim).
    _, descriptors = tensogram.decode_descriptors(msg)
    return [d.hash for d in descriptors]


def _make_transparent_msg(threads: int, n: int = 200_000) -> bytes:
    """Encode a pipeline with no codec — pure passthrough."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(n).astype(np.float64)
    meta = {"version": 2, "base": [{}]}
    desc = {"type": "ntensor", "shape": [n], "dtype": "float64"}
    return tensogram.encode(meta, [(desc, data)], threads=threads)


def _make_sp_msg(threads: int, bits: int, n: int = 200_000) -> bytes:
    rng = np.random.default_rng(42)
    values = 250.0 + rng.standard_normal(n) * 30.0
    params = tensogram.compute_packing_params(values.astype(np.float64).ravel(), bits, 0)
    meta = {"version": 2, "base": [{}]}
    desc = {
        "type": "ntensor",
        "shape": [n],
        "dtype": "float64",
        "encoding": "simple_packing",
        "compression": "szip",
        **params,
        "szip_rsi": 128,
        "szip_block_size": 16,
        "szip_flags": 8,
    }
    return tensogram.encode(meta, [(desc, values)], threads=threads)


def _make_blosc2_msg(threads: int, n: int = 100_000) -> bytes:
    rng = np.random.default_rng(42)
    data = rng.standard_normal(n).astype(np.float64)
    meta = {"version": 2, "base": [{}]}
    desc = {
        "type": "ntensor",
        "shape": [n],
        "dtype": "float64",
        "compression": "blosc2",
        "blosc2_clevel": 5,
        "blosc2_codec": "lz4",
    }
    return tensogram.encode(meta, [(desc, data)], threads=threads)


# ── Default behaviour ────────────────────────────────────────────────


class TestDefaultBehaviour:
    def test_encode_threads_kwarg_accepted(self):
        """The kwarg exists on encode/decode and accepts 0/1/N."""
        data = np.arange(100, dtype=np.float32)
        meta = {"version": 2, "base": [{}]}
        desc = {"type": "ntensor", "shape": [100], "dtype": "float32"}
        for t in THREAD_COUNTS:
            msg = tensogram.encode(meta, [(desc, data)], threads=t)
            assert isinstance(msg, bytes)
            decoded = tensogram.decode(msg, threads=t)
            assert decoded.objects[0][1].tolist() == data.tolist()

    def test_threads_zero_is_default(self):
        """threads=0 is the default and matches an explicit 0."""
        data = np.arange(100, dtype=np.float32)
        meta = {"version": 2, "base": [{}]}
        desc = {"type": "ntensor", "shape": [100], "dtype": "float32"}
        m1 = tensogram.encode(meta, [(desc, data)])
        m2 = tensogram.encode(meta, [(desc, data)], threads=0)
        # Both have fresh uuid/time, so compare encoded payloads.
        assert _encoded_payloads(m1) == _encoded_payloads(m2)


# ── Transparent byte-identity ────────────────────────────────────────


class TestTransparentByteIdentity:
    def test_no_codec_byte_identical(self):
        baseline = _encoded_payloads(_make_transparent_msg(threads=0))
        for t in THREAD_COUNTS:
            got = _encoded_payloads(_make_transparent_msg(threads=t))
            assert got == baseline, f"threads={t} payload hash mismatch"

    @pytest.mark.parametrize("bits", [16, 24, 32])
    def test_simple_packing_plus_szip_byte_identical(self, bits):
        baseline = _encoded_payloads(_make_sp_msg(threads=0, bits=bits))
        for t in THREAD_COUNTS:
            got = _encoded_payloads(_make_sp_msg(threads=t, bits=bits))
            assert got == baseline, f"sp+szip bits={bits} threads={t} mismatch"


# ── Opaque codec round-trip ──────────────────────────────────────────


class TestOpaqueRoundTrip:
    def test_blosc2_round_trip_lossless_across_threads(self):
        """blosc2 compressed bytes may differ; decoded values must not."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100_000).astype(np.float64)

        for t in THREAD_COUNTS:
            msg = _make_blosc2_msg(threads=t)
            decoded = tensogram.decode(msg)
            np.testing.assert_array_equal(decoded.objects[0][1], data)


# ── Axis-A order preservation ────────────────────────────────────────


class TestAxisAOrderPreservation:
    def test_multi_object_order_preserved(self):
        """16 objects, no codec → axis A.  Output order matches input."""
        meta = {"version": 2, "base": [{}] * 16}
        descs_and_data = []
        for i in range(16):
            arr = np.arange(50_000, dtype=np.uint32) + i * 1_000_000
            desc = {"type": "ntensor", "shape": [50_000], "dtype": "uint32"}
            descs_and_data.append((desc, arr))

        baseline = _encoded_payloads(tensogram.encode(meta, descs_and_data, threads=0))
        for t in THREAD_COUNTS:
            msg = tensogram.encode(meta, descs_and_data, threads=t)
            assert _encoded_payloads(msg) == baseline

            decoded = tensogram.decode(msg, threads=t)
            for i, (_d, arr) in enumerate(decoded.objects):
                expected = np.arange(50_000, dtype=np.uint32) + i * 1_000_000
                np.testing.assert_array_equal(arr, expected)


# ── Threshold ────────────────────────────────────────────────────────


class TestSmallMessageThreshold:
    def test_threads_ignored_below_threshold(self):
        """Tiny payloads never take the parallel path."""
        data = np.arange(64, dtype=np.float64)
        meta = {"version": 2, "base": [{}]}
        desc = {"type": "ntensor", "shape": [64], "dtype": "float64"}
        baseline = _encoded_payloads(tensogram.encode(meta, [(desc, data)], threads=0))
        for t in (1, 2, 4, 8):
            got = _encoded_payloads(tensogram.encode(meta, [(desc, data)], threads=t))
            assert got == baseline


# ── Decode thread budget ─────────────────────────────────────────────


class TestDecodeThreads:
    def test_decode_threads_byte_identical_transparent(self):
        """decode(threads=N) must return identical bytes to decode(threads=0)."""
        msg = _make_transparent_msg(threads=0)
        baseline = tensogram.decode(msg, threads=0)
        for t in THREAD_COUNTS:
            got = tensogram.decode(msg, threads=t)
            np.testing.assert_array_equal(got.objects[0][1], baseline.objects[0][1])

    def test_decode_object_threads(self):
        msg = _make_transparent_msg(threads=0)
        _, _desc, data = tensogram.decode_object(msg, 0, threads=8)
        assert data.shape == (200_000,)

    def test_decode_range_threads(self):
        msg = _make_transparent_msg(threads=0)
        parts = tensogram.decode_range(msg, 0, [(0, 100), (1000, 100)], threads=4)
        assert len(parts) == 2
        assert parts[0].shape == (100,)
        assert parts[1].shape == (100,)


# ── Async binding parity ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_decode_message_threads_matches_sync(tmp_path):
    """`AsyncTensogramFile.decode_message(threads=N)` parity with sync path.

    Keeps the async wrapper honest: the new `threads` kwarg on the async
    API must reach the core the same way the sync kwarg does.
    """
    path = str(tmp_path / "async_threads.tgm")
    msg = _make_transparent_msg(threads=0)
    with open(path, "wb") as fh:
        fh.write(msg)

    async with await tensogram.AsyncTensogramFile.open(path) as f:
        result_seq = await f.decode_message(0)
        result_par = await f.decode_message(0, threads=4)
        np.testing.assert_array_equal(result_seq.objects[0][1], result_par.objects[0][1])
