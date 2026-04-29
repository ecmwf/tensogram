# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""6-cell matrix for `verify_hash` on the Python decode surface.

Mirrors `rust/tensogram/tests/decode_verify_hash.rs` for the
binding-level tests (`tensogram.decode`, `tensogram.decode_object`,
`tensogram.iter_messages`, `TensogramFile.decode_message`,
`TensogramFile.file_decode_object`).  See
`PLAN_DECODE_HASH_VERIFICATION.md` §5.2 for the matrix definition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensogram

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "rust/tensogram/tests/golden"


def _read(name: str) -> bytes:
    return (GOLDEN_DIR / name).read_bytes()


def _build_unhashed_message() -> bytes:
    """Single-object message with `hashing=false` (HASH_PRESENT flag clear).

    The committed goldens are all `hashing=true`; we build the
    hashless fixture in-process so cell C is testable without a
    dedicated on-disk file.
    """
    meta = {"version": 3}
    desc = {
        "type": "ntensor",
        "ndim": 1,
        "shape": [4],
        "strides": [1],
        "dtype": "float32",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return bytes(tensogram.encode(meta, [(desc, data)], hash=None))


# ── Cells A & B — verify on/off, hashed message, all entry points ────


class TestCellsAAndB:
    def test_decode_no_verify_succeeds(self):
        data = _read("hash_xxh3.tgm")
        _meta, objects = tensogram.decode(data)
        assert len(objects) == 1

    def test_decode_with_verify_succeeds(self):
        data = _read("hash_xxh3.tgm")
        _meta, objects = tensogram.decode(data, verify_hash=True)
        assert len(objects) == 1

    def test_decode_object_with_verify_succeeds(self):
        data = _read("hash_xxh3.tgm")
        _, _, arr = tensogram.decode_object(data, 0, verify_hash=True)
        assert arr.shape == (4,)

    def test_iter_messages_with_verify_succeeds(self):
        data = _read("hash_xxh3.tgm")
        items = list(tensogram.iter_messages(data, verify_hash=True))
        assert len(items) == 1


# ── Cell C — unhashed message + verify=True ──────────────────────────


class TestCellC:
    def test_decode_returns_missing_hash_on_unhashed(self):
        msg = _build_unhashed_message()
        with pytest.raises(tensogram.MissingHashError) as excinfo:
            _ = tensogram.decode(msg, verify_hash=True)
        assert excinfo.value.object_index == 0
        assert isinstance(excinfo.value, tensogram.IntegrityError)

    def test_decode_object_returns_missing_hash_on_unhashed(self):
        msg = _build_unhashed_message()
        with pytest.raises(tensogram.MissingHashError) as excinfo:
            _ = tensogram.decode_object(msg, 0, verify_hash=True)
        assert excinfo.value.object_index == 0

    def test_no_verify_silently_decodes_unhashed(self):
        msg = _build_unhashed_message()
        _meta, objects = tensogram.decode(msg)
        assert len(objects) == 1


# ── Cell D — tampered hash slot, single-object golden ────────────────


def _locate_first_object_frame(buf: bytes) -> tuple[int, int]:
    """Return (frame_start, frame_total_length) for the first
    NTensorFrame in a single-message buffer.  Mirrors the helper in
    the Rust matrix tests.
    """
    pos = 24  # past the 24-byte preamble
    while pos + 16 <= len(buf):
        if buf[pos : pos + 2] != b"FR":
            pos += 1
            continue
        # Frame header: magic(2) + frame_type u16 BE + version u16 BE
        # + flags u16 BE + total_length u64 BE.
        frame_type = int.from_bytes(buf[pos + 2 : pos + 4], "big")
        total_length = int.from_bytes(buf[pos + 8 : pos + 16], "big")
        if frame_type == 9:  # NTensorFrame
            return pos, total_length
        # 8-byte alignment between frames.
        pos = (pos + total_length + 7) & ~7
    raise AssertionError("no NTensorFrame found")


class TestCellD:
    def test_decode_reports_hash_mismatch_on_tampered_slot(self):
        data = bytearray(_read("hash_xxh3.tgm"))
        frame_start, frame_total_length = _locate_first_object_frame(bytes(data))
        # Inline hash slot lives at frame_end - 12.
        slot_byte = frame_start + frame_total_length - 12
        data[slot_byte] ^= 0xFF
        with pytest.raises(tensogram.HashMismatchError) as excinfo:
            _ = tensogram.decode(bytes(data), verify_hash=True)
        assert excinfo.value.object_index == 0
        assert excinfo.value.expected != excinfo.value.actual


# ── Cell E — tampered payload byte (slot intact, body modified) ───────


class TestCellE:
    """Body-tamper path is symmetric to Cell D's slot-tamper but
    exercises the *recomputed* side of the hash equation: the
    stored slot is left alone and a single byte of the encoded
    tensor payload is flipped, so check_frame_hash sees stored ≠
    recomputed.  Mirrors `cell_e_*` in the Rust matrix tests."""

    def test_decode_reports_hash_mismatch_on_tampered_payload(self):
        data = bytearray(_read("hash_xxh3.tgm"))
        frame_start, _ = _locate_first_object_frame(bytes(data))
        # First byte of the hashed body region — guaranteed to
        # land in the encoded tensor payload (cbor-after-payload
        # is the buffered-encoder default for `simple_f32_hashed`-
        # style fixtures), not in the CBOR descriptor.
        payload_byte = frame_start + 16  # skip frame header
        data[payload_byte] ^= 0xFF
        with pytest.raises(tensogram.HashMismatchError) as excinfo:
            _ = tensogram.decode(bytes(data), verify_hash=True)
        assert excinfo.value.object_index == 0
        assert excinfo.value.expected != excinfo.value.actual


# ── Cell F — multi-object: tamper object 1, expect index 1 ───────────


class TestCellF:
    def _multi_obj(self) -> bytearray:
        return bytearray(_read("multi_object_xxh3.tgm"))

    def test_decode_reports_correct_object_index(self):
        data = self._multi_obj()
        # Walk to the SECOND data-object frame and tamper its slot.
        pos = 24
        objects_seen = 0
        target_slot_byte = None
        while pos + 16 <= len(data):
            if data[pos : pos + 2] != b"FR":
                pos += 1
                continue
            frame_type = int.from_bytes(data[pos + 2 : pos + 4], "big")
            total_length = int.from_bytes(data[pos + 8 : pos + 16], "big")
            if frame_type == 9:
                if objects_seen == 1:
                    target_slot_byte = pos + total_length - 12
                    break
                objects_seen += 1
            pos = (pos + total_length + 7) & ~7
        assert target_slot_byte is not None
        data[target_slot_byte] ^= 0xFF

        with pytest.raises(tensogram.HashMismatchError) as excinfo:
            _ = tensogram.decode(bytes(data), verify_hash=True)
        assert excinfo.value.object_index == 1, (
            "must surface the *tampered* object's index, not 0"
        )

    def test_decode_object_targets_specific_object(self):
        data = bytes(self._multi_obj())
        # Tamper object 1 in the same way.
        mutable = bytearray(data)
        pos = 24
        objects_seen = 0
        target = None
        while pos + 16 <= len(mutable):
            if mutable[pos : pos + 2] != b"FR":
                pos += 1
                continue
            frame_type = int.from_bytes(mutable[pos + 2 : pos + 4], "big")
            total_length = int.from_bytes(mutable[pos + 8 : pos + 16], "big")
            if frame_type == 9:
                if objects_seen == 1:
                    target = pos + total_length - 12
                    break
                objects_seen += 1
            pos = (pos + total_length + 7) & ~7
        assert target is not None
        mutable[target] ^= 0xFF
        tampered = bytes(mutable)

        # Object 0 still verifies cleanly.
        tensogram.decode_object(tampered, 0, verify_hash=True)
        # Object 2 still verifies cleanly.
        tensogram.decode_object(tampered, 2, verify_hash=True)
        # Object 1 raises.
        with pytest.raises(tensogram.HashMismatchError) as excinfo:
            tensogram.decode_object(tampered, 1, verify_hash=True)
        assert excinfo.value.object_index == 1


# ── decode_range deliberately ignores verify_hash ────────────────────


class TestDecodeRangeUnverified:
    def test_decode_range_does_not_accept_verify_hash_kwarg(self):
        """Negative test: ``decode_range`` is *unverified by
        construction* (PLAN §6) and does not accept the kwarg in any
        binding.  This is what makes the asymmetry from
        `decode_object` visible at the API surface."""
        msg = _build_unhashed_message()
        with pytest.raises(TypeError, match="verify_hash"):
            _ = tensogram.decode_range(msg, 0, [(0, 4)], verify_hash=True)

    def test_file_decode_range_does_not_accept_verify_hash_kwarg(self, tmp_path):
        """Same negative test for the file-handle variant."""
        path = tmp_path / "x.tgm"
        path.write_bytes(_build_unhashed_message())
        f = tensogram.TensogramFile.open(str(path))
        with pytest.raises(TypeError, match="verify_hash"):
            _ = f.file_decode_range(0, obj_index=0, ranges=[(0, 4)], verify_hash=True)
