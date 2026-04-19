# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the ``allow_nan`` / ``allow_inf`` Python bindings.

Covers the NaN / Inf bitmask companion frame (type 9,
``NTensorMaskedFrame``) introduced in 0.17 (see
``plans/BITMASK_FRAME.md``).  These tests exercise:

- default-reject policy (``allow_nan`` / ``allow_inf`` both False)
- masked-encode round-trip (NaN restored on decode)
- ``restore_non_finite=False`` returns 0-substituted bytes
- per-kind ``*_mask_method`` selection (all six methods)
- small-mask auto-fallback
- ``StreamingEncoder`` + ``TensogramFile.append`` parity
"""

from __future__ import annotations

import numpy as np
import pytest
import tensogram


def _meta(version: int = 2) -> dict:
    return {"version": version}


def _desc(shape: list[int], dtype: str = "float64") -> dict:
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": dtype,
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


# ── Default reject ──────────────────────────────────────────────────────────


def test_default_encode_rejects_nan_f64() -> None:
    data = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    with pytest.raises(Exception, match="NaN"):
        tensogram.encode(_meta(), [(_desc([3]), data)])


def test_default_encode_rejects_positive_inf_f32() -> None:
    data = np.array([1.0, np.inf], dtype=np.float32)
    with pytest.raises(Exception, match=r"(\+Inf|Inf)"):
        tensogram.encode(_meta(), [(_desc([2], "float32"), data)])


def test_allow_inf_only_still_rejects_nan() -> None:
    data = np.array([np.nan], dtype=np.float64)
    with pytest.raises(Exception, match="NaN"):
        tensogram.encode(_meta(), [(_desc([1]), data)], allow_inf=True)


# ── allow_nan round-trip ────────────────────────────────────────────────────


def test_allow_nan_round_trip_f64_restores_nan_at_masked_positions() -> None:
    data = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
    msg = tensogram.encode(_meta(), [(_desc([5]), data)], allow_nan=True)
    decoded = tensogram.decode(msg)
    out = decoded.objects[0][1]
    assert out[0] == 1.0
    assert np.isnan(out[1])
    assert out[2] == 3.0
    assert np.isnan(out[3])
    assert out[4] == 5.0


def test_restore_non_finite_false_returns_substituted_zero() -> None:
    data = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    msg = tensogram.encode(_meta(), [(_desc([3]), data)], allow_nan=True)
    decoded = tensogram.decode(msg, restore_non_finite=False)
    out = decoded.objects[0][1]
    # NaN position now holds 0.0 (the substituted bit pattern).
    assert out[0] == 1.0
    assert out[1] == 0.0
    assert out[2] == 3.0


def test_allow_nan_and_inf_f32_all_kinds_round_trip() -> None:
    data = np.array(
        [1.0, np.nan, np.inf, -np.inf, 5.0],
        dtype=np.float32,
    )
    msg = tensogram.encode(
        _meta(),
        [(_desc([5], "float32"), data)],
        allow_nan=True,
        allow_inf=True,
        small_mask_threshold_bytes=0,  # force requested methods
    )
    decoded = tensogram.decode(msg)
    out = decoded.objects[0][1]
    assert out[0] == 1.0
    assert np.isnan(out[1])
    assert np.isposinf(out[2])
    assert np.isneginf(out[3])
    assert out[4] == 5.0


# ── All mask methods round-trip ────────────────────────────────────────────


@pytest.mark.parametrize("method", ["none", "rle", "roaring", "lz4", "zstd"])
def test_mask_method_round_trip(method: str) -> None:
    # 128-element payload with 3 NaNs — large enough that the default
    # small-mask threshold (128 bytes) does not force an auto-fallback
    # to "none" except for the explicit "none" case.
    data = np.arange(128, dtype=np.float64)
    data[10] = np.nan
    data[50] = np.nan
    data[100] = np.nan
    msg = tensogram.encode(
        _meta(),
        [(_desc([128]), data)],
        allow_nan=True,
        nan_mask_method=method,
        small_mask_threshold_bytes=0,
    )
    decoded = tensogram.decode(msg)
    out = decoded.objects[0][1]
    nan_positions = np.nonzero(np.isnan(out))[0]
    assert sorted(nan_positions.tolist()) == [10, 50, 100]


def test_unknown_mask_method_errors_cleanly() -> None:
    data = np.array([np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="unknown mask method"):
        tensogram.encode(
            _meta(),
            [(_desc([1]), data)],
            allow_nan=True,
            nan_mask_method="bogus",
        )


def test_unknown_mask_method_error_lists_all_accepted_names() -> None:
    """Regression: the error message enumerates every accepted mask
    method name.  Keeps cross-language parity with the Rust / CLI /
    TS / WASM / FFI frontends (single source of truth in
    ``MaskError::UnknownMethod``).
    """
    data = np.array([np.nan], dtype=np.float64)
    with pytest.raises(ValueError) as exc_info:
        tensogram.encode(
            _meta(),
            [(_desc([1]), data)],
            allow_nan=True,
            nan_mask_method="znorfle",
        )
    message = str(exc_info.value)
    for name in ("none", "rle", "roaring", "lz4", "zstd", "blosc2"):
        assert f'"{name}"' in message, f"expected quoted {name!r} in error: {message!r}"


# ── Small-mask auto-fallback ───────────────────────────────────────────────


def test_small_mask_auto_fallback_still_round_trips() -> None:
    # A 4-element mask packs to 1 byte — way under 128.  The encoder
    # emits method="none" regardless of the requested method.
    data = np.array([np.nan, 1.0, np.nan, 2.0], dtype=np.float64)
    msg = tensogram.encode(
        _meta(),
        [(_desc([4]), data)],
        allow_nan=True,
        nan_mask_method="roaring",
        # leave small_mask_threshold_bytes at the default (128)
    )
    decoded = tensogram.decode(msg)
    out = decoded.objects[0][1]
    assert np.isnan(out[0])
    assert out[1] == 1.0
    assert np.isnan(out[2])
    assert out[3] == 2.0


# ── StreamingEncoder parity ────────────────────────────────────────────────


def test_streaming_encoder_allow_nan_round_trip() -> None:
    data = np.array([1.0, np.nan, 2.0], dtype=np.float64)
    enc = tensogram.StreamingEncoder(
        _meta(),
        allow_nan=True,
        small_mask_threshold_bytes=0,
    )
    enc.write_object(_desc([3]), data)
    msg = bytes(enc.finish())
    decoded = tensogram.decode(msg)
    out = decoded.objects[0][1]
    assert out[0] == 1.0
    assert np.isnan(out[1])
    assert out[2] == 2.0


# ── TensogramFile.append ───────────────────────────────────────────────────


def test_tensogram_file_append_allow_nan(tmp_path) -> None:
    path = str(tmp_path / "with_nan.tgm")
    data = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    with tensogram.TensogramFile.create(path) as f:
        f.append(
            _meta(),
            [(_desc([3]), data)],
            allow_nan=True,
            small_mask_threshold_bytes=0,
        )
    with tensogram.TensogramFile.open(path) as f:
        msg = f.decode_message(0)
        out = msg.objects[0][1]
        assert out[0] == 1.0
        assert np.isnan(out[1])
        assert out[2] == 3.0


# ── decode_range + decode_object honour restore_non_finite ─────────────────


def test_decode_range_restores_nan_inside_range() -> None:
    data = np.array(
        [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
        dtype=np.float64,
    )
    msg = tensogram.encode(
        _meta(),
        [(_desc([8]), data)],
        allow_nan=True,
        small_mask_threshold_bytes=0,
    )
    # Range covers elements 1..5 → [NaN, 3.0, NaN, 5.0].
    parts = tensogram.decode_range(msg, 0, [(1, 4)])
    got = parts[0]
    assert np.isnan(got[0])
    assert got[1] == 3.0
    assert np.isnan(got[2])
    assert got[3] == 5.0


def test_decode_object_restores_nan() -> None:
    data = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    msg = tensogram.encode(
        _meta(),
        [(_desc([3]), data)],
        allow_nan=True,
        small_mask_threshold_bytes=0,
    )
    _, _desc_out, obj = tensogram.decode_object(msg, 0)
    assert obj[0] == 1.0
    assert np.isnan(obj[1])
    assert obj[2] == 3.0
