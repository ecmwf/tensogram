# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Python parity tests for ``EncodeOptions::reject_nan`` / ``reject_inf``.

The strict-finite checks run upstream of the encoding pipeline and
surface as ``ValueError`` with ``"strict-NaN"`` / ``"strict-Inf"`` in
the message. These tests pin the cross-language contract documented
in ``plans/RESEARCH_NAN_HANDLING.md`` §4.1.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensogram


def _meta():
    return {"version": 2}


def _desc(shape, dtype, encoding="none", compression="none"):
    return {
        "type": "ntensor",
        "shape": list(shape),
        "dtype": dtype,
        "byte_order": "little",
        "encoding": encoding,
        "filter": "none",
        "compression": compression,
    }


# ── Defaults preserve current behaviour ───────────────────────────────────────


def test_default_encode_accepts_nan_float32():
    data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    msg = tensogram.encode(_meta(), [(_desc([3], "float32"), data)])
    # Should round-trip bit-exactly
    _, objs = tensogram.decode(bytes(msg))
    out = np.frombuffer(objs[0][1], dtype=np.float32)
    assert np.isnan(out[1])


def test_default_encode_accepts_inf_float64():
    data = np.array([1.0, np.inf, -np.inf, 4.0], dtype=np.float64)
    tensogram.encode(_meta(), [(_desc([4], "float64"), data)])


# ── reject_nan rejects across float dtypes ────────────────────────────────────


def test_reject_nan_rejects_float32():
    data = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    with pytest.raises(ValueError, match=r"(?i)nan"):
        tensogram.encode(_meta(), [(_desc([4], "float32"), data)], reject_nan=True)


def test_reject_nan_rejects_float64():
    data = np.array([1.0, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match=r"(?i)nan"):
        tensogram.encode(_meta(), [(_desc([2], "float64"), data)], reject_nan=True)


def test_reject_nan_rejects_complex64_real_component():
    # complex64 = (f32 real, f32 imag). Put NaN in real of element 1.
    interleaved = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError, match=r"(?i)nan.*real"):
        tensogram.encode(
            _meta(),
            [(_desc([3], "complex64"), interleaved)],
            reject_nan=True,
        )


def test_reject_nan_rejects_complex64_imag_component():
    # Imag of element 2 is NaN
    interleaved = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], dtype=np.float32)
    with pytest.raises(ValueError, match=r"(?i)(nan.*imag|imag.*nan)"):
        tensogram.encode(
            _meta(),
            [(_desc([3], "complex64"), interleaved)],
            reject_nan=True,
        )


# ── reject_inf rejects +Inf and -Inf ─────────────────────────────────────────


def test_reject_inf_rejects_positive_inf():
    data = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match=r"(?i)inf"):
        tensogram.encode(_meta(), [(_desc([3], "float32"), data)], reject_inf=True)


def test_reject_inf_rejects_negative_inf():
    data = np.array([1.0, 2.0, -np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match=r"(?i)inf"):
        tensogram.encode(_meta(), [(_desc([3], "float64"), data)], reject_inf=True)


# ── Orthogonality ────────────────────────────────────────────────────────────


def test_reject_inf_does_not_reject_nan():
    data = np.array([1.0, np.nan], dtype=np.float32)
    tensogram.encode(_meta(), [(_desc([2], "float32"), data)], reject_inf=True)


def test_reject_nan_does_not_reject_inf():
    data = np.array([1.0, np.inf], dtype=np.float32)
    tensogram.encode(_meta(), [(_desc([2], "float32"), data)], reject_nan=True)


def test_reject_both_catches_either():
    nan_data = np.array([1.0, np.nan], dtype=np.float32)
    inf_data = np.array([1.0, np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match=r"(?i)nan"):
        tensogram.encode(
            _meta(),
            [(_desc([2], "float32"), nan_data)],
            reject_nan=True,
            reject_inf=True,
        )
    with pytest.raises(ValueError, match=r"(?i)inf"):
        tensogram.encode(
            _meta(),
            [(_desc([2], "float32"), inf_data)],
            reject_nan=True,
            reject_inf=True,
        )


# ── Integer dtypes skip the scan ─────────────────────────────────────────────


def test_reject_nan_skips_uint32_payload():
    # 0xFFFFFFFF would be NaN if interpreted as f32 bits.
    data = np.full(8, 0xFFFFFFFF, dtype=np.uint32)
    tensogram.encode(
        _meta(),
        [(_desc([8], "uint32"), data)],
        reject_nan=True,
        reject_inf=True,
    )


# ── Interaction with simple_packing: mitigates the §3.1 gotcha ──────────────


def test_reject_inf_blocks_simple_packing_silent_corruption():
    # Without the flag, this data through simple_packing silently
    # decodes to NaN everywhere (binary_scale_factor overflows to
    # i32::MAX). With reject_inf=True we catch it at encode time.
    data = np.array([1.0, np.inf, 3.0], dtype=np.float64)
    desc = _desc([3], "float64", encoding="simple_packing")
    # simple_packing requires its params on the descriptor before
    # encode() can proceed — we fill them out so the pipeline would
    # otherwise run. The strict scan fires upstream.
    desc["params"] = {
        "bits_per_value": 16,
        "reference_value": 1.0,
        "binary_scale_factor": 0,
        "decimal_scale_factor": 0,
    }
    with pytest.raises(ValueError, match=r"(?i)inf"):
        tensogram.encode(_meta(), [(desc, data)], reject_inf=True)


# ── Interaction with compression ─────────────────────────────────────────────


def test_reject_nan_fires_before_lz4():
    data = np.array([1.0, np.nan], dtype=np.float64)
    desc = _desc([2], "float64", compression="lz4")
    with pytest.raises(ValueError, match=r"(?i)nan"):
        tensogram.encode(_meta(), [(desc, data)], reject_nan=True)


# ── encode_pre_encoded does NOT expose strict flags ──────────────────────────


def test_encode_pre_encoded_has_no_strict_flags():
    """``encode_pre_encoded`` intentionally does not accept
    ``reject_nan`` / ``reject_inf`` kwargs.  Passing them raises
    ``TypeError`` at the Python binding layer, before reaching Rust.

    At the Rust layer, the buffered and streaming pre-encoded APIs
    also error if these flags are set on ``EncodeOptions`` — see
    ``rust/tensogram/tests/strict_finite.rs::encode_pre_encoded_errors_when_reject_nan_is_set``
    for the parity contract.  Python users never hit that path because
    the kwargs cannot be passed through this binding.
    """
    data = np.array([1.0, np.nan], dtype=np.float32).tobytes()
    desc = _desc([2], "float32")
    with pytest.raises(TypeError):
        tensogram.encode_pre_encoded(_meta(), [(desc, data)], reject_nan=True)


# ── StreamingEncoder honours the flags ───────────────────────────────────────


def test_streaming_encoder_rejects_nan_with_flag():
    enc = tensogram.StreamingEncoder(_meta(), reject_nan=True)
    data = np.array([1.0, np.nan], dtype=np.float32)
    with pytest.raises(ValueError, match=r"(?i)nan"):
        enc.write_object(_desc([2], "float32"), data)


def test_streaming_encoder_rejects_inf_with_flag():
    enc = tensogram.StreamingEncoder(_meta(), reject_inf=True)
    data = np.array([1.0, np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match=r"(?i)inf"):
        enc.write_object(_desc([2], "float64"), data)


def test_streaming_encoder_default_accepts_nan():
    enc = tensogram.StreamingEncoder(_meta())
    data = np.array([1.0, np.nan], dtype=np.float32)
    enc.write_object(_desc([2], "float32"), data)
    result = enc.finish()
    assert len(result) > 0


# ── TensogramFile.append honours the flags ───────────────────────────────────


def test_tensogram_file_append_rejects_nan(tmp_path):
    path = str(tmp_path / "strict.tgm")
    data = np.array([1.0, np.nan], dtype=np.float32)
    desc = _desc([2], "float32")
    with (
        tensogram.TensogramFile.create(path) as f,
        pytest.raises(ValueError, match=r"(?i)nan"),
    ):
        f.append(_meta(), [(desc, data)], reject_nan=True)


def test_tensogram_file_append_default_accepts_nan(tmp_path):
    path = str(tmp_path / "strict_default.tgm")
    data = np.array([1.0, np.nan], dtype=np.float32)
    with tensogram.TensogramFile.create(path) as f:
        f.append(_meta(), [(_desc([2], "float32"), data)])
    # File is valid
    with tensogram.TensogramFile.open(path) as f:
        assert f.message_count() == 1


# ── Parallel path ────────────────────────────────────────────────────────────


def test_reject_nan_parallel_large_input():
    values = np.ones(16_384, dtype=np.float64)
    values[10_000] = np.nan
    with pytest.raises(ValueError, match=r"(?i)nan"):
        tensogram.encode(
            _meta(),
            [(_desc([16_384], "float64"), values)],
            reject_nan=True,
            threads=4,
        )


# ── Edge cases ──────────────────────────────────────────────────────────────


def test_reject_nan_negative_zero_passes():
    data = np.array([1.0, -0.0, 2.0], dtype=np.float64)
    tensogram.encode(
        _meta(),
        [(_desc([3], "float64"), data)],
        reject_nan=True,
        reject_inf=True,
    )


def test_reject_nan_empty_array_passes():
    data = np.zeros(0, dtype=np.float32)
    desc = _desc([0], "float32")
    tensogram.encode(_meta(), [(desc, data)], reject_nan=True, reject_inf=True)


def test_error_message_mentions_dtype_and_index():
    data = np.array([1.0, 2.0, 3.0, np.nan, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match=r"NaN.*element 3.*float64") as excinfo:
        tensogram.encode(
            _meta(), [(_desc([5], "float64"), data)], reject_nan=True
        )
    msg = str(excinfo.value)
    assert "NaN" in msg
    assert "element 3" in msg
    assert "float64" in msg


# ── Standalone-API safety net (plans/RESEARCH_NAN_HANDLING.md §4.2.3) ────────
#
# simple_packing::encode_with_threads now validates SimplePackingParams
# against silent-corruption-producing values.  Here we exercise the
# validation through the high-level `tensogram.encode` path — the
# caller supplies a descriptor with a degenerate `binary_scale_factor`,
# and the error must surface as ValueError.


def _simple_packing_desc(
    ref_value: float = 0.0,
    binary_scale_factor: int = 0,
    bits_per_value: int = 16,
):
    # The PyO3 binding folds every non-reserved descriptor key into
    # `params`, so simple_packing's reference_value / binary_scale_factor
    # / etc. go at the top level of the dict, NOT nested under "params".
    return {
        "type": "ntensor",
        "shape": [4],
        "dtype": "float64",
        "byte_order": "little",
        "encoding": "simple_packing",
        "filter": "none",
        "compression": "none",
        "reference_value": ref_value,
        "binary_scale_factor": binary_scale_factor,
        "decimal_scale_factor": 0,
        "bits_per_value": bits_per_value,
    }


def test_standalone_safety_net_rejects_huge_binary_scale_factor():
    """Caller supplies ``binary_scale_factor=i32::MAX`` — the
    fingerprint of feeding Inf through compute_params's range
    arithmetic.  The safety net catches it at the Rust core."""
    desc = _simple_packing_desc(binary_scale_factor=2**31 - 1)
    data = np.array([273.15, 283.0, 293.0, 303.0], dtype=np.float64)
    with pytest.raises(ValueError, match=r"binary_scale_factor"):
        tensogram.encode(_meta(), [(desc, data)])


def test_standalone_safety_net_threshold_is_256():
    """Pinning the 256-step threshold is inclusive: 256 passes, 257 fails."""
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    # At threshold → accepted
    desc_ok = _simple_packing_desc(binary_scale_factor=256)
    tensogram.encode(_meta(), [(desc_ok, data)])
    # Above threshold → rejected
    desc_fail = _simple_packing_desc(binary_scale_factor=257)
    with pytest.raises(ValueError, match=r"binary_scale_factor"):
        tensogram.encode(_meta(), [(desc_fail, data)])


def test_standalone_safety_net_accepts_realistic_binary_scale_factors():
    """Regression guard: real-world weather values (|bsf| ≤ 60) pass."""
    data = np.array([273.15, 283.0, 293.0, 303.0], dtype=np.float64)
    for bsf in (-60, -20, 0, 20, 60):
        desc = _simple_packing_desc(ref_value=273.15, binary_scale_factor=bsf)
        tensogram.encode(_meta(), [(desc, data)])


def test_standalone_safety_net_constant_field_still_works():
    """``bits_per_value=0`` is a legitimate constant-field encoding;
    the safety net must not reject it."""
    desc = _simple_packing_desc(ref_value=42.0, bits_per_value=0)
    data = np.array([42.0] * 4, dtype=np.float64)
    # Should succeed and the packed payload is empty (0 bits per value).
    tensogram.encode(_meta(), [(desc, data)])
