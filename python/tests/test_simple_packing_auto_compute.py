# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Python-side BDD scenarios for the simple_packing auto-compute path.

Mirrors ``rust/tensogram/tests/simple_packing_auto_compute.rs`` so the
Python binding doesn't silently regress.  The user story:

> Write a descriptor with just ``encoding="simple_packing"`` and
> ``sp_bits_per_value=16`` and have the encoder auto-compute
> ``sp_reference_value`` / ``sp_binary_scale_factor``.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensogram


def _auto_desc(shape: list[int], bits: int = 16) -> dict:
    """A minimal simple_packing descriptor relying on auto-compute."""
    return {
        "type": "ntensor",
        "dtype": "float64",
        "ndim": len(shape),
        "shape": list(shape),
        "encoding": "simple_packing",
        "filter": "none",
        "compression": "none",
        "sp_bits_per_value": bits,
    }


class TestAutoComputeHappyPath:
    def test_s1_bits_per_value_only_descriptor_roundtrips(self) -> None:
        """The ergonomic form — only ``sp_bits_per_value`` on the user side."""
        data = np.linspace(270.0, 285.0, 64, dtype=np.float64)
        desc = _auto_desc([data.size])
        buf = tensogram.encode({}, [(desc, data)])

        _meta, objects = tensogram.decode(buf)
        assert len(objects) == 1
        returned_desc, decoded = objects[0]

        # All 4 sp_* keys must be populated on the wire after encode.
        params = returned_desc.params
        assert params["sp_bits_per_value"] == 16
        assert params["sp_decimal_scale_factor"] == 0
        assert "sp_reference_value" in params
        assert "sp_binary_scale_factor" in params

        # Values round-trip within 16-bit quantization tolerance.
        tol = float(data.max() - data.min()) / (1 << 16)
        np.testing.assert_allclose(decoded.reshape(data.shape), data, atol=tol)

    def test_parity_with_explicit_compute_packing_params(self) -> None:
        """Auto-compute must produce identical decoded values to the
        explicit ``compute_packing_params`` path.
        """
        data = np.array(
            [100.0, 105.5, 110.25, 115.125, 120.0625, 125.0, 130.0],
            dtype=np.float64,
        )

        # Explicit path — user calls compute_packing_params then spreads
        # the returned sp_*-keyed dict into the descriptor.
        sp_params = tensogram.compute_packing_params(
            data.ravel(), bits_per_value=16, decimal_scale_factor=0
        )
        explicit_desc = _auto_desc([data.size]) | sp_params

        # Auto path — user supplies only sp_bits_per_value.
        auto_desc = _auto_desc([data.size])

        exp_bytes = tensogram.encode({}, [(explicit_desc, data)])
        auto_bytes = tensogram.encode({}, [(auto_desc, data)])

        _, exp_obj = tensogram.decode(exp_bytes)
        _, auto_obj = tensogram.decode(auto_bytes)

        # Decoded descriptor params match
        for key in (
            "sp_reference_value",
            "sp_binary_scale_factor",
            "sp_decimal_scale_factor",
            "sp_bits_per_value",
        ):
            assert exp_obj[0][0].params[key] == auto_obj[0][0].params[key]

        # Decoded payloads match bit-for-bit
        np.testing.assert_array_equal(exp_obj[0][1], auto_obj[0][1])


class TestAutoComputeExplicitWins:
    def test_s2_explicit_ref_and_bsf_used_verbatim(self) -> None:
        """User-provided ``sp_reference_value`` + ``sp_binary_scale_factor``
        are NOT recomputed (Q2 option b)."""
        data = np.array([270.0, 275.0, 280.0, 285.0], dtype=np.float64)
        desc = _auto_desc([4]) | {
            "sp_reference_value": 200.0,
            "sp_binary_scale_factor": 5,
        }
        buf = tensogram.encode({}, [(desc, data)])
        _, objects = tensogram.decode(buf)
        params = objects[0][0].params
        assert params["sp_reference_value"] == 200.0
        assert params["sp_binary_scale_factor"] == 5


class TestAutoComputeErrors:
    def test_s3_missing_bits_per_value_is_a_clear_error(self) -> None:
        desc = _auto_desc([4])
        del desc["sp_bits_per_value"]
        data = np.array([270.0, 275.0, 280.0, 285.0], dtype=np.float64)
        with pytest.raises(Exception, match="sp_bits_per_value"):
            tensogram.encode({}, [(desc, data)])

    def test_s4_nan_in_data_is_rejected(self) -> None:
        desc = _auto_desc([4])
        data = np.array([270.0, float("nan"), 280.0, 285.0], dtype=np.float64)
        with pytest.raises(Exception, match=r"[Nn][Aa][Nn]"):
            tensogram.encode({}, [(desc, data)])

    def test_s7_non_float64_dtype_rejected(self) -> None:
        desc = _auto_desc([4])
        desc["dtype"] = "float32"
        data = np.array([270.0, 275.0, 280.0, 285.0], dtype=np.float32)
        with pytest.raises(Exception, match="simple_packing only supports float64"):
            tensogram.encode({}, [(desc, data)])

    def test_e1_only_sp_reference_value_rejected(self) -> None:
        """Half-explicit params (only ref, not bsf) are ambiguous."""
        desc = _auto_desc([4])
        desc["sp_reference_value"] = 200.0
        data = np.array([270.0, 275.0, 280.0, 285.0], dtype=np.float64)
        with pytest.raises(Exception, match=r"sp_reference_value.*sp_binary_scale_factor"):
            tensogram.encode({}, [(desc, data)])

    def test_e1_only_sp_binary_scale_factor_rejected(self) -> None:
        """Half-explicit params (only bsf, not ref) are ambiguous."""
        desc = _auto_desc([4])
        desc["sp_binary_scale_factor"] = 5
        data = np.array([270.0, 275.0, 280.0, 285.0], dtype=np.float64)
        with pytest.raises(Exception, match=r"sp_binary_scale_factor.*sp_reference_value"):
            tensogram.encode({}, [(desc, data)])


class TestAutoComputeDefaults:
    def test_s6_decimal_scale_factor_defaults_to_zero(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        desc = _auto_desc([4])
        buf = tensogram.encode({}, [(desc, data)])
        _, objects = tensogram.decode(buf)
        assert objects[0][0].params["sp_decimal_scale_factor"] == 0


class TestComputePackingParamsReturnedKeys:
    """The helper function returns ``sp_*``-prefixed keys so the user
    can spread the result directly into a descriptor."""

    def test_returned_dict_uses_sp_prefix(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        params = tensogram.compute_packing_params(
            values, bits_per_value=16, decimal_scale_factor=0
        )
        assert set(params.keys()) == {
            "sp_reference_value",
            "sp_binary_scale_factor",
            "sp_decimal_scale_factor",
            "sp_bits_per_value",
        }
        assert params["sp_bits_per_value"] == 16
        assert params["sp_decimal_scale_factor"] == 0


class TestAutoComputeStreamingEncoder:
    def test_streaming_encoder_auto_computes(self, tmp_path) -> None:
        """``StreamingEncoder.write_object`` must take the same
        auto-compute shortcut as buffered ``encode``."""
        data = np.linspace(270.0, 285.0, 32, dtype=np.float64)
        desc = _auto_desc([data.size])

        path = tmp_path / "streamed.tgm"
        enc = tensogram.StreamingEncoder({})
        enc.write_object(desc, data)
        buf = enc.finish()
        path.write_bytes(buf)

        with tensogram.TensogramFile.open(str(path)) as f:
            assert len(f) == 1
            _, objects = tensogram.decode(f.read_message(0))
            params = objects[0][0].params
            assert "sp_reference_value" in params
            assert "sp_binary_scale_factor" in params
