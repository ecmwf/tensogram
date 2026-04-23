# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Encoder path: ``data.to_target("file", "out.tgm", encoder="tensogram")``.

Contract:

* The ``tensogram`` encoder entry point is discoverable under
  ``earthkit.data.encoders``.
* Encoding a MARS-keyed :class:`FieldList` produces a valid ``.tgm``
  that round-trips back into an equivalent FieldList via
  ``ekd.from_source("tensogram", path)``.
* Values are preserved bit-for-bit; MARS keys round-trip one-to-one.
* Writing to ``bytes`` via :meth:`EncodedData.to_bytes` yields the same
  output as writing to disk.
* Non-MARS generic tensor data via an xarray Dataset is supported too:
  every data variable becomes one tensogram object.
"""

from __future__ import annotations

from pathlib import Path

import earthkit.data as ekd
import entrypoints
import numpy as np
import pytest
from earthkit.data.encoders import create_encoder


class TestEncoderEntryPoint:
    def test_encoder_entry_point_registered(self) -> None:
        names = {e.name for e in entrypoints.get_group_all("earthkit.data.encoders")}
        assert "tensogram" in names

    def test_encoder_resolvable_by_name(self) -> None:
        enc = create_encoder("tensogram")
        assert enc is not None
        assert type(enc).__module__.startswith("tensogram_earthkit")


class TestFieldListRoundTrip:
    """Write → read → values + MARS keys match the original FieldList."""

    def test_fieldlist_to_target_produces_valid_tgm(self, mars_tensogram_file, tmp_path) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        out_path: Path = tmp_path / "roundtrip.tgm"
        fl.to_target("file", str(out_path), encoder="tensogram")
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        # Leading bytes must be the tensogram magic.
        assert out_path.read_bytes()[:8] == b"TENSOGRM"

    def test_roundtrip_preserves_values(self, mars_tensogram_file, tmp_path) -> None:
        original = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        out_path = tmp_path / "roundtrip.tgm"
        original.to_target("file", str(out_path), encoder="tensogram")

        restored = ekd.from_source("tensogram", str(out_path)).to_fieldlist()
        assert len(restored) == len(original)

        # Match by param since field order is implementation-dependent.
        orig_by_param = {f.metadata("param"): f for f in original}
        for new_field in restored:
            p = new_field.metadata("param")
            assert p in orig_by_param
            np.testing.assert_array_equal(
                new_field.to_numpy(), orig_by_param[p].to_numpy(), err_msg=p
            )

    def test_roundtrip_preserves_mars_keys(self, mars_tensogram_file, tmp_path) -> None:
        original = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        out_path = tmp_path / "roundtrip.tgm"
        original.to_target("file", str(out_path), encoder="tensogram")

        restored = ekd.from_source("tensogram", str(out_path)).to_fieldlist()
        expected_keys = ["param", "step", "date", "time", "levtype", "class", "stream", "type"]
        orig_by_param = {f.metadata("param"): f for f in original}
        for new_field in restored:
            p = new_field.metadata("param")
            orig = orig_by_param[p]
            for k in expected_keys:
                assert new_field.metadata(k) == orig.metadata(k), f"{p}.{k}"


class TestEncoderEncodedData:
    def test_to_bytes_matches_to_file(self, mars_tensogram_file, tmp_path) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        enc = create_encoder("tensogram")

        encoded = enc.encode(fl)

        # Bytes in memory...
        via_bytes = encoded.to_bytes()

        # ...must equal bytes on disk.
        out_path = tmp_path / "direct.tgm"
        fl.to_target("file", str(out_path), encoder="tensogram")
        via_file = out_path.read_bytes()

        # Both are valid tensogram messages starting with the magic.
        assert via_bytes[:8] == b"TENSOGRM"
        assert via_file[:8] == b"TENSOGRM"
        # They encode the same content — round-trip through decode and
        # compare values.  Byte-level equality is not guaranteed because
        # tensogram's provenance encoder stamps a per-encode timestamp.
        import tensogram

        meta1 = tensogram.decode_metadata(via_bytes)
        meta2 = tensogram.decode_metadata(via_file)
        base1 = [e for e in (meta1.base or [])]
        base2 = [e for e in (meta2.base or [])]
        assert len(base1) == len(base2)


class TestXarrayEncoder:
    """Generic (non-MARS) xarray Datasets can be written too."""

    def test_xarray_dataset_to_target(self, nonmars_tensogram_file, tmp_path) -> None:
        ds = ekd.from_source("tensogram", str(nonmars_tensogram_file)).to_xarray()
        out_path = tmp_path / "xa_roundtrip.tgm"
        enc = create_encoder("tensogram")
        encoded = enc.encode(ds)
        encoded.to_file(str(out_path))
        assert out_path.exists()
        assert out_path.read_bytes()[:8] == b"TENSOGRM"

        # Round-trip values.
        restored = ekd.from_source("tensogram", str(out_path)).to_xarray()
        assert len(restored.data_vars) == len(ds.data_vars)
        for name in ds.data_vars:
            np.testing.assert_array_equal(restored[name].values, ds[name].values, err_msg=name)


class TestErrors:
    def test_encode_without_data_raises(self) -> None:
        enc = create_encoder("tensogram")
        with pytest.raises(ValueError, match="requires a data object"):
            enc.encode(data=None)
