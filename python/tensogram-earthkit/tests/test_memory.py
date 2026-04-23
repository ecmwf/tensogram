# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Memory-reader path: ``ekd.from_source("tensogram", <bytes>)``.

Contract:

* Passing ``bytes`` or ``bytearray`` to the tensogram source yields the
  same data-object API as passing a file path — ``to_xarray()``,
  ``to_numpy()``, and ``to_fieldlist()`` (for MARS content) all work.
* ``memoryview`` inputs are accepted too.
* The integration is a logical equivalent to the file path — not
  implementation-coupled to a temp file, so we test the behaviour, not
  the mechanism.
"""

from __future__ import annotations

import earthkit.data as ekd
import numpy as np
import pytest
import xarray as xr


class TestBytesInput:
    def test_bytes_nonmars_to_xarray(self, nonmars_tensogram_bytes) -> None:
        data = ekd.from_source("tensogram", nonmars_tensogram_bytes)
        ds = data.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 1
        var = next(iter(ds.data_vars.values()))
        assert var.shape == (2, 3, 4)

    def test_bytes_nonmars_to_numpy(self, nonmars_tensogram_bytes) -> None:
        data = ekd.from_source("tensogram", nonmars_tensogram_bytes)
        arr = data.to_numpy()
        expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        np.testing.assert_array_equal(arr, expected)

    def test_bytearray_accepted(self, nonmars_tensogram_bytes) -> None:
        """bytearray input works like bytes."""
        buf = bytearray(nonmars_tensogram_bytes)
        data = ekd.from_source("tensogram", buf)
        ds = data.to_xarray()
        assert isinstance(ds, xr.Dataset)

    def test_memoryview_accepted(self, nonmars_tensogram_bytes) -> None:
        """memoryview input works like bytes."""
        view = memoryview(nonmars_tensogram_bytes)
        data = ekd.from_source("tensogram", view)
        ds = data.to_xarray()
        assert isinstance(ds, xr.Dataset)

    def test_bytes_mars_to_fieldlist(self, mars_tensogram_bytes) -> None:
        data = ekd.from_source("tensogram", mars_tensogram_bytes)
        fl = data.to_fieldlist()
        assert len(fl) == 2
        params = sorted(f.metadata("param") for f in fl)
        assert params == ["2t", "tp"]

    def test_bytes_nonmars_to_fieldlist_raises(self, nonmars_tensogram_bytes) -> None:
        data = ekd.from_source("tensogram", nonmars_tensogram_bytes)
        with pytest.raises(NotImplementedError, match="MARS"):
            data.to_fieldlist()

    def test_bytes_too_small_for_magic_is_rejected(self) -> None:
        """Garbage input should raise a clear error."""
        with pytest.raises((ValueError, OSError, Exception)):
            ekd.from_source("tensogram", b"\x00\x00\x00")


class TestBytesMatchesFileParity:
    """Bytes-mode and file-mode must produce equivalent decoded content."""

    def test_nonmars_parity(self, nonmars_tensogram_file, nonmars_tensogram_bytes) -> None:
        via_file = ekd.from_source("tensogram", str(nonmars_tensogram_file)).to_numpy()
        via_bytes = ekd.from_source("tensogram", nonmars_tensogram_bytes).to_numpy()
        np.testing.assert_array_equal(via_file, via_bytes)

    def test_mars_parity(self, mars_tensogram_file, mars_tensogram_bytes) -> None:
        file_fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        bytes_fl = ekd.from_source("tensogram", mars_tensogram_bytes).to_fieldlist()
        assert len(file_fl) == len(bytes_fl)
        for ff, bf in zip(file_fl, bytes_fl, strict=True):
            assert ff.metadata("param") == bf.metadata("param")
            np.testing.assert_array_equal(ff.to_numpy(), bf.to_numpy())
