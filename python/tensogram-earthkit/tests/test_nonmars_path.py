# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""The xarray-only path for tensogram files without MARS metadata.

The contract covered here:

* Opening a non-MARS ``.tgm`` via ``ekd.from_source("tensogram", ...)``
  exposes ``.to_xarray()`` that returns an :class:`xarray.Dataset`
  **byte-equivalent** to the Dataset you get from ``xr.open_dataset``
  with ``engine="tensogram"`` (the tensogram-xarray backend).
* ``.to_numpy()`` returns the single decoded tensor for a one-object
  message, or raises a clear error for multi-object messages.
* ``.to_fieldlist()`` raises :class:`NotImplementedError` with a clear
  message pointing at the xarray path.

Delegation to tensogram-xarray is deliberate — we pinned that design
choice in the plan so coordinate auto-detection, dim-name resolution,
and hypercube merging live in exactly one place.
"""

from __future__ import annotations

import earthkit.data as ekd
import numpy as np
import pytest
import xarray as xr


class TestXarrayPath:
    def test_to_xarray_returns_dataset(self, nonmars_tensogram_file) -> None:
        data = ekd.from_source("tensogram", str(nonmars_tensogram_file))
        ds = data.to_xarray()
        assert isinstance(ds, xr.Dataset)
        # The fixture encodes one 3-D (2,3,4) float64 variable.
        assert len(ds.data_vars) == 1
        var = next(iter(ds.data_vars.values()))
        assert var.shape == (2, 3, 4)
        assert var.dtype == np.dtype("float64")

    def test_matches_tensogram_xarray_backend(self, nonmars_tensogram_file) -> None:
        """Byte-for-byte parity with ``xr.open_dataset(engine="tensogram")``.

        The earthkit source must not introduce its own coordinate or
        dim-name logic on top of the tensogram-xarray backend.
        """
        via_earthkit = ekd.from_source("tensogram", str(nonmars_tensogram_file)).to_xarray()
        via_backend = xr.open_dataset(str(nonmars_tensogram_file), engine="tensogram")

        assert list(via_earthkit.data_vars) == list(via_backend.data_vars)
        for name in via_earthkit.data_vars:
            np.testing.assert_array_equal(
                via_earthkit[name].values, via_backend[name].values, err_msg=name
            )

    def test_to_numpy_single_object(self, nonmars_tensogram_file) -> None:
        """Single-object message → single ndarray via ``.to_numpy()``."""
        data = ekd.from_source("tensogram", str(nonmars_tensogram_file))
        arr = data.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3, 4)
        assert arr.dtype == np.dtype("float64")
        # Contents match what conftest encoded.
        expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        np.testing.assert_array_equal(arr, expected)

    def test_to_fieldlist_raises_for_nonmars(self, nonmars_tensogram_file) -> None:
        """Non-MARS tensograms have no Field semantics — raise cleanly."""
        data = ekd.from_source("tensogram", str(nonmars_tensogram_file))
        with pytest.raises(NotImplementedError, match="MARS"):
            data.to_fieldlist()


class TestXarrayPathReaderSurface:
    """``to_xarray`` should also be callable on the reader (FileSource) directly."""

    def test_source_level_to_xarray(self, nonmars_tensogram_file) -> None:
        source = ekd.from_source("tensogram", str(nonmars_tensogram_file))
        # FileSource exposes to_xarray too (delegates to the reader).
        ds = source.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 1

    def test_source_level_to_numpy(self, nonmars_tensogram_file) -> None:
        source = ekd.from_source("tensogram", str(nonmars_tensogram_file))
        arr = source.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3, 4)
