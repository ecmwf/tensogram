# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the :class:`TensogramData` facade.

The wrapper is a thin delegation layer over :class:`TensogramFileReader`.
Covered here so the ``available_types`` advertisement and the three
``to_*`` methods cannot silently diverge from the reader's behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tensogram_earthkit import TensogramData
from tensogram_earthkit.readers.file import TensogramFileReader


class _Src:
    """Fake source — only the attributes the Reader base class needs."""

    def __init__(self) -> None:
        self.source_filename = None
        self.storage_options = None


def _make_reader(path) -> TensogramFileReader:
    return TensogramFileReader(_Src(), str(path))


class TestTensogramDataAvailableTypes:
    def test_advertises_three_conversions(self) -> None:
        assert TensogramData.available_types == ("xarray", "numpy", "fieldlist")


class TestTensogramDataDelegation:
    def test_to_xarray_delegates(self, nonmars_tensogram_file) -> None:
        data = TensogramData(_make_reader(nonmars_tensogram_file))
        ds = data.to_xarray()
        assert isinstance(ds, xr.Dataset)

    def test_to_numpy_delegates(self, nonmars_tensogram_file) -> None:
        data = TensogramData(_make_reader(nonmars_tensogram_file))
        arr = data.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3, 4)

    def test_to_fieldlist_delegates_for_mars(self, mars_tensogram_file) -> None:
        data = TensogramData(_make_reader(mars_tensogram_file))
        fl = data.to_fieldlist()
        assert len(fl) == 2

    def test_to_fieldlist_raises_for_nonmars(self, nonmars_tensogram_file) -> None:
        data = TensogramData(_make_reader(nonmars_tensogram_file))
        with pytest.raises(NotImplementedError, match="MARS"):
            data.to_fieldlist()


class TestReaderProducesData:
    """``reader.to_data_object()`` produces a TensogramData instance."""

    def test_to_data_object(self, nonmars_tensogram_file) -> None:
        reader = _make_reader(nonmars_tensogram_file)
        data = reader.to_data_object()
        assert isinstance(data, TensogramData)
        # Round-trip through the wrapper.
        arr = data.to_numpy()
        assert arr.shape == (2, 3, 4)
