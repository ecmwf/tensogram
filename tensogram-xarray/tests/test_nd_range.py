# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for N-dimensional slice to flat range mapping and ratio heuristic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensogram
import xarray as xr
from tensogram_xarray.array import (
    DEFAULT_RANGE_THRESHOLD,
    _nd_slice_to_flat_ranges,
)

# ---------------------------------------------------------------------------
# Unit tests for _nd_slice_to_flat_ranges
# ---------------------------------------------------------------------------


class TestNdSliceToFlatRanges:
    """Verify the N-D -> flat range mapping math."""

    def test_1d_full(self):
        """Full 1-D slice -> single range."""
        ranges, shape = _nd_slice_to_flat_ranges((10,), (slice(None),))
        assert ranges == [(0, 10)]
        assert shape == (10,)

    def test_1d_partial(self):
        """Partial 1-D slice -> single range."""
        ranges, shape = _nd_slice_to_flat_ranges((100,), (slice(10, 20),))
        assert ranges == [(10, 10)]
        assert shape == (10,)

    def test_2d_full_row(self):
        """arr[1:3, :] on shape (6, 10) -> one contiguous range of 20."""
        ranges, shape = _nd_slice_to_flat_ranges((6, 10), (slice(1, 3), slice(None)))
        assert ranges == [(10, 20)]
        assert shape == (2, 10)

    def test_2d_partial_columns(self):
        """arr[1:3, 2:5] on shape (6, 10) -> two ranges of 3 each."""
        ranges, shape = _nd_slice_to_flat_ranges((6, 10), (slice(1, 3), slice(2, 5)))
        assert ranges == [(12, 3), (22, 3)]
        assert shape == (2, 3)

    def test_2d_full_array(self):
        """arr[:, :] -> single range covering everything."""
        ranges, shape = _nd_slice_to_flat_ranges((6, 10), (slice(None), slice(None)))
        assert ranges == [(0, 60)]
        assert shape == (6, 10)

    def test_2d_single_element(self):
        """arr[2:3, 5:6] -> single range of 1."""
        ranges, shape = _nd_slice_to_flat_ranges((6, 10), (slice(2, 3), slice(5, 6)))
        assert ranges == [(25, 1)]
        assert shape == (1, 1)

    def test_3d_inner_partial(self):
        """arr[1:3, :, 2:4] on shape (4, 3, 5) -> 6 ranges of 2."""
        ranges, shape = _nd_slice_to_flat_ranges(
            (4, 3, 5), (slice(1, 3), slice(None), slice(2, 4))
        )
        # 2 outer-dim-0 indices x 3 outer-dim-1 indices = 6 ranges
        assert len(ranges) == 6
        assert all(count == 2 for _, count in ranges)
        assert shape == (2, 3, 2)
        # Total elements = 2 * 3 * 2 = 12
        assert sum(c for _, c in ranges) == 12

    def test_3d_trailing_full(self):
        """arr[1:2, 0:2, :] on shape (4, 3, 5) -> one range of 10."""
        ranges, shape = _nd_slice_to_flat_ranges(
            (4, 3, 5), (slice(1, 2), slice(0, 2), slice(None))
        )
        # dim 1 is partial (0:2 out of 3), dim 2 is full -> block = 2 * 5 = 10
        # dim 0 has 1 index -> 1 range of 10
        assert ranges == [(15, 10)]
        assert shape == (1, 2, 5)

    def test_adjacent_merge(self):
        """arr[:, 2:5] on shape (3, 10) -> 3 ranges of 3, non-adjacent.

        These should NOT merge because there are gaps between rows.
        """
        ranges, _shape = _nd_slice_to_flat_ranges((3, 10), (slice(None), slice(2, 5)))
        assert len(ranges) == 3
        assert ranges == [(2, 3), (12, 3), (22, 3)]

    def test_merge_possible(self):
        """arr[1:3, :] on shape (6, 10) merges two rows into one range."""
        ranges, _shape = _nd_slice_to_flat_ranges((6, 10), (slice(1, 3), slice(None)))
        # Two consecutive full rows -- should be one merged range.
        assert ranges == [(10, 20)]

    def test_empty_slice(self):
        """arr[3:3, :] -> empty result."""
        ranges, shape = _nd_slice_to_flat_ranges((6, 10), (slice(3, 3), slice(None)))
        assert ranges == []
        assert shape == (0, 10)


# ---------------------------------------------------------------------------
# Helpers for building test .tgm files
# ---------------------------------------------------------------------------


def _make_desc(shape, dtype="float32"):
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": dtype,
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


# ---------------------------------------------------------------------------
# Integration: N-D partial reads through the xarray backend
# ---------------------------------------------------------------------------


class TestNdPartialReadIntegration:
    """Verify that 2D slices via xarray trigger partial reads correctly."""

    @pytest.fixture
    def tgm_2d(self, tmp_path: Path) -> tuple[Path, np.ndarray]:
        """Create a 2D float32 .tgm file with known data."""
        data = np.arange(60, dtype=np.float32).reshape(6, 10)
        meta = {"version": 2}
        desc = _make_desc([6, 10])
        path = tmp_path / "2d.tgm"
        with tensogram.TensogramFile.create(str(path)) as f:
            f.append(meta, [(desc, data)])
        return path, data

    def test_2d_slice_values(self, tgm_2d):
        """Slicing a 2D array via xarray returns correct values."""
        path, data = tgm_2d
        ds = xr.open_dataset(str(path), engine="tensogram")
        sliced = ds["object_0"][1:3, 2:5].values
        np.testing.assert_array_equal(sliced, data[1:3, 2:5])

    def test_2d_full_row_values(self, tgm_2d):
        """Full-row slice uses single range."""
        path, data = tgm_2d
        ds = xr.open_dataset(str(path), engine="tensogram")
        sliced = ds["object_0"][1:3, :].values
        np.testing.assert_array_equal(sliced, data[1:3, :])

    def test_2d_full_load_values(self, tgm_2d):
        """Full load returns correct values."""
        path, data = tgm_2d
        ds = xr.open_dataset(str(path), engine="tensogram")
        np.testing.assert_array_equal(ds["object_0"].values, data)


# ---------------------------------------------------------------------------
# Ratio heuristic tests
# ---------------------------------------------------------------------------


class TestRangeThreshold:
    """Verify that the range_threshold parameter controls behavior."""

    @pytest.fixture
    def tgm_small(self, tmp_path: Path) -> tuple[Path, np.ndarray]:
        """10-element 1D float32 array."""
        data = np.arange(10, dtype=np.float32)
        meta = {"version": 2}
        desc = _make_desc([10])
        path = tmp_path / "small.tgm"
        with tensogram.TensogramFile.create(str(path)) as f:
            f.append(meta, [(desc, data)])
        return path, data

    def test_default_threshold(self):
        assert DEFAULT_RANGE_THRESHOLD == 0.5

    def test_small_slice_uses_partial(self, tgm_small):
        """Slice of <50% should use partial decode (default threshold)."""
        path, data = tgm_small
        ds = xr.open_dataset(str(path), engine="tensogram")
        # 3 out of 10 elements = 30%, below threshold
        sliced = ds["object_0"][2:5].values
        np.testing.assert_array_equal(sliced, data[2:5])

    def test_large_slice_correct(self, tgm_small):
        """Slice of >50% falls back to full decode, still correct."""
        path, data = tgm_small
        ds = xr.open_dataset(str(path), engine="tensogram")
        # 8 out of 10 elements = 80%, above threshold
        sliced = ds["object_0"][1:9].values
        np.testing.assert_array_equal(sliced, data[1:9])

    def test_custom_threshold(self, tgm_small):
        """User can set a custom range_threshold."""
        path, data = tgm_small
        ds = xr.open_dataset(str(path), engine="tensogram", range_threshold=0.1)
        # With threshold=0.1, even a 20% slice uses full decode, still correct.
        sliced = ds["object_0"][0:2].values
        np.testing.assert_array_equal(sliced, data[0:2])
