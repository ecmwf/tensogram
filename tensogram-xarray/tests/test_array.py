"""Tests for TensogramBackendArray lazy loading."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import xarray as xr

from tensogram_xarray.array import (
    TensogramBackendArray,
    _supports_range_decode,
)


class TestSupportsRangeDecodeLogic:
    """Unit tests for the _supports_range_decode helper."""

    def test_none_compression(self):
        """Uncompressed data supports range decode."""

        class _Desc:
            compression = "none"
            filter = "none"
            params = {}

        assert _supports_range_decode(_Desc()) is True

    def test_szip_supports(self):
        class _Desc:
            compression = "szip"
            filter = "none"
            params = {}

        assert _supports_range_decode(_Desc()) is True

    def test_blosc2_supports(self):
        class _Desc:
            compression = "blosc2"
            filter = "none"
            params = {}

        assert _supports_range_decode(_Desc()) is True

    def test_zfp_fixed_rate_supports(self):
        class _Desc:
            compression = "zfp"
            filter = "none"
            params = {"zfp_mode": "fixed_rate"}

        assert _supports_range_decode(_Desc()) is True

    def test_zfp_non_fixed_rate_not_supported(self):
        class _Desc:
            compression = "zfp"
            filter = "none"
            params = {"zfp_mode": "fixed_precision"}

        assert _supports_range_decode(_Desc()) is False

    def test_zstd_not_supported(self):
        class _Desc:
            compression = "zstd"
            filter = "none"
            params = {}

        assert _supports_range_decode(_Desc()) is False

    def test_lz4_not_supported(self):
        class _Desc:
            compression = "lz4"
            filter = "none"
            params = {}

        assert _supports_range_decode(_Desc()) is False

    def test_sz3_not_supported(self):
        class _Desc:
            compression = "sz3"
            filter = "none"
            params = {}

        assert _supports_range_decode(_Desc()) is False

    def test_shuffle_blocks_range(self):
        """Shuffle filter prevents range decode regardless of compressor."""

        class _Desc:
            compression = "none"
            filter = "shuffle"
            params = {}

        assert _supports_range_decode(_Desc()) is False

    def test_shuffle_plus_szip_blocks(self):
        class _Desc:
            compression = "szip"
            filter = "shuffle"
            params = {}

        assert _supports_range_decode(_Desc()) is False


class TestBackendArrayPickle:
    """Pickle safety for dask multiprocessing."""

    def test_picklable(self, simple_tgm: Path):
        arr = TensogramBackendArray(
            file_path=str(simple_tgm),
            msg_index=0,
            obj_index=0,
            shape=(6, 10),
            dtype=np.dtype("float32"),
            supports_range=True,
        )
        data = pickle.dumps(arr)
        restored = pickle.loads(data)
        assert restored.shape == (6, 10)
        assert restored.dtype == np.dtype("float32")
        assert restored.file_path == str(simple_tgm)


class TestLazyLoading:
    """Verify that data is loaded lazily."""

    def test_data_not_loaded_on_open(self, simple_tgm: Path):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram")
        # The internal data should be a LazilyIndexedArray, not numpy.
        var = ds["object_0"].variable
        assert not isinstance(var._data, np.ndarray)

    def test_data_loads_on_access(self, simple_tgm: Path, simple_data: np.ndarray):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram")
        values = ds["object_0"].values  # triggers load
        np.testing.assert_array_equal(values, simple_data)

    def test_slice_access(self, simple_tgm: Path, simple_data: np.ndarray):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram")
        sliced = ds["object_0"][1:3, 2:5].values
        np.testing.assert_array_equal(sliced, simple_data[1:3, 2:5])
