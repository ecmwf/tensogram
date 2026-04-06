"""Integration tests using zarr.open_group/open_array with TensogramStore.

These tests verify that the Zarr API works end-to-end on top of TensogramStore.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensogram
import zarr
from tensogram_zarr import TensogramStore


class TestZarrReadGroup:
    """Test reading TGM files through zarr.open_group()."""

    def test_open_group_simple(self, simple_tgm: str):
        """Open a simple .tgm as a Zarr group and read the array."""
        store = TensogramStore.open_tgm(simple_tgm)
        root = zarr.open_group(store=store, mode="r")

        # Root should have one array
        members = list(root.members())
        assert len(members) == 1

        _name, arr = members[0]
        assert arr.shape == (6, 10)
        assert arr.dtype == np.float32

        # Read actual data
        data = arr[:]
        expected = np.arange(60, dtype=np.float32).reshape(6, 10)
        np.testing.assert_array_equal(data, expected)

    def test_open_group_multi_object(self, multi_object_tgm: str):
        """Open a multi-object .tgm as a Zarr group."""
        store = TensogramStore.open_tgm(multi_object_tgm)
        root = zarr.open_group(store=store, mode="r")

        members = list(root.members())
        assert len(members) == 3

        # Access by name (mars.param values: 2t, sp, q)
        names = [name for name, _ in members]
        assert "2t" in names
        assert "sp" in names
        assert "q" in names

    def test_group_attributes(self, mars_metadata_tgm: str):
        """Verify group-level attributes from GlobalMetadata."""
        store = TensogramStore.open_tgm(mars_metadata_tgm)
        root = zarr.open_group(store=store, mode="r")

        attrs = dict(root.attrs)
        assert "mars" in attrs
        assert attrs["mars"]["class"] == "od"

    def test_array_dtype_preservation(self, int_types_tgm: str):
        """Verify integer dtypes are preserved through the Zarr layer."""
        store = TensogramStore.open_tgm(int_types_tgm)
        root = zarr.open_group(store=store, mode="r")

        counts = root["counts"]
        assert counts.dtype == np.int32
        np.testing.assert_array_equal(counts[:], [1, 2, 3, 4])

        flags = root["flags"]
        assert flags.dtype == np.uint16
        np.testing.assert_array_equal(flags[:], [10, 20, 30])


class TestZarrReadArray:
    """Test reading individual arrays through zarr.open_array()."""

    def test_slicing(self, simple_tgm: str):
        """Test that Zarr slicing works on TGM-backed arrays."""
        store = TensogramStore.open_tgm(simple_tgm)
        root = zarr.open_group(store=store, mode="r")

        members = list(root.members())
        _, arr = members[0]

        # Full read
        full = arr[:]
        assert full.shape == (6, 10)

        # Partial slice
        subset = arr[2:4, 3:7]
        expected = np.arange(60, dtype=np.float32).reshape(6, 10)[2:4, 3:7]
        np.testing.assert_array_equal(subset, expected)

    def test_scalar_indexing(self, simple_tgm: str):
        store = TensogramStore.open_tgm(simple_tgm)
        root = zarr.open_group(store=store, mode="r")
        members = list(root.members())
        _, arr = members[0]

        val = arr[0, 0]
        assert float(val) == 0.0

        val = arr[5, 9]
        assert float(val) == 59.0


class TestZarrReadEmpty:
    """Test edge case: empty TGM file."""

    def test_empty_file_opens(self, empty_tgm: str):
        store = TensogramStore.open_tgm(empty_tgm)
        root = zarr.open_group(store=store, mode="r")
        members = list(root.members())
        assert len(members) == 0


class TestZarrFullRoundTrip:
    """Full end-to-end: tensogram encode → Zarr read → verify."""

    @pytest.mark.parametrize(
        ("dtype_str", "np_dtype"),
        [
            ("float32", np.float32),
            ("float64", np.float64),
            ("int32", np.int32),
            ("int64", np.int64),
            ("uint8", np.uint8),
            ("uint16", np.uint16),
        ],
    )
    def test_dtype_round_trip(self, tmp_path: Path, dtype_str: str, np_dtype):
        """Test that various dtypes survive the TGM → Zarr round-trip."""
        path = str(tmp_path / f"rt_{dtype_str}.tgm")
        original = np.arange(12, dtype=np_dtype).reshape(3, 4)

        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {"version": 2, "base": [{"name": "data"}]},
                [({"type": "ntensor", "shape": [3, 4], "dtype": dtype_str}, original)],
            )

        store = TensogramStore.open_tgm(path)
        root = zarr.open_group(store=store, mode="r")
        arr = root["data"]
        np.testing.assert_array_equal(arr[:], original)

    def test_large_array(self, tmp_path: Path):
        """Test with a larger array to verify no truncation."""
        path = str(tmp_path / "large.tgm")
        original = np.random.rand(100, 200).astype(np.float64)

        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {"version": 2, "base": [{"name": "big"}]},
                [({"type": "ntensor", "shape": [100, 200], "dtype": "float64"}, original)],
            )

        store = TensogramStore.open_tgm(path)
        root = zarr.open_group(store=store, mode="r")
        arr = root["big"]
        assert arr.shape == (100, 200)
        np.testing.assert_array_almost_equal(arr[:], original)

    def test_1d_array(self, tmp_path: Path):
        """Test 1-D array round-trip."""
        path = str(tmp_path / "1d.tgm")
        original = np.array([10, 20, 30, 40, 50], dtype=np.int32)

        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {"version": 2, "base": [{"name": "vector"}]},
                [({"type": "ntensor", "shape": [5], "dtype": "int32"}, original)],
            )

        store = TensogramStore.open_tgm(path)
        root = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(root["vector"][:], original)

    def test_3d_array(self, tmp_path: Path):
        """Test 3-D array round-trip (the sea wave spectra use case)."""
        path = str(tmp_path / "3d.tgm")
        original = np.random.rand(10, 20, 30).astype(np.float32)

        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {"version": 2, "base": [{"name": "spectra"}]},
                [({"type": "ntensor", "shape": [10, 20, 30], "dtype": "float32"}, original)],
            )

        store = TensogramStore.open_tgm(path)
        root = zarr.open_group(store=store, mode="r")
        arr = root["spectra"]
        assert arr.shape == (10, 20, 30)
        np.testing.assert_array_almost_equal(arr[:], original)
