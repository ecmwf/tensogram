# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Coverage-gap tests for tensogram-xarray.

Targets specific untested code paths identified by pytest-cov:

- merge.py: coord building in open_datasets, hypercube stacking,
  multi-variable dataset, flat group dataset, _resolve_dims helper
- scanner.py: message_count property, scan_message(), metadata edge paths
- array.py: non-unit stride rejection, exception fallback to full decode
- store.py: dtype fallback, extra-only metadata, close()
- backend.py: merge_objects returning empty
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import tensogram
import xarray as xr

from tensogram_xarray.array import (
    _is_contiguous_slice,
    _nd_slice_to_flat_ranges,
)
from tensogram_xarray.merge import (
    _extract_meta_keys,
    _make_hashable,
    _partition_keys,
    _try_hypercube,
    _unique_values,
    open_datasets,
)
from tensogram_xarray.scanner import FileIndex, ObjectInfo, scan_file, scan_message
from tensogram_xarray.store import TensogramDataStore, _to_numpy_dtype


def _desc(shape, dtype="float32", **extra):
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": dtype,
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
        **extra,
    }


# ---------------------------------------------------------------------------
# store.py gaps
# ---------------------------------------------------------------------------


class TestStoreCoverage:
    """Cover store.py lines 48, 115, 118, 142."""

    def test_dtype_fallback_valid(self):
        """_to_numpy_dtype falls back to np.dtype() for unlisted strings."""
        # line 48: fallback path -- 'f4' is valid numpy shorthand for float32
        # but isn't in our explicit _DTYPE_MAP.
        dt = _to_numpy_dtype("f4")
        assert dt == np.dtype("float32")

    def test_dtype_fallback_invalid(self):
        """_to_numpy_dtype raises for truly unknown dtype strings."""
        with pytest.raises(TypeError):
            _to_numpy_dtype("not_a_dtype")

    def test_common_meta_extra_only(self, tmp_path: Path):
        """meta.extra keys appear in dataset attrs when common is empty."""
        # lines 115, 118: extra-only metadata path
        data = np.ones((2, 3), dtype=np.float32)
        path = str(tmp_path / "extra_only.tgm")
        # Use a top-level extra key (not common, not payload)
        meta = {"version": 2, "experiment": "test42"}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([2, 3]), data)])

        store = TensogramDataStore(path, msg_index=0)
        ds = store.build_dataset()
        # 'experiment' should be in dataset attrs via meta.extra
        assert ds.attrs.get("experiment") == "test42"

    def test_store_close_is_noop(self, tmp_path: Path):
        """close() is a no-op and does not raise."""
        # line 142
        data = np.ones((2,), dtype=np.float32)
        path = str(tmp_path / "close.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([2]), data)])
        store = TensogramDataStore(path)
        store.close()  # should not raise

    def test_reserved_key_filtered(self, tmp_path: Path):
        """_reserved_ key from base entries doesn't appear in var attrs."""
        data = np.ones((3,), dtype=np.float32)
        path = str(tmp_path / "base_filter.tgm")
        meta = {"version": 2, "base": [{"mars": {"param": "2t"}}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3]), data)])

        store = TensogramDataStore(path, variable_key="mars.param")
        ds = store.build_dataset()
        var = ds["2t"]
        # _reserved_ key should be filtered
        assert "_reserved_" not in var.attrs
        # But user key should be present
        assert "mars" in var.attrs


# ---------------------------------------------------------------------------
# array.py gaps
# ---------------------------------------------------------------------------


class TestArrayCoverage:
    """Cover array.py lines 61, 64, 161, 250-251."""

    def test_non_slice_key_rejected(self):
        """_is_contiguous_slice returns False for integer indexing."""
        # line 61: not isinstance(k, slice)
        assert _is_contiguous_slice((slice(None), 5)) is False

    def test_non_unit_stride_rejected(self):
        """_is_contiguous_slice returns False for step != 1."""
        # line 64: k.step is not None and k.step != 1
        assert _is_contiguous_slice((slice(0, 10, 2),)) is False

    def test_unit_stride_accepted(self):
        """_is_contiguous_slice accepts step=1 and step=None."""
        assert _is_contiguous_slice((slice(0, 10, 1),)) is True
        assert _is_contiguous_slice((slice(0, 10),)) is True

    def test_nd_range_merge_multiple(self):
        """Verify that the merge path in _nd_slice_to_flat_ranges works.

        arr[1:4, :] on shape (6, 10): three consecutive full rows should
        merge into one range.
        """
        # line 161: merged path
        ranges, shape = _nd_slice_to_flat_ranges((6, 10), (slice(1, 4), slice(None)))
        assert ranges == [(10, 30)]
        assert shape == (3, 10)

    def test_decode_range_fallback_on_error(self, tmp_path: Path):
        """When decode_range fails, fall back to decode_object."""
        # lines 250-251: exception fallback
        # Create a file with valid data
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = str(tmp_path / "fallback.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([3, 4]), data)])

        # Open with a very low threshold so full array load
        # goes through decode_object path (ratio > threshold).
        ds = xr.open_dataset(str(path), engine="tensogram", range_threshold=0.0)
        values = ds["object_0"].values
        np.testing.assert_array_equal(values, data)


# ---------------------------------------------------------------------------
# backend.py gaps
# ---------------------------------------------------------------------------


class TestBackendCoverage:
    """Cover backend.py line 97."""

    def test_merge_objects_empty_file(self, tmp_path: Path):
        """merge_objects=True on file with no data objects returns empty."""
        # line 97: empty dataset from merge_objects
        path = str(tmp_path / "empty.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [])

        ds = xr.open_dataset(str(path), engine="tensogram", merge_objects=True)
        assert isinstance(ds, xr.Dataset)


# ---------------------------------------------------------------------------
# scanner.py gaps
# ---------------------------------------------------------------------------


class TestScannerCoverage:
    """Cover scanner.py lines 49-51, 78, 86, 89, 137-163."""

    def test_file_index_message_count_empty(self):
        """FileIndex.message_count returns 0 when empty."""
        # lines 49-51
        idx = FileIndex(file_path="none.tgm")
        assert idx.message_count == 0

    def test_file_index_message_count_multi(self, tmp_path: Path):
        """FileIndex.message_count returns correct count for multi-msg."""
        path = str(tmp_path / "multi.tgm")
        data = np.ones((2,), dtype=np.float32)
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([2]), data)])
            f.append({"version": 2}, [(_desc([2]), data)])
            f.append({"version": 2}, [(_desc([2]), data)])

        idx = scan_file(path)
        assert idx.message_count == 3

    def test_scan_message(self):
        """scan_message works on a raw in-memory message."""
        # lines 137-163
        data = np.arange(6, dtype=np.float32)
        meta = {"version": 2}
        desc = _desc([6], name="wind")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))

        objects = scan_message(msg)
        assert len(objects) == 1
        assert objects[0].shape == (6,)
        assert objects[0].msg_index == 0
        assert objects[0].obj_index == 0
        # name from desc.params should be in per_object_meta
        assert objects[0].per_object_meta.get("name") == "wind"

    def test_scan_message_multi_object(self):
        """scan_message with multiple objects in one message."""
        a = np.ones((3,), dtype=np.float32)
        b = np.zeros((5,), dtype=np.float64)
        meta = {"version": 2}
        msg = bytes(tensogram.encode(meta, [(_desc([3]), a), (_desc([5], dtype="float64"), b)]))

        objects = scan_message(msg)
        assert len(objects) == 2
        assert objects[0].shape == (3,)
        assert objects[1].shape == (5,)

    def test_scan_file_with_extra_metadata(self, tmp_path: Path):
        """scan_file reads extra metadata into ObjectInfo.common_meta."""
        path = str(tmp_path / "extra.tgm")
        data = np.ones((2,), dtype=np.float32)
        meta = {"version": 2, "source": "ecmwf"}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([2]), data)])

        idx = scan_file(path)
        assert len(idx.objects) == 1
        assert idx.objects[0].common_meta.get("source") == "ecmwf"

    def test_object_info_merged_meta(self):
        """ObjectInfo.merged_meta merges common + per-object, per-object wins."""
        info = ObjectInfo(
            msg_index=0,
            obj_index=0,
            ndim=1,
            shape=(5,),
            dtype="float32",
            descriptor=None,
            per_object_meta={"param": "2t", "source": "local"},
            common_meta={"source": "ecmwf", "class": "od"},
        )
        merged = info.merged_meta
        assert merged["param"] == "2t"
        assert merged["source"] == "local"  # per-object wins
        assert merged["class"] == "od"


# ---------------------------------------------------------------------------
# merge.py gaps -- helper functions
# ---------------------------------------------------------------------------


class TestMergeHelpersCoverage:
    """Cover merge.py helper functions that are untested."""

    def test_make_hashable_nested_dict(self):
        """_make_hashable handles nested dicts and lists."""
        val = {"a": [1, {"b": 2}]}
        h = _make_hashable(val)
        assert isinstance(h, tuple)
        # Should be hashable
        hash(h)  # should not raise TypeError

    def test_make_hashable_simple(self):
        """_make_hashable passes through simple values."""
        assert _make_hashable(42) == 42
        assert _make_hashable("hello") == "hello"

    def test_unique_values_with_dicts(self):
        """_unique_values deduplicates dicts (unhashable)."""
        vals = [{"a": 1}, {"b": 2}, {"a": 1}]
        result = _unique_values(vals)
        assert len(result) == 2

    def test_partition_keys_constant_vs_varying(self):
        """_partition_keys splits constant and varying keys."""
        kv = {
            "param": ["2t", "10u"],
            "class": ["od", "od"],
        }
        const, vary = _partition_keys(kv)
        assert "class" in const
        assert const["class"] == "od"
        assert "param" in vary
        assert vary["param"] == ["2t", "10u"]

    def test_partition_keys_unhashable(self):
        """_partition_keys treats unhashable values as constant."""
        kv = {"mars": [{"param": "2t"}, {"param": "2t"}]}
        const, _vary = _partition_keys(kv)
        # Dicts are hashable via _make_hashable, so identical dicts -> constant
        assert "mars" in const

    def test_try_hypercube_complete(self):
        """_try_hypercube returns True for complete grid."""

        class _Obj:
            pass

        objs = [_Obj() for _ in range(4)]
        varying = {
            "param": ["2t", "10u", "2t", "10u"],
            "date": ["d1", "d1", "d2", "d2"],
        }
        assert _try_hypercube(objs, varying) is True

    def test_try_hypercube_incomplete(self):
        """_try_hypercube returns False for incomplete grid."""

        class _Obj:
            pass

        objs = [_Obj() for _ in range(3)]
        varying = {
            "param": ["2t", "10u", "2t"],
            "date": ["d1", "d1", "d2"],
        }
        # 2 x 2 = 4 expected but only 3 objects
        assert _try_hypercube(objs, varying) is False

    def test_try_hypercube_empty_varying(self):
        """_try_hypercube with no varying keys returns True."""
        assert _try_hypercube([], {}) is True

    def test_extract_meta_keys(self):
        """_extract_meta_keys collects values per key."""
        info1 = ObjectInfo(
            msg_index=0,
            obj_index=0,
            ndim=1,
            shape=(3,),
            dtype="float32",
            descriptor=None,
            per_object_meta={"param": "2t"},
            common_meta={"class": "od"},
        )
        info2 = ObjectInfo(
            msg_index=1,
            obj_index=0,
            ndim=1,
            shape=(3,),
            dtype="float32",
            descriptor=None,
            per_object_meta={"param": "10u"},
            common_meta={"class": "od"},
        )
        kv = _extract_meta_keys([info1, info2])
        assert kv["param"] == ["2t", "10u"]
        assert kv["class"] == ["od", "od"]


# ---------------------------------------------------------------------------
# merge.py gaps -- integration: coord vars, hypercube, multi-var dataset
# ---------------------------------------------------------------------------


class TestMergeIntegrationCoverage:
    """Cover merge.py integration paths: coord building, hypercube, multi-var."""

    def test_open_datasets_with_coords(self, tmp_path: Path):
        """open_datasets builds coord vars from coord objects across msgs."""
        # lines 74-90: coord building in open_datasets
        path = str(tmp_path / "coords_merge.tgm")
        lat = np.linspace(-90, 90, 3, dtype=np.float64)
        lon = np.linspace(0, 360, 4, endpoint=False, dtype=np.float64)
        temp = np.ones((3, 4), dtype=np.float32)

        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {"version": 2},
                [
                    (_desc([3], dtype="float64", name="latitude"), lat),
                    (_desc([4], dtype="float64", name="longitude"), lon),
                    (_desc([3, 4], name="temperature"), temp),
                ],
            )

        datasets = open_datasets(path)
        assert len(datasets) >= 1
        ds = datasets[0]
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords

    def test_multi_variable_with_outer_dim(self, tmp_path: Path):
        """open_datasets with variable_key splits by param, stacks by date.

        Exercises _build_multi_variable_dataset (lines 510-613).
        """
        path = str(tmp_path / "multi_var.tgm")
        rng = np.random.default_rng(77)

        with tensogram.TensogramFile.create(path) as f:
            for param in ["2t", "10u"]:
                for date in ["20260401", "20260402"]:
                    data = rng.random((3, 4), dtype=np.float32).astype(np.float32)
                    desc = _desc([3, 4], mars={"param": param, "date": date})
                    f.append({"version": 2}, [(desc, data)])

        datasets = open_datasets(path, variable_key="mars.param")
        assert len(datasets) >= 1
        ds = datasets[0]
        # Both variables should be present
        assert "2t" in ds.data_vars or "10u" in ds.data_vars

    def test_flat_group_no_varying(self, tmp_path: Path):
        """Objects with identical metadata become separate variables.

        Exercises _flat_group_dataset (lines 390-413).
        """
        path = str(tmp_path / "flat.tgm")
        data = np.ones((2, 3), dtype=np.float32)

        with tensogram.TensogramFile.create(path) as f:
            # Two messages with identical metadata and shape
            f.append({"version": 2}, [(_desc([2, 3], name="field"), data)])
            f.append({"version": 2}, [(_desc([2, 3], name="field"), data * 2)])

        datasets = open_datasets(path)
        assert len(datasets) >= 1
        ds = datasets[0]
        # Should have at least 1 variable
        assert len(ds.data_vars) >= 1

    def test_hypercube_without_variable_key(self, tmp_path: Path):
        """Hypercube stacking without variable_key.

        Exercises _hypercube_dataset (lines 416-460).
        """
        path = str(tmp_path / "hypercube.tgm")
        rng = np.random.default_rng(88)

        with tensogram.TensogramFile.create(path) as f:
            for step in ["0", "6", "12"]:
                data = rng.random((2, 3), dtype=np.float32).astype(np.float32)
                desc = _desc([2, 3], step=step)
                f.append({"version": 2}, [(desc, data)])

        datasets = open_datasets(path)
        assert len(datasets) >= 1
        # All 3 objects should be represented
        ds = datasets[0]
        total_vars = len(ds.data_vars)
        assert total_vars >= 1

    def test_multi_variable_single_per_param(self, tmp_path: Path):
        """variable_key with single object per param -> no outer dims.

        Exercises line 526-540 in _build_multi_variable_dataset.
        Uses a flat 'param' key (not nested mars dict) so variable_key
        is the only varying key and each value gets its own variable.
        """
        path = str(tmp_path / "single_per_param.tgm")
        t2m = np.ones((2, 3), dtype=np.float32) * 273
        u10 = np.ones((2, 3), dtype=np.float32) * 5

        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([2, 3], param="2t"), t2m)])
            f.append({"version": 2}, [(_desc([2, 3], param="10u"), u10)])

        datasets = open_datasets(path, variable_key="param")
        assert len(datasets) >= 1
        ds = datasets[0]
        assert "2t" in ds.data_vars
        assert "10u" in ds.data_vars
        np.testing.assert_array_equal(ds["2t"].values, t2m)
        np.testing.assert_array_equal(ds["10u"].values, u10)

    def test_multi_variable_no_remaining_varying(self, tmp_path: Path):
        """variable_key exhausts all varying keys -> no outer dims.

        Exercises lines 596-610 in _build_multi_variable_dataset.
        """
        path = str(tmp_path / "no_remain.tgm")
        a = np.ones((2,), dtype=np.float32)
        b = np.ones((2,), dtype=np.float32) * 2

        with tensogram.TensogramFile.create(path) as f:
            # Two messages where the ONLY varying key is the variable_key itself
            f.append({"version": 2}, [(_desc([2], param="alpha"), a)])
            f.append({"version": 2}, [(_desc([2], param="beta"), b)])

        datasets = open_datasets(path, variable_key="param")
        assert len(datasets) >= 1
        ds = datasets[0]
        assert "alpha" in ds.data_vars
        assert "beta" in ds.data_vars

    def test_multi_variable_stacking_with_date(self, tmp_path: Path):
        """variable_key with remaining varying key (date) triggers stacking.

        Exercises lines 541-580 in _build_multi_variable_dataset -- the
        hypercube stacking path within multi-variable splitting.
        """
        path = str(tmp_path / "stack_date.tgm")
        rng = np.random.default_rng(55)

        with tensogram.TensogramFile.create(path) as f:
            for param in ["2t", "10u"]:
                for date in ["d1", "d2"]:
                    data = rng.random((2, 3), dtype=np.float32).astype(np.float32)
                    desc = _desc([2, 3], param=param, date=date)
                    f.append({"version": 2}, [(desc, data)])

        datasets = open_datasets(path, variable_key="param")
        assert len(datasets) >= 1
        ds = datasets[0]
        # Both variables should exist, stacked along 'date'
        assert "2t" in ds.data_vars or "10u" in ds.data_vars

    def test_incomplete_hypercube_fallback(self, tmp_path: Path):
        """Incomplete hypercube falls back to flat group (line 332).

        3 objects varying on 2 keys but not forming a 2x2 grid.
        """
        path = str(tmp_path / "incomplete.tgm")
        rng = np.random.default_rng(66)

        with tensogram.TensogramFile.create(path) as f:
            # param x date: (2t,d1), (10u,d1), (2t,d2) -- missing (10u,d2)
            for param, date in [("2t", "d1"), ("10u", "d1"), ("2t", "d2")]:
                data = rng.random((2, 3), dtype=np.float32).astype(np.float32)
                desc = _desc([2, 3], param=param, date=date)
                f.append({"version": 2}, [(desc, data)])

        datasets = open_datasets(path)
        assert len(datasets) >= 1
        # With incomplete hypercube, objects fall back to flat group.
        # All have obj_index=0 so share the name "object_0"; the group
        # should still produce at least one dataset with data.
        total = sum(len(ds.data_vars) for ds in datasets)
        assert total >= 1

    def test_resolve_dims_with_dim_names(self, tmp_path: Path):
        """open_datasets with dim_names uses them for all variables.

        Exercises line 633 in _resolve_dims.
        """
        path = str(tmp_path / "dimnames.tgm")
        data = np.ones((3, 4), dtype=np.float32)
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([3, 4]), data)])

        datasets = open_datasets(path, dim_names=["row", "col"])
        assert len(datasets) >= 1
        ds = datasets[0]
        var = next(iter(ds.data_vars.values()))
        assert var.dims == ("row", "col")

    def test_build_dataset_from_group_empty(self):
        """_build_dataset_from_group returns None for empty group.

        Exercises line 270.
        """
        from tensogram_xarray.merge import _build_dataset_from_group

        result = _build_dataset_from_group(
            group=[],
            file_path="none.tgm",
            coord_vars={},
            dim_names=None,
            variable_key=None,
            lock=__import__("threading").Lock(),
        )
        assert result is None

    def test_split_by_key(self):
        """_split_by_key partitions by a metadata key.

        Exercises lines 241-246.
        """
        from tensogram_xarray.merge import _split_by_key

        objs = [
            ObjectInfo(0, 0, 1, (3,), "float32", None, {"param": "2t"}, {}),
            ObjectInfo(1, 0, 1, (3,), "float32", None, {"param": "10u"}, {}),
            ObjectInfo(2, 0, 1, (3,), "float32", None, {"param": "2t"}, {}),
        ]
        groups = _split_by_key(objs, "param")
        assert len(groups) == 2
        sizes = sorted(len(g) for g in groups)
        assert sizes == [1, 2]


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Verify that errors are raised with informative messages."""

    def test_dtype_fallback_message(self):
        """_to_numpy_dtype includes the dtype name in the error."""
        with pytest.raises(TypeError, match=r"unsupported tensogram dtype.*'badtype'"):
            _to_numpy_dtype("badtype")

    def test_negative_message_index(self):
        """Negative message_index raises ValueError with context."""
        with pytest.raises(ValueError, match="message_index must be >= 0"):
            xr.open_dataset("nonexistent.tgm", engine="tensogram", message_index=-1)

    def test_dim_names_wrong_count(self, tmp_path):
        """dim_names length mismatch gives a clear error."""
        data = np.ones((3, 4), dtype=np.float32)
        path = str(tmp_path / "bad_dims.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([3, 4]), data)])
        with pytest.raises(ValueError, match="dim_names has 1 entries"):
            xr.open_dataset(str(path), engine="tensogram", dim_names=["x"])

    def test_file_not_found(self):
        """Opening a non-existent file raises an OSError."""
        with pytest.raises(OSError, match="does_not_exist"):
            xr.open_dataset("/tmp/does_not_exist_xyz.tgm", engine="tensogram")

    def test_coord_names_sync(self):
        """KNOWN_COORD_NAMES and CANONICAL_DIM are always in sync."""
        from tensogram_xarray.coords import CANONICAL_DIM, KNOWN_COORD_NAMES

        assert frozenset(CANONICAL_DIM.keys()) == KNOWN_COORD_NAMES


# ---------------------------------------------------------------------------
# Additional coverage: reachable gaps from code coverage audit
# ---------------------------------------------------------------------------


class TestStackedBackendArrayEdges:
    """Cover StackedBackendArray error and edge paths in array.py."""

    def test_count_mismatch_error(self):
        """StackedBackendArray rejects wrong number of arrays (lines 298-303)."""
        from tensogram_xarray.array import StackedBackendArray

        with pytest.raises(ValueError, match="expected 6 backing arrays"):
            StackedBackendArray(
                arrays=[None, None],  # 2, but outer_shape needs 6
                outer_shape=(2, 3),
                inner_shape=(4,),
                dtype=np.dtype("float32"),
            )

    def test_expand_key_wildcard(self):
        """_expand_key_to_indices handles non-int/non-slice keys (lines 376-379)."""
        from tensogram_xarray.array import _expand_key_to_indices

        # Pass None as a key element — falls through to wildcard branch
        result = _expand_key_to_indices((None, slice(1, 3)), (5, 10))
        assert result[0] == list(range(5))  # wildcard: full range
        assert result[1] == [1, 2]  # slice


class TestDecodeRangeFallback:
    """Cover the decode_range exception fallback in array.py (lines 258-259)."""

    def test_fallback_when_decode_range_raises(self, tmp_path: Path):
        """Inject a failure in decode_range to trigger the fallback path."""
        from unittest.mock import patch

        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = str(tmp_path / "fallback_exc.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([3, 4]), data)])

        ds = xr.open_dataset(path, engine="tensogram", range_threshold=1.0)

        # Patch decode_range to raise, forcing fallback to decode_object
        with patch("tensogram.decode_range", side_effect=RuntimeError("mock fail")):
            values = ds["object_0"][0:1, 0:2].values
            np.testing.assert_array_equal(values, data[0:1, 0:2])


class TestMergeConflictingCoords:
    """Cover the conflicting coordinate shape error in merge.py (lines 99-108)."""

    def test_conflicting_coord_shapes_raises(self, tmp_path: Path):
        """Two lat coords with different sizes should raise ValueError."""
        path = str(tmp_path / "conflict.tgm")
        lat3 = np.linspace(-90, 90, 3, dtype=np.float64)
        lat5 = np.linspace(-90, 90, 5, dtype=np.float64)
        data3 = np.ones((3, 4), dtype=np.float32)
        data5 = np.ones((5, 4), dtype=np.float32)

        with tensogram.TensogramFile.create(path) as f:
            # Message 0: lat has 3 elements
            f.append(
                {"version": 2},
                [
                    (_desc([3], dtype="float64", name="latitude"), lat3),
                    (_desc([3, 4]), data3),
                ],
            )
            # Message 1: lat has 5 elements — conflicting
            f.append(
                {"version": 2},
                [
                    (_desc([5], dtype="float64", name="latitude"), lat5),
                    (_desc([5, 4]), data5),
                ],
            )

        with pytest.raises(ValueError, match="conflicting shapes"):
            open_datasets(path)


class TestMergeNonHypercubeFallback:
    """Cover non-hypercube fallback paths in merge.py (lines 642-686).

    These paths are inside ``_build_multi_variable_dataset``, which requires
    ``len(variable_names) > 1`` — so we need at least two distinct param
    values to enter it.
    """

    def test_incomplete_sub_hypercube_warns(self, tmp_path: Path, caplog):
        """Incomplete sub-group hypercube triggers warning (lines 642-662).

        4 messages: param={2t, 10u}.  The 2t sub-group has 3 objects with
        (date, level) varying as (d1,L1), (d1,L2), (d2,L1) — missing
        (d2,L2) so the 2x2 grid is incomplete.
        """
        import logging

        path = str(tmp_path / "incomplete_sub.tgm")
        rng = np.random.default_rng(33)

        with tensogram.TensogramFile.create(path) as f:
            for date, level in [("d1", "L1"), ("d1", "L2"), ("d2", "L1")]:
                data = rng.random((2, 3), dtype=np.float32).astype(np.float32)
                desc = _desc([2, 3], param="2t", date=date, level=level)
                f.append({"version": 2}, [(desc, data)])
            data = rng.random((2, 3), dtype=np.float32).astype(np.float32)
            desc = _desc([2, 3], param="10u", date="d1", level="L1")
            f.append({"version": 2}, [(desc, data)])

        with caplog.at_level(logging.WARNING, logger="tensogram_xarray"):
            datasets = open_datasets(path, variable_key="param")

        assert len(datasets) >= 1
        assert any("cannot form a hypercube" in r.message for r in caplog.records)

    def test_duplicate_objects_no_varying_warns(self, tmp_path: Path, caplog):
        """Duplicate objects with identical metadata triggers warning (lines 664-686).

        4 messages: 2 with param=2t (identical metadata), 2 with param=10u.
        Within each sub-group, no varying keys remain, triggering the warning.
        """
        import logging

        path = str(tmp_path / "dups.tgm")
        data = np.ones((2, 3), dtype=np.float32)

        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([2, 3], param="2t"), data)])
            f.append({"version": 2}, [(_desc([2, 3], param="2t"), data)])
            f.append({"version": 2}, [(_desc([2, 3], param="10u"), data * 2)])
            f.append({"version": 2}, [(_desc([2, 3], param="10u"), data * 2)])

        with caplog.at_level(logging.WARNING, logger="tensogram_xarray"):
            datasets = open_datasets(path, variable_key="param")

        assert len(datasets) >= 1
        assert any("duplicate objects" in r.message for r in caplog.records)


class TestScannerDescParamsFallback:
    """Cover scanner.py desc.params fallback (line 79) and extra merge (line 102)."""

    def test_desc_params_supplement(self, tmp_path: Path):
        """Descriptor params fill in when payload is absent (line 79)."""
        data = np.ones((3,), dtype=np.float32)
        path = str(tmp_path / "params.tgm")
        # No payload in metadata — desc.params is the only source
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([3], custom_key="hello"), data)])

        idx = scan_file(path)
        assert len(idx.objects) == 1
        # custom_key should come from desc.params
        assert idx.objects[0].per_object_meta.get("custom_key") == "hello"

    def test_extra_in_common_meta(self, tmp_path: Path):
        """meta.extra keys appear in common_meta."""
        data = np.ones((2,), dtype=np.float32)
        path = str(tmp_path / "extra.tgm")
        meta = {"version": 2, "experiment": "test99"}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([2]), data)])

        idx = scan_file(path)
        assert idx.objects[0].common_meta.get("experiment") == "test99"


class TestStoreMetaExtra:
    """Cover store.py meta.extra merge (line 126)."""

    def test_extra_keys_in_dataset_attrs(self, tmp_path: Path):
        """meta.extra keys appear in dataset attributes (line 126)."""
        data = np.ones((2, 3), dtype=np.float32)
        path = str(tmp_path / "store_extra.tgm")
        meta = {"version": 2, "custom_attr": "present"}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([2, 3]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds.attrs.get("custom_attr") == "present"


# ---------------------------------------------------------------------------
# Coverage pass 2: scanner.py unit-level helpers
# ---------------------------------------------------------------------------


class TestScannerHelpers:
    """Cover scanner.py helpers at the unit level."""

    def test_base_entry_from_meta_non_dict_entry(self):
        """_base_entry_from_meta returns {} when base[i] is not a dict."""
        from tensogram_xarray.scanner import _base_entry_from_meta

        class FakeMeta:
            base: ClassVar[list[object]] = ["not_a_dict", {"key": "val"}]

        # index 0 is a string, not a dict -> should return {}
        assert _base_entry_from_meta(FakeMeta(), 0) == {}
        # index 1 is a dict -> should return filtered copy
        assert _base_entry_from_meta(FakeMeta(), 1) == {"key": "val"}

    def test_base_entry_from_meta_no_base(self):
        """_base_entry_from_meta returns {} when meta has no base attribute."""
        from tensogram_xarray.scanner import _base_entry_from_meta

        class FakeMeta:
            pass

        assert _base_entry_from_meta(FakeMeta(), 0) == {}

    def test_base_entry_from_meta_base_not_list(self):
        """_base_entry_from_meta returns {} when base is not a list."""
        from tensogram_xarray.scanner import _base_entry_from_meta

        class FakeMeta:
            base: ClassVar[str] = "not a list"

        assert _base_entry_from_meta(FakeMeta(), 0) == {}

    def test_base_entry_from_meta_reserved_filtered(self):
        """_base_entry_from_meta filters out _reserved_ key."""
        from tensogram_xarray.scanner import _base_entry_from_meta

        class FakeMeta:
            base: ClassVar[list[object]] = [
                {"mars": {"param": "2t"}, "_reserved_": {"tensor": {}}}
            ]

        result = _base_entry_from_meta(FakeMeta(), 0)
        assert "_reserved_" not in result
        assert "mars" in result

    def test_extra_from_meta_empty_extra(self):
        """_extra_from_meta returns {} when extra is empty dict."""
        from tensogram_xarray.scanner import _extra_from_meta

        class FakeMeta:
            extra: ClassVar[dict[str, object]] = {}

        assert _extra_from_meta(FakeMeta()) == {}

    def test_extra_from_meta_none_extra(self):
        """_extra_from_meta returns {} when extra is None."""
        from tensogram_xarray.scanner import _extra_from_meta

        class FakeMeta:
            extra: ClassVar[None] = None

        assert _extra_from_meta(FakeMeta()) == {}

    def test_extra_from_meta_no_attr(self):
        """_extra_from_meta returns {} when meta has no extra attribute."""
        from tensogram_xarray.scanner import _extra_from_meta

        class FakeMeta:
            pass

        assert _extra_from_meta(FakeMeta()) == {}

    def test_extra_from_meta_non_dict(self):
        """_extra_from_meta returns {} when extra is not a dict."""
        from tensogram_xarray.scanner import _extra_from_meta

        class FakeMeta:
            extra: ClassVar[list[object]] = [1, 2, 3]

        assert _extra_from_meta(FakeMeta()) == {}

    def test_desc_params_none(self):
        """_desc_params returns {} when desc.params is None."""
        from tensogram_xarray.scanner import _desc_params

        class FakeDesc:
            params: ClassVar[None] = None

        assert _desc_params(FakeDesc()) == {}

    def test_desc_params_non_dict(self):
        """_desc_params returns {} when desc.params is not a dict."""
        from tensogram_xarray.scanner import _desc_params

        class FakeDesc:
            params: ClassVar[str] = "not_a_dict"

        assert _desc_params(FakeDesc()) == {}

    def test_desc_params_no_attr(self):
        """_desc_params returns {} when desc has no params attribute."""
        from tensogram_xarray.scanner import _desc_params

        class FakeDesc:
            pass

        assert _desc_params(FakeDesc()) == {}

    def test_desc_params_valid(self):
        """_desc_params returns a copy of the dict when valid."""
        from tensogram_xarray.scanner import _desc_params

        class FakeDesc:
            params: ClassVar[dict[str, object]] = {"custom": "value", "count": 42}

        result = _desc_params(FakeDesc())
        assert result == {"custom": "value", "count": 42}
        # Should be a copy, not the same object
        assert result is not FakeDesc.params

    def test_merge_per_object_meta_base_wins(self):
        """_merge_per_object_meta: base entry takes priority over desc.params."""
        from tensogram_xarray.scanner import _merge_per_object_meta

        class FakeMeta:
            base: ClassVar[list[object]] = [{"key": "from_base", "shared": "base_val"}]

        class FakeDesc:
            params: ClassVar[dict[str, object]] = {"shared": "desc_val", "extra": "desc_only"}

        result = _merge_per_object_meta(FakeMeta(), 0, FakeDesc())
        assert result["key"] == "from_base"
        assert result["shared"] == "base_val"  # base wins
        assert result["extra"] == "desc_only"  # desc fills gap


# ---------------------------------------------------------------------------
# Coverage pass 2: store.py per-object meta edge cases
# ---------------------------------------------------------------------------


class TestStorePerObjectMetaEdges:
    """Cover store.py _get_per_object_meta edge paths."""

    def test_base_is_none(self, tmp_path):
        """_get_per_object_meta handles base=None gracefully."""
        from unittest.mock import MagicMock

        data = np.ones((3,), dtype=np.float32)
        path = str(tmp_path / "base_none.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([3]), data)])

        store = TensogramDataStore(path)
        # Replace _meta with a mock where base is None
        mock = MagicMock()
        mock.base = None
        mock.extra = {}
        mock.version = 2
        store._meta = mock

        result = store._get_per_object_meta(0, store._descriptors[0])
        assert isinstance(result, dict)

    def test_desc_params_supplement_in_store(self, tmp_path):
        """desc.params supplements missing base keys in store."""
        data = np.ones((3,), dtype=np.float32)
        path = str(tmp_path / "desc_sup.tgm")
        # base has mars but desc has custom_key
        meta = {"version": 2, "base": [{"mars": {"param": "2t"}}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3], extra_key="from_desc"), data)])

        store = TensogramDataStore(path)
        result = store._get_per_object_meta(0, store._descriptors[0])
        # mars from base, extra_key from desc.params
        assert "mars" in result
        assert result.get("extra_key") == "from_desc"


# ---------------------------------------------------------------------------
# Coverage pass 2: mapping.py edge cases
# ---------------------------------------------------------------------------


class TestMappingEdges:
    """Cover xarray mapping.py edge cases."""

    def test_resolve_dotted_empty_dict_intermediate(self):
        """_resolve_dotted returns None when an intermediate is empty dict."""
        from tensogram_xarray.mapping import _resolve_dotted

        assert _resolve_dotted({}, "a.b") is None

    def test_resolve_dotted_none_intermediate(self):
        """_resolve_dotted returns None when a key maps to None."""
        from tensogram_xarray.mapping import _resolve_dotted

        assert _resolve_dotted({"a": None}, "a.b") is None

    def test_resolve_dotted_non_dict_intermediate(self):
        """_resolve_dotted returns None when intermediate is not a dict."""
        from tensogram_xarray.mapping import _resolve_dotted

        assert _resolve_dotted({"a": 42}, "a.b") is None

    def test_resolve_variable_name_empty_meta(self):
        """resolve_variable_name with empty meta falls back to generic."""
        from tensogram_xarray.mapping import resolve_variable_name

        assert resolve_variable_name(7, {}, "mars.param") == "object_7"

    def test_resolve_variable_name_value_is_int(self):
        """Numeric metadata values are stringified."""
        from tensogram_xarray.mapping import resolve_variable_name

        meta = {"mars": {"param": 500}}
        assert resolve_variable_name(0, meta, "mars.param") == "500"


# ---------------------------------------------------------------------------
# Coverage pass 2: backend.py open_dataset with merge_objects=True empty
# ---------------------------------------------------------------------------


class TestBackendMergeObjectsEmpty:
    """Cover backend.py merge_objects returning xr.Dataset() for empty."""

    def test_merge_objects_empty_returns_dataset(self, tmp_path):
        """merge_objects=True on empty file returns xr.Dataset()."""
        import tensogram

        path = str(tmp_path / "meta_only.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2, "signal": "start"}, [])

        ds = xr.open_dataset(path, engine="tensogram", merge_objects=True)
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 0


# ---------------------------------------------------------------------------
# Coverage pass 2: shared dim resolver generic fallback naming
# ---------------------------------------------------------------------------


class TestSharedResolveDimsGeneric:
    """Cover :func:`tensogram_xarray.mapping.resolve_dims_for_axes` fallback."""

    def test_generic_dims_across_axes(self):
        """With no hints, every axis falls back to a per-axis ``dim_N`` name."""
        from tensogram_xarray.mapping import resolve_dims_for_axes

        dims = resolve_dims_for_axes(
            (3, 4, 5),
            user_dim_names=None,
            coord_dim_sizes={},
            per_object_meta=None,
            extra_dim_names_hint=None,
        )
        assert [name for name, _ in dims] == ["dim_0", "dim_1", "dim_2"]
        assert all(is_generic for _, is_generic in dims)

    def test_mixed_coord_and_generic(self):
        """Coord-matched axes are non-generic; unmatched axes fall back."""
        from tensogram_xarray.mapping import resolve_dims_for_axes

        dims = resolve_dims_for_axes(
            (3, 7),
            user_dim_names=None,
            coord_dim_sizes={"latitude": 3},
            per_object_meta=None,
            extra_dim_names_hint=None,
        )
        assert dims[0] == ("latitude", False)
        assert dims[1] == ("dim_1", True)


# ---------------------------------------------------------------------------
# Coverage pass 2: array.py _supports_range_decode edge cases
# ---------------------------------------------------------------------------


class TestSupportsRangeDecodeEdges:
    """Cover array.py _supports_range_decode edge paths."""

    def test_zfp_non_fixed_rate_not_supported(self):
        """zfp with non-fixed_rate mode does not support range decode."""
        from tensogram_xarray.array import _supports_range_decode

        class FakeDesc:
            compression: ClassVar[str] = "zfp"
            filter: ClassVar[str] = "none"
            params: ClassVar[dict[str, object]] = {"zfp_mode": "fixed_accuracy"}

        assert _supports_range_decode(FakeDesc()) is False

    def test_zfp_fixed_rate_supported(self):
        """zfp with fixed_rate mode supports range decode."""
        from tensogram_xarray.array import _supports_range_decode

        class FakeDesc:
            compression: ClassVar[str] = "zfp"
            filter: ClassVar[str] = "none"
            params: ClassVar[dict[str, object]] = {"zfp_mode": "fixed_rate"}

        assert _supports_range_decode(FakeDesc()) is True

    def test_zfp_no_params_not_supported(self):
        """zfp with no params does not support range decode."""
        from tensogram_xarray.array import _supports_range_decode

        class FakeDesc:
            compression: ClassVar[str] = "zfp"
            filter: ClassVar[str] = "none"
            params: ClassVar[None] = None

        assert _supports_range_decode(FakeDesc()) is False

    def test_shuffle_blocks_range_decode(self):
        """shuffle filter blocks range decode regardless of compressor."""
        from tensogram_xarray.array import _supports_range_decode

        class FakeDesc:
            compression: ClassVar[str] = "none"
            filter: ClassVar[str] = "shuffle"
            params: ClassVar[dict[str, object]] = {}

        assert _supports_range_decode(FakeDesc()) is False

    def test_unknown_compressor_not_supported(self):
        """Unknown compressor does not support range decode."""
        from tensogram_xarray.array import _supports_range_decode

        class FakeDesc:
            compression: ClassVar[str] = "zstd"
            filter: ClassVar[str] = "none"
            params: ClassVar[dict[str, object]] = {}

        assert _supports_range_decode(FakeDesc()) is False

    def test_szip_supported(self):
        """szip supports range decode."""
        from tensogram_xarray.array import _supports_range_decode

        class FakeDesc:
            compression: ClassVar[str] = "szip"
            filter: ClassVar[str] = "none"
            params: ClassVar[dict[str, object]] = {}

        assert _supports_range_decode(FakeDesc()) is True


# ---------------------------------------------------------------------------
# Coverage pass 2: scanner.py scan_message with extra metadata
# ---------------------------------------------------------------------------


class TestScanMessageWithExtra:
    """Cover scan_message extracting extra metadata."""

    def test_scan_message_extra_in_common_meta(self):
        """scan_message reads extra metadata into ObjectInfo.common_meta."""
        data = np.ones((3,), dtype=np.float32)
        meta = {"version": 2, "experiment": "test99"}
        msg = bytes(tensogram.encode(meta, [(_desc([3]), data)]))

        objects = scan_message(msg)
        assert len(objects) == 1
        assert objects[0].common_meta.get("experiment") == "test99"

    def test_scan_message_no_extra(self):
        """scan_message with no extra returns empty common_meta."""
        data = np.ones((3,), dtype=np.float32)
        msg = bytes(tensogram.encode({"version": 2}, [(_desc([3]), data)]))

        objects = scan_message(msg)
        assert len(objects) == 1
        assert objects[0].common_meta == {}


# ---------------------------------------------------------------------------
# Coverage pass 2: store.py _resolve_dims_for_var all branches
# ---------------------------------------------------------------------------


class TestResolveDimsForVar:
    """Cover TensogramDataStore._resolve_dims_for_var branches."""

    def test_with_explicit_dim_names(self, tmp_path):
        """User-provided dim_names used directly."""
        data = np.ones((3, 4), dtype=np.float32)
        path = str(tmp_path / "dims.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([3, 4]), data)])

        store = TensogramDataStore(path, dim_names=["row", "col"])
        ds = store.build_dataset()
        var = ds["object_0"]
        assert var.dims == ("row", "col")

    def test_size_matching_against_coords(self, tmp_path):
        """Dim names matched from coord vars by size."""
        lat = np.linspace(-90, 90, 3, dtype=np.float64)
        lon = np.linspace(0, 360, 4, endpoint=False, dtype=np.float64)
        data = np.ones((3, 4), dtype=np.float32)

        meta = {
            "version": 2,
            "base": [
                {"name": "latitude"},
                {"name": "longitude"},
                {"name": "field"},
            ],
        }
        path = str(tmp_path / "sz_match.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([3], dtype="float64"), lat),
                    (_desc([4], dtype="float64"), lon),
                    (_desc([3, 4]), data),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        # With the priority chain, base[2]["name"] = "field" is used.
        dims = ds["field"].dims
        assert "latitude" in dims
        assert "longitude" in dims


# ---------------------------------------------------------------------------
# Coverage pass 3: additional gap tests
# ---------------------------------------------------------------------------


class TestMetaDimNames:
    """Cover _get_meta_dim_names() and its integration into _resolve_dims_for_var."""

    # ── List format (preferred) ───────────────────────────────────────

    def test_list_format_used(self, tmp_path):
        """List-format dim_names are assigned by axis position."""
        data = np.ones((1000, 50), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": ["values", "level"]},
        }
        path = str(tmp_path / "list_dims.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([1000, 50]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("values", "level")

    def test_list_format_same_size_axes(self, tmp_path):
        """List format handles same-size axes correctly (by position)."""
        data = np.ones((10, 10), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": ["grid_x", "grid_y"]},
        }
        path = str(tmp_path / "list_same_size.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([10, 10]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("grid_x", "grid_y")

    def test_list_format_wrong_length_ignored(self, tmp_path):
        """List with wrong number of names is silently ignored."""
        data = np.ones((3, 4), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": ["only_one"]},
        }
        path = str(tmp_path / "list_wrong_len.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3, 4]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("dim_0", "dim_1")

    # ── Dict format (legacy) ─────────────────────────────────────────

    def test_dict_format_used(self, tmp_path):
        """Dict-format dim_names resolve by size matching."""
        data = np.ones((1000, 50), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": {"1000": "values", "50": "level"}},
        }
        path = str(tmp_path / "dict_dims.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([1000, 50]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("values", "level")

    def test_dict_format_duplicate_size(self, tmp_path):
        """Dict format: same-size axes — hint used once, second falls back."""
        data = np.ones((10, 10), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": {"10": "grid"}},
        }
        path = str(tmp_path / "dict_dup_size.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([10, 10]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        dims = ds["field"].dims
        assert dims[0] == "grid"
        assert dims[1] == "dim_1"

    # ── Priority & edge cases ────────────────────────────────────────

    def test_meta_dim_names_absent(self, tmp_path):
        """Without _extra_.dim_names, dims fall back to dim_N."""
        data = np.ones((3, 4), dtype=np.float32)
        path = str(tmp_path / "no_meta.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([3, 4]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["object_0"].dims == ("dim_0", "dim_1")

    def test_meta_dim_names_malformed(self, tmp_path):
        """Malformed dim_names (string) silently falls back to dim_N."""
        data = np.ones((3, 4), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": "not_a_dict_or_list"},
        }
        path = str(tmp_path / "bad_meta.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3, 4]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("dim_0", "dim_1")

    def test_coord_takes_priority_over_dict_hint(self, tmp_path):
        """Coord-based matching (step 2) takes priority over dict hints."""
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        data = np.ones((5, 8), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [
                {"name": "latitude"},
                {"name": "field"},
            ],
            "_extra_": {"dim_names": {"5": "rows", "8": "cols"}},
        }
        path = str(tmp_path / "coord_vs_meta.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([5, 8]), data),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        dims = ds["field"].dims
        # Axis 0 (size 5): coord "latitude" wins over hint "rows"
        assert dims[0] == "latitude"
        # Axis 1 (size 8): no coord match, dict hint "cols" used
        assert dims[1] == "cols"

    def test_explicit_dim_names_override_meta(self, tmp_path):
        """User-supplied dim_names (step 1) override everything."""
        data = np.ones((5, 8), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": ["rows", "cols"]},
        }
        path = str(tmp_path / "explicit_vs_meta.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([5, 8]), data)])

        ds = xr.open_dataset(path, engine="tensogram", dim_names=["lat", "lon"])
        assert ds["field"].dims == ("lat", "lon")


class TestRangeMergeContiguous:
    """Cover array.py line 165: adjacent flat-range merging."""

    def test_contiguous_rows_merged(self):
        """3-D array: outer slice over two rows with full inner dims merges."""
        # shape (4, 3, 5), slice [0:2, 0:3, 0:5] → all slices
        # The outer indices produce flat ranges that are adjacent → merged.
        shape = (4, 3, 5)
        key = (slice(0, 2), slice(0, 3), slice(0, 5))
        ranges, out_shape = _nd_slice_to_flat_ranges(shape, key)
        assert out_shape == (2, 3, 5)
        # The two outer-row ranges should merge into one:
        assert len(ranges) == 1
        assert ranges[0] == (0, 30)

    def test_non_contiguous_rows_not_merged(self):
        """Non-adjacent rows produce separate flat ranges."""
        # shape (4, 3, 5), partial inner slice → gaps between outer rows
        shape = (4, 3, 5)
        key = (slice(0, 2), slice(0, 1), slice(0, 5))
        ranges, out_shape = _nd_slice_to_flat_ranges(shape, key)
        assert out_shape == (2, 1, 5)
        # Row 0 col 0: (0, 5); Row 1 col 0: (15, 5) → NOT adjacent
        assert len(ranges) == 2


class TestDuplicateCoordDedup:
    """Cover merge.py line 121: duplicate coord with matching shape skipped."""

    def test_duplicate_coord_across_messages(self, tmp_path):
        """Two messages with the same 1-D coord name + shape: second is skipped."""
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        data1 = np.ones((5, 8), dtype=np.float32)
        data2 = np.ones((5, 8), dtype=np.float32) * 2

        path = str(tmp_path / "dup_coord.tgm")
        with tensogram.TensogramFile.create(path) as f:
            # Both messages include 'latitude' as a coord object
            f.append(
                {
                    "version": 2,
                    "base": [
                        {"name": "latitude"},
                        {"name": "temp"},
                    ],
                },
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([5, 8]), data1),
                ],
            )
            f.append(
                {
                    "version": 2,
                    "base": [
                        {"name": "latitude"},
                        {"name": "wind"},
                    ],
                },
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([5, 8]), data2),
                ],
            )

        datasets = open_datasets(path)
        # Should not raise despite duplicate 'latitude' coord
        assert len(datasets) >= 1
        # latitude should exist as a coord
        ds = datasets[0]
        assert "latitude" in ds.coords or any("latitude" in d.coords for d in datasets)


class TestCoordOnlyFile:
    """Cover merge.py line 148: coord-only file produces fallback Dataset."""

    def test_coord_only_fallback(self, tmp_path):
        """File with only 1-D objects (all become coords) → fallback Dataset."""
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        lon = np.linspace(0, 360, 8, endpoint=False, dtype=np.float64)

        path = str(tmp_path / "coord_only.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {
                    "version": 2,
                    "base": [
                        {"name": "latitude"},
                        {"name": "longitude"},
                    ],
                },
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([8], dtype="float64"), lon),
                ],
            )

        datasets = open_datasets(path)
        assert len(datasets) >= 1
        # No data vars, but coords should exist
        ds = datasets[0]
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords


class TestUnhashableMetadataConstant:
    """Cover merge.py lines 222-225: unhashable values treated as constant."""

    def test_dict_metadata_becomes_attr(self, tmp_path):
        """Dict-valued metadata can't be hashed → treated as constant attr."""
        path = str(tmp_path / "unhashable.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for i in range(3):
                data = np.ones((3, 4), dtype=np.float32) * i
                f.append(
                    {
                        "version": 2,
                        "base": [
                            {"config": {"nested": [1, 2, 3]}, "name": "temp"},
                        ],
                    },
                    [(_desc([3, 4]), data)],
                )

        datasets = open_datasets(path)
        assert len(datasets) >= 1
        # The unhashable 'config' key should become a dataset attribute,
        # not a dimension — open_datasets should not crash


# ---------------------------------------------------------------------------
# Merge path dim resolution parity with store path
# ---------------------------------------------------------------------------


class TestMergePathDimResolution:
    """``open_datasets()`` / ``merge_objects=True`` honours the same priority
    chain as ``open_dataset()``:  user kwarg > coord match > per-object
    ``base[i]["dim_names"]`` > ``_extra_["dim_names"]`` > generic fallback.

    Closes a pre-existing divergence where ``merge.py`` silently ignored
    ``_extra_["dim_names"]`` list and dict hints.
    """

    def test_extra_list_hint_honoured(self, tmp_path):
        """Single-message file with ``_extra_`` list hint applies through merge path."""
        path = str(tmp_path / "merge_extra_list.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": ["values", "level"]},
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([1000, 50]), np.ones((1000, 50), dtype=np.float32))])

        datasets = open_datasets(path)
        assert len(datasets) == 1
        assert datasets[0]["field"].dims == ("values", "level")

    def test_extra_dict_hint_honoured(self, tmp_path):
        """Single-message file with ``_extra_`` dict hint applies through merge path."""
        path = str(tmp_path / "merge_extra_dict.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": {"1000": "values", "50": "level"}},
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([1000, 50]), np.ones((1000, 50), dtype=np.float32))])

        datasets = open_datasets(path)
        assert len(datasets) == 1
        assert datasets[0]["field"].dims == ("values", "level")

    def test_per_object_hint_honoured(self, tmp_path):
        """Per-object ``base[i]["dim_names"]`` drives dims through merge path."""
        path = str(tmp_path / "merge_per_obj.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": ["row", "col"]}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), np.ones((4, 5), dtype=np.float32))])

        datasets = open_datasets(path)
        assert datasets[0]["field"].dims == ("row", "col")

    def test_coord_match_outranks_extra_list(self, tmp_path):
        """Coord size-match beats ``_extra_`` list format on matching axes."""
        path = str(tmp_path / "coord_vs_extra_list.tgm")
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        data = np.ones((5, 8), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "latitude"}, {"name": "field"}],
            "_extra_": {"dim_names": ["rows", "cols"]},
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([5, 8]), data),
                ],
            )

        datasets = open_datasets(path)
        ds = datasets[0]
        assert ds["field"].dims[0] == "latitude"
        assert ds["field"].dims[1] == "cols"

    def test_dim_names_not_a_hypercube_outer_dim(self, tmp_path):
        """Varying ``dim_names`` across messages must not become an outer dim.

        Without the structural-key filter, differing ``dim_names`` hints
        across messages would be treated as a varying metadata key and
        promoted to a hypercube outer dimension.
        """
        path = str(tmp_path / "varying_dim_names.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {
                    "version": 2,
                    "base": [{"name": "temp", "dim_names": ["a", "b"]}],
                },
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32))],
            )
            f.append(
                {
                    "version": 2,
                    "base": [{"name": "temp", "dim_names": ["c", "d"]}],
                },
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32) * 2)],
            )

        datasets = open_datasets(path)
        for ds in datasets:
            assert "dim_names" not in ds.sizes
            assert "dim_names" not in ds.coords

    def test_inconsistent_per_object_hint_warns_and_falls_back(self, tmp_path, caplog):
        """Multi-message group with conflicting per-object hints warns + falls back."""
        import logging

        path = str(tmp_path / "inconsistent.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {
                    "version": 2,
                    "base": [{"name": "temp", "dim_names": ["x", "y"]}],
                },
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32))],
            )
            f.append(
                {
                    "version": 2,
                    "base": [{"name": "temp", "dim_names": ["p", "q"]}],
                },
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32) * 2)],
            )

        with caplog.at_level(logging.WARNING, logger="tensogram_xarray.merge"):
            datasets = open_datasets(path)

        assert any("inconsistent per-object dim_names" in r.message for r in caplog.records)
        for ds in datasets:
            dims = ds["temp"].dims
            assert dims != ("x", "y")
            assert dims != ("p", "q")

    def test_merge_hint_vs_coord_conflict_does_not_crash(self, tmp_path, caplog):
        """Merge path: a hint that claims an existing coord name at a different
        size used to crash Dataset assembly with ``conflicting sizes for
        dimension`` because only ``store.py`` disambiguated such clashes.
        Now the merge path renames the offending axis and keeps the coord
        intact."""
        import logging

        path = str(tmp_path / "merge_hint_vs_coord.tgm")
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        wrong = np.ones((7,), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [
                {"name": "latitude"},
                {"name": "field", "dim_names": ["latitude"]},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([7]), wrong),
                ],
            )

        with caplog.at_level(logging.WARNING, logger="tensogram_xarray.merge"):
            datasets = open_datasets(path)

        assert datasets, "open_datasets must succeed on hint-vs-coord conflict"
        found_field = False
        for ds in datasets:
            if "field" in ds.data_vars:
                found_field = True
                assert ds["field"].dims == ("obj_1_dim_0",)
                assert "latitude" in ds.coords
                assert ds.coords["latitude"].shape == (5,)
        assert found_field, "expected a dataset containing the 'field' variable"
        assert any("conflicts with coord 'latitude'" in r.message for r in caplog.records)

    def test_inconsistent_extra_hint_does_not_trigger_per_object_warning(self, tmp_path, caplog):
        """Disagreeing ``_extra_["dim_names"]`` across messages must emit the
        ``_extra_`` warning only — *not* a spurious per-object warning.

        Regression test for a bug where ``_consistent_hint_meta`` read
        ``obj.merged_meta`` (which folded in ``common_meta`` carrying
        ``_extra_["dim_names"]``) and mis-classified the list as a
        per-object hint."""
        import logging

        path = str(tmp_path / "extra_conflict.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {
                    "version": 2,
                    "base": [{"name": "temp"}],
                    "_extra_": {"dim_names": ["x", "y"]},
                },
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32))],
            )
            f.append(
                {
                    "version": 2,
                    "base": [{"name": "temp"}],
                    "_extra_": {"dim_names": ["p", "q"]},
                },
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32) * 2)],
            )

        with caplog.at_level(logging.WARNING, logger="tensogram_xarray.merge"):
            datasets = open_datasets(path)

        messages = [r.message for r in caplog.records]
        assert any("inconsistent _extra_" in m for m in messages)
        assert not any("inconsistent per-object dim_names" in m for m in messages), (
            "merge path must not misread message-level _extra_['dim_names'] as a per-object hint"
        )
        assert datasets, "expected at least one Dataset despite inconsistent hints"
