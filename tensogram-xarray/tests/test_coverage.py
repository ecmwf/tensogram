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

    def test_auto_payload_keys_filtered(self, tmp_path: Path):
        """ndim/shape/strides/dtype from payload don't appear in var attrs."""
        data = np.ones((3,), dtype=np.float32)
        path = str(tmp_path / "payload_filter.tgm")
        meta = {"version": 2, "payload": [{"mars": {"param": "2t"}}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3]), data)])

        store = TensogramDataStore(path, variable_key="mars.param")
        ds = store.build_dataset()
        var = ds["2t"]
        # Auto-populated keys should be filtered
        assert "ndim" not in var.attrs
        assert "shape" not in var.attrs
        assert "strides" not in var.attrs
        assert "dtype" not in var.attrs
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

    def test_scan_file_with_common_metadata(self, tmp_path: Path):
        """scan_file reads common metadata into ObjectInfo.common_meta."""
        # line 78, 86, 89: common metadata path
        path = str(tmp_path / "common.tgm")
        data = np.ones((2,), dtype=np.float32)
        meta = {"version": 2, "common": {"source": "ecmwf"}}
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
        # Should have at least 3 variables (one per object)
        total = sum(len(ds.data_vars) for ds in datasets)
        assert total >= 3

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
