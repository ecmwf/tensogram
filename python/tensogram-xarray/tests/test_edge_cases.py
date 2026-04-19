# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Edge case tests for tensogram-xarray.

Covers: meta.base out-of-range, _reserved_ filtering consistency,
variable naming with dot paths, multi-message merge with different
base structures, messages with zero base entries, and lazy loading
shape/dtype sourcing after refactor.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import tensogram
import xarray as xr
from tensogram_xarray.mapping import resolve_variable_name
from tensogram_xarray.merge import open_datasets
from tensogram_xarray.scanner import scan_file
from tensogram_xarray.store import TensogramDataStore


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
# meta.base access edge cases
# ---------------------------------------------------------------------------


class TestBaseIndexOutOfRange:
    """meta.base[obj_index] when obj_index >= len(meta.base).

    Note: the standard encoder always pads `base` to match the object
    count, so this edge case can't be triggered through normal encoding.
    We test the warning logic via unit-level mocking instead.
    """

    def test_encoder_pads_base_to_match_objects(self, tmp_path: Path):
        """The encoder auto-extends base to match the number of objects.

        Even if the user provides fewer base entries, the encoder pads
        the rest with auto-populated _reserved_ entries.
        """
        path = str(tmp_path / "short_base.tgm")
        data1 = np.ones((3, 4), dtype=np.float32)
        data2 = np.ones((3, 4), dtype=np.float32) * 2.0

        # Only 1 user base entry for 2 objects
        meta = {"version": 2, "base": [{"tag": "first"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [(_desc([3, 4]), data1), (_desc([3, 4]), data2)],
            )

        store = TensogramDataStore(path)
        ds = store.build_dataset()
        # Encoder padded base to 2 entries, so no warning
        assert len(ds.data_vars) == 2
        # First object should have "tag" in its attrs
        assert ds["object_0"].attrs.get("tag") == "first"

    def test_warning_on_short_base_via_mock(self, tmp_path: Path, caplog):
        """Directly test the warning path by mocking a short base list.

        The PyO3 metadata object returns copies of base, so we mock
        the _meta attribute with a plain Python object.
        """
        from unittest.mock import MagicMock

        path = str(tmp_path / "mock_base.tgm")
        data = np.ones((3, 4), dtype=np.float32)
        meta = {"version": 2}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3, 4]), data)])

        store = TensogramDataStore(path)
        # Replace _meta with a mock that has an empty base list
        mock_meta = MagicMock()
        mock_meta.base = []  # empty base, but we request index 0
        store._meta = mock_meta

        with caplog.at_level(logging.WARNING):
            result = store._get_per_object_meta(0, store._descriptors[0])

        assert any(
            "meta.base has 0 entries but object index 0" in r.message for r in caplog.records
        )
        # Should still return a dict (from desc.params fallback)
        assert isinstance(result, dict)

    def test_empty_base_list_no_crash(self, tmp_path: Path):
        """meta.base = [] should not crash."""
        path = str(tmp_path / "empty_base.tgm")
        data = np.ones((3,), dtype=np.float32)
        meta = {"version": 2, "base": []}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3]), data)])

        store = TensogramDataStore(path)
        ds = store.build_dataset()
        assert len(ds.data_vars) == 1

    def test_no_base_attribute(self, tmp_path: Path):
        """Message with no 'base' key in metadata should not crash."""
        path = str(tmp_path / "no_base.tgm")
        data = np.ones((3,), dtype=np.float32)
        # version only, no base
        meta = {"version": 2}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3]), data)])

        store = TensogramDataStore(path)
        ds = store.build_dataset()
        assert len(ds.data_vars) == 1


# ---------------------------------------------------------------------------
# _reserved_ filtering consistency
# ---------------------------------------------------------------------------


class TestReservedFiltering:
    """_reserved_ must be consistently filtered everywhere."""

    def test_reserved_not_in_variable_attrs(self, tmp_path: Path):
        """_reserved_ auto-populated by encoder must not leak to user attrs."""
        path = str(tmp_path / "reserved.tgm")
        data = np.ones((3,), dtype=np.float32)
        meta = {"version": 2, "base": [{"mars": {"param": "2t"}}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3]), data)])

        ds = xr.open_dataset(path, engine="tensogram", variable_key="mars.param")
        assert "_reserved_" not in ds["2t"].attrs

    def test_reserved_not_in_scanner_per_object_meta(self, tmp_path: Path):
        """scanner.py must filter _reserved_ from per-object metadata."""
        path = str(tmp_path / "scan_reserved.tgm")
        data = np.ones((3,), dtype=np.float32)
        meta = {"version": 2, "base": [{"name": "wind"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3]), data)])

        idx = scan_file(path)
        assert "_reserved_" not in idx.objects[0].per_object_meta

    def test_reserved_not_in_open_datasets_var_attrs(self, tmp_path: Path):
        """open_datasets must also filter _reserved_ from variable attrs."""
        path = str(tmp_path / "merge_reserved.tgm")
        data = np.ones((3,), dtype=np.float32)
        meta = {"version": 2, "base": [{"name": "temp"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3]), data)])

        datasets = open_datasets(path)
        assert len(datasets) >= 1
        for ds in datasets:
            for var in ds.data_vars.values():
                assert "_reserved_" not in var.attrs


# ---------------------------------------------------------------------------
# Variable naming with dot paths
# ---------------------------------------------------------------------------


class TestVariableNamingEdgeCases:
    """resolve_variable_name with dot paths like 'mars.param'."""

    def test_dotted_path_resolution(self):
        meta = {"mars": {"param": "2t"}}
        assert resolve_variable_name(0, meta, "mars.param") == "2t"

    def test_missing_dotted_path_fallback(self):
        meta = {"mars": {"type": "fc"}}
        # mars.param doesn't exist → fallback to object_0
        assert resolve_variable_name(0, meta, "mars.param") == "object_0"

    def test_deeply_nested_dotted_path(self):
        meta = {"a": {"b": {"c": "deep_value"}}}
        assert resolve_variable_name(0, meta, "a.b.c") == "deep_value"

    def test_non_string_value_converted(self):
        """Numeric metadata values should be stringified."""
        meta = {"mars": {"param": 128}}
        assert resolve_variable_name(0, meta, "mars.param") == "128"

    def test_variable_key_none_generic_name(self):
        """No variable_key → generic name."""
        assert resolve_variable_name(5, {}, None) == "object_5"

    def test_obj_index_in_generic_name(self):
        """Generic name uses the object index, not position in var_indices."""
        assert resolve_variable_name(3, {}, None) == "object_3"


# ---------------------------------------------------------------------------
# Multi-message merge with different base structures
# ---------------------------------------------------------------------------


class TestMultiMessageMerge:
    """Messages with different base entry structures."""

    def test_messages_with_different_keys_stack(self, tmp_path: Path):
        """Messages where base entries have extra/different keys.

        The extra keys become Dataset attributes (constant) or outer
        dimensions (varying).
        """
        path = str(tmp_path / "diff_keys.tgm")
        with tensogram.TensogramFile.create(path) as f:
            # Message 0: has 'source' key
            f.append(
                {"version": 2, "base": [{"name": "temp", "source": "model"}]},
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32))],
            )
            # Message 1: has 'source' key with different value
            f.append(
                {"version": 2, "base": [{"name": "temp", "source": "obs"}]},
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32) * 2)],
            )

        datasets = open_datasets(path)
        assert len(datasets) >= 1

    def test_message_with_extra_keys_become_attrs(self, tmp_path: Path):
        """Message-level extra keys appear as Dataset attributes."""
        path = str(tmp_path / "extra_keys.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {"version": 2, "experiment": "test42"},
                [(_desc([3, 4]), np.ones((3, 4), dtype=np.float32))],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds.attrs.get("experiment") == "test42"

    def test_message_with_zero_objects(self, tmp_path: Path):
        """Message with zero data objects returns an empty list of datasets."""
        path = str(tmp_path / "zero_obj.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2, "signal": "start"}, [])

        datasets = open_datasets(path)
        # Zero data objects → empty list (no datasets to construct)
        assert isinstance(datasets, list)
        assert len(datasets) == 0


# ---------------------------------------------------------------------------
# Lazy loading shape/dtype sourcing
# ---------------------------------------------------------------------------


class TestLazyLoadingSources:
    """BackendArray gets ndim/shape/dtype from DataObjectDescriptor, not base."""

    def test_shape_from_descriptor_not_base(self, tmp_path: Path):
        """Shape comes from the descriptor, not from base metadata."""
        path = str(tmp_path / "shape_source.tgm")
        data = np.arange(20, dtype=np.float64).reshape(4, 5)
        meta = {"version": 2, "base": [{"tag": "field"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5], dtype="float64"), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        # Without variable_key, name is generic "object_0"
        var = ds["object_0"]
        assert var.shape == (4, 5)
        assert var.dtype == np.float64

    def test_partial_decode_after_refactor(self, tmp_path: Path):
        """Partial decode (slicing) still works after metadata refactor."""
        path = str(tmp_path / "partial.tgm")
        data = np.arange(60, dtype=np.float32).reshape(6, 10)
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(_desc([6, 10]), data)])

        ds = xr.open_dataset(path, engine="tensogram", range_threshold=1.0)
        # Slice a subset
        sliced = ds["object_0"][1:3, 2:5].values
        np.testing.assert_array_equal(sliced, data[1:3, 2:5])


# ---------------------------------------------------------------------------
# Coord detection edge cases
# ---------------------------------------------------------------------------


class TestCoordDetectionEdges:
    """Edge cases in coordinate auto-detection."""

    def test_coord_name_case_insensitive(self, tmp_path: Path):
        """Coordinate names are matched case-insensitively."""
        path = str(tmp_path / "case_coords.tgm")
        lat = np.linspace(-90, 90, 3, dtype=np.float64)
        data = np.ones((3, 4), dtype=np.float32)

        meta = {
            "version": 2,
            "base": [
                {"name": "LATITUDE"},  # uppercase
                {"name": "field"},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([3], dtype="float64"), lat),
                    (_desc([3, 4]), data),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert "latitude" in ds.coords

    def test_non_coord_name_becomes_data_var(self, tmp_path: Path):
        """Objects with non-coordinate names become data variables.

        Uses variable_key="name" so the base entry name is used.
        """
        path = str(tmp_path / "non_coord.tgm")
        data = np.ones((3,), dtype=np.float32)
        meta = {"version": 2, "base": [{"name": "custom_field"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3]), data)])

        ds = xr.open_dataset(path, engine="tensogram", variable_key="name")
        assert "custom_field" not in ds.coords
        assert "custom_field" in ds.data_vars


# ---------------------------------------------------------------------------
# Dimension resolution edge cases
# ---------------------------------------------------------------------------


class TestWireFormatEdgeCases:
    """Wire format edge cases for xarray integration."""

    def test_single_element_variable(self, tmp_path: Path):
        """Shape [1] variable opens correctly with xarray."""
        path = str(tmp_path / "single_elem.tgm")
        data = np.array([42.0], dtype=np.float32)
        meta = {"version": 2, "base": [{"name": "scalar_like"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([1]), data)])

        ds = xr.open_dataset(path, engine="tensogram", variable_key="name")
        var = ds["scalar_like"]
        assert var.shape == (1,)
        np.testing.assert_array_equal(var.values, data)

    @pytest.mark.skip(
        reason="All-NaN round-trip requires allow_nan bitmask opt-in "
        "(BITMASK_FRAME.md Commit 5). Until then, NaN input is rejected at encode."
    )
    def test_all_nan_float32_preserved(self, tmp_path: Path):
        """All-NaN float32 array opens correctly with xarray, NaN preserved."""
        path = str(tmp_path / "all_nan.tgm")
        data = np.full((3, 4), np.nan, dtype=np.float32)
        meta = {"version": 2, "base": [{"name": "nan_field"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([3, 4]), data)])

        ds = xr.open_dataset(path, engine="tensogram", variable_key="name")
        var = ds["nan_field"]
        assert var.shape == (3, 4)
        assert var.dtype == np.float32
        assert np.all(np.isnan(var.values))

    def test_file_with_20_messages_open_datasets_count(self, tmp_path: Path):
        """File with 20 messages → open_datasets returns correct count.

        open_datasets groups messages by compatible shape/dtype/name and
        stacks along varying metadata keys.  Here 'step' varies, so the
        20 messages form a hypercube along 'step'.
        """
        path = str(tmp_path / "twenty_msgs.tgm")
        n_messages = 20
        with tensogram.TensogramFile.create(path) as f:
            for i in range(n_messages):
                data = np.full((3, 4), float(i), dtype=np.float32)
                meta = {
                    "version": 2,
                    "base": [{"name": "temp", "step": i}],
                }
                f.append(meta, [(_desc([3, 4]), data)])

        datasets = open_datasets(path)
        assert isinstance(datasets, list)
        assert len(datasets) >= 1

        # All 20 messages should be present in the dataset(s).
        # With varying 'step', they form a hypercube with step as outer dim.
        # The 'step' dim should have size 20.
        total_steps = 0
        for ds in datasets:
            if "step" in ds.sizes:
                total_steps += ds.sizes["step"]
            else:
                # No step dim means each dataset has 1 message
                total_steps += len(ds.data_vars)
        assert total_steps == n_messages


class TestDimResolutionEdges:
    """Edge cases in dimension name resolution."""

    def test_ambiguous_size_match(self, tmp_path: Path):
        """When two coords have the same size, the first match is used
        and subsequent axes get the second coord name."""
        path = str(tmp_path / "ambiguous.tgm")
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        depth_arr = np.linspace(0, 1000, 5, dtype=np.float64)
        data = np.ones((5, 5), dtype=np.float32)

        # Coord detection uses "name" key in per-object metadata
        meta = {
            "version": 2,
            "base": [
                {"name": "latitude"},
                {"name": "depth"},
                {"tag": "data_field"},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([5], dtype="float64"), depth_arr),
                    (_desc([5, 5]), data),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        # The data variable is "object_2" (no variable_key set)
        dims = ds["object_2"].dims
        # One axis should get 'latitude', other 'depth'
        assert "latitude" in dims
        assert "depth" in dims

    def test_no_matching_coord_uses_generic(self, tmp_path: Path):
        """When no coord has matching size, generic dim names are used."""
        path = str(tmp_path / "no_match.tgm")
        lat = np.linspace(-90, 90, 3, dtype=np.float64)
        data = np.ones((7, 9), dtype=np.float32)

        meta = {
            "version": 2,
            "base": [
                {"name": "latitude"},
                {"tag": "data_field"},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([3], dtype="float64"), lat),
                    (_desc([7, 9]), data),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        # Data var is "object_1" (no variable_key)
        dims = ds["object_1"].dims
        assert dims == ("dim_0", "dim_1")
