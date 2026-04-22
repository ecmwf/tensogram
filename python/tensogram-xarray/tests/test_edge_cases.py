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
from pathlib import Path

import numpy as np
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

    def test_all_nan_float32_preserved(self, tmp_path: Path):
        """All-NaN float32 array opens correctly with xarray, NaN preserved."""
        path = str(tmp_path / "all_nan.tgm")
        data = np.full((3, 4), np.nan, dtype=np.float32)
        meta = {"version": 2, "base": [{"name": "nan_field"}]}
        with tensogram.TensogramFile.create(path) as f:
            # allow_nan=True opts into the NaN companion-mask wire
            # format (see docs/src/guide/nan-inf-handling.md).
            f.append(meta, [(_desc([3, 4]), data)], allow_nan=True)

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


# ---------------------------------------------------------------------------
# Mixed-rank dim collision (issue #66)
# ---------------------------------------------------------------------------


class TestMixedRankDimCollision:
    """Issue #66: xr.open_dataset must not crash on mixed-dimensionality
    messages when non-coordinate objects of different shapes would otherwise
    fall back to the same generic ``dim_N`` name.
    """

    def test_issue_66_repro(self, tmp_path: Path):
        """Exact shape profile from the bug report opens without ValueError.

        A 3-D variable alongside three 1-D arrays whose names fall outside
        the CF coord allowlist used to raise::

            ValueError: conflicting sizes for dimension 'dim_0':
              length X on 'nx' and length Y on {'dim_0': 'count_flash_all', ...}

        With collision-aware fallback disambiguation, the Dataset opens and
        each clashing ``dim_0`` axis is renamed to ``obj_{i}_dim_0``.
        """
        path = str(tmp_path / "issue_66.tgm")

        t_size, y_size, x_size = 4, 5, 6
        data_3d = np.ones((t_size, y_size, x_size), dtype=np.float32)
        coord_t = np.arange(t_size, dtype=np.float32)
        coord_y = np.arange(y_size, dtype=np.float32)
        coord_x = np.arange(x_size, dtype=np.float32)

        meta = {
            "version": 2,
            "base": [
                {"name": "reflectance"},
                {"name": "count_flash_all"},
                {"name": "ny"},
                {"name": "nx"},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([t_size, y_size, x_size]), data_3d),
                    (_desc([t_size]), coord_t),
                    (_desc([y_size]), coord_y),
                    (_desc([x_size]), coord_x),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert set(ds.data_vars) == {"reflectance", "count_flash_all", "ny", "nx"}
        assert ds["reflectance"].dims == ("obj_0_dim_0", "dim_1", "dim_2")
        assert ds["count_flash_all"].dims == ("obj_1_dim_0",)
        assert ds["ny"].dims == ("obj_2_dim_0",)
        assert ds["nx"].dims == ("obj_3_dim_0",)

    def test_only_conflicting_axes_renamed(self, tmp_path: Path):
        """Non-conflicting generic axes must keep their ``dim_N`` names."""
        path = str(tmp_path / "partial.tgm")

        meta = {"version": 2, "base": [{"name": "a"}, {"name": "b"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([4, 9]), np.zeros((4, 9), dtype=np.float32)),
                    (_desc([7, 9]), np.zeros((7, 9), dtype=np.float32)),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["a"].dims == ("obj_0_dim_0", "dim_1")
        assert ds["b"].dims == ("obj_1_dim_0", "dim_1")

    def test_no_collision_keeps_generic_names(self, tmp_path: Path):
        """Regression guard: same-shape vars share ``dim_N`` compatibly."""
        path = str(tmp_path / "no_collision.tgm")

        meta = {"version": 2, "base": [{"name": "a"}, {"name": "b"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([4, 5]), np.zeros((4, 5), dtype=np.float32)),
                    (_desc([4, 5]), np.zeros((4, 5), dtype=np.float32)),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["a"].dims == ("dim_0", "dim_1")
        assert ds["b"].dims == ("dim_0", "dim_1")

    def test_coord_untouched_by_disambiguation(self, tmp_path: Path):
        """Detected coord dims never get renamed by the disambiguation pass."""
        path = str(tmp_path / "coord_preserved.tgm")

        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        data_a = np.ones((5, 7), dtype=np.float32)
        data_b = np.ones((3,), dtype=np.float32)

        meta = {
            "version": 2,
            "base": [
                {"name": "latitude"},
                {"name": "field_a"},
                {"name": "field_b"},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([5, 7]), data_a),
                    (_desc([3]), data_b),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field_a"].dims == ("latitude", "dim_1")
        assert ds["field_b"].dims == ("dim_0",)

    def test_disambiguate_helper_pure(self):
        """Unit-level check of :func:`_disambiguate_fallback_dims`."""
        from tensogram_xarray.store import _DataVarPlan, _disambiguate_fallback_dims

        plans = [
            _DataVarPlan(
                obj_index=0,
                var_name="a",
                shape=(3,),
                dims_with_provenance=[("dim_0", True)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
            _DataVarPlan(
                obj_index=1,
                var_name="b",
                shape=(5,),
                dims_with_provenance=[("dim_0", True)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
        ]
        assert _disambiguate_fallback_dims(plans, {}) == [
            ("obj_0_dim_0",),
            ("obj_1_dim_0",),
        ]

    def test_disambiguate_hinted_conflict_warns_and_falls_back(self, caplog):
        """Non-generic name claimed at different sizes across objects warns
        and falls back to per-object names rather than crashing at Dataset
        assembly (producer-error safety net)."""
        import logging

        from tensogram_xarray.store import _DataVarPlan, _disambiguate_fallback_dims

        plans = [
            _DataVarPlan(
                obj_index=0,
                var_name="a",
                shape=(3,),
                dims_with_provenance=[("time", False)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
            _DataVarPlan(
                obj_index=1,
                var_name="b",
                shape=(5,),
                dims_with_provenance=[("time", False)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
        ]
        with caplog.at_level(logging.WARNING, logger="tensogram_xarray.store"):
            resolved = _disambiguate_fallback_dims(plans, {})
        assert resolved == [("obj_0_dim_0",), ("obj_1_dim_0",)]
        assert any("'time'" in r.message for r in caplog.records)
        assert sum("'time'" in r.message for r in caplog.records) == 1, (
            "warning must be emitted once per conflicting hint, not per offending axis"
        )

    def test_disambiguate_hinted_same_size_preserved(self):
        """Non-generic names that agree on size are never renamed — hint
        sharing across objects is the whole point of per-object dim_names."""
        from tensogram_xarray.store import _DataVarPlan, _disambiguate_fallback_dims

        plans = [
            _DataVarPlan(
                obj_index=0,
                var_name="a",
                shape=(5,),
                dims_with_provenance=[("time", False)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
            _DataVarPlan(
                obj_index=1,
                var_name="b",
                shape=(5,),
                dims_with_provenance=[("time", False)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
        ]
        assert _disambiguate_fallback_dims(plans, {}) == [("time",), ("time",)]

    def test_disambiguate_considers_coord_sizes(self):
        """Coord dim sizes participate in conflict detection but coord names
        themselves are never renamed."""
        from tensogram_xarray.store import _DataVarPlan, _disambiguate_fallback_dims

        plans = [
            _DataVarPlan(
                obj_index=0,
                var_name="a",
                shape=(7,),
                dims_with_provenance=[("latitude", True)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
        ]
        resolved = _disambiguate_fallback_dims(plans, {"latitude": 5})
        assert resolved == [("obj_0_dim_0",)]

    def test_disambiguate_preserves_legit_coord_match_under_conflict(self):
        """An axis that legitimately matches a coord (same name AND same size)
        must survive disambiguation even when another plan wrongly claims the
        same coord name at a different size — the coord-sharing binding is
        the correct semantics and only the offender should be renamed."""
        from tensogram_xarray.store import _DataVarPlan, _disambiguate_fallback_dims

        plans = [
            _DataVarPlan(
                obj_index=0,
                var_name="legit",
                shape=(5, 7),
                dims_with_provenance=[("latitude", False), ("dim_1", True)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
            _DataVarPlan(
                obj_index=1,
                var_name="wrong",
                shape=(7,),
                dims_with_provenance=[("latitude", False)],
                backend_array=None,  # type: ignore[arg-type]
                var_attrs={},
            ),
        ]
        resolved = _disambiguate_fallback_dims(plans, {"latitude": 5})
        assert resolved[0] == ("latitude", "dim_1")
        assert resolved[1] == ("obj_1_dim_0",)

    def test_build_dataset_shares_coord_when_other_hint_is_wrong(self, tmp_path: Path):
        """End-to-end: a data var legitimately using coord 'latitude' keeps
        that dim when another var's bad hint triggers disambiguation — and
        xarray actually treats the shared dim as aligned for indexing."""
        path = str(tmp_path / "coord_shared.tgm")
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        legit = np.arange(35, dtype=np.float32).reshape(5, 7)
        wrong = np.ones((7,), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [
                {"name": "latitude"},
                {"name": "legit"},
                {"name": "wrong", "dim_names": ["latitude"]},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([5, 7]), legit),
                    (_desc([7]), wrong),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["legit"].dims[0] == "latitude"
        assert ds["wrong"].dims == ("obj_2_dim_0",)
        assert ds.coords["latitude"].shape == (5,)
        # Runtime alignment: indexing via the shared coord must work.
        slice_at_lat0 = ds["legit"].isel(latitude=0).values
        np.testing.assert_array_equal(slice_at_lat0, legit[0])
        assert ds["legit"].sel(latitude=lat[2]).values.tolist() == legit[2].tolist()


# ---------------------------------------------------------------------------
# Per-object dim_names hint (base[i]["dim_names"])
# ---------------------------------------------------------------------------


class TestPerObjectDimNames:
    """The per-object ``base[i]["dim_names"]`` opt-in reader convention.

    Producers may embed an axis-ordered list of dim names in each base
    entry so mixed-rank messages open with semantically meaningful dims
    without requiring callers to pass ``dim_names=`` or the message-level
    ``_extra_["dim_names"]`` fallback.
    """

    def test_hint_applied_simple(self, tmp_path: Path):
        """Per-object hint drives dim names for a single variable."""
        path = str(tmp_path / "po_simple.tgm")
        data = np.ones((4, 5), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": ["time", "level"]}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("time", "level")

    def test_hint_applied_mixed_ranks(self, tmp_path: Path):
        """Mixed-rank message with per-object hints opens with semantic dims."""
        path = str(tmp_path / "po_mixed.tgm")
        t_size, y_size, x_size = 4, 5, 6
        meta = {
            "version": 2,
            "base": [
                {"name": "reflectance", "dim_names": ["time", "y", "x"]},
                {"name": "count", "dim_names": ["time"]},
                {"name": "ny", "dim_names": ["y"]},
                {"name": "nx", "dim_names": ["x"]},
            ],
        }
        data_3d = np.ones((t_size, y_size, x_size), dtype=np.float32)
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([t_size, y_size, x_size]), data_3d),
                    (_desc([t_size]), np.arange(t_size, dtype=np.float32)),
                    (_desc([y_size]), np.arange(y_size, dtype=np.float32)),
                    (_desc([x_size]), np.arange(x_size, dtype=np.float32)),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["reflectance"].dims == ("time", "y", "x")
        assert ds["count"].dims == ("time",)
        assert ds["ny"].dims == ("y",)
        assert ds["nx"].dims == ("x",)

    def test_shared_dim_across_objects(self, tmp_path: Path):
        """Two objects pinning matching-size axes to the same name share the dim."""
        path = str(tmp_path / "po_shared.tgm")
        meta = {
            "version": 2,
            "base": [
                {"name": "temp", "dim_names": ["time", "level"]},
                {"name": "humid", "dim_names": ["time", "level"]},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([4, 3]), np.ones((4, 3), dtype=np.float32)),
                    (_desc([4, 3]), np.ones((4, 3), dtype=np.float32)),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["temp"].dims == ("time", "level")
        assert ds["humid"].dims == ("time", "level")
        assert ds.sizes == {"time": 4, "level": 3}

    def test_user_kwarg_overrides_per_object_hint(self, tmp_path: Path):
        """User-supplied ``dim_names=`` outranks the per-object hint."""
        path = str(tmp_path / "po_vs_user.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": ["producer_x", "producer_y"]}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), np.ones((4, 5), dtype=np.float32))])

        ds = xr.open_dataset(path, engine="tensogram", dim_names=["user_x", "user_y"])
        assert ds["field"].dims == ("user_x", "user_y")

    def test_coord_match_overrides_per_object_hint(self, tmp_path: Path):
        """Detected coord wins over per-object hint on the matching axis."""
        path = str(tmp_path / "po_vs_coord.tgm")
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        data = np.ones((5, 7), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [
                {"name": "latitude"},
                {"name": "field", "dim_names": ["row", "col"]},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([5], dtype="float64"), lat),
                    (_desc([5, 7]), data),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("latitude", "col")

    def test_per_object_overrides_extra_hint(self, tmp_path: Path):
        """Per-object hint outranks the message-level ``_extra_`` hint."""
        path = str(tmp_path / "po_vs_extra.tgm")
        data = np.ones((4, 5), dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": ["per_obj_x", "per_obj_y"]}],
            "_extra_": {"dim_names": ["extra_x", "extra_y"]},
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), data)])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("per_obj_x", "per_obj_y")

    def test_malformed_wrong_length_ignored(self, tmp_path: Path):
        """Hint with wrong number of names silently falls through."""
        path = str(tmp_path / "mal_len.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": ["only_one"]}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), np.ones((4, 5), dtype=np.float32))])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("dim_0", "dim_1")

    def test_malformed_non_list_ignored(self, tmp_path: Path):
        """Scalar/dict hints (wrong type) silently fall through."""
        path = str(tmp_path / "mal_type.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": "not_a_list"}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), np.ones((4, 5), dtype=np.float32))])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("dim_0", "dim_1")

    def test_malformed_duplicates_ignored(self, tmp_path: Path):
        """Duplicate entries in the hint silently fall through."""
        path = str(tmp_path / "mal_dup.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": ["x", "x"]}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), np.ones((4, 5), dtype=np.float32))])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("dim_0", "dim_1")

    def test_malformed_empty_string_ignored(self, tmp_path: Path):
        """Empty-string entries in the hint silently fall through."""
        path = str(tmp_path / "mal_empty.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": ["", "y"]}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), np.ones((4, 5), dtype=np.float32))])

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["field"].dims == ("dim_0", "dim_1")

    def test_malformed_non_string_entry_ignored(self, tmp_path: Path):
        """Non-string entries (ints, None, etc.) in the hint silently fall through."""
        from tensogram_xarray.mapping import parse_per_object_dim_names

        assert parse_per_object_dim_names(2, {"dim_names": ["x", 42]}) is None
        assert parse_per_object_dim_names(2, {"dim_names": ["x", None]}) is None
        assert parse_per_object_dim_names(1, {"dim_names": [42]}) is None

    def test_validator_accepts_zero_dim_empty_list(self):
        """A zero-dim tensor (ndim=0) accepts an empty ``dim_names`` list."""
        from tensogram_xarray.mapping import parse_per_object_dim_names

        assert parse_per_object_dim_names(0, {"dim_names": []}) == []
        assert parse_per_object_dim_names(0, {"dim_names": ["extra"]}) is None

    def test_validator_accepts_tuple(self):
        """Tuple hints (any non-str/bytes Sequence) are accepted."""
        from tensogram_xarray.mapping import parse_per_object_dim_names

        assert parse_per_object_dim_names(2, {"dim_names": ("x", "y")}) == ["x", "y"]

    def test_validator_rejects_string(self):
        """A plain ``str`` is not a valid dim-names sequence."""
        from tensogram_xarray.mapping import parse_per_object_dim_names

        assert parse_per_object_dim_names(2, {"dim_names": "xy"}) is None

    def test_validator_rejects_bytes(self):
        """A ``bytes`` / ``bytearray`` is not a valid dim-names sequence."""
        from tensogram_xarray.mapping import parse_per_object_dim_names

        assert parse_per_object_dim_names(2, {"dim_names": b"xy"}) is None

    def test_dim_names_not_in_var_attrs(self, tmp_path: Path):
        """``dim_names`` is structural metadata; it must not leak into attrs."""
        path = str(tmp_path / "no_leak.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field", "dim_names": ["x", "y"], "units": "K"}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), np.ones((4, 5), dtype=np.float32))])

        ds = xr.open_dataset(path, engine="tensogram")
        assert "dim_names" not in ds["field"].attrs
        assert ds["field"].attrs.get("units") == "K"
        assert ds["field"].dims == ("x", "y")

    def test_extra_dim_names_not_in_dataset_attrs(self, tmp_path: Path):
        """Message-level ``_extra_["dim_names"]`` must not leak into ``ds.attrs``."""
        path = str(tmp_path / "no_leak_extra.tgm")
        meta = {
            "version": 2,
            "base": [{"name": "field"}],
            "_extra_": {"dim_names": ["x", "y"], "experiment": "e42"},
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(_desc([4, 5]), np.ones((4, 5), dtype=np.float32))])

        ds = xr.open_dataset(path, engine="tensogram")
        assert "dim_names" not in ds.attrs
        assert ds.attrs.get("experiment") == "e42"
        assert ds["field"].dims == ("x", "y")


# ---------------------------------------------------------------------------
# Hinted-name conflict (producer error safety net)
# ---------------------------------------------------------------------------


class TestHintedNameConflict:
    """Two objects claiming the same dim name at different sizes must not crash.

    Hinted-name conflicts indicate a producer error.  The backend warns
    and falls back to per-object dim names so the Dataset still opens.
    """

    def test_inconsistent_per_object_hint_same_size_ok(self, tmp_path: Path):
        """Different per-object hints on different-size axes do not conflict."""
        path = str(tmp_path / "hint_distinct.tgm")
        meta = {
            "version": 2,
            "base": [
                {"name": "a", "dim_names": ["alpha"]},
                {"name": "b", "dim_names": ["beta"]},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([4]), np.zeros(4, dtype=np.float32)),
                    (_desc([7]), np.zeros(7, dtype=np.float32)),
                ],
            )

        ds = xr.open_dataset(path, engine="tensogram")
        assert ds["a"].dims == ("alpha",)
        assert ds["b"].dims == ("beta",)

    def test_hinted_conflict_warns_and_falls_back(self, tmp_path: Path, caplog):
        """Same hinted name at different sizes across objects → warn + fallback."""
        import logging

        path = str(tmp_path / "hint_conflict.tgm")
        meta = {
            "version": 2,
            "base": [
                {"name": "a", "dim_names": ["time"]},
                {"name": "b", "dim_names": ["time"]},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    (_desc([4]), np.zeros(4, dtype=np.float32)),
                    (_desc([7]), np.zeros(7, dtype=np.float32)),
                ],
            )

        with caplog.at_level(logging.WARNING, logger="tensogram_xarray.store"):
            ds = xr.open_dataset(path, engine="tensogram")

        assert ds["a"].dims == ("obj_0_dim_0",)
        assert ds["b"].dims == ("obj_1_dim_0",)
        assert any("'time'" in r.message for r in caplog.records)

    def test_hint_colliding_with_coord_size_renames_hinted_axis(self, tmp_path: Path, caplog):
        """A per-object hint that claims an existing coord name at a different
        size triggers warn + fallback for that axis; the coord itself is left
        intact because coord dim names are never auto-renamed."""
        import logging

        path = str(tmp_path / "hint_vs_coord.tgm")
        lat = np.linspace(-90, 90, 5, dtype=np.float64)
        data = np.ones((7,), dtype=np.float32)
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
                    (_desc([7]), data),
                ],
            )

        with caplog.at_level(logging.WARNING, logger="tensogram_xarray.store"):
            ds = xr.open_dataset(path, engine="tensogram")

        assert ds.coords["latitude"].shape == (5,)
        assert ds["field"].dims == ("obj_1_dim_0",)
        assert any("'latitude'" in r.message for r in caplog.records)
