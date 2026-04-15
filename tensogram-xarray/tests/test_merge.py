# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for auto-merge and auto-split of multi-message files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from tensogram_xarray.merge import open_datasets


class TestOpenDatasetsMultiMessage:
    """Multi-message file grouping."""

    def test_returns_list(self, multi_msg_tgm: Path):
        datasets = open_datasets(str(multi_msg_tgm))
        assert isinstance(datasets, list)
        assert len(datasets) >= 1

    def test_all_data_present(self, multi_msg_tgm: Path):
        datasets = open_datasets(str(multi_msg_tgm))
        # Should have data from all 4 messages somewhere.
        total_vars = sum(len(ds.data_vars) for ds in datasets)
        assert total_vars >= 1

    def test_with_variable_key(self, multi_msg_tgm: Path):
        datasets = open_datasets(str(multi_msg_tgm), variable_key="mars.param")
        # With variable_key, param values become variable names.
        all_var_names: set[str] = set()
        for ds in datasets:
            all_var_names.update(ds.data_vars)
        # Should have 2t and/or 10u as variable names.
        assert "2t" in all_var_names or "10u" in all_var_names


class TestAutoSplit:
    """Heterogeneous files should be split into compatible groups."""

    def test_different_shapes_split(self, heterogeneous_tgm: Path):
        datasets = open_datasets(str(heterogeneous_tgm))
        # Should produce at least 2 groups (3x4 float32 vs 5 int32).
        assert len(datasets) >= 2

    def test_shapes_correct(self, heterogeneous_tgm: Path):
        datasets = open_datasets(str(heterogeneous_tgm))
        # Collect all variable shapes across datasets.
        shapes = set()
        for ds in datasets:
            for var in ds.data_vars.values():
                shapes.add(var.shape)
        # Expect shapes from both groups to be present.
        assert len(shapes) >= 2, f"Expected >=2 unique shapes, got {shapes}"


class TestEmptyFile:
    """Edge case: empty file or file with zero objects."""

    def test_empty_returns_list(self, tmp_path: Path):
        import tensogram

        path = tmp_path / "empty.tgm"
        with tensogram.TensogramFile.create(str(path)) as f:
            # Append a zero-object message.
            meta = {"version": 2}
            f.append(meta, [])
        datasets = open_datasets(str(path))
        assert isinstance(datasets, list)


class TestSingleMessageMerge:
    """Single-message file with merge_objects flag."""

    def test_merge_single_message(self, simple_tgm: Path, simple_data: np.ndarray):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram", merge_objects=True)
        assert "object_0" in ds.data_vars
        np.testing.assert_array_equal(ds["object_0"].values, simple_data)


class TestHypercubeDataCorrectness:
    """Verify stacked hypercube data values are placed in correct positions.

    Regression test for the StackedBackendArray unravel bug where
    column-major unraveling placed elements in wrong positions for
    multi-dimensional outer shapes.
    """

    def test_2d_outer_shape_values_correct(self, multi_msg_tgm: Path):
        """multi_msg_tgm has param x date = 2x2 outer shape.

        With variable_key="mars.param", each variable (2t, 10u) gets a
        1-D outer shape from the date dimension. This tests stacking with
        a single varying key.
        """
        import tensogram

        datasets = open_datasets(str(multi_msg_tgm), variable_key="mars.param")
        assert len(datasets) == 1
        ds = datasets[0]

        # Read expected values directly from each message.
        with tensogram.TensogramFile.open(str(multi_msg_tgm)) as f:
            expected = {}
            for msg_idx in range(len(f)):
                raw = f.read_message(msg_idx)
                meta, descs_and_data = tensogram.decode(raw)
                _desc, arr = descs_and_data[0]
                # Per-object metadata is in meta.base[0]
                base_entry = meta.base[0] if meta.base else {}
                mars = base_entry.get("mars", {})
                param = mars.get("param", "")
                date = mars.get("date", "")
                expected[(param, date)] = np.asarray(arr)

        expected_dates = ["20260401", "20260402"]

        # Check each variable's stacked values.
        for var_name in ds.data_vars:
            var = ds[var_name]
            vals = var.values
            # The variable should have an outer dimension from date.
            assert vals.ndim == 3, (
                f"Expected {var_name} to be stacked as 3-D (date, y, x), got shape {vals.shape}"
            )
            # Outer dim is date (2 values), inner is (3,4).
            assert vals.shape[0] == len(expected_dates), (
                f"Expected {var_name} to have {len(expected_dates)} date slices, "
                f"got shape {vals.shape}"
            )
            for date_idx, date_val in enumerate(expected_dates):
                key = (var_name, date_val)
                assert key in expected, (
                    f"Missing expected source message for {var_name} date={date_val}"
                )
                np.testing.assert_array_equal(
                    vals[date_idx],
                    expected[key],
                    err_msg=f"Mismatch for {var_name} date={date_val}",
                )

    def test_stacked_backend_array_2d_outer(self):
        """Direct unit test: StackedBackendArray with 2-D outer shape.

        Creates mock backing arrays with known values and verifies
        the full stacked result has every element in the right position.
        """
        from unittest.mock import MagicMock

        from tensogram_xarray.array import StackedBackendArray

        # 2x3 outer shape, (2,2) inner shape => total (2,3,2,2)
        outer_shape = (2, 3)
        inner_shape = (2, 2)
        dtype = np.dtype("float32")

        # Create 6 mock backing arrays with distinct fill values.
        arrays = []
        for i in range(6):
            arr = MagicMock()
            arr.file_path = "/fake"
            arr.shape = inner_shape
            arr.dtype = dtype
            # Each backing array returns a constant fill for identification.
            fill = np.full(inner_shape, float(i), dtype=dtype)
            arr._raw_indexing_method = MagicMock(return_value=fill)
            arrays.append(arr)

        stacked = StackedBackendArray(arrays, outer_shape, inner_shape, dtype)
        assert stacked.shape == (2, 3, 2, 2)

        # Full read: key = all slices.
        full_key = (slice(None), slice(None), slice(None), slice(None))
        result = stacked._raw_indexing_method(full_key)

        # Verify: result[i, j] should contain float(i*3 + j) everywhere.
        for i in range(2):
            for j in range(3):
                expected_val = float(i * 3 + j)
                np.testing.assert_array_equal(
                    result[i, j],
                    np.full(inner_shape, expected_val, dtype=dtype),
                    err_msg=f"Wrong data at outer position [{i},{j}]",
                )
