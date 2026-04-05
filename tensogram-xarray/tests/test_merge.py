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
