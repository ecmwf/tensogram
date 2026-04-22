# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Issue #67 regression: zarr variable naming falls back to descriptor keys.

The Python binding silently routes unknown descriptor keys to
``DataObjectDescriptor.params``.  xarray has long merged ``base[i]`` +
``desc.params``; zarr did not, yielding ``object_0, object_1, ...``
instead of the intended names.  These tests pin the new per-key
precedence: ``base[i]`` wins for the same key, but a higher-priority
key from ``desc.params`` beats a lower-priority key from ``base[i]``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import tensogram
import zarr
from tensogram_zarr import TensogramStore


def _write(path: str, meta: dict, objs: list) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, objs)


@pytest.fixture
def descriptor_name_tgm(tmp_path: Path) -> str:
    path = str(tmp_path / "descriptor_name.tgm")
    meta = {"version": 2}
    objects = [
        (
            {
                "type": "ntensor",
                "shape": [10, 8],
                "dtype": "float32",
                "compression": "zstd",
                "name": "temperature",
            },
            np.random.rand(10, 8).astype("float32"),
        ),
        (
            {
                "type": "ntensor",
                "shape": [10, 8],
                "dtype": "float32",
                "compression": "zstd",
                "name": "humidity",
            },
            np.random.rand(10, 8).astype("float32"),
        ),
    ]
    _write(path, meta, objects)
    return path


class TestIssue67Reproducer:
    def test_keys_match_descriptor_names(self, descriptor_name_tgm: str):
        with TensogramStore(descriptor_name_tgm, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            assert sorted(root.keys()) == ["humidity", "temperature"]

    def test_raw_store_keys_use_descriptor_names(self, descriptor_name_tgm: str):
        with TensogramStore(descriptor_name_tgm, mode="r") as store:
            names = sorted(
                k.split("/")[0]
                for k in store._keys
                if k.endswith("/zarr.json") and k != "zarr.json"
            )
            assert names == ["humidity", "temperature"]


class TestPrecedenceAcrossSources:
    def test_base_wins_over_descriptor_for_same_key(self, tmp_path: Path):
        path = str(tmp_path / "base_wins.tgm")
        meta = {"version": 2, "base": [{"name": "from_base"}]}
        desc = {
            "type": "ntensor",
            "shape": [4, 4],
            "dtype": "float32",
            "name": "from_descriptor",
        }
        _write(path, meta, [(desc, np.zeros((4, 4), dtype=np.float32))])

        with TensogramStore(path, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            assert list(root.keys()) == ["from_base"]

    def test_higher_priority_descriptor_key_beats_lower_priority_base_key(
        self, tmp_path: Path
    ):
        path = str(tmp_path / "priority.tgm")
        meta = {"version": 2, "base": [{"param": "T"}]}
        desc = {
            "type": "ntensor",
            "shape": [4, 4],
            "dtype": "float32",
            "name": "temperature",
        }
        _write(path, meta, [(desc, np.zeros((4, 4), dtype=np.float32))])

        with TensogramStore(path, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            assert list(root.keys()) == ["temperature"]


class TestVariableKeyWithMergedMeta:
    def test_dotted_variable_key_resolves_against_descriptor(self, tmp_path: Path):
        path = str(tmp_path / "namespaced.tgm")
        meta = {"version": 2}
        desc = {
            "type": "ntensor",
            "shape": [4, 4],
            "dtype": "float32",
            "product": {"name": "intensity"},
        }
        _write(path, meta, [(desc, np.zeros((4, 4), dtype=np.float32))])

        with TensogramStore(path, mode="r", variable_key="product.name") as store:
            root = zarr.open_group(store=store, mode="r")
            assert list(root.keys()) == ["intensity"]


class TestPerObjectFallback:
    def test_object_with_base_and_object_without_resolve_independently(
        self, tmp_path: Path
    ):
        path = str(tmp_path / "mixed.tgm")
        meta = {
            "version": 2,
            "base": [
                {"name": "alpha"},
                {},
            ],
        }
        descs_and_data = [
            (
                {"type": "ntensor", "shape": [3], "dtype": "float32"},
                np.zeros(3, dtype=np.float32),
            ),
            (
                {
                    "type": "ntensor",
                    "shape": [3],
                    "dtype": "float32",
                    "name": "beta",
                },
                np.zeros(3, dtype=np.float32),
            ),
        ]
        _write(path, meta, descs_and_data)

        with TensogramStore(path, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            assert sorted(root.keys()) == ["alpha", "beta"]


class TestDuplicateNamesStillSuffixed:
    def test_duplicate_descriptor_names_get_numeric_suffix(self, tmp_path: Path):
        path = str(tmp_path / "duplicates.tgm")
        meta = {"version": 2}
        descs_and_data = [
            (
                {
                    "type": "ntensor",
                    "shape": [2],
                    "dtype": "float32",
                    "name": "temp",
                },
                np.zeros(2, dtype=np.float32),
            ),
            (
                {
                    "type": "ntensor",
                    "shape": [2],
                    "dtype": "float32",
                    "name": "temp",
                },
                np.zeros(2, dtype=np.float32),
            ),
        ]
        _write(path, meta, descs_and_data)

        with TensogramStore(path, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            assert sorted(root.keys()) == ["temp", "temp_1"]


class TestNestedRootIsAtomic:
    """The base+params merge is shallow, root-key atomic: if ``base[i]`` has
    a root key, the entire sub-tree in ``desc.params`` under that key is
    shadowed — no deep merge.  Pinned to lock semantics against future
    accidental deep-merge "fixes".  Matches xarray's ``_merge_per_object_meta``.
    """

    def test_base_mars_without_param_shadows_descriptor_mars_param(self, tmp_path: Path):
        path = str(tmp_path / "nested_shadow.tgm")
        meta = {"version": 2, "base": [{"mars": {"levtype": "sfc"}}]}
        desc = {
            "type": "ntensor",
            "shape": [4, 4],
            "dtype": "float32",
            "mars": {"param": "2t"},
        }
        _write(path, meta, [(desc, np.zeros((4, 4), dtype=np.float32))])

        with TensogramStore(path, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            assert list(root.keys()) == ["object_0"]


class TestExistingBehaviourUnchanged:
    def test_canonical_base_name_still_works(self, tmp_path: Path):
        path = str(tmp_path / "base_only.tgm")
        meta = {"version": 2, "base": [{"name": "canonical"}]}
        desc = {"type": "ntensor", "shape": [4, 4], "dtype": "float32"}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(desc, np.zeros((4, 4), dtype=np.float32))])

        with TensogramStore(path, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            assert list(root.keys()) == ["canonical"]

    def test_unnamed_object_falls_back_to_object_index(self, tmp_path: Path):
        path = str(tmp_path / "unnamed.tgm")
        meta = {"version": 2}
        desc = {"type": "ntensor", "shape": [4, 4], "dtype": "float32"}
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [(desc, np.zeros((4, 4), dtype=np.float32))])

        with TensogramStore(path, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            assert list(root.keys()) == ["object_0"]
