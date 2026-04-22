# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Issue #67 regression: xarray must continue merging base[i] with desc.params.

This pins the pre-existing ``scanner._merge_per_object_meta`` behaviour so
it cannot silently regress.  The zarr backend was recently taught the
same merge (see ``tensogram_zarr.store._merge_base_with_desc_params``);
the two paths must stay in sync.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import tensogram
import xarray as xr


def _write(path: str, meta: dict, objs: list) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, objs)


class TestXarrayFallbackUnchanged:
    def test_descriptor_name_surfaces_as_variable_name(self, tmp_path: Path):
        path = str(tmp_path / "descriptor_name.tgm")
        meta = {"version": 2}
        objects = [
            (
                {
                    "type": "ntensor",
                    "shape": [10, 8],
                    "dtype": "float32",
                    "name": "temperature",
                },
                np.random.rand(10, 8).astype("float32"),
            ),
            (
                {
                    "type": "ntensor",
                    "shape": [10, 8],
                    "dtype": "float32",
                    "name": "humidity",
                },
                np.random.rand(10, 8).astype("float32"),
            ),
        ]
        _write(path, meta, objects)

        ds = xr.open_dataset(path, engine="tensogram")
        assert sorted(ds.data_vars) == ["humidity", "temperature"]

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

        ds = xr.open_dataset(path, engine="tensogram")
        assert list(ds.data_vars) == ["from_base"]
