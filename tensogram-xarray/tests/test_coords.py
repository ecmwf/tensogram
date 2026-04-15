# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for coordinate auto-detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from tensogram_xarray.coords import KNOWN_COORD_NAMES, detect_coords


class TestDetectCoords:
    """Unit tests for the detect_coords function."""

    def test_lat_lon_detected(self):
        metas = [
            {"name": "latitude"},
            {"name": "longitude"},
            {"name": "temperature"},
        ]
        coord_idx, var_idx, dim_map = detect_coords(metas)
        assert coord_idx == [0, 1]
        assert var_idx == [2]
        assert dim_map == {0: "latitude", 1: "longitude"}

    def test_case_insensitive(self):
        metas = [{"name": "LAT"}, {"name": "LON"}, {"name": "data"}]
        coord_idx, _var_idx, dim_map = detect_coords(metas)
        assert coord_idx == [0, 1]
        assert dim_map == {0: "latitude", 1: "longitude"}

    def test_param_key(self):
        metas = [{"param": "time"}, {"param": "temperature"}]
        coord_idx, var_idx, _ = detect_coords(metas)
        assert coord_idx == [0]
        assert var_idx == [1]

    def test_mars_param_key(self):
        metas = [{"mars": {"param": "level"}}, {"mars": {"param": "wind"}}]
        coord_idx, var_idx, _ = detect_coords(metas)
        assert coord_idx == [0]
        assert var_idx == [1]

    def test_no_coords(self):
        metas = [{"name": "temp"}, {"name": "wind"}]
        coord_idx, var_idx, _ = detect_coords(metas)
        assert coord_idx == []
        assert var_idx == [0, 1]

    def test_all_known_names_recognized(self):
        for name in KNOWN_COORD_NAMES:
            coord_idx, _, _ = detect_coords([{"name": name}])
            assert coord_idx == [0], f"Failed for {name}"


class TestCoordIntegration:
    """Integration: coordinates appear in the opened Dataset."""

    def test_coords_detected_in_dataset(
        self,
        tgm_with_coords: Path,
        coord_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ):
        lat, lon, _temp = coord_arrays
        ds = xr.open_dataset(str(tgm_with_coords), engine="tensogram")

        # Coordinates should be present.
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords

        np.testing.assert_allclose(ds.coords["latitude"].values, lat)
        np.testing.assert_allclose(ds.coords["longitude"].values, lon)

    def test_data_var_uses_coord_dims(
        self,
        tgm_with_coords: Path,
    ):
        ds = xr.open_dataset(str(tgm_with_coords), engine="tensogram")

        # The temperature variable should use latitude/longitude dims.
        temp_var = [v for v in ds.data_vars if v != "latitude" and v != "longitude"]
        assert len(temp_var) >= 1
        var = ds[temp_var[0]]
        assert "latitude" in var.dims or "longitude" in var.dims
