# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the xarray BackendEntrypoint integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from tensogram_xarray.backend import TensogramBackendEntrypoint


class TestGuessCanOpen:
    """``guess_can_open`` file-extension detection."""

    def test_tgm_extension(self):
        backend = TensogramBackendEntrypoint()
        assert backend.guess_can_open("data.tgm") is True

    def test_uppercase_extension(self):
        backend = TensogramBackendEntrypoint()
        assert backend.guess_can_open("data.TGM") is True

    def test_wrong_extension(self):
        backend = TensogramBackendEntrypoint()
        assert backend.guess_can_open("data.grib") is False
        assert backend.guess_can_open("data.nc") is False

    def test_non_string(self):
        backend = TensogramBackendEntrypoint()
        assert backend.guess_can_open(12345) is False  # type: ignore[arg-type]


class TestOpenDatasetSimple:
    """Basic open_dataset: single message, no mapping."""

    def test_opens_simple_tgm(self, simple_tgm: Path, simple_data: np.ndarray):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram")
        assert "object_0" in ds.data_vars
        np.testing.assert_array_equal(ds["object_0"].values, simple_data)

    def test_generic_dim_names(self, simple_tgm: Path):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram")
        assert ds["object_0"].dims == ("dim_0", "dim_1")

    def test_shape_preserved(self, simple_tgm: Path, simple_data: np.ndarray):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram")
        assert ds["object_0"].shape == simple_data.shape

    def test_dtype_preserved(self, simple_tgm: Path):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram")
        assert ds["object_0"].dtype == np.float32

    def test_metadata_in_attrs(self, simple_tgm: Path):
        ds = xr.open_dataset(str(simple_tgm), engine="tensogram")
        assert "tensogram_version" in ds.attrs


class TestOpenDatasetWithDimNames:
    """User-specified dim_names."""

    def test_custom_dim_names(self, simple_tgm: Path):
        ds = xr.open_dataset(
            str(simple_tgm),
            engine="tensogram",
            dim_names=["latitude", "longitude"],
        )
        assert ds["object_0"].dims == ("latitude", "longitude")

    def test_wrong_dim_count_raises(self, simple_tgm: Path):
        with pytest.raises(ValueError, match="dim_names has 3 entries"):
            xr.open_dataset(
                str(simple_tgm),
                engine="tensogram",
                dim_names=["a", "b", "c"],
            )


class TestOpenDatasetWithVariableKey:
    """Variable naming from metadata."""

    def test_variable_key_mars_param(self, tgm_with_mars: Path):
        ds = xr.open_dataset(
            str(tgm_with_mars),
            engine="tensogram",
            variable_key="mars.param",
        )
        assert "2t" in ds.data_vars
        assert "10u" in ds.data_vars

    def test_variable_key_fallback(self, simple_tgm: Path):
        ds = xr.open_dataset(
            str(simple_tgm),
            engine="tensogram",
            variable_key="mars.param",  # doesn't exist in metadata
        )
        # Falls back to object_0.
        assert "object_0" in ds.data_vars


class TestDropVariables:
    """drop_variables support."""

    def test_drop_variable(self, tgm_with_mars: Path):
        ds = xr.open_dataset(
            str(tgm_with_mars),
            engine="tensogram",
            variable_key="mars.param",
            drop_variables=["10u"],
        )
        assert "2t" in ds.data_vars
        assert "10u" not in ds.data_vars


class TestMessageIndex:
    """Selecting a specific message in a multi-message file."""

    def test_message_index(self, multi_msg_tgm: Path):
        ds0 = xr.open_dataset(str(multi_msg_tgm), engine="tensogram", message_index=0)
        ds1 = xr.open_dataset(str(multi_msg_tgm), engine="tensogram", message_index=1)
        # Messages 0 and 1 both have param="2t" (different dates).
        # Variable names come from the priority chain (mars.param → "2t").
        assert "2t" in ds0.data_vars
        assert "2t" in ds1.data_vars
        assert ds0["2t"].shape == ds1["2t"].shape
