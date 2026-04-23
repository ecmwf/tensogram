# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""The FieldList path for tensogram files carrying MARS metadata.

Contract:

* ``.to_fieldlist()`` returns an :class:`earthkit.data.FieldList`
  containing one :class:`Field` per MARS data object.  Coordinate
  objects (entries without a ``mars`` sub-map) are skipped.
* Each field exposes MARS keys via ``.metadata(...)`` and ``.get(...)``.
* ``.sel(param="2t")`` returns only fields matching that MARS value.
* ``.to_xarray()`` on the FieldList delegates transparently to
  tensogram-xarray (same Dataset as the non-MARS path).
"""

from __future__ import annotations

import earthkit.data as ekd
import numpy as np
import pytest
from earthkit.data.core.fieldlist import FieldList


class TestFieldListConstruction:
    def test_to_fieldlist_returns_fieldlist(self, mars_tensogram_file) -> None:
        data = ekd.from_source("tensogram", str(mars_tensogram_file))
        fl = data.to_fieldlist()
        assert isinstance(fl, FieldList)

    def test_skips_coord_objects(self, mars_tensogram_file) -> None:
        """The fixture encodes 2 coordinate + 2 MARS objects → 2 fields."""
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        assert len(fl) == 2

    def test_field_shape_matches_descriptor(self, mars_tensogram_file) -> None:
        """The fixture's MARS fields are (4, 6) float32 grids."""
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        for field in fl:
            assert field.shape == (4, 6)

    def test_field_values_match_fixture_data(self, mars_tensogram_file) -> None:
        """Values round-trip bit-for-bit."""
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        # The fixture encodes field 0 as 273.15 + arange(24) reshape(4, 6)
        expected = np.full((4, 6), 273.15, dtype=np.float32) + np.arange(
            24, dtype=np.float32
        ).reshape(4, 6)
        # Figure out which field is 2t and compare
        found = False
        for field in fl:
            if field.metadata("param") == "2t":
                arr = field.to_numpy()
                # ArrayField stores the decoded tensor directly — compare shape first.
                assert arr.shape == (4, 6) or arr.shape == (24,)
                # If flattened, reshape before compare.
                np.testing.assert_allclose(arr.reshape(4, 6), expected)
                found = True
                break
        assert found, "expected a '2t' field in FieldList"


class TestFieldListMetadata:
    def test_metadata_exposes_mars_keys(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        params = [f.metadata("param") for f in fl]
        # Fixture has '2t' and 'tp' as MARS params.
        assert set(params) == {"2t", "tp"}

    def test_get_method_returns_scalar_per_field(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        steps = [f.metadata("step") for f in fl]
        # Fixture: step 0 for 2t, step 6 for tp.
        assert sorted(steps) == [0, 6]


class TestFieldListSelection:
    def test_sel_param_filters(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        subset = fl.sel(param="2t")
        assert len(subset) == 1
        assert subset[0].metadata("param") == "2t"

    def test_sel_empty_filter_returns_empty(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        subset = fl.sel(param="nonexistent")
        assert len(subset) == 0


class TestFieldListToXarray:
    """FieldList must delegate ``to_xarray`` to the tensogram-xarray backend.

    This is the invariant the design plan locked in: coordinate detection
    and dim-name resolution live in exactly one place.
    """

    def test_fieldlist_to_xarray_matches_backend(self, mars_tensogram_file) -> None:
        import xarray as xr

        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        ds_fl = fl.to_xarray()

        ds_backend = xr.open_dataset(str(mars_tensogram_file), engine="tensogram")

        # Same variable set, same values.
        assert set(ds_fl.data_vars) == set(ds_backend.data_vars)
        for name in ds_fl.data_vars:
            np.testing.assert_array_equal(
                ds_fl[name].values, ds_backend[name].values, err_msg=name
            )


class TestFieldListSourceSurface:
    """The FileSource-level ``sel()`` etc. must also work for MARS files."""

    def test_source_level_sel(self, mars_tensogram_file) -> None:
        source = ekd.from_source("tensogram", str(mars_tensogram_file))
        subset = source.sel(param="2t")
        assert len(subset) == 1

    def test_source_level_len(self, mars_tensogram_file) -> None:
        source = ekd.from_source("tensogram", str(mars_tensogram_file))
        # len() on a source forwards to the reader → FieldList len = 2.
        assert len(source) == 2

    def test_source_level_iter(self, mars_tensogram_file) -> None:
        source = ekd.from_source("tensogram", str(mars_tensogram_file))
        fields = list(source)
        assert len(fields) == 2


class TestNonMarsStillRaises:
    """Sanity: the MARS path must not regress the non-MARS guard."""

    def test_nonmars_to_fieldlist_raises(self, nonmars_tensogram_file) -> None:
        data = ekd.from_source("tensogram", str(nonmars_tensogram_file))
        with pytest.raises(NotImplementedError, match="MARS"):
            data.to_fieldlist()
