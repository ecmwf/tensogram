# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for dimension and variable name mapping."""

from __future__ import annotations

import pytest
from tensogram_xarray.mapping import resolve_dim_names, resolve_variable_name


class TestResolveDimNames:
    """Unit tests for resolve_dim_names."""

    def test_user_names(self):
        assert resolve_dim_names(3, ["x", "y", "z"]) == ["x", "y", "z"]

    def test_generic_names(self):
        assert resolve_dim_names(3, None) == ["dim_0", "dim_1", "dim_2"]

    def test_zero_dims(self):
        assert resolve_dim_names(0, None) == []
        assert resolve_dim_names(0, []) == []

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="dim_names has 2 entries"):
            resolve_dim_names(3, ["x", "y"])


class TestResolveVariableName:
    """Unit tests for resolve_variable_name."""

    def test_simple_key(self):
        meta = {"param": "2t"}
        assert resolve_variable_name(0, meta, "param") == "2t"

    def test_dotted_key(self):
        meta = {"mars": {"param": "10u"}}
        assert resolve_variable_name(0, meta, "mars.param") == "10u"

    def test_missing_key_falls_through_priority_chain(self):
        """When explicit variable_key is not found, the priority chain
        (name → mars.param → param → mars.shortName → shortName) is tried."""
        meta = {"name": "something"}
        # mars.param not found, but "name" is in the priority chain → "something"
        assert resolve_variable_name(5, meta, "mars.param") == "something"

    def test_missing_key_all_absent_fallback(self):
        """When no priority-chain key is found, falls back to object_<index>."""
        meta = {"other_key": "value"}
        assert resolve_variable_name(5, meta, "mars.param") == "object_5"

    def test_no_variable_key_uses_priority_chain(self):
        """Without variable_key, the priority chain is still applied."""
        meta = {"param": "2t"}
        assert resolve_variable_name(3, meta, None) == "2t"

    def test_no_variable_key_no_match(self):
        """Without variable_key and no priority-chain match, falls back."""
        meta = {"other_key": "value"}
        assert resolve_variable_name(3, meta, None) == "object_3"

    def test_deeply_nested_key(self):
        meta = {"a": {"b": {"c": "deep_value"}}}
        assert resolve_variable_name(0, meta, "a.b.c") == "deep_value"
