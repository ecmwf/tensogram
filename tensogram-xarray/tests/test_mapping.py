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

    def test_missing_key_fallback(self):
        meta = {"name": "something"}
        assert resolve_variable_name(5, meta, "mars.param") == "object_5"

    def test_no_variable_key(self):
        meta = {"param": "2t"}
        assert resolve_variable_name(3, meta, None) == "object_3"

    def test_deeply_nested_key(self):
        meta = {"a": {"b": {"c": "deep_value"}}}
        assert resolve_variable_name(0, meta, "a.b.c") == "deep_value"
