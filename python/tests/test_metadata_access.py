# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Metadata access parity (see plans/METADATA_ACCESS_PARITY.md §6.5).

Covers the new ``Metadata`` Mapping surface and dot-path helpers:
  - ``get`` / ``keys`` / ``values`` / ``items`` / ``__iter__`` / ``__len__``
    consistency, and their agreement with ``in`` / ``__getitem__``.
  - ``has_path`` / ``get_path`` and the per-object ``*_at`` variants.
  - absent-vs-empty (``""`` and ``0`` are found; a missing key gives the
    default / ``False``).
  - per-object scoping (no ``_extra_`` fallback, no cross-object match).
  - ``_reserved_`` hidden from the path getters but visible in ``base[i]``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import tensogram


def _descriptor(shape: list[int]) -> dict:
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": "float32",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


def _build_meta() -> tensogram.Metadata:
    """A two-object message exercising every access edge case.

    ``base[0]`` carries an empty string, a zero int, a ``False`` bool, a
    nested map, an array, a key present only in object 0, and a key that
    also lives in ``_extra_`` (to prove first-match precedence + dedup).
    ``base[1]`` re-uses ``name``/``mars``/``levels`` (dedup) but omits the
    object-0-only keys.  ``_extra_`` adds a zero and an empty string too.
    The top-level ``version`` key flows into ``_extra_``.
    """
    meta = {
        "version": 3,
        "base": [
            {
                "name": "obj0",
                "mars": {"class": "od", "param": "2t"},
                "levels": [1000, 850, 500],
                "empty": "",
                "zero": 0,
                "flag": False,
                "only_in_0": "yes",
                "shared": "from_base",
            },
            {
                "name": "obj1",
                "mars": {"class": "rd", "param": "msl"},
                "levels": [1000],
            },
        ],
        "_extra_": {
            "producer": "test-suite",
            "count": 0,
            "note": "",
            "shared": "from_extra",
        },
    }
    pairs = [
        (_descriptor([3]), np.ones(3, dtype=np.float32)),
        (_descriptor([2]), np.ones(2, dtype=np.float32)),
    ]
    msg = bytes(tensogram.encode(meta, pairs))
    return tensogram.decode_metadata(msg)


# Ordered keys reachable via ``__getitem__``: base[0]'s keys (sorted,
# minus ``_reserved_``) first, then base[1] (nothing new), then _extra_'s
# new keys (sorted).  ``shared`` is de-duplicated to its base position.
EXPECTED_KEYS = [
    "empty",
    "flag",
    "levels",
    "mars",
    "name",
    "only_in_0",
    "shared",
    "zero",
    "count",
    "note",
    "producer",
    "version",
]


@pytest.fixture
def meta() -> tensogram.Metadata:
    return _build_meta()


# ---------------------------------------------------------------------------
# Message-level Mapping surface
# ---------------------------------------------------------------------------


class TestMessageLevelMapping:
    def test_num_objects(self, meta):
        assert meta.num_objects == 2

    def test_keys_exact_order(self, meta):
        """keys() = de-duplicated first-seen union (base then extra)."""
        assert list(meta.keys()) == EXPECTED_KEYS

    def test_keys_return_list(self, meta):
        assert isinstance(meta.keys(), list)
        assert isinstance(meta.values(), list)
        assert isinstance(meta.items(), list)

    def test_len_matches_keys(self, meta):
        assert len(meta) == len(EXPECTED_KEYS)
        assert len(meta) == len(meta.keys())

    def test_iter_matches_keys(self, meta):
        assert list(meta) == list(meta.keys())
        assert list(iter(meta)) == EXPECTED_KEYS

    def test_values_match_getitem(self, meta):
        keys = meta.keys()
        assert meta.values() == [meta[k] for k in keys]

    def test_items_match_keys_and_values(self, meta):
        assert meta.items() == list(zip(meta.keys(), meta.values()))
        for pair in meta.items():
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_get_matches_getitem(self, meta):
        # `keys()` returns a plain list; bind it so the loop exercises the
        # method without tripping SIM118's dict-view heuristic.
        keys = meta.keys()
        for k in keys:
            assert meta.get(k) == meta[k]

    def test_get_agrees_with_get_path_for_flat_keys(self, meta):
        # Flat `get` and single-segment `get_path` agree for ordinary keys.
        # ``version`` is the one deliberate exception (see below): the
        # dot-path walker refuses it because it is a preamble field.
        keys = meta.keys()
        for k in keys:
            if k == "version":
                continue
            assert meta.get(k) == meta.get_path(k)

    def test_version_is_a_plain_key(self, meta):
        # A literal ``version`` CBOR key (here in _extra_) resolves the same way
        # through flat access and the dot-path walker — the walker does not
        # special-case it. The wire-format version is a separate concept,
        # exposed as the ``meta.version`` property (a preamble field).
        assert meta.get("version") == 3  # flat _extra_ key
        assert "version" in meta
        assert meta.get_path("version") == 3  # dot-path agrees
        assert meta.has_path("version") is True
        assert meta.version == 3  # preamble field, distinct concept

    def test_contains_agrees_with_keys(self, meta):
        keys = meta.keys()
        for k in keys:
            assert k in meta
        assert "missing" not in meta

    def test_first_match_precedence_and_dedup(self, meta):
        """``shared`` lives in base[0] and _extra_ → base wins, once."""
        assert list(meta.keys()).count("shared") == 1
        assert meta["shared"] == "from_base"
        assert meta.get("shared") == "from_base"

    def test_base_precedence_over_extra(self, meta):
        # producer only in extra; name/mars only reachable from base.
        assert meta["name"] == "obj0"
        assert meta["producer"] == "test-suite"

    def test_version_flows_into_keys(self, meta):
        assert "version" in meta
        assert meta.get("version") == 3

    def test_dict_from_mapping(self, meta):
        as_dict = dict(meta)
        assert as_dict == {k: meta[k] for k in meta}
        assert as_dict["name"] == "obj0"
        assert as_dict["shared"] == "from_base"

    def test_double_star_unpacking(self, meta):
        assert {**meta} == dict(meta)

    def test_is_mapping_instance(self, meta):
        assert isinstance(meta, Mapping)


# ---------------------------------------------------------------------------
# Absent vs empty vs a value equal to a default
# ---------------------------------------------------------------------------


class TestAbsentVsEmpty:
    def test_empty_string_is_found(self, meta):
        assert "empty" in meta
        assert meta["empty"] == ""
        assert meta.get("empty") == ""
        # A sentinel default must NOT shadow a real empty-string value.
        assert meta.get("empty", "DEFAULT") == ""

    def test_zero_int_is_found(self, meta):
        assert "zero" in meta
        assert meta["zero"] == 0
        assert meta.get("zero") == 0
        assert meta.get("zero", 999) == 0

    def test_false_bool_is_found(self, meta):
        assert "flag" in meta
        assert meta.get("flag") is False
        assert meta.get("flag", True) is False

    def test_extra_zero_and_empty_found(self, meta):
        assert meta.get("count") == 0
        assert meta.get("note") == ""

    def test_missing_returns_none_by_default(self, meta):
        assert meta.get("does_not_exist") is None

    def test_missing_returns_supplied_default(self, meta):
        sentinel = object()
        assert meta.get("does_not_exist", sentinel) is sentinel

    def test_missing_not_in(self, meta):
        assert "does_not_exist" not in meta

    def test_getitem_missing_raises(self, meta):
        with pytest.raises(KeyError):
            _ = meta["does_not_exist"]


# ---------------------------------------------------------------------------
# Dot-path helpers (message level)
# ---------------------------------------------------------------------------


class TestDotPath:
    def test_has_path_nested(self, meta):
        assert meta.has_path("mars.class") is True
        assert meta.has_path("mars.param") is True

    def test_get_path_nested_first_match(self, meta):
        # First-match across base → base[0] wins.
        assert meta.get_path("mars.class") == "od"
        assert meta.get_path("mars.param") == "2t"

    def test_get_path_returns_container(self, meta):
        assert meta.get_path("mars") == {"class": "od", "param": "2t"}
        assert meta.get_path("levels") == [1000, 850, 500]

    def test_has_path_absent(self, meta):
        assert meta.has_path("nope") is False
        assert meta.has_path("mars.nope") is False
        assert meta.has_path("nope.nope") is False

    def test_get_path_absent_returns_default(self, meta):
        assert meta.get_path("nope.nope") is None
        assert meta.get_path("mars.nope", "D") == "D"

    def test_get_path_extra_fallback(self, meta):
        # Message-level path lookup falls back to _extra_.
        assert meta.has_path("producer") is True
        assert meta.get_path("producer") == "test-suite"

    def test_get_path_empty_and_zero(self, meta):
        # Absent-vs-empty must hold for the dot-path getter too.
        assert meta.has_path("empty") is True
        assert meta.get_path("empty", "DEFAULT") == ""
        assert meta.has_path("zero") is True
        assert meta.get_path("zero", 999) == 0


# ---------------------------------------------------------------------------
# Per-object scoping
# ---------------------------------------------------------------------------


class TestPerObjectScoping:
    def test_has_path_at_each_object(self, meta):
        assert meta.has_path_at(0, "mars.class") is True
        assert meta.has_path_at(1, "mars.class") is True

    def test_get_path_at_each_object(self, meta):
        assert meta.get_path_at(0, "mars.class") == "od"
        assert meta.get_path_at(1, "mars.class") == "rd"
        assert meta.get_path_at(0, "mars.param") == "2t"
        assert meta.get_path_at(1, "mars.param") == "msl"

    def test_get_path_at_scoped_absent_key(self, meta):
        # only_in_0 exists in base[0] but not base[1].
        assert meta.has_path_at(0, "only_in_0") is True
        assert meta.get_path_at(0, "only_in_0") == "yes"
        assert meta.has_path_at(1, "only_in_0") is False
        assert meta.get_path_at(1, "only_in_0") is None
        assert meta.get_path_at(1, "only_in_0", "D") == "D"

    def test_get_path_at_no_extra_fallback(self, meta):
        # producer is a message-level _extra_ key → not per-object visible.
        assert meta.has_path("producer") is True
        assert meta.has_path_at(0, "producer") is False
        assert meta.has_path_at(1, "producer") is False
        assert meta.get_path_at(0, "producer") is None

    def test_get_path_at_out_of_range(self, meta):
        assert meta.has_path_at(99, "mars.class") is False
        assert meta.get_path_at(99, "mars.class") is None
        assert meta.get_path_at(99, "mars.class", "D") == "D"

    def test_get_path_at_absent_vs_empty(self, meta):
        assert meta.get_path_at(0, "empty", "DEFAULT") == ""
        assert meta.get_path_at(0, "zero", 999) == 0


# ---------------------------------------------------------------------------
# _reserved_ visibility
# ---------------------------------------------------------------------------


class TestReservedVisibility:
    def test_reserved_hidden_from_has_path(self, meta):
        assert meta.has_path("_reserved_") is False
        assert meta.has_path("_reserved_.tensor") is False

    def test_reserved_hidden_from_get_path(self, meta):
        assert meta.get_path("_reserved_") is None
        assert meta.get_path("_reserved_.tensor") is None

    def test_reserved_hidden_per_object(self, meta):
        assert meta.has_path_at(0, "_reserved_") is False
        assert meta.has_path_at(0, "_reserved_.tensor") is False
        assert meta.get_path_at(0, "_reserved_.tensor") is None

    def test_reserved_not_a_message_key(self, meta):
        keys = meta.keys()
        assert "_reserved_" not in meta
        assert "_reserved_" not in keys

    def test_reserved_visible_in_base_dict(self, meta):
        # The encoder auto-populates base[i]["_reserved_"]["tensor"]; it stays
        # visible through the native dict even though the path getters hide it.
        base0 = meta.base[0]
        assert "_reserved_" in base0
        tensor = base0["_reserved_"]["tensor"]
        assert tensor["ndim"] == 1
        assert tensor["shape"] == [3]
