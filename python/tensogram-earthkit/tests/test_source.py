# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Smoke tests for the ``earthkit.data.sources.tensogram`` entry point.

Covers the wiring only — not the data paths (those live in
:mod:`test_nonmars_path` and :mod:`test_mars_path`).  The three invariants
pinned here are:

* The entry point is discoverable by ``entrypoints.get_group_all``.
* ``earthkit.data.sources.get_source("tensogram")`` resolves to our class.
* A smoke ``ekd.from_source("tensogram", path)`` succeeds for both MARS
  and non-MARS files and returns an object exposing ``to_xarray``.
"""

from __future__ import annotations

import earthkit.data as ekd
import entrypoints
import pytest

# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


class TestEntryPoint:
    def test_tensogram_source_entry_point_registered(self) -> None:
        """The ``tensogram`` source entry point is installed."""
        names = {e.name for e in entrypoints.get_group_all("earthkit.data.sources")}
        assert "tensogram" in names

    def test_source_resolves_via_earthkit_machinery(self) -> None:
        """``find_plugin`` resolves the entry point to our class.

        ``get_source("tensogram")`` eagerly instantiates the plugin,
        which would require a ``path`` argument.  Using ``find_plugin``
        directly proves the plugin is wired without the side effect.
        """
        import os

        import earthkit.data.sources as sources
        from earthkit.data.core.plugins import find_plugin

        resolved = find_plugin(
            os.path.dirname(sources.__file__), "tensogram", sources.SourceLoader()
        )
        assert callable(resolved)
        # Must live inside our package so we know we got OUR entry point.
        assert resolved.__module__.startswith("tensogram_earthkit")


# ---------------------------------------------------------------------------
# Smoke: from_source returns something usable
# ---------------------------------------------------------------------------


class TestFromSourceSmoke:
    def test_from_source_with_mars_file(self, mars_tensogram_file) -> None:
        data = ekd.from_source("tensogram", str(mars_tensogram_file))
        assert data is not None
        assert hasattr(data, "to_xarray")

    def test_from_source_with_nonmars_file(self, nonmars_tensogram_file) -> None:
        data = ekd.from_source("tensogram", str(nonmars_tensogram_file))
        assert data is not None
        assert hasattr(data, "to_xarray")

    def test_from_source_with_missing_file(self, tmp_path) -> None:
        """Missing files raise a clear ``FileNotFoundError``.

        earthkit-data already raises this for unknown paths; we just
        assert our source doesn't hide the error under a different type.
        """
        missing = tmp_path / "does_not_exist.tgm"
        with pytest.raises(FileNotFoundError):
            ekd.from_source("tensogram", str(missing))
