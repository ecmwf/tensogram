# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""User-facing data wrapper for tensogram content.

Mirrors the pattern used by earthkit-data's other data wrappers
(``GribData``, ``NetCDFData``) — a minimal facade over the
:class:`TensogramFileReader` that exposes ``to_xarray``,
``to_fieldlist``, and ``to_numpy``.

The wrapper adds no behaviour beyond delegation so that future upstream
migration to earthkit-data's own ``data/`` tree is a copy of this file.
"""

from __future__ import annotations

from typing import Any


class TensogramData:
    """Thin facade over a :class:`TensogramFileReader`."""

    #: Conversions advertised to earthkit-data's API discovery.
    available_types: tuple[str, ...] = ("xarray", "numpy", "fieldlist")

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def to_xarray(self, **kwargs: Any) -> Any:
        return self._reader.to_xarray(**kwargs)

    def to_fieldlist(self, **kwargs: Any) -> Any:
        return self._reader.to_fieldlist(**kwargs)

    def to_numpy(self, **kwargs: Any) -> Any:
        return self._reader.to_numpy(**kwargs)
