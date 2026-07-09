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

__all__ = ["TensogramData"]


class TensogramData:
    """Thin facade over a :class:`TensogramFileReader`.

    In earthkit-data 1.x, ``from_source`` returns this wrapper (via the
    reader's ``to_data_object``), so it also carries the FieldList-shaped
    conveniences (``sel`` / ``order_by`` / ``metadata`` / ``len`` / ``iter``)
    and the temp-file lifecycle (``close``) that used to live on the source.
    """

    #: Conversions advertised to earthkit-data's API discovery.
    available_types: tuple[str, ...] = ("xarray", "numpy", "fieldlist")

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def to_xarray(self, **kwargs: Any) -> Any:
        """Decode the file as an :class:`xarray.Dataset`."""
        return self._reader.to_xarray(**kwargs)

    def to_fieldlist(self, **kwargs: Any) -> Any:
        """Build a MARS :class:`FieldList`; raises for non-MARS content."""
        return self._reader.to_fieldlist(**kwargs)

    def to_numpy(self, **kwargs: Any) -> Any:
        """Return the single decoded ndarray for a one-variable message."""
        return self._reader.to_numpy(**kwargs)

    # -- FieldList-shaped conveniences (MARS content only) ----------------------

    def sel(self, *args: Any, **kwargs: Any) -> Any:
        return self._reader.sel(*args, **kwargs)

    def order_by(self, *args: Any, **kwargs: Any) -> Any:
        return self._reader.order_by(*args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._reader.get(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._reader)

    def __iter__(self) -> Any:
        return iter(self._reader)

    # -- lifecycle ----------------------------------------------------------------

    @property
    def storage_options(self) -> Any:
        """The remote ``storage_options`` given to the source (``None`` when local)."""
        return self._reader._storage_options

    @property
    def _tmp_path(self) -> Any:
        """The backing temp file for bytes-based input (``None`` otherwise)."""
        source = getattr(self._reader, "_owned_source", None)
        return getattr(source, "_tmp_path", None)

    def close(self) -> None:
        """Release any backing temp file early (safe no-op otherwise)."""
        close = getattr(self._reader, "close", None)
        if callable(close):
            close()
