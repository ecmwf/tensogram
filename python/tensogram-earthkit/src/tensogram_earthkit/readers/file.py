# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""File-backed tensogram reader.

The :class:`TensogramFileReader` is a :class:`earthkit.data.readers.Reader`
that owns the path to a local ``.tgm`` file and serves the three exports
earthkit-data's data model expects: :meth:`to_xarray`, :meth:`to_numpy`,
and (in Cycle 3, when MARS keys are present) :meth:`to_fieldlist`.

Delegation:

* :meth:`to_xarray` hands off to
  :class:`tensogram_xarray.backend.TensogramBackendEntrypoint` so the
  coordinate auto-detection, dim-name resolution, and lazy backing-array
  logic live in exactly one place.
* :meth:`to_numpy` resolves through the xarray path for non-MARS files
  (single-variable convenience); the MARS path in Cycle 3 routes through
  :class:`FieldList`.

The module-level :func:`reader` callable matches earthkit-data's reader
protocol — both the discovery path (with ``magic`` + ``deeper_check``)
and the ``source.reader`` hook path (no ``magic``, trust the source).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from earthkit.data.readers import Reader

from tensogram_earthkit.detection import TENSOGRM_MAGIC, _match_magic, is_mars_tensogram

LOG = logging.getLogger(__name__)


def _is_remote(path: str) -> bool:
    """Thin wrapper around ``tensogram.is_remote_url`` for test-seam reasons."""
    import tensogram

    return tensogram.is_remote_url(path)


class TensogramFileReader(Reader):
    """Reader for a tensogram ``.tgm`` file — local path or remote URL.

    Construction is cheap — no CBOR is parsed, no payload is decoded.
    Remote URLs (``http://``, ``https://``, ``s3://``, ``gs://``,
    ``az://``) are handed off to ``tensogram.TensogramFile.open_remote``
    and ``xr.open_dataset(engine="tensogram", storage_options=…)`` when
    needed.
    """

    appendable = True
    binary = True

    def mutate_source(self) -> Any:
        """Keep the owning source — no multi-file or zip promotion."""
        return None

    @property
    def _is_remote_path(self) -> bool:
        return _is_remote(str(self.path))

    @property
    def _storage_options(self) -> dict[str, Any] | None:
        """Read ``storage_options`` off the source — ``None`` when local."""
        source = self.source
        if source is None:
            return None
        return getattr(source, "storage_options", None)

    # -- format-sensing helpers -------------------------------------------------

    def _decode_first_metadata(self) -> Any:
        """Decode only the first message's metadata.  Used to branch paths."""
        import tensogram

        path = str(self.path)
        if self._is_remote_path:
            with tensogram.TensogramFile.open_remote(path, self._storage_options or {}) as f:
                result = f.file_decode_descriptors(0)
            return result["metadata"]

        with tensogram.TensogramFile.open(path) as f:
            raw = f.read_message(0)
        return tensogram.decode_metadata(raw)

    def _is_mars(self) -> bool:
        """``True`` when the first message carries per-object MARS metadata."""
        try:
            meta = self._decode_first_metadata()
        except Exception:  # pragma: no cover - defensive
            LOG.debug("could not read first message metadata from %s", self.path)
            return False
        return is_mars_tensogram(meta)

    # -- public data-model surface ----------------------------------------------

    def to_xarray(self, **kwargs: Any) -> Any:
        """Return an :class:`xarray.Dataset` via the tensogram-xarray backend."""
        import xarray as xr

        # Forward storage_options for remote URLs unless the caller
        # already specified one.
        if self._is_remote_path and "storage_options" not in kwargs:
            kwargs["storage_options"] = self._storage_options or {}

        # Defer to the backend engine so dim-name / coord logic is single-sourced.
        return xr.open_dataset(str(self.path), engine="tensogram", **kwargs)

    def to_numpy(self, **kwargs: Any) -> Any:
        """Return a single decoded ndarray.

        For a one-variable Dataset, returns ``ds[var].values``.  For
        multi-variable Datasets, raises :class:`ValueError` — callers
        should use :meth:`to_xarray` or, on MARS data, :meth:`to_fieldlist`
        and iterate fields explicitly.
        """
        ds = self.to_xarray(**kwargs)
        if len(ds.data_vars) == 1:
            name = next(iter(ds.data_vars))
            return ds[name].values
        raise ValueError(
            f"to_numpy() requires a single-variable tensogram message; "
            f"{self.path} has {len(ds.data_vars)} variables — "
            "use to_xarray() or to_fieldlist() instead"
        )

    def to_fieldlist(self, **_kwargs: Any) -> Any:
        """Return a MARS-backed :class:`FieldList`.

        Eagerly decodes every MARS-flavoured object.  Coordinate-only
        objects (``base[i]`` entries with no ``mars`` sub-map) are
        skipped — they do not have the geography semantics earthkit's
        :class:`Field` expects.  Non-MARS files raise
        :class:`NotImplementedError` pointing at :meth:`to_xarray`.
        """
        if not self._is_mars():
            raise NotImplementedError(
                f"{self.path} has no MARS metadata; use to_xarray() for non-MARS tensograms"
            )

        from tensogram_earthkit.fieldlist import build_fieldlist_from_path

        return build_fieldlist_from_path(str(self.path), storage_options=self._storage_options)

    # -- FieldList-shape convenience (source uses these) ----------------------

    def _fieldlist_or_none(self) -> Any:
        """Return the MARS FieldList, or ``None`` for non-MARS files."""
        if not self._is_mars():
            return None
        from tensogram_earthkit.fieldlist import build_fieldlist_from_path

        return build_fieldlist_from_path(str(self.path), storage_options=self._storage_options)

    # -- FieldList-shape convenience (source uses these) ----------------------

    def _fieldlist_or_none(self) -> Any:
        """Return the MARS FieldList, or ``None`` for non-MARS files."""
        if not self._is_mars():
            return None
        from tensogram_earthkit.fieldlist import build_fieldlist_from_path

        return build_fieldlist_from_path(str(self.path))

    def __len__(self) -> int:
        fl = self._fieldlist_or_none()
        if fl is None:
            raise TypeError(
                f"{self.path} has no MARS metadata; len() requires a FieldList — "
                "use to_xarray() instead"
            )
        return len(fl)

    def __iter__(self) -> Any:
        fl = self._fieldlist_or_none()
        if fl is None:
            raise TypeError(
                f"{self.path} has no MARS metadata; iter() requires a FieldList — "
                "use to_xarray() instead"
            )
        return iter(fl)

    def sel(self, *args: Any, **kwargs: Any) -> Any:
        return self.to_fieldlist().sel(*args, **kwargs)

    def order_by(self, *args: Any, **kwargs: Any) -> Any:
        return self.to_fieldlist().order_by(*args, **kwargs)

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self.to_fieldlist().metadata(*args, **kwargs)

    # -- data-object wrapper ----------------------------------------------------

    def to_data_object(self) -> Any:
        """Return the user-facing :class:`TensogramData` wrapper."""
        from tensogram_earthkit.data import TensogramData

        return TensogramData(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path})"


def reader(
    source: Any,
    path: str,
    *,
    magic: bytes | None = None,
    deeper_check: bool = False,
    **_kwargs: Any,
) -> TensogramFileReader | None:
    """Create a :class:`TensogramFileReader` — two entry protocols supported.

    *Discovery path* — earthkit-data's :func:`_find_reader` calls us with
    peeked bytes and returns ``None`` to pass through on mismatch.

    *Source-hook path* — our :class:`TensogramSource` sets itself as the
    reader provider; earthkit-data then calls us with just ``(source, path)``.
    Since the source has already committed to the tensogram format, we
    validate the magic ourselves by peeking the file; a real mismatch
    raises :class:`ValueError` rather than silently returning ``None``.
    """
    if magic is not None:
        # Discovery path.
        if not _match_magic(magic, deeper_check=deeper_check):
            return None
        return TensogramFileReader(source, path)

    # Source-hook path — remote URLs can't be peeked cheaply, so we
    # trust the source's format claim and defer validation to the first
    # real read (where a non-tensogram URL will fail with a clear
    # FramingError from the tensogram core).
    if _is_remote(path):
        return TensogramFileReader(source, path)

    try:
        with Path(path).open("rb") as fh:
            peeked = fh.read(len(TENSOGRM_MAGIC))
    except OSError as exc:  # pragma: no cover - FileSource already guards this
        raise FileNotFoundError(f"cannot open {path}") from exc

    if not _match_magic(peeked, deeper_check=False):
        raise ValueError(
            f"{path!r} does not look like a tensogram file "
            f"(expected magic {TENSOGRM_MAGIC!r}, got {peeked!r})"
        )
    return TensogramFileReader(source, path)
