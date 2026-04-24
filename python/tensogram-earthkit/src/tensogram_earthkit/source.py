# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""earthkit-data source plugin — ``ekd.from_source("tensogram", …)``.

The module exposes a ``source`` attribute at module scope, which is the
hook earthkit-data's ``SourceLoader`` / ``EntryPointLoader`` grabs when
it resolves ``earthkit.data.sources.tensogram``.

At its core the class is a :class:`FileSource` subclass that tells the
earthkit-data reader machinery *which* reader callable to use, bypassing
the usual magic-byte scan of ``earthkit/data/readers/``.  Three input
shapes are accepted:

* a filesystem path (``str`` / :class:`pathlib.Path`) — read directly;
* a remote URL (``s3://``, ``gs://``, ``az://``, ``http://``,
  ``https://``) — handed off to ``tensogram.TensogramFile.open_remote``;
* bytes-like content (``bytes`` / ``bytearray`` / ``memoryview``) —
  materialised to a temp file owned by the source; the temp file is
  unlinked when the source is garbage-collected or explicitly closed.

Byte streams are routed through :mod:`.readers.stream`.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import weakref
from typing import Any

from earthkit.data.sources.file import FileSource

from tensogram_earthkit.readers.file import reader as file_reader

_BytesLike = (bytes, bytearray, memoryview)


def _materialise_bytes(buf: bytes | bytearray | memoryview) -> str:
    """Write *buf* to a fresh temp file and return its path.

    The caller is responsible for unlinking the returned path once the
    source is done with it — :class:`TensogramSource` uses a
    :func:`weakref.finalize` to guarantee this even on non-graceful
    teardown.
    """
    fd, path = tempfile.mkstemp(suffix=".tgm", prefix="tensogram_earthkit_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(bytes(buf))
    except Exception:
        os.unlink(path)
        raise
    return path


def _cleanup_tempfile(path: str) -> None:
    """Finaliser callback — unlink silently if the file is still there."""
    with contextlib.suppress(OSError):
        os.unlink(path)


class TensogramSource(FileSource):
    """Source for tensogram ``.tgm`` content.

    Parameters
    ----------
    path
        A filesystem path, a remote URL, or bytes-like content
        (``bytes`` / ``bytearray`` / ``memoryview``).
    storage_options
        Forwarded to ``tensogram.TensogramFile.open_remote`` for remote
        URLs.  Ignored for local paths and bytes.
    **kwargs
        Forwarded to the earthkit-data :class:`FileSource` constructor.
    """

    # Class-level ``reader`` attribute — earthkit-data's
    # :func:`earthkit.data.readers.reader` function checks
    # ``hasattr(source, "reader")`` and, when the value is callable,
    # invokes it with ``(source, path)`` bypassing the built-in
    # readers/ directory scan.  ``staticmethod`` ensures the attribute
    # returns the bare callable (not a bound method) when accessed via
    # an instance.
    reader = staticmethod(file_reader)

    def __init__(
        self,
        path: str | os.PathLike | bytes | bytearray | memoryview,
        *,
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Stashed for the remote path — see TensogramFileReader._storage_options.
        self.storage_options = storage_options

        if isinstance(path, _BytesLike):
            # Materialise the bytes to a temp file.  A weakref finaliser
            # unlinks it deterministically when the source is collected,
            # even if the user never calls :meth:`close`.
            tmp_path = _materialise_bytes(path)
            self._tmp_path: str | None = tmp_path
            weakref.finalize(self, _cleanup_tempfile, tmp_path)
            super().__init__(tmp_path, **kwargs)
            return

        self._tmp_path = None
        super().__init__(str(path), **kwargs)

    # -- additions over FileSource ---------------------------------------------

    def to_fieldlist(self, **kwargs: Any) -> Any:
        """Delegate to the reader — raises ``NotImplementedError`` for non-MARS."""
        return self._reader.to_fieldlist(**kwargs)

    def close(self) -> None:
        """Unlink any backing temp file early (optional)."""
        if self._tmp_path is not None:
            _cleanup_tempfile(self._tmp_path)
            self._tmp_path = None


# Module-level attribute that earthkit-data's SourceLoader picks up.
source = TensogramSource
