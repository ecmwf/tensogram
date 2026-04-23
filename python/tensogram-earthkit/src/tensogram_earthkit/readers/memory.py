# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""In-memory (``bytes`` / ``bytearray`` / ``memoryview``) tensogram reader.

The user-facing memory path is driven by :class:`TensogramSource` —
when it receives bytes it materialises them to a temp file and falls
through to the file reader, which is simpler than running a parallel
bytes-based pipeline and keeps coordinate/dim-name logic single-sourced
in the tensogram-xarray backend.

This module still provides a :func:`memory_reader` callable in the
shape that earthkit-data's reader registry expects so the module layout
stays upstream-mergeable: anyone dropping this package into
``earthkit-data/readers/tensogram/`` later gets a complete reader-trio
(``reader``, ``memory_reader``, ``stream_reader``) with no further
refactoring.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from tensogram_earthkit.detection import _match_magic


def memory_reader(
    source: Any,
    buffer: bytes | bytearray | memoryview,
    *,
    magic: bytes | None = None,
    deeper_check: bool = False,
    **_kwargs: Any,
) -> Any | None:
    """Entry-point callable for in-memory tensogram content.

    Materialises *buffer* to a temp file and delegates to
    :class:`TensogramFileReader`.  Returns ``None`` when the magic
    does not match (discovery path).
    """
    # Use the buffer as magic when nothing was pre-peeked.
    if magic is None:
        magic = bytes(buffer[:8]) if len(buffer) >= 8 else bytes(buffer)

    if not _match_magic(magic, deeper_check=deeper_check):
        return None

    from tensogram_earthkit.readers.file import TensogramFileReader

    fd, path = tempfile.mkstemp(suffix=".tgm", prefix="tensogram_earthkit_mem_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(bytes(buffer))
    except Exception:
        os.unlink(path)
        raise

    reader = TensogramFileReader(source, path)
    # Lash the temp-file lifetime onto the reader so GC cleans it up.
    reader._tgm_owned_tempfile = path
    return reader
