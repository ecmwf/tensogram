# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Stream-backed tensogram reader.

The implementation drains the provided stream into bytes and dispatches
through :func:`tensogram_earthkit.readers.memory.memory_reader`.  This
keeps the stream path semantically identical to the memory path — the
downstream xarray backend still needs a concrete file on disk and the
MARS :class:`FieldList` is materialised eagerly in either case.

Progressive yield-as-each-message-arrives streaming is explicitly out
of scope for now: tensogram-xarray's coordinate auto-detection needs
the full message graph before it can resolve dim names, and the
FieldList contract requires ``__len__`` to be known.  A true progressive
reader (yielding Fields one by one as they decode) can slot in here
later without touching the source or file reader.
"""

from __future__ import annotations

import contextlib
from typing import Any

from tensogram_earthkit.detection import _match_magic


def _drain(stream: Any) -> bytes:
    """Read the stream to EOF and return the bytes.

    Tolerates any BinaryIO-compatible object (BytesIO, BufferedReader,
    file handle, network stream adapter).  Closes the stream after
    draining so the caller doesn't have to.
    """
    buf = stream.read()
    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        stream.close()
    return buf


def stream_reader(
    source: Any,
    stream: Any,
    magic: bytes | None = None,
    *,
    deeper_check: bool = False,
    memory: bool = False,
    **_kwargs: Any,
) -> Any | None:
    """Entry-point callable for stream-backed tensogram content.

    Parameters
    ----------
    source
        Calling source — kept weakly by the constructed reader.
    stream
        Any BinaryIO-compatible stream.
    magic
        Bytes pre-peeked by earthkit-data's machinery; may be ``None``
        when the stream did not support ``peek``.  In that case we
        drain first, then test the leading bytes ourselves.
    deeper_check
        ``True`` on the second discovery pass.
    memory
        Ignored — the reader always materialises, see module docstring.

    Returns
    -------
    A :class:`TensogramFileReader` when the stream carries tensogram
    content, ``None`` when the magic does not match.
    """
    # Earthkit-data asks whether we recognise the stream based on
    # pre-peeked bytes.  When peek succeeded we can fast-fail on
    # mismatch without draining.
    if (
        magic is not None
        and len(magic) >= 8
        and not _match_magic(magic, deeper_check=deeper_check)
    ):
        return None

    buf = _drain(stream)
    if not buf:
        return None

    # If we did not have a real magic, validate now from the drained
    # buffer's leading bytes.
    if (magic is None or len(magic) < 8) and not _match_magic(buf[:8], deeper_check=deeper_check):
        return None

    # Hand off to the memory path — same tempfile materialisation,
    # same TensogramFileReader on the other side.
    from tensogram_earthkit.readers.memory import memory_reader

    return memory_reader(source, buf, magic=buf[:8], deeper_check=False)
