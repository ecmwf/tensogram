# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Magic-byte and MARS-flavour detection for tensogram messages.

Two helpers:

* :func:`_match_magic` — returns ``True`` when a byte buffer starts with
  (or, in the *deeper* pass, contains) the tensogram preamble magic
  ``TENSOGRM``.  The two-pass signature matches the contract
  earthkit-data's reader registry expects from every reader module.

* :func:`is_mars_tensogram` — returns ``True`` when a decoded
  ``tensogram.Metadata`` object carries at least one non-empty
  ``base[i]["mars"]`` dict.  Drives the branch between the MARS
  :class:`FieldList` path and the xarray-only path in the source.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TENSOGRM_MAGIC", "is_mars_tensogram"]

# Message preamble magic.  See ``plans/WIRE_FORMAT.md`` §3 in the main
# tensogram documentation for the full preamble byte layout.
TENSOGRM_MAGIC: bytes = b"TENSOGRM"

# Minimum bytes required to confirm a magic match.  The preamble magic is
# 8 bytes wide; anything shorter is ambiguous.
_MAGIC_LEN: int = len(TENSOGRM_MAGIC)


def _match_magic(magic: bytes | None, deeper_check: bool) -> bool:
    """Return ``True`` when *magic* identifies a tensogram message.

    Parameters
    ----------
    magic
        The first *N* bytes of the file, in-memory buffer, or stream as
        supplied by earthkit-data's reader machinery.  ``None`` signals
        that earthkit-data could not peek — treat as non-match.
    deeper_check
        When ``False`` (first pass), require the magic at offset 0.
        When ``True`` (second pass), search the whole buffer.  Matches
        the semantics documented by earthkit-data's two-pass
        ``_find_reader`` loop.
    """
    if not magic or len(magic) < _MAGIC_LEN:
        return False

    if not deeper_check:
        return magic[:_MAGIC_LEN] == TENSOGRM_MAGIC

    return TENSOGRM_MAGIC in magic


def is_mars_tensogram(meta: Any) -> bool:
    """Return ``True`` if *meta* carries per-object MARS metadata.

    A tensogram message counts as *MARS-flavoured* when at least one
    entry in ``meta.base`` is a dict with a non-empty ``"mars"`` sub-map.
    Message-level MARS keys (e.g. in ``meta.extra``) are deliberately
    ignored: the earthkit-data FieldList model is per-object, so the
    discriminator must also be per-object.

    The function is duck-typed to work with both ``tensogram.Metadata``
    instances and plain dict-shaped stand-ins (used in unit tests).
    """
    base = getattr(meta, "base", None)
    if not isinstance(base, list) or not base:
        return False

    for entry in base:
        if not isinstance(entry, dict):
            continue
        mars = entry.get("mars")
        if isinstance(mars, dict) and mars:
            return True

    return False
