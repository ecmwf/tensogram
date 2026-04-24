# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Helpers for bridging tensogram ``base[i]["mars"]`` entries to earthkit fields.

Two small functions:

* :func:`extract_mars_keys` flattens a ``base[i]`` entry into the dict
  shape that :class:`earthkit.data.utils.metadata.dict.UserMetadata`
  understands.  MARS keys come from the ``mars`` sub-map; any sibling
  keys (``name``, CF attrs, ``grib`` etc.) are merged on top but with
  MARS keys winning on conflict — that matches GRIB's semantics.
* :func:`base_entry_to_usermetadata` wraps the flattened dict in a
  :class:`UserMetadata` with a concrete ``shape`` so
  :attr:`UserMetadata.geography` can report the grid shape.

Keeping this isolated from :mod:`.fieldlist` means the encoder in
:mod:`.encoder` can reuse the reverse direction
(:func:`field_to_base_entry`) without importing the reader side.
"""

from __future__ import annotations

import contextlib
from typing import Any

__all__ = [
    "base_entry_to_usermetadata",
    "extract_mars_keys",
    "field_to_base_entry",
    "has_mars_namespace",
]


def extract_mars_keys(base_entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten a tensogram ``base[i]`` entry into a plain metadata dict.

    Rules:

    * The ``"_reserved_"`` key (library-managed tensor info) is dropped.
    * Every non-MARS sibling key is copied first (e.g. ``name``,
      ``cf``, ``grib``); those with dict values are left nested for the
      caller's inspection — earthkit's :class:`UserMetadata` can look
      up ``"cf.standard_name"`` via dotted paths on its own.
    * The ``"mars"`` sub-map is then merged on top, flattening its
      entries into the top-level dict.  MARS keys therefore win on
      conflict, which matches how GRIB-converted tensograms are
      typically consumed.

    An entry that is not a dict (or contains no ``mars`` sub-map) yields
    an empty dict — the caller decides whether such objects count as
    MARS fields or coordinate/auxiliary objects.
    """
    if not isinstance(base_entry, dict):
        return {}

    out: dict[str, Any] = {}
    for k, v in base_entry.items():
        if k in ("_reserved_", "mars"):
            continue
        out[k] = v

    mars = base_entry.get("mars")
    if isinstance(mars, dict):
        out.update(mars)

    return out


def base_entry_to_usermetadata(
    base_entry: dict[str, Any],
    shape: tuple[int, ...],
) -> Any:
    """Build an earthkit ``UserMetadata`` from a tensogram base entry.

    Parameters
    ----------
    base_entry
        One element of ``tensogram.Metadata.base``.
    shape
        The tensor shape — forwarded to :class:`UserMetadata` so its
        ``geography.shape()`` reports the right grid extent.

    Returns
    -------
    An ``earthkit.data.utils.metadata.dict.UserMetadata`` instance.
    """
    from earthkit.data.utils.metadata.dict import UserMetadata

    flat = extract_mars_keys(base_entry)
    return UserMetadata(flat, shape=shape)


def has_mars_namespace(base_entry: Any) -> bool:
    """Return ``True`` if ``base_entry`` carries a non-empty MARS sub-map.

    Mirrors the per-object discriminator used by
    :func:`tensogram_earthkit.detection.is_mars_tensogram`.
    """
    if not isinstance(base_entry, dict):
        return False
    mars = base_entry.get("mars")
    return isinstance(mars, dict) and bool(mars)


# Canonical MARS key set — everything not in this set falls to ``_extra_``
# when an earthkit Field is serialised back to a tensogram base entry.
# Kept conservative: adding more keys only moves them out of `_extra_`
# into `mars`, which is always the safer default for MARS consumers.
_MARS_CANONICAL_KEYS: frozenset[str] = frozenset(
    {
        "class",
        "stream",
        "expver",
        "type",
        "param",
        "shortName",
        "date",
        "time",
        "step",
        "levtype",
        "levelist",
        "number",
        "ensemble",
        "domain",
        "grid",
        "area",
    }
)


def field_to_base_entry(field: Any) -> dict[str, Any]:
    """Build a tensogram ``base[i]`` entry from an earthkit :class:`Field`.

    Collects every metadata key exposed by the field, routes known
    MARS keys into the ``mars`` sub-map, and stashes the rest in
    ``_extra_`` so nothing is dropped on round-trips.
    """
    items = _collect_field_metadata(field)

    mars: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for k, v in items.items():
        if k in _MARS_CANONICAL_KEYS:
            mars[k] = v
        else:
            extras[k] = v

    entry: dict[str, Any] = {}
    if mars:
        entry["mars"] = mars
    if extras:
        entry["_extra_"] = extras
    return entry


def _collect_field_metadata(field: Any) -> dict[str, Any]:
    """Return a flat metadata dict for *field* — robust to Field variants.

    Not every earthkit :class:`Field` implementation exposes the same
    metadata interface.  Three strategies are attempted in order:

    1. ``field.metadata().items()`` — works for dict-backed
       :class:`UserMetadata`.
    2. ``field.metadata(k)`` per canonical MARS key — works for
       read-only metadata objects that only support ``__getitem__``
       or ``metadata(key)``.
    3. Empty dict — the field has no readable metadata, caller's
       resulting base entry will be empty.
    """
    metadata_obj = None
    with contextlib.suppress(AttributeError, TypeError):
        metadata_obj = field.metadata()

    if metadata_obj is not None:
        try:
            return dict(metadata_obj.items())
        except (AttributeError, TypeError):
            pass

    # Fallback: ask for each canonical MARS key individually.  Each
    # lookup is guarded because older Field implementations may raise
    # on missing keys instead of returning None.
    out: dict[str, Any] = {}
    for k in _MARS_CANONICAL_KEYS:
        try:
            v = field.metadata(k)
        except (KeyError, AttributeError, TypeError, ValueError):
            continue
        if v is not None:
            out[k] = v
    return out
