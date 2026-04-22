# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Dimension and variable naming for the xarray backend.

Handles the ``dim_names`` and ``variable_key`` parameters that let callers
control how tensogram data maps to xarray dimensions and variable names,
including the per-object ``base[i]["dim_names"]`` opt-in convention.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

PER_OBJECT_DIM_NAMES_KEY = "dim_names"
EXTRA_DIM_NAMES_KEY = "dim_names"

# Metadata keys that encode xarray structure rather than user attributes.
# These are read by the backend to shape the Dataset but must not leak into
# :attr:`xarray.Variable.attrs` or participate in the :mod:`merge` path's
# hypercube grouping (otherwise hint-like keys could become outer dims).
STRUCTURAL_META_KEYS: frozenset[str] = frozenset({PER_OBJECT_DIM_NAMES_KEY})


def strip_structural_keys(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *meta* without :data:`STRUCTURAL_META_KEYS`."""
    return {k: v for k, v in meta.items() if k not in STRUCTURAL_META_KEYS}


def resolve_dim_names(
    ndim: int,
    user_dim_names: Sequence[str] | None,
) -> list[str]:
    """Return dimension names for a tensor with *ndim* axes.

    If *user_dim_names* is provided, it must have exactly *ndim* entries.
    Otherwise generic ``dim_0``, ``dim_1``, ... names are generated.
    """
    if user_dim_names is not None:
        names = list(user_dim_names)
        if len(names) != ndim:
            msg = (
                f"dim_names has {len(names)} entries but tensor has {ndim} "
                f"dimensions. Provide exactly {ndim} names."
            )
            raise ValueError(msg)
        return names
    return [f"dim_{i}" for i in range(ndim)]


def _looks_like_string_sequence(raw: Any) -> bool:
    """True if *raw* is a list/tuple/sequence of items, excluding ``str``/``bytes``."""
    if isinstance(raw, (str, bytes, bytearray)):
        return False
    return isinstance(raw, Sequence)


def parse_per_object_dim_names(
    ndim: int,
    obj_meta: Mapping[str, Any] | None,
) -> list[str] | None:
    """Return validated per-object dim names or ``None`` when absent/malformed.

    The per-object hint lives at ``base[i]["dim_names"]`` and must be a
    sequence (but not ``str``) of exactly *ndim* non-empty distinct strings.
    Any deviation yields ``None`` (logged at DEBUG), so malformed hints
    silently fall through the priority chain rather than crashing or
    corrupting dim assignment.
    """
    if not obj_meta:
        return None
    raw = obj_meta.get(PER_OBJECT_DIM_NAMES_KEY)
    if raw is None:
        return None
    if not _looks_like_string_sequence(raw):
        logger.debug(
            "per-object %s hint is not a list/sequence (got %s); ignoring",
            PER_OBJECT_DIM_NAMES_KEY,
            type(raw).__name__,
        )
        return None
    names = list(raw)
    if len(names) != ndim:
        logger.debug(
            "per-object %s hint has %d entries but ndim=%d; ignoring",
            PER_OBJECT_DIM_NAMES_KEY,
            len(names),
            ndim,
        )
        return None
    if not all(isinstance(n, str) and n for n in names):
        logger.debug(
            "per-object %s hint contains non-string or empty entries; ignoring",
            PER_OBJECT_DIM_NAMES_KEY,
        )
        return None
    if len(set(names)) != ndim:
        logger.debug(
            "per-object %s hint contains duplicate entries %r; ignoring",
            PER_OBJECT_DIM_NAMES_KEY,
            names,
        )
        return None
    return names


def parse_extra_dim_names_hint(
    ndim: int,
    raw: Any,
) -> list[str] | dict[int, str]:
    """Return parsed ``_extra_["dim_names"]`` hint.

    Accepts two legacy formats:

    * list (preferred) — axis-ordered names, length must equal *ndim*
    * dict — size-to-name mapping (string keys coerced to int)

    Invalid hints yield an empty dict so callers can iterate uniformly.
    """
    if raw is None:
        return {}
    if isinstance(raw, list):
        try:
            names = [str(n) for n in raw]
        except (TypeError, ValueError):
            return {}
        if len(names) == ndim:
            return names
        return {}
    if isinstance(raw, dict):
        try:
            return {int(k): str(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def resolve_dims_for_axes(
    shape: tuple[int, ...],
    *,
    user_dim_names: Sequence[str] | None,
    coord_dim_sizes: Mapping[str, int],
    per_object_meta: Mapping[str, Any] | None,
    extra_dim_names_hint: Any,
) -> list[tuple[str, bool]]:
    """Return ``(name, is_generic_fallback)`` per axis using the full priority chain.

    Priority (highest to lowest):

    1. ``user_dim_names`` — explicit caller kwarg.
    2. Coord size-match — an existing coord dim whose size equals the axis size.
    3. Per-object ``base[i]["dim_names"]`` — validated by
       :func:`parse_per_object_dim_names`.
    4. ``_extra_["dim_names"]`` — list or size-to-name dict, parsed by
       :func:`parse_extra_dim_names_hint`.
    5. Generic ``dim_{axis}`` fallback — flagged ``is_generic_fallback=True``
       so the caller can disambiguate on collision.

    Only axes from step 5 are flagged generic; all earlier sources count as
    user-visible hints and are never auto-renamed.  A hinted-name collision
    is surfaced separately by the caller's disambiguation pass.
    """
    ndim = len(shape)

    if user_dim_names is not None:
        return [(name, False) for name in resolve_dim_names(ndim, user_dim_names)]

    per_obj = parse_per_object_dim_names(ndim, per_object_meta)

    size_to_coord: dict[int, list[str]] = {}
    for cname, csize in coord_dim_sizes.items():
        size_to_coord.setdefault(csize, []).append(cname)

    extra_hints = parse_extra_dim_names_hint(ndim, extra_dim_names_hint)

    dims: list[tuple[str, bool]] = []
    used: set[str] = set()
    for axis, axis_size in enumerate(shape):
        name: str | None = None
        if axis_size in size_to_coord:
            for cname in size_to_coord[axis_size]:
                if cname not in used:
                    name = cname
                    break
        if name is None and per_obj is not None:
            candidate = per_obj[axis]
            if candidate not in used:
                name = candidate
        if name is None and isinstance(extra_hints, list):
            candidate = extra_hints[axis]
            if candidate not in used:
                name = candidate
        if name is None and isinstance(extra_hints, dict) and axis_size in extra_hints:
            candidate = extra_hints[axis_size]
            if candidate not in used:
                name = candidate
        if name is None:
            dims.append((f"dim_{axis}", True))
            continue
        dims.append((name, False))
        used.add(name)
    return dims


def _resolve_dotted(meta: dict[str, Any], dotted_key: str) -> Any:
    """Resolve a dotted key path like ``mars.param`` in a nested dict."""
    parts = dotted_key.split(".")
    current: Any = meta
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


# Dotted-path metadata keys to try for variable naming, in priority order.
# Must match the priority chain in tensogram-zarr's mapping.py.
_VARIABLE_NAME_KEYS = [
    "name",
    "mars.param",
    "param",
    "mars.shortName",
    "shortName",
]


def resolve_variable_name(
    obj_index: int,
    per_object_meta: dict[str, Any],
    variable_key: str | None,
) -> str:
    """Determine the xarray variable name for a data object.

    If *variable_key* is given (e.g. ``"mars.param"``), the value at that
    dotted path in the per-object metadata is used.  Otherwise the function
    tries ``_VARIABLE_NAME_KEYS`` in priority order, then falls back to a
    generic ``"object_<index>"`` name.

    The priority chain matches ``tensogram-zarr``'s ``resolve_variable_name``
    so that the same ``.tgm`` file produces consistent variable names
    regardless of which backend opens it.
    """
    source = per_object_meta or {}

    # Try explicit key first, then the standard priority chain.
    keys_to_try = [variable_key] if variable_key else []
    keys_to_try.extend(_VARIABLE_NAME_KEYS)

    for key in keys_to_try:
        val = _resolve_dotted(source, key)
        if val is not None:
            return str(val)

    return f"object_{obj_index}"
