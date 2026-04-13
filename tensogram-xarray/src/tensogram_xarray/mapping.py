"""User-specified dimension and variable mapping.

Handles the ``dim_names`` and ``variable_key`` parameters that let callers
control how tensogram data maps to xarray dimensions and variable names.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


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
