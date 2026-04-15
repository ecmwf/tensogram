# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Coordinate detection by name matching.

When a data object's per-object metadata contains a ``name`` or ``param`` key
whose value matches a known coordinate name (case-insensitive), that object is
treated as a coordinate array rather than a data variable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Known coordinate names (all lower-case for case-insensitive matching).
KNOWN_COORD_NAMES: frozenset[str] = frozenset(
    {
        "lat",
        "latitude",
        "lon",
        "longitude",
        "x",
        "y",
        "time",
        "level",
        "pressure",
        "height",
        "depth",
        "frequency",
        "step",
    }
)

# Canonical name mapping: aliases -> preferred dimension name.
CANONICAL_DIM: dict[str, str] = {
    "lat": "latitude",
    "latitude": "latitude",
    "lon": "longitude",
    "longitude": "longitude",
    "x": "x",
    "y": "y",
    "time": "time",
    "level": "level",
    "pressure": "pressure",
    "height": "height",
    "depth": "depth",
    "frequency": "frequency",
    "step": "step",
}


# Module-level assertion: every known name must have a canonical mapping.
_canonical_keys = frozenset(CANONICAL_DIM.keys())
assert _canonical_keys == KNOWN_COORD_NAMES, (
    f"KNOWN_COORD_NAMES and CANONICAL_DIM keys are out of sync: "
    f"missing from CANONICAL_DIM: {KNOWN_COORD_NAMES - _canonical_keys}, "
    f"extra in CANONICAL_DIM: {_canonical_keys - KNOWN_COORD_NAMES}"
)


def _get_object_name(meta: dict[str, Any]) -> str | None:
    """Extract the name/param identifier from per-object metadata.

    Checks ``name``, ``param``, and nested ``mars.param`` in that order.
    """
    if "name" in meta:
        return str(meta["name"])
    if "param" in meta:
        return str(meta["param"])
    mars = meta.get("mars")
    if isinstance(mars, dict) and "param" in mars:
        return str(mars["param"])
    return None


def detect_coords(
    object_metas: Sequence[dict[str, Any]],
) -> tuple[list[int], list[int], dict[int, str]]:
    """Partition data objects into coordinates and variables.

    Parameters
    ----------
    object_metas
        Per-object metadata dicts (one per data object in the message).

    Returns
    -------
    coord_indices
        Indices of objects identified as coordinates.
    var_indices
        Indices of objects identified as data variables.
    coord_dim_names
        Mapping from coord object index to canonical dimension name.
    """
    coord_indices: list[int] = []
    var_indices: list[int] = []
    coord_dim_names: dict[int, str] = {}

    for i, meta in enumerate(object_metas):
        obj_name = _get_object_name(meta)
        if obj_name is not None and obj_name.lower() in KNOWN_COORD_NAMES:
            coord_indices.append(i)
            coord_dim_names[i] = CANONICAL_DIM[obj_name.lower()]
        else:
            var_indices.append(i)

    return coord_indices, var_indices, coord_dim_names
