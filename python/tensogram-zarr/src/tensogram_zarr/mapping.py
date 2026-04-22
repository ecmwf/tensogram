# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Bidirectional mapping between Tensogram and Zarr v3 metadata.

Converts TGM dtypes, descriptors, and global metadata into Zarr v3
``zarr.json`` structures and back.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

# TGM dtype string  →  (Zarr v3 data_type, numpy dtype string)
_TGM_TO_ZARR_DTYPE: dict[str, tuple[str, str]] = {
    "float16": ("float16", "<f2"),
    "bfloat16": ("bfloat16", "<V2"),  # no native numpy; raw 2-byte
    "float32": ("float32", "<f4"),
    "float64": ("float64", "<f8"),
    "complex64": ("complex64", "<c8"),
    "complex128": ("complex128", "<c16"),
    "int8": ("int8", "|i1"),
    "int16": ("int16", "<i2"),
    "int32": ("int32", "<i4"),
    "int64": ("int64", "<i8"),
    "uint8": ("uint8", "|u1"),
    "uint16": ("uint16", "<u2"),
    "uint32": ("uint32", "<u4"),
    "uint64": ("uint64", "<u8"),
    "bitmask": ("uint8", "|u1"),  # bitmask exposed as uint8
}

# Zarr v3 data_type  →  TGM dtype string
_ZARR_TO_TGM_DTYPE: dict[str, str] = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float64": "float64",
    "complex64": "complex64",
    "complex128": "complex128",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
}

# numpy dtype  →  TGM dtype string (used on write path)
_NP_TO_TGM_DTYPE: dict[np.dtype, str] = {
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("complex64"): "complex64",
    np.dtype("complex128"): "complex128",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
}


def tgm_dtype_to_zarr(tgm_dtype: str) -> str:
    """Convert a TGM dtype string to a Zarr v3 data_type string."""
    pair = _TGM_TO_ZARR_DTYPE.get(tgm_dtype)
    if pair is None:
        raise ValueError(f"unsupported TGM dtype: {tgm_dtype!r}")
    return pair[0]


def tgm_dtype_to_numpy(tgm_dtype: str) -> np.dtype:
    """Convert a TGM dtype string to a numpy dtype."""
    pair = _TGM_TO_ZARR_DTYPE.get(tgm_dtype)
    if pair is None:
        raise ValueError(f"unsupported TGM dtype: {tgm_dtype!r}")
    return np.dtype(pair[1])


def zarr_dtype_to_tgm(zarr_dtype: str) -> str:
    """Convert a Zarr v3 data_type string to a TGM dtype string."""
    result = _ZARR_TO_TGM_DTYPE.get(zarr_dtype)
    if result is None:
        raise ValueError(f"unsupported Zarr dtype: {zarr_dtype!r}")
    return result


def numpy_dtype_to_tgm(dtype: np.dtype) -> str:
    """Convert a numpy dtype to a TGM dtype string."""
    result = _NP_TO_TGM_DTYPE.get(dtype)
    if result is None:
        raise ValueError(f"unsupported numpy dtype: {dtype!r}")
    return result


# ---------------------------------------------------------------------------
# Zarr v3 metadata synthesis (read path: TGM → zarr.json)
# ---------------------------------------------------------------------------


def build_group_zarr_json(
    meta: Any,
    variable_names: list[str],
) -> dict[str, Any]:
    """Synthesize a Zarr v3 group ``zarr.json`` from TGM GlobalMetadata.

    Parameters
    ----------
    meta : tensogram.Metadata
        Decoded TGM global metadata.
    variable_names : list[str]
        Names of the arrays in this group (for informational attributes).

    Returns
    -------
    dict
        A Zarr v3 group metadata dict ready for JSON serialization.
    """
    attrs: dict[str, Any] = {}

    # Merge extra metadata (message-level annotations)
    if hasattr(meta, "extra") and meta.extra:
        attrs.update(meta.extra)

    # Wire-format version comes from the preamble (see
    # ``plans/WIRE_FORMAT.md`` §3).  ``meta.version`` surfaces that
    # preamble-sourced value; we record it as an underscore-prefixed
    # attribute so the Zarr group advertises the wire format it came
    # from.  Renamed from ``_tensogram_version`` → ``_tensogram_wire_version``
    # in 0.17 to avoid any collision with a user-supplied free-form
    # ``version`` annotation, which now flows through ``_extra_``.
    attrs["_tensogram_wire_version"] = meta.version
    attrs["_tensogram_variables"] = variable_names

    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attrs,
    }


def build_array_zarr_json(
    desc: Any,
    per_object_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize a Zarr v3 array ``zarr.json`` from a TGM DataObjectDescriptor.

    The array is treated as a single chunk (chunk_shape == shape) since each
    TGM data object is a monolithic tensor.

    Parameters
    ----------
    desc : tensogram.DataObjectDescriptor
        The decoded object descriptor.
    per_object_meta : dict, optional
        Per-object metadata from ``meta.base[i]``.

    Returns
    -------
    dict
        A Zarr v3 array metadata dict ready for JSON serialization.
    """
    shape = list(desc.shape)
    zarr_dtype = tgm_dtype_to_zarr(desc.dtype)

    # Single chunk covering the whole array
    chunk_shape = list(shape) if shape else [1]

    # Build codec chain: just bytes (no Zarr-level compression; data is
    # already encoded/compressed inside TGM)
    codecs = [
        {
            "name": "bytes",
            "configuration": {"endian": "little"},
        },
    ]

    # Attributes from per-object metadata + descriptor params
    attrs: dict[str, Any] = {}
    if per_object_meta:
        attrs.update(per_object_meta)
    if desc.params:
        # Encoding params stored under _tensogram prefix to avoid clashes
        attrs["_tensogram_params"] = dict(desc.params)

    attrs["_tensogram_encoding"] = desc.encoding
    attrs["_tensogram_filter"] = desc.filter
    attrs["_tensogram_compression"] = desc.compression
    if desc.hash:
        attrs["_tensogram_hash"] = desc.hash

    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": zarr_dtype,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": chunk_shape},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": codecs,
        "fill_value": _default_fill_value(zarr_dtype),
        "attributes": attrs,
    }


_FLOAT_LIKE_PREFIXES = ("float", "bfloat", "complex")


def _default_fill_value(zarr_dtype: str) -> Any:
    """Return a sensible default fill value for a Zarr dtype."""
    if any(zarr_dtype.startswith(p) for p in _FLOAT_LIKE_PREFIXES):
        return float("nan")
    return 0


# ---------------------------------------------------------------------------
# TGM metadata reconstruction (write path: zarr.json → TGM)
# ---------------------------------------------------------------------------


def parse_array_zarr_json(zarr_meta: dict[str, Any]) -> dict[str, Any]:
    """Extract TGM-relevant fields from a Zarr v3 array ``zarr.json``.

    Returns a dict with keys: ``shape``, ``dtype``, ``byte_order``,
    ``encoding``, ``filter``, ``compression``, ``attrs``.
    """
    shape = zarr_meta["shape"]
    zarr_dtype = zarr_meta["data_type"]
    tgm_dtype = zarr_dtype_to_tgm(zarr_dtype)

    # Try to recover byte order from codecs
    byte_order = "little"
    for codec in zarr_meta.get("codecs", []):
        if codec.get("name") == "bytes":
            byte_order = codec.get("configuration", {}).get("endian", "little")

    # Work on a copy to avoid mutating the caller's dict
    attrs = dict(zarr_meta.get("attributes", {}))

    encoding = attrs.pop("_tensogram_encoding", "none")
    filt = attrs.pop("_tensogram_filter", "none")
    compression = attrs.pop("_tensogram_compression", "none")

    return {
        "shape": shape,
        "dtype": tgm_dtype,
        "byte_order": byte_order,
        "encoding": encoding,
        "filter": filt,
        "compression": compression,
        "attrs": {k: v for k, v in attrs.items() if not k.startswith("_tensogram_")},
    }


def serialize_zarr_json(meta: dict[str, Any]) -> bytes:
    """Serialize a zarr.json dict to UTF-8 JSON bytes.

    Non-finite float values (NaN, Infinity, -Infinity) are converted to
    their Zarr v3 string sentinels so the output is valid RFC 8259 JSON.
    """
    safe = _json_safe_metadata(meta)
    return json.dumps(safe, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")


def _json_safe_metadata(obj: Any) -> Any:
    """Recursively replace non-finite floats with Zarr v3 string sentinels.

    RFC 8259 forbids bare ``NaN`` / ``Infinity`` tokens in JSON.  Zarr v3
    uses the string values ``"NaN"``, ``"Infinity"``, ``"-Infinity"`` for
    fill_value and similar fields.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe_metadata(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_metadata(v) for v in obj]
    return obj


def deserialize_zarr_json(data: bytes) -> dict[str, Any]:
    """Deserialize UTF-8 JSON bytes to a dict.

    Raises ``ValueError`` with context if the data is not valid JSON.
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        preview = data[:80].hex()
        raise ValueError(
            f"invalid zarr.json content ({len(data)} bytes, starts {preview!s}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Variable naming
# ---------------------------------------------------------------------------

# Dotted-path metadata keys to try for variable naming, in priority order.
_VARIABLE_NAME_KEYS = [
    "name",
    "mars.param",
    "param",
    "mars.shortName",
    "shortName",
]


def resolve_variable_name(
    obj_index: int,
    per_object_meta: dict[str, Any] | None,
    common_meta: dict[str, Any] | None = None,
    variable_key: str | None = None,
) -> str:
    """Determine the Zarr variable name for a TGM data object.

    Tries ``variable_key`` first if given, then ``_VARIABLE_NAME_KEYS``,
    then falls back to ``object_<index>``.

    The lookup runs against whatever per-object dict the caller supplies.
    ``TensogramStore._resolve_names`` passes a shallow root-key merge of
    ``meta.base[i]`` and ``desc.params`` (base wins same-key), matching
    the xarray backend's ``scanner._merge_per_object_meta``.  Callers
    that want pure ``meta.base[i]`` semantics can simply pass that.

    ``common_meta`` (historically from ``meta.extra``) is accepted for
    API compatibility but is **not** searched — variable names must come
    from per-object metadata to avoid all objects in a message sharing
    the same name.
    """
    source = per_object_meta or {}

    # Try explicit key first
    keys_to_try = [variable_key] if variable_key else []
    keys_to_try.extend(_VARIABLE_NAME_KEYS)

    for key in keys_to_try:
        val = _dotted_get(source, key)
        if val is not None:
            return str(val)

    return f"object_{obj_index}"


def _dotted_get(d: dict[str, Any], path: str) -> Any:
    """Resolve a dotted key path like ``mars.param`` in a nested dict."""
    parts = path.split(".")
    current: Any = d
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current
