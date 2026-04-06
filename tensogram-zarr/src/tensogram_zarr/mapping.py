"""Bidirectional mapping between Tensogram and Zarr v3 metadata.

Converts TGM dtypes, descriptors, and global metadata into Zarr v3
``zarr.json`` structures and back.
"""

from __future__ import annotations

import json
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

    # Merge common metadata into attributes
    if hasattr(meta, "common") and meta.common:
        attrs.update(meta.common)

    # Merge extra metadata
    if hasattr(meta, "extra") and meta.extra:
        attrs.update(meta.extra)

    attrs["_tensogram_version"] = meta.version
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
        Per-object metadata from ``meta.payload[i]``.

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
    """Serialize a zarr.json dict to UTF-8 JSON bytes."""
    return json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")


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
    """
    sources = [per_object_meta or {}, common_meta or {}]

    # Try explicit key first
    keys_to_try = [variable_key] if variable_key else []
    keys_to_try.extend(_VARIABLE_NAME_KEYS)

    for key in keys_to_try:
        if key is None:
            continue
        for src in sources:
            val = _dotted_get(src, key)
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
