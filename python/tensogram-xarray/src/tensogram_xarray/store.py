# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Data store: bridge between tensogram messages and xarray Variables.

``TensogramDataStore`` reads a single tensogram message (identified by file
path and message index) and produces :class:`xarray.Variable` objects with
lazy-loaded data backed by :class:`TensogramBackendArray`.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from xarray.core import indexing

from tensogram_xarray.array import TensogramBackendArray, _supports_range_decode
from tensogram_xarray.coords import detect_coords
from tensogram_xarray.mapping import resolve_dim_names, resolve_variable_name
from tensogram_xarray.scanner import RESERVED_KEY

logger = logging.getLogger(__name__)

# Map tensogram dtype strings to numpy dtypes.
try:
    import ml_dtypes

    _BFLOAT16_DTYPE = ml_dtypes.bfloat16
except ImportError:
    _BFLOAT16_DTYPE = np.dtype("uint16")  # fallback: raw 2-byte words

_DTYPE_MAP: dict[str, np.dtype] = {
    "float16": np.dtype("float16"),
    "bfloat16": np.dtype(_BFLOAT16_DTYPE),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "complex64": np.dtype("complex64"),
    "complex128": np.dtype("complex128"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "uint64": np.dtype("uint64"),
    "bitmask": np.dtype("uint8"),  # packed bits as raw bytes
}


def _to_numpy_dtype(tgm_dtype: str) -> np.dtype:
    """Convert a tensogram dtype string to a numpy dtype."""
    key = tgm_dtype.lower()
    if key in _DTYPE_MAP:
        return _DTYPE_MAP[key]
    # Fallback: try numpy directly.
    try:
        return np.dtype(tgm_dtype)
    except TypeError as exc:
        msg = f"unsupported tensogram dtype {tgm_dtype!r}: {exc}"
        raise TypeError(msg) from exc


class TensogramDataStore:
    """Read-only data store for a single tensogram message.

    Parameters
    ----------
    file_path
        Path or remote URL to the ``.tgm`` file.
    msg_index
        Index of the message within the file.
    dim_names
        Optional user-specified dimension names for data variables.
    variable_key
        Optional dotted metadata path for variable naming.
    verify_hash
        Whether to verify object hashes on decode.
    range_threshold
        Maximum fraction of total array elements (0.0-1.0) for which
        partial ``decode_range()`` is used.  Default ``0.5``.
    storage_options
        Key-value pairs forwarded to the object store backend when
        ``file_path`` is a remote URL (S3, GCS, Azure, HTTP).  Used
        for credentials, region, endpoint overrides, etc.  Ignored
        for local files.
    """

    def __init__(
        self,
        file_path: str,
        msg_index: int = 0,
        dim_names: Sequence[str] | None = None,
        variable_key: str | None = None,
        verify_hash: bool = False,
        range_threshold: float = 0.5,
        storage_options: dict[str, Any] | None = None,
    ):
        import tensogram

        self._is_remote = tensogram.is_remote_url(file_path)
        self.file_path = file_path if self._is_remote else os.path.abspath(file_path)
        self.msg_index = msg_index
        self.dim_names = dim_names
        self.variable_key = variable_key
        self.verify_hash = verify_hash
        self.range_threshold = range_threshold
        self.storage_options = storage_options
        self._lock = threading.Lock()
        self._backend_arrays: list[TensogramBackendArray] = []

        self._file = self._open_file()
        self._meta, self._descriptors = self._read_metadata()

    def _open_file(self) -> Any:
        import tensogram

        if self._is_remote:
            return tensogram.TensogramFile.open_remote(self.file_path, self.storage_options or {})
        return tensogram.TensogramFile.open(self.file_path)

    def _read_metadata(self) -> tuple[Any, list]:
        import tensogram

        if self._is_remote:
            result = self._file.file_decode_descriptors(self.msg_index)
            return result["metadata"], result["descriptors"]

        raw = self._file.read_message(self.msg_index)
        meta = tensogram.decode_metadata(raw)
        _, descriptors = tensogram.decode_descriptors(raw)
        return meta, descriptors

    def _get_common_meta(self) -> dict[str, Any]:
        """Extract message-level metadata for Dataset attributes.

        Reads from ``meta.extra`` (message-level annotations).
        """
        attrs: dict[str, Any] = {}
        extra = getattr(self._meta, "extra", None)
        if extra and isinstance(extra, dict):
            attrs.update(extra)
        attrs["tensogram_version"] = getattr(self._meta, "version", 2)
        return attrs

    def _get_per_object_meta(self, obj_index: int, desc: Any) -> dict[str, Any]:
        """Extract per-object metadata.

        Reads from ``meta.base[obj_index]`` (primary), filtering out the
        ``_reserved_`` key (encoder-populated tensor info).  Then merges in
        ``desc.params`` (fallback for extra keys in the descriptor dict).

        If ``obj_index`` is out of range (fewer base entries than objects),
        a warning is logged and the base entry is treated as empty.
        """
        meta: dict[str, Any] = {}
        # Primary source: meta.base[i]
        base = getattr(self._meta, "base", None)
        if base is not None and isinstance(base, list):
            if obj_index < len(base):
                entry = base[obj_index]
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        if k != RESERVED_KEY:
                            meta[k] = v
            else:
                logger.warning(
                    "meta.base has %d entries but object index %d requested; "
                    "per-object metadata will be empty for this object",
                    len(base),
                    obj_index,
                )
        # Fallback/supplement: desc.params (extra descriptor keys)
        params = getattr(desc, "params", None)
        if params and isinstance(params, dict):
            for k, v in params.items():
                if k not in meta:
                    meta[k] = v
        return meta

    def build_dataset(
        self,
        drop_variables: set[str] | None = None,
    ) -> xr.Dataset:
        """Construct an :class:`xr.Dataset` from this message.

        Coordinate objects are auto-detected by name matching.  Data objects
        become lazy-loaded variables.  All metadata flows to attributes.
        """
        dataset_attrs = self._get_common_meta()

        # Gather per-object metadata from payload + descriptor params.
        obj_metas = [self._get_per_object_meta(i, d) for i, d in enumerate(self._descriptors)]

        # Detect coordinates vs data variables.
        coord_indices, var_indices, coord_dim_names = detect_coords(obj_metas)

        # Build coordinate variables from detected coord objects.
        coord_vars: dict[str, xr.Variable] = {}
        for ci in coord_indices:
            desc = self._descriptors[ci]
            dim_name = coord_dim_names[ci]
            np_dtype = _to_numpy_dtype(desc.dtype)
            shape = tuple(desc.shape)

            backend_array = TensogramBackendArray(
                file_path=self.file_path,
                msg_index=self.msg_index,
                obj_index=ci,
                shape=shape,
                dtype=np_dtype,
                supports_range=_supports_range_decode(desc),
                verify_hash=self.verify_hash,
                range_threshold=self.range_threshold,
                lock=self._lock,
                storage_options=self.storage_options,
                shared_file=self._file,
            )
            self._backend_arrays.append(backend_array)
            lazy_data = indexing.LazilyIndexedArray(backend_array)

            coord_dims = (dim_name,)
            coord_attrs = dict(obj_metas[ci])
            coord_vars[dim_name] = xr.Variable(coord_dims, lazy_data, coord_attrs)

        data_vars: dict[str, xr.Variable] = {}
        for vi in var_indices:
            desc = self._descriptors[vi]
            var_name = resolve_variable_name(vi, obj_metas[vi], self.variable_key)
            if drop_variables and var_name in drop_variables:
                continue

            np_dtype = _to_numpy_dtype(desc.dtype)
            shape = tuple(desc.shape)

            dims = self._resolve_dims_for_var(shape, coord_vars)

            backend_array = TensogramBackendArray(
                file_path=self.file_path,
                msg_index=self.msg_index,
                obj_index=vi,
                shape=shape,
                dtype=np_dtype,
                supports_range=_supports_range_decode(desc),
                verify_hash=self.verify_hash,
                range_threshold=self.range_threshold,
                lock=self._lock,
                storage_options=self.storage_options,
                shared_file=self._file,
            )
            self._backend_arrays.append(backend_array)
            lazy_data = indexing.LazilyIndexedArray(backend_array)

            var_attrs = dict(obj_metas[vi])
            data_vars[var_name] = xr.Variable(dims, lazy_data, var_attrs)

        # Assemble Dataset.
        ds = xr.Dataset(data_vars, coords=coord_vars, attrs=dataset_attrs)
        return ds

    def _get_meta_dim_names(self, ndim: int) -> list[str] | dict[int, str]:
        """Return dimension name hints from ``_extra_["dim_names"]``.

        Any writer can embed hints at ``_extra_["dim_names"]`` to provide
        meaningful dimension names without the reader passing ``dim_names``
        explicitly. Two formats are accepted:

        **List (preferred)** — axis-ordered names, one per dimension::

            "_extra_": {"dim_names": ["values", "level"]}

        This handles axes with identical sizes correctly because names
        are assigned by position.

        **Dict (legacy)** — size-to-name mapping::

            "_extra_": {"dim_names": {"1000": "values", "50": "level"}}

        A dict cannot disambiguate axes with the same size; only the
        first matching axis receives the hint.

        Returns an empty list/dict when the key is absent or malformed.
        """
        try:
            raw = self._meta.extra["dim_names"]
        except (AttributeError, KeyError, TypeError):
            return {}

        # List format: axis-ordered names.
        if isinstance(raw, list):
            try:
                names = [str(n) for n in raw]
                if len(names) == ndim:
                    return names
                # Length mismatch — ignore the hint rather than crash.
                return {}
            except (TypeError, ValueError):
                return {}

        # Dict format: size -> name (legacy).
        if isinstance(raw, dict):
            try:
                return {int(k): str(v) for k, v in raw.items()}
            except (TypeError, ValueError):
                return {}

        return {}

    def _resolve_dims_for_var(
        self,
        shape: tuple[int, ...],
        coord_vars: dict[str, xr.Variable],
    ) -> tuple[str, ...]:
        """Assign dimension names for a data variable.

        Strategy:
        1. If user provided ``dim_names``, use them directly.
        2. Try to match each axis size against a known coordinate variable.
        3. Use producer hints from ``_extra_["dim_names"]`` (list or dict).
        4. Fall back to ``dim_0``, ``dim_1``, ...
        """
        ndim = len(shape)

        # (1) Explicit user mapping.
        if self.dim_names is not None:
            return tuple(resolve_dim_names(ndim, self.dim_names))

        # (2) Match by size against detected coordinates.
        # Build a map: size -> list of coord dim names with that size.
        size_to_coord: dict[int, list[str]] = {}
        for cname, cvar in coord_vars.items():
            csize = cvar.shape[0]
            size_to_coord.setdefault(csize, []).append(cname)

        # (3) Metadata hints written by the producer.
        meta_hints = self._get_meta_dim_names(ndim)

        # If the producer supplied an axis-ordered list, use it directly.
        if isinstance(meta_hints, list):
            return tuple(meta_hints)

        # Otherwise meta_hints is a dict (size -> name); match by size.
        dims: list[str] = []
        used: set[str] = set()
        for axis, axis_size in enumerate(shape):
            matched = False
            # Prefer a detected coordinate dimension.
            if axis_size in size_to_coord:
                for cname in size_to_coord[axis_size]:
                    if cname not in used:
                        dims.append(cname)
                        used.add(cname)
                        matched = True
                        break
            # Fall back to producer hint (dict).
            if not matched and axis_size in meta_hints:
                hint = meta_hints[axis_size]
                if hint not in used:
                    dims.append(hint)
                    used.add(hint)
                    matched = True
            # Final fallback.
            if not matched:
                dims.append(f"dim_{axis}")

        return tuple(dims)

    def close(self) -> None:
        for arr in self._backend_arrays:
            arr._shared_file = None
        self._backend_arrays.clear()
        self._file = None
