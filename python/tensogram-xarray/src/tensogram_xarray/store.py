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
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from xarray.core import indexing

from tensogram_xarray.array import TensogramBackendArray, _supports_range_decode
from tensogram_xarray.coords import detect_coords
from tensogram_xarray.mapping import (
    EXTRA_DIM_NAMES_KEY,
    resolve_dims_for_axes,
    resolve_variable_name,
    strip_structural_keys,
)
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


# ---------------------------------------------------------------------------
# Dimension resolution helpers
# ---------------------------------------------------------------------------


@dataclass
class _DataVarPlan:
    """Per-variable plan gathered during Dataset construction.

    Collected in a first pass so that a second pass can disambiguate generic
    fallback dimension names that would otherwise cause xarray Dataset size
    collisions across variables of different shapes.
    """

    obj_index: int
    var_name: str
    shape: tuple[int, ...]
    # One entry per axis: (dim_name, came_from_generic_fallback).  A generic
    # fallback is the `dim_{axis}` name produced when no user kwarg, coord
    # size-match, per-object ``base[i]["dim_names"]``, or
    # ``_extra_["dim_names"]`` hint provides a name for that axis.
    dims_with_provenance: list[tuple[str, bool]]
    backend_array: TensogramBackendArray
    var_attrs: dict[str, Any]


def _disambiguate_fallback_dims(
    plans: list[_DataVarPlan],
    coord_dim_sizes: dict[str, int],
) -> list[tuple[str, ...]]:
    """Return final dimension names per *plan*, resolving every size conflict.

    A conflict exists when the same dim name is claimed at two or more
    distinct sizes — either across data variables or against an existing
    coordinate dim.

    Resolution per axis:

    * **Legitimate coord match** — an axis whose name matches an existing
      coord dim AND whose size equals the coord's size is preserved, even
      if some *other* plan wrongly claims the same name at a different
      size.  The coord-sharing binding is the correct semantics here; only
      the offending plan's axis is renamed.
    * **Generic fallback** conflicts (``dim_{axis}`` with
      ``is_generic_fallback=True``) are silently renamed to
      ``f"obj_{obj_index}_dim_{axis}"``.
    * **Hinted** conflicts (names from user kwargs, coord-match on a
      non-matching size, per-object ``base[i]["dim_names"]``, or
      ``_extra_["dim_names"]``) are renamed to the same
      ``obj_{i}_dim_{axis}`` form *with a warning* so the Dataset stays
      openable rather than crashing at :class:`xr.Dataset` assembly.

    Coordinate dim names themselves are never renamed (coords claim a
    single size in ``coord_dim_sizes`` and so never conflict with
    themselves); they only participate in detection.

    Parameters
    ----------
    plans
        One :class:`_DataVarPlan` per data variable, in the order they
        will appear in the resulting Dataset.
    coord_dim_sizes
        Map from existing coordinate dim name to its size.

    Returns
    -------
    list[tuple[str, ...]]
        Final dim names per plan, parallel to *plans*.
    """
    name_to_sizes: dict[str, set[int]] = {
        cname: {csize} for cname, csize in coord_dim_sizes.items()
    }
    for plan in plans:
        for axis, (dim_name, _is_generic) in enumerate(plan.dims_with_provenance):
            name_to_sizes.setdefault(dim_name, set()).add(plan.shape[axis])

    conflicting = {name for name, sizes in name_to_sizes.items() if len(sizes) > 1}
    warned: set[str] = set()

    resolved: list[tuple[str, ...]] = []
    for plan in plans:
        new_dims: list[str] = []
        for axis, (dim_name, is_generic) in enumerate(plan.dims_with_provenance):
            if dim_name in conflicting:
                if dim_name in coord_dim_sizes and plan.shape[axis] == coord_dim_sizes[dim_name]:
                    new_dims.append(dim_name)
                    continue
                if not is_generic and dim_name not in warned:
                    logger.warning(
                        "dimension name %r claimed at conflicting sizes %s across "
                        "objects; falling back to object-local names for the "
                        "conflicting axes (check user dim_names kwarg, _extra_, "
                        "or per-object hints for an inconsistent producer)",
                        dim_name,
                        sorted(name_to_sizes[dim_name]),
                    )
                    warned.add(dim_name)
                new_dims.append(f"obj_{plan.obj_index}_dim_{axis}")
            else:
                new_dims.append(dim_name)
        resolved.append(tuple(new_dims))

    return resolved


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
        range_threshold: float = 0.5,
        storage_options: dict[str, Any] | None = None,
    ):
        import tensogram

        self._is_remote = tensogram.is_remote_url(file_path)
        self.file_path = file_path if self._is_remote else os.path.abspath(file_path)
        self.msg_index = msg_index
        self.dim_names = dim_names
        self.variable_key = variable_key
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

        Reads from ``meta.extra`` (message-level annotations) and filters
        out structural keys such as ``dim_names`` so they do not leak from
        the wire format into user-visible :attr:`xr.Dataset.attrs`.
        """
        attrs: dict[str, Any] = {}
        extra = getattr(self._meta, "extra", None)
        if extra and isinstance(extra, dict):
            attrs.update(strip_structural_keys(extra))
        # The wire-format version lives in the preamble (see
        # ``plans/WIRE_FORMAT.md`` §3).  ``meta.version`` sources from
        # that preamble-derived value; we surface it as a namespaced
        # attribute so downstream xarray tooling knows which wire
        # format the file was encoded against.  The attribute name
        # changed from ``tensogram_version`` to
        # ``tensogram_wire_version`` in 0.17 to avoid collisions with
        # any user-supplied ``version`` key, which now flows through
        # ``_extra_`` like every other free-form annotation.
        attrs["tensogram_wire_version"] = getattr(
            self._meta,
            "version",
            # Fallback for producers that somehow decoded without a
            # preamble — should never happen in practice.
            3,
        )
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

        obj_metas = [self._get_per_object_meta(i, d) for i, d in enumerate(self._descriptors)]
        extra_dim_names_hint = self._extra_dim_names_hint()

        coord_indices, var_indices, coord_dim_names = detect_coords(obj_metas)

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
                range_threshold=self.range_threshold,
                lock=self._lock,
                storage_options=self.storage_options,
                shared_file=self._file,
            )
            self._backend_arrays.append(backend_array)
            lazy_data = indexing.LazilyIndexedArray(backend_array)

            coord_vars[dim_name] = xr.Variable(
                (dim_name,), lazy_data, strip_structural_keys(obj_metas[ci])
            )

        coord_dim_sizes = {name: var.shape[0] for name, var in coord_vars.items()}

        plans: list[_DataVarPlan] = []
        for vi in var_indices:
            desc = self._descriptors[vi]
            var_name = resolve_variable_name(vi, obj_metas[vi], self.variable_key)
            if drop_variables and var_name in drop_variables:
                continue

            np_dtype = _to_numpy_dtype(desc.dtype)
            shape = tuple(desc.shape)

            dims_with_provenance = resolve_dims_for_axes(
                shape,
                user_dim_names=self.dim_names,
                coord_dim_sizes=coord_dim_sizes,
                per_object_meta=obj_metas[vi],
                extra_dim_names_hint=extra_dim_names_hint,
            )

            backend_array = TensogramBackendArray(
                file_path=self.file_path,
                msg_index=self.msg_index,
                obj_index=vi,
                shape=shape,
                dtype=np_dtype,
                supports_range=_supports_range_decode(desc),
                range_threshold=self.range_threshold,
                lock=self._lock,
                storage_options=self.storage_options,
                shared_file=self._file,
            )
            self._backend_arrays.append(backend_array)

            plans.append(
                _DataVarPlan(
                    obj_index=vi,
                    var_name=var_name,
                    shape=shape,
                    dims_with_provenance=dims_with_provenance,
                    backend_array=backend_array,
                    var_attrs=strip_structural_keys(obj_metas[vi]),
                )
            )

        resolved_dims = _disambiguate_fallback_dims(plans, coord_dim_sizes)

        data_vars: dict[str, xr.Variable] = {}
        for plan, dims in zip(plans, resolved_dims, strict=True):
            lazy_data = indexing.LazilyIndexedArray(plan.backend_array)
            data_vars[plan.var_name] = xr.Variable(dims, lazy_data, plan.var_attrs)

        return xr.Dataset(data_vars, coords=coord_vars, attrs=dataset_attrs)

    def _extra_dim_names_hint(self) -> Any:
        """Return the raw ``_extra_["dim_names"]`` payload, or ``None``."""
        try:
            return self._meta.extra[EXTRA_DIM_NAMES_KEY]
        except (AttributeError, KeyError, TypeError):
            return None

    def close(self) -> None:
        for arr in self._backend_arrays:
            arr._shared_file = None
        self._backend_arrays.clear()
        self._file = None
