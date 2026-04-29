# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Auto-merge and auto-split for multi-message tensogram files.

``open_datasets()`` scans all messages in a ``.tgm`` file, groups compatible
data objects (same shape, dtype, metadata structure) into hypercubes, and
returns a list of :class:`xr.Dataset` instances.  Incompatible objects are
automatically split into separate Datasets.
"""

from __future__ import annotations

import itertools
import logging
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import xarray as xr
from xarray.core import indexing

from tensogram_xarray.array import (
    StackedBackendArray,
    TensogramBackendArray,
    _supports_range_decode,
)
from tensogram_xarray.coords import detect_coords
from tensogram_xarray.mapping import (
    EXTRA_DIM_NAMES_KEY,
    STRUCTURAL_META_KEYS,
    parse_per_object_dim_names,
    resolve_dims_for_axes,
    resolve_variable_name,
    strip_structural_keys,
)
from tensogram_xarray.scanner import ObjectInfo, scan_file
from tensogram_xarray.store import _to_numpy_dtype

logger = logging.getLogger(__name__)


def open_datasets(
    path: str,
    *,
    dim_names: Sequence[str] | None = None,
    variable_key: str | None = None,
    range_threshold: float = 0.5,
    storage_options: dict[str, Any] | None = None,
) -> list[xr.Dataset]:
    """Open a ``.tgm`` file, auto-grouping into compatible Datasets.

    Each returned Dataset represents a group of data objects that share
    compatible shapes and metadata structure.  Objects whose metadata varies
    on certain keys are stacked along new outer dimensions.

    Parameters
    ----------
    path
        Path or remote URL (S3, GCS, Azure, HTTP) to the ``.tgm`` file.
    dim_names
        Explicit dimension names for the innermost tensor axes.
    variable_key
        Dotted metadata key path for variable naming.
    range_threshold
        Maximum fraction of total array elements for which partial
        ``decode_range()`` is used.  Default ``0.5``.
    storage_options
        Key-value pairs forwarded to the object store backend for
        remote URLs.  Ignored for local files.

    Returns
    -------
    list[xr.Dataset]
        One Dataset per compatible group.
    """
    file_index = scan_file(path, storage_options=storage_options)

    if not file_index.objects:
        return []

    import tensogram

    is_remote = tensogram.is_remote_url(path)
    shared_file = None
    if is_remote:
        shared_file = tensogram.TensogramFile.open_remote(path, storage_options or {})

    all_metas = [o.merged_meta for o in file_index.objects]
    coord_indices, var_indices, coord_dim_map = detect_coords(all_metas)

    lock = threading.Lock()
    all_backend_arrays: list[TensogramBackendArray] = []
    coord_vars: dict[str, xr.Variable] = {}
    for ci in coord_indices:
        obj = file_index.objects[ci]
        dim_name = coord_dim_map[ci]
        np_dtype = _to_numpy_dtype(obj.dtype)
        shape = obj.shape

        backend_array = TensogramBackendArray(
            file_path=path,
            msg_index=obj.msg_index,
            obj_index=obj.obj_index,
            shape=shape,
            dtype=np_dtype,
            supports_range=_supports_range_decode(obj.descriptor),
            range_threshold=range_threshold,
            lock=lock,
            storage_options=storage_options,
            shared_file=shared_file,
        )
        all_backend_arrays.append(backend_array)
        lazy_data = indexing.LazilyIndexedArray(backend_array)

        if dim_name in coord_vars:
            existing = coord_vars[dim_name]
            if existing.shape != shape:
                msg = (
                    f"coordinate {dim_name!r} has conflicting shapes: "
                    f"existing {existing.shape} vs new {shape} "
                    f"(msg_index={obj.msg_index}, obj_index={obj.obj_index})"
                )
                raise ValueError(msg)
            # Duplicate with matching shape -- skip (keep the first).
            continue

        coord_vars[dim_name] = xr.Variable(
            (dim_name,), lazy_data, strip_structural_keys(obj.per_object_meta)
        )

    # Group data objects by structural compatibility.
    data_objects = [file_index.objects[i] for i in var_indices]
    groups = _group_by_structure(data_objects)

    datasets: list[xr.Dataset] = []
    for group in groups:
        ds = _build_dataset_from_group(
            group,
            file_path=path,
            coord_vars=coord_vars,
            dim_names=dim_names,
            variable_key=variable_key,
            lock=lock,
            range_threshold=range_threshold,
            storage_options=storage_options,
            shared_file=shared_file,
            backend_arrays=all_backend_arrays,
        )
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        datasets = [xr.Dataset(coords=coord_vars, attrs={"source": path})]

    if shared_file is not None:

        def _close_shared():
            nonlocal shared_file
            for arr in all_backend_arrays:
                arr._shared_file = None
            all_backend_arrays.clear()
            shared_file = None

        for ds in datasets:
            ds.set_close(_close_shared)

    return datasets


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

_StructureKey = tuple[tuple[int, ...], str]  # (shape, dtype)


def _group_by_structure(
    objects: list[ObjectInfo],
) -> list[list[ObjectInfo]]:
    """Group objects by (shape, dtype) -- structural compatibility."""
    buckets: dict[_StructureKey, list[ObjectInfo]] = defaultdict(list)
    for obj in objects:
        key: _StructureKey = (obj.shape, obj.dtype)
        buckets[key].append(obj)
    return list(buckets.values())


# ---------------------------------------------------------------------------
# Hypercube construction
# ---------------------------------------------------------------------------


def _extract_meta_keys(objects: list[ObjectInfo]) -> dict[str, list[Any]]:
    """For each (non-structural) metadata key, collect values across all objects.

    :data:`STRUCTURAL_META_KEYS` are skipped so xarray-structural hints
    (currently ``dim_names``) never become a hypercube outer dimension
    or a partition key in the grouping pipeline.
    """
    all_keys: set[str] = set()
    for obj in objects:
        all_keys.update(k for k in obj.merged_meta if k not in STRUCTURAL_META_KEYS)

    key_values: dict[str, list[Any]] = {}
    for k in sorted(all_keys):
        values = []
        for obj in objects:
            values.append(obj.merged_meta.get(k))
        key_values[k] = values
    return key_values


def _partition_keys(
    key_values: dict[str, list[Any]],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Split keys into constant (attrs) and varying (candidate dims).

    Returns
    -------
    constant
        Keys with a single unique value -> Dataset attributes.
    varying
        Keys with multiple unique values -> candidate outer dimensions.
    """
    constant: dict[str, Any] = {}
    varying: dict[str, list[Any]] = {}

    for k, values in key_values.items():
        # Convert to hashable for uniqueness check.
        try:
            unique = set(_make_hashable(v) for v in values)
        except TypeError:
            # Unhashable values (dicts, lists) -> treat as attribute.
            constant[k] = values[0]
            continue

        if len(unique) == 1:
            constant[k] = values[0]
        else:
            varying[k] = values

    return constant, varying


def _make_hashable(val: Any) -> Any:
    """Convert a value to a hashable form for set operations."""
    if isinstance(val, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
    if isinstance(val, list):
        return tuple(_make_hashable(v) for v in val)
    return val


def _unique_values(values: list[Any]) -> list[Any]:
    """Return unique values preserving order, handling unhashable types."""
    seen: set[Any] = set()
    result: list[Any] = []
    for v in values:
        h = _make_hashable(v)
        if h not in seen:
            seen.add(h)
            result.append(v)
    return result


def _try_hypercube(
    objects: list[ObjectInfo],
    varying: dict[str, list[Any]],
) -> bool:
    """Check whether the varying keys form a complete hypercube.

    A complete hypercube means every combination of unique values across
    all varying keys has exactly one corresponding object.
    """
    if not varying:
        return True

    # Unique values per key (use _make_hashable for unhashable types like dicts).
    unique_per_key: dict[str, list[Any]] = {}
    for k, v in varying.items():
        seen: set[Any] = set()
        unique: list[Any] = []
        for val in v:
            h = _make_hashable(val)
            if h not in seen:
                seen.add(h)
                unique.append(val)
        unique_per_key[k] = unique

    # Total expected combinations.
    expected = 1
    for vals in unique_per_key.values():
        expected *= len(vals)

    return len(objects) == expected


def _split_by_key(
    objects: list[ObjectInfo],
    key: str,
) -> list[list[ObjectInfo]]:
    """Split objects into sub-groups by the values of *key*."""
    buckets: dict[Any, list[ObjectInfo]] = defaultdict(list)
    for obj in objects:
        val = obj.merged_meta.get(key)
        hval = _make_hashable(val)
        buckets[hval].append(obj)
    return list(buckets.values())


# ---------------------------------------------------------------------------
# Dataset construction from a compatible group
# ---------------------------------------------------------------------------


def _build_dataset_from_group(
    group: list[ObjectInfo],
    file_path: str,
    coord_vars: dict[str, xr.Variable],
    dim_names: Sequence[str] | None,
    variable_key: str | None,
    lock: threading.Lock,
    range_threshold: float = 0.5,
    storage_options: dict[str, Any] | None = None,
    *,
    shared_file: Any = None,
    backend_arrays: list | None = None,
) -> xr.Dataset | None:
    """Build a Dataset from a group of structurally compatible objects.

    If the group has a single object, it produces a simple Dataset.
    Multiple objects are merged along varying metadata keys when they
    form a clean hypercube; otherwise auto-split is attempted.
    """
    if not group:
        return None

    # Single object -> simple Dataset.
    if len(group) == 1:
        return _single_object_dataset(
            group[0],
            file_path,
            coord_vars,
            dim_names,
            variable_key,
            lock,
            range_threshold=range_threshold,
            storage_options=storage_options,
            shared_file=shared_file,
            backend_arrays=backend_arrays,
        )

    # Multiple objects -> try hypercube merge.
    key_values = _extract_meta_keys(group)
    constant, varying = _partition_keys(key_values)

    # If variable_key is specified, split by it first (each unique value
    # becomes a separate variable in the Dataset).  Use resolved variable
    # names rather than membership in ``varying`` so dotted keys such as
    # "mars.param" are handled correctly.
    if variable_key is not None:
        variable_names = {
            resolve_variable_name(obj.obj_index, obj.per_object_meta, variable_key)
            for obj in group
        }
        if len(variable_names) > 1:
            return _build_multi_variable_dataset(
                group,
                file_path,
                coord_vars,
                dim_names,
                variable_key,
                constant,
                varying,
                lock,
                range_threshold=range_threshold,
                storage_options=storage_options,
                shared_file=shared_file,
                backend_arrays=backend_arrays,
            )

    # Check if the varying keys form a hypercube.
    if not varying:
        # All metadata identical -> can't distinguish objects.
        # Return each as object_0, object_1, ...
        return _flat_group_dataset(
            group,
            file_path,
            coord_vars,
            dim_names,
            variable_key,
            constant,
            lock,
            range_threshold=range_threshold,
            storage_options=storage_options,
            shared_file=shared_file,
            backend_arrays=backend_arrays,
        )

    if _try_hypercube(group, varying):
        return _hypercube_dataset(
            group,
            file_path,
            coord_vars,
            dim_names,
            variable_key,
            constant,
            varying,
            lock,
            range_threshold=range_threshold,
            storage_options=storage_options,
            shared_file=shared_file,
            backend_arrays=backend_arrays,
        )

    # Hypercube incomplete -> just return as separate variables.
    return _flat_group_dataset(
        group,
        file_path,
        coord_vars,
        dim_names,
        variable_key,
        constant,
        lock,
        range_threshold=range_threshold,
        storage_options=storage_options,
        shared_file=shared_file,
    )


def _single_object_dataset(
    obj: ObjectInfo,
    file_path: str,
    coord_vars: dict[str, xr.Variable],
    dim_names: Sequence[str] | None,
    variable_key: str | None,
    lock: threading.Lock,
    range_threshold: float = 0.5,
    storage_options: dict[str, Any] | None = None,
    *,
    shared_file: Any = None,
    backend_arrays: list | None = None,
) -> xr.Dataset:
    """Build a Dataset from a single object."""
    np_dtype = _to_numpy_dtype(obj.dtype)
    shape = obj.shape

    var_name = resolve_variable_name(obj.obj_index, obj.merged_meta, variable_key)
    dims = _resolve_inner_dims(
        [obj],
        shape,
        user_dim_names=dim_names,
        coord_vars=coord_vars,
    )

    backend_array = TensogramBackendArray(
        file_path=file_path,
        msg_index=obj.msg_index,
        obj_index=obj.obj_index,
        shape=shape,
        dtype=np_dtype,
        supports_range=_supports_range_decode(obj.descriptor),
        range_threshold=range_threshold,
        lock=lock,
        storage_options=storage_options,
        shared_file=shared_file,
    )
    if backend_arrays is not None:
        backend_arrays.append(backend_array)
    lazy_data = indexing.LazilyIndexedArray(backend_array)
    var = xr.Variable(dims, lazy_data, strip_structural_keys(obj.merged_meta))

    ds_attrs = strip_structural_keys(obj.common_meta)
    ds = xr.Dataset({var_name: var}, coords=coord_vars, attrs=ds_attrs)
    return ds


def _flat_group_dataset(
    group: list[ObjectInfo],
    file_path: str,
    coord_vars: dict[str, xr.Variable],
    dim_names: Sequence[str] | None,
    variable_key: str | None,
    constant: dict[str, Any],
    lock: threading.Lock,
    range_threshold: float = 0.5,
    storage_options: dict[str, Any] | None = None,
    *,
    shared_file: Any = None,
    backend_arrays: list | None = None,
) -> xr.Dataset:
    """Build a Dataset with one variable per object (no stacking)."""
    data_vars: dict[str, xr.Variable] = {}

    inner_shape = group[0].shape
    dims = _resolve_inner_dims(
        group,
        inner_shape,
        user_dim_names=dim_names,
        coord_vars=coord_vars,
    )

    for obj in group:
        np_dtype = _to_numpy_dtype(obj.dtype)
        var_name = resolve_variable_name(obj.obj_index, obj.per_object_meta, variable_key)

        backend_array = TensogramBackendArray(
            file_path=file_path,
            msg_index=obj.msg_index,
            obj_index=obj.obj_index,
            shape=obj.shape,
            dtype=np_dtype,
            supports_range=_supports_range_decode(obj.descriptor),
            range_threshold=range_threshold,
            lock=lock,
            storage_options=storage_options,
            shared_file=shared_file,
        )
        if backend_arrays is not None:
            backend_arrays.append(backend_array)
        lazy_data = indexing.LazilyIndexedArray(backend_array)
        data_vars[var_name] = xr.Variable(dims, lazy_data, strip_structural_keys(obj.merged_meta))

    ds_attrs = strip_structural_keys(constant)
    ds = xr.Dataset(data_vars, coords=coord_vars, attrs=ds_attrs)
    return ds


def _hypercube_dataset(
    group: list[ObjectInfo],
    file_path: str,
    coord_vars: dict[str, xr.Variable],
    dim_names: Sequence[str] | None,
    variable_key: str | None,
    constant: dict[str, Any],
    varying: dict[str, list[Any]],
    lock: threading.Lock,
    range_threshold: float = 0.5,
    storage_options: dict[str, Any] | None = None,
    *,
    shared_file: Any = None,
    backend_arrays: list | None = None,
) -> xr.Dataset:
    """Stack objects into a Dataset with outer dimensions from varying keys.

    All objects in *group* must have the same inner shape.  Varying metadata
    keys become outer dimensions whose coordinate values are the unique
    metadata values.
    """
    inner_shape = group[0].shape
    np_dtype = _to_numpy_dtype(group[0].dtype)
    inner_dims = _resolve_inner_dims(
        group,
        inner_shape,
        user_dim_names=dim_names,
        coord_vars=coord_vars,
    )

    # Determine outer dimension names and coordinate values.
    outer_keys = sorted(varying.keys())
    outer_coords: dict[str, list] = {}
    for k in outer_keys:
        outer_coords[k] = _unique_values(varying[k])

    # Build N-D index mapping: (val_for_key0, val_for_key1, ...) -> ObjectInfo
    obj_by_coord: dict[tuple, ObjectInfo] = {}
    for i, obj in enumerate(group):
        coord_key = tuple(_make_hashable(varying[k][i]) for k in outer_keys)
        obj_by_coord[coord_key] = obj

    # Compute outer shape.
    outer_shape = tuple(len(outer_coords[k]) for k in outer_keys)
    outer_dims = tuple(outer_keys)
    full_dims = outer_dims + inner_dims

    # Build lazy backing arrays for each position in the outer grid
    # (row-major order).  No payload data is decoded here.
    backing_arrays: list[TensogramBackendArray] = []
    for idx_tuple in itertools.product(*(range(s) for s in outer_shape)):
        coord_key = tuple(
            _make_hashable(outer_coords[outer_keys[d]][idx_tuple[d]])
            for d in range(len(outer_keys))
        )
        obj = obj_by_coord.get(coord_key)
        if obj is None:
            msg = (
                f"hypercube has a missing entry at {dict(zip(outer_keys, idx_tuple))} "
                f"in {file_path}"
            )
            raise ValueError(msg)
        backing_arrays.append(
            TensogramBackendArray(
                file_path=file_path,
                msg_index=obj.msg_index,
                obj_index=obj.obj_index,
                shape=inner_shape,
                dtype=np_dtype,
                supports_range=_supports_range_decode(obj.descriptor),
                range_threshold=range_threshold,
                lock=lock,
                storage_options=storage_options,
                shared_file=shared_file,
            )
        )

    if backend_arrays is not None:
        backend_arrays.extend(backing_arrays)
    stacked = StackedBackendArray(backing_arrays, outer_shape, inner_shape, np_dtype)
    lazy_data = indexing.LazilyIndexedArray(stacked)

    var_name = resolve_variable_name(group[0].obj_index, group[0].merged_meta, variable_key)

    # Add outer coordinates.
    merged_coords = dict(coord_vars)
    for k in outer_keys:
        merged_coords[k] = xr.Variable((k,), outer_coords[k])

    var = xr.Variable(full_dims, lazy_data, strip_structural_keys(constant))
    ds = xr.Dataset({var_name: var}, coords=merged_coords, attrs=strip_structural_keys(constant))
    return ds


def _build_multi_variable_dataset(
    group: list[ObjectInfo],
    file_path: str,
    coord_vars: dict[str, xr.Variable],
    dim_names: Sequence[str] | None,
    variable_key: str,
    constant: dict[str, Any],
    varying: dict[str, list[Any]],
    lock: threading.Lock,
    range_threshold: float = 0.5,
    storage_options: dict[str, Any] | None = None,
    *,
    shared_file: Any = None,
    backend_arrays: list | None = None,
) -> xr.Dataset:
    """Split group by variable_key, then stack each sub-group.

    Each unique value of *variable_key* becomes a separate variable in the
    Dataset.  Remaining varying keys become outer dimensions.
    """
    # Split by variable_key value.
    sub_groups: dict[str, list[ObjectInfo]] = defaultdict(list)
    for obj in group:
        val = resolve_variable_name(obj.obj_index, obj.per_object_meta, variable_key)
        sub_groups[val].append(obj)

    # Remaining varying keys (exclude variable_key).
    remaining_varying = {k: v for k, v in varying.items() if k != variable_key}

    data_vars: dict[str, xr.Variable] = {}
    merged_coords = dict(coord_vars)
    inner_shape = group[0].shape
    np_dtype = _to_numpy_dtype(group[0].dtype)

    for var_name, sub_group in sub_groups.items():
        inner_dims = _resolve_inner_dims(
            sub_group,
            inner_shape,
            user_dim_names=dim_names,
            coord_vars=coord_vars,
        )
        if len(sub_group) == 1:
            # Single object for this variable -> no outer dims.
            obj = sub_group[0]
            backend_array = TensogramBackendArray(
                file_path=file_path,
                msg_index=obj.msg_index,
                obj_index=obj.obj_index,
                shape=inner_shape,
                dtype=np_dtype,
                supports_range=_supports_range_decode(obj.descriptor),
                range_threshold=range_threshold,
                lock=lock,
                storage_options=storage_options,
                shared_file=shared_file,
            )
            if backend_arrays is not None:
                backend_arrays.append(backend_array)
            lazy_data = indexing.LazilyIndexedArray(backend_array)
            data_vars[var_name] = xr.Variable(
                inner_dims, lazy_data, strip_structural_keys(obj.merged_meta)
            )
        elif remaining_varying:
            # Re-extract varying keys for this sub-group.
            sub_kv = _extract_meta_keys(sub_group)
            sub_const, sub_vary = _partition_keys(sub_kv)

            if sub_vary and _try_hypercube(sub_group, sub_vary):
                # Build stacked variable.
                outer_keys = sorted(sub_vary.keys())
                outer_coords_local: dict[str, list] = {}
                for k in outer_keys:
                    outer_coords_local[k] = _unique_values(sub_vary[k])

                outer_shape = tuple(len(outer_coords_local[k]) for k in outer_keys)
                outer_dims = tuple(outer_keys)
                full_dims = outer_dims + inner_dims

                obj_by_coord: dict[tuple, ObjectInfo] = {}
                for j, obj in enumerate(sub_group):
                    coord_key = tuple(_make_hashable(sub_vary[k][j]) for k in outer_keys)
                    obj_by_coord[coord_key] = obj

                # Build lazy stacked array (no payload decode here).
                backing: list[TensogramBackendArray] = []
                for idx_tuple in itertools.product(*(range(s) for s in outer_shape)):
                    coord_key = tuple(
                        _make_hashable(outer_coords_local[outer_keys[d]][idx_tuple[d]])
                        for d in range(len(outer_keys))
                    )
                    obj = obj_by_coord.get(coord_key)
                    if obj is None:
                        msg = (
                            f"hypercube has a missing entry at "
                            f"{dict(zip(outer_keys, idx_tuple))} in {file_path}"
                        )
                        raise ValueError(msg)
                    backing.append(
                        TensogramBackendArray(
                            file_path=file_path,
                            msg_index=obj.msg_index,
                            obj_index=obj.obj_index,
                            shape=inner_shape,
                            dtype=np_dtype,
                            supports_range=_supports_range_decode(obj.descriptor),
                            range_threshold=range_threshold,
                            lock=lock,
                            storage_options=storage_options,
                            shared_file=shared_file,
                        )
                    )

                if backend_arrays is not None:
                    backend_arrays.extend(backing)
                stacked = StackedBackendArray(backing, outer_shape, inner_shape, np_dtype)
                lazy_data = indexing.LazilyIndexedArray(stacked)

                for k in outer_keys:
                    merged_coords[k] = xr.Variable((k,), outer_coords_local[k])
                data_vars[var_name] = xr.Variable(
                    full_dims, lazy_data, strip_structural_keys(sub_const)
                )
            else:
                # Can't form hypercube -> use first object only.
                logger.warning(
                    "variable %r: %d objects cannot form a hypercube, "
                    "using only the first object (dropping %d)",
                    var_name,
                    len(sub_group),
                    len(sub_group) - 1,
                )
                obj = sub_group[0]
                backend_array = TensogramBackendArray(
                    file_path=file_path,
                    msg_index=obj.msg_index,
                    obj_index=obj.obj_index,
                    shape=inner_shape,
                    dtype=np_dtype,
                    supports_range=_supports_range_decode(obj.descriptor),
                    range_threshold=range_threshold,
                    lock=lock,
                    storage_options=storage_options,
                    shared_file=shared_file,
                )
                if backend_arrays is not None:
                    backend_arrays.append(backend_array)
                lazy_data = indexing.LazilyIndexedArray(backend_array)
                data_vars[var_name] = xr.Variable(
                    inner_dims, lazy_data, strip_structural_keys(obj.merged_meta)
                )
        else:
            # No remaining varying keys -> use first object.
            if len(sub_group) > 1:
                logger.warning(
                    "variable %r: %d duplicate objects with no distinguishing "
                    "metadata, using only the first (dropping %d)",
                    var_name,
                    len(sub_group),
                    len(sub_group) - 1,
                )
            obj = sub_group[0]
            backend_array = TensogramBackendArray(
                file_path=file_path,
                msg_index=obj.msg_index,
                obj_index=obj.obj_index,
                shape=inner_shape,
                dtype=np_dtype,
                supports_range=_supports_range_decode(obj.descriptor),
                range_threshold=range_threshold,
                lock=lock,
                storage_options=storage_options,
                shared_file=shared_file,
            )
            if backend_arrays is not None:
                backend_arrays.append(backend_array)
            lazy_data = indexing.LazilyIndexedArray(backend_array)
            data_vars[var_name] = xr.Variable(
                inner_dims, lazy_data, strip_structural_keys(obj.merged_meta)
            )

    ds = xr.Dataset(data_vars, coords=merged_coords, attrs=strip_structural_keys(constant))
    return ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _consistent_hint_meta(
    objects: Sequence[ObjectInfo],
    ndim: int,
) -> Mapping[str, Any] | None:
    """Return a representative per-object meta if all hints agree, else ``None``.

    Inspects ``obj.per_object_meta`` (from ``meta.base[i]``) exclusively —
    not ``merged_meta``, which also folds in ``common_meta`` and would
    mis-classify a message-level ``_extra_["dim_names"]`` list as a
    per-object hint.

    Objects without a valid per-object ``dim_names`` hint are skipped.
    If every object with a hint supplies the *same* list, that object's
    ``per_object_meta`` is returned (its ``dim_names`` drives the resolver).
    If two objects supply *different* valid hints, a warning is logged
    and ``None`` is returned so the resolver falls through to
    ``_extra_["dim_names"]`` and the generic fallback.
    """
    seen: list[tuple[str, ...]] = []
    representative: Mapping[str, Any] | None = None
    for obj in objects:
        parsed = parse_per_object_dim_names(ndim, obj.per_object_meta)
        if parsed is None:
            continue
        parsed_tuple = tuple(parsed)
        if not seen:
            seen.append(parsed_tuple)
            representative = obj.per_object_meta
        elif parsed_tuple not in seen:
            seen.append(parsed_tuple)
    if len(seen) > 1:
        logger.warning(
            "group has inconsistent per-object dim_names hints %s; "
            "falling back to coord/_extra_/generic resolution",
            [list(t) for t in seen],
        )
        return None
    return representative


def _consistent_extra_hint(objects: Sequence[ObjectInfo]) -> Any:
    """Return a representative ``_extra_["dim_names"]`` if all agree, else ``None``.

    Each :class:`ObjectInfo.common_meta` carries the per-message
    ``_extra_`` dict.  When messages disagree on the hint, the resolver
    cannot safely apply any single value across the group, so the hint
    is discarded with a warning.
    """
    seen: list[Any] = []
    seen_hashable: set[Any] = set()
    for obj in objects:
        raw = obj.common_meta.get(EXTRA_DIM_NAMES_KEY)
        if raw is None:
            continue
        frozen = _make_hashable(raw)
        if frozen not in seen_hashable:
            seen_hashable.add(frozen)
            seen.append(raw)
    if len(seen) > 1:
        logger.warning(
            "group has inconsistent _extra_ %s hints across messages "
            "(%d distinct); ignoring this hint and falling back to coord "
            "matches, consistent per-object hints, or generic dim names",
            EXTRA_DIM_NAMES_KEY,
            len(seen),
        )
        return None
    return seen[0] if seen else None


def _resolve_inner_dims(
    objects: Sequence[ObjectInfo],
    inner_shape: tuple[int, ...],
    *,
    user_dim_names: Sequence[str] | None,
    coord_vars: Mapping[str, xr.Variable],
) -> tuple[str, ...]:
    """Resolve dimension names for a group of structurally compatible objects.

    Uses the same priority chain as :func:`TensogramDataStore.build_dataset`
    (via :func:`resolve_dims_for_axes`).  Per-object and message-level
    ``dim_names`` hints must agree across the group; disagreement triggers
    a warning and falls back to lower-priority sources.

    A final pass drops any hint name that would collide with an existing
    coord dim at a different size — those axes are renamed to
    ``obj_{i}_dim_{axis}`` to prevent ``xr.Dataset`` assembly from raising
    ``conflicting sizes for dimension ...``.  Within-group fallback
    collisions cannot occur because structurally compatible objects share
    their inner shape.
    """
    ndim = len(inner_shape)
    coord_dim_sizes = {name: var.shape[0] for name, var in coord_vars.items()}
    per_object_meta = _consistent_hint_meta(objects, ndim)
    extra_hint = _consistent_extra_hint(objects)
    dims_with_provenance = resolve_dims_for_axes(
        inner_shape,
        user_dim_names=user_dim_names,
        coord_dim_sizes=coord_dim_sizes,
        per_object_meta=per_object_meta,
        extra_dim_names_hint=extra_hint,
    )

    representative_obj_index = objects[0].obj_index if objects else 0
    warned: set[str] = set()
    result: list[str] = []
    for axis, (dim_name, is_generic) in enumerate(dims_with_provenance):
        coord_size = coord_dim_sizes.get(dim_name)
        if coord_size is not None and coord_size != inner_shape[axis]:
            if not is_generic and dim_name not in warned:
                logger.warning(
                    "dim name %r at axis %d size %d conflicts with coord %r "
                    "size %d; renaming to obj_%d_dim_%d to keep the Dataset "
                    "openable (check user dim_names kwarg, _extra_, or "
                    "per-object hints)",
                    dim_name,
                    axis,
                    inner_shape[axis],
                    dim_name,
                    coord_size,
                    representative_obj_index,
                    axis,
                )
                warned.add(dim_name)
            result.append(f"obj_{representative_obj_index}_dim_{axis}")
        else:
            result.append(dim_name)
    return tuple(result)
