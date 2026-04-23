# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""MARS :class:`FieldList` assembly for tensogram files.

:func:`build_fieldlist_from_path` opens a ``.tgm`` file, walks every
message, and produces one :class:`earthkit.data.ArrayField` per object
whose ``base[i]`` carries a non-empty ``mars`` sub-map.  Coordinate
objects (``base[i]`` entries without ``mars``) are skipped — they
would not have the grid semantics the earthkit :class:`Field` model
expects, and users who want them should use the xarray path.

The result is wrapped in a :class:`SimpleFieldList` so the downstream
:meth:`sel` / :meth:`order_by` / :meth:`get` / :meth:`metadata` APIs
work out of the box.  :meth:`TensogramSimpleFieldList.to_xarray`
delegates to the tensogram-xarray backend, preserving the single
source of truth for coordinate detection and dim-name resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from earthkit.data.indexing.fieldlist import SimpleFieldList
from earthkit.data.sources.array_list import ArrayField

from tensogram_earthkit.mars import base_entry_to_usermetadata, has_mars_namespace


class TensogramSimpleFieldList(SimpleFieldList):
    """:class:`SimpleFieldList` that remembers its backing tensogram file.

    Overrides :meth:`to_xarray` to delegate to the tensogram-xarray
    backend rather than the default earthkit FieldList → xarray
    conversion.  This keeps coordinate auto-detection, dim-name
    resolution, and the lazy backing-array logic single-sourced in
    :mod:`tensogram_xarray`.
    """

    def __init__(self, fields: list[Any], file_path: str | None = None) -> None:
        super().__init__(fields)
        self._tgm_file_path = file_path

    def to_xarray(self, **kwargs: Any) -> Any:
        """Delegate to the tensogram-xarray backend when possible."""
        if self._tgm_file_path is not None:
            import xarray as xr

            return xr.open_dataset(self._tgm_file_path, engine="tensogram", **kwargs)
        # Fallback — no path available (e.g. built from bytes).
        return super().to_xarray(**kwargs)


def _open_tgm_file(path: str, storage_options: dict[str, Any] | None) -> Any:
    """Open a tensogram file locally or via ``open_remote`` as appropriate."""
    import tensogram

    if tensogram.is_remote_url(path):
        return tensogram.TensogramFile.open_remote(path, storage_options or {})
    return tensogram.TensogramFile.open(path)


def _decode_one_message(
    path: str,
    msg_index: int,
    storage_options: dict[str, Any] | None,
) -> tuple[Any, list[Any], list[Any]]:
    """Decode a single message: (metadata, descriptors, data arrays).

    Handles both local and remote backends.  For remote files we go
    through ``file_decode_object`` to stay on the ranged-read path;
    for local files the buffer-level ``decode`` is fine.
    """
    import tensogram

    if tensogram.is_remote_url(path):
        with _open_tgm_file(path, storage_options) as fh:
            result = fh.file_decode_descriptors(msg_index)
            meta = result["metadata"]
            descriptors = list(result["descriptors"])
            arrays: list[Any] = []
            for obj_index in range(len(descriptors)):
                obj = fh.file_decode_object(msg_index, obj_index)
                arrays.append(np.asarray(obj["data"]))
        return meta, descriptors, arrays

    with _open_tgm_file(path, storage_options) as fh:
        raw = fh.read_message(msg_index)
    meta, objects = tensogram.decode(raw)
    descriptors = [obj[0] for obj in objects]
    arrays = [np.asarray(obj[1]) for obj in objects]
    return meta, descriptors, arrays


def build_fieldlist_from_path(
    file_path: str,
    storage_options: dict[str, Any] | None = None,
) -> TensogramSimpleFieldList:
    """Open ``file_path`` and build a MARS FieldList.

    Every tensogram object whose ``base[i]`` entry has a non-empty
    ``mars`` sub-map becomes one :class:`ArrayField`.  All other
    objects are silently skipped.

    Remote URLs (``http(s)``, ``s3``, ``gs``, ``az``) are supported —
    ``storage_options`` is forwarded to
    ``tensogram.TensogramFile.open_remote``.
    """
    import tensogram

    path_str = file_path if tensogram.is_remote_url(file_path) else str(Path(file_path))
    fields: list[Any] = []

    with _open_tgm_file(path_str, storage_options) as fh:
        n_messages = len(fh)

    for msg_index in range(n_messages):
        meta, descriptors, arrays = _decode_one_message(path_str, msg_index, storage_options)
        base = list(getattr(meta, "base", None) or [])

        for obj_index, (desc, arr) in enumerate(zip(descriptors, arrays, strict=True)):
            if obj_index >= len(base):
                continue
            if not has_mars_namespace(base[obj_index]):
                continue
            shape = tuple(desc.shape)
            arr = arr.reshape(shape) if arr.shape != shape else arr
            metadata = base_entry_to_usermetadata(base[obj_index], shape=shape)
            fields.append(ArrayField(arr, metadata))

    return TensogramSimpleFieldList(fields, file_path=path_str)
