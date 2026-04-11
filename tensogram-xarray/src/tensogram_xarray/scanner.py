"""File scanner: extract metadata from all messages/objects without decoding payloads.

The scanner opens a ``.tgm`` file via ``tensogram.TensogramFile``, reads each
message's metadata, and builds an index of per-object metadata dicts that
downstream merge/split logic can consume.

Per-object metadata is read from ``meta.base[i]`` (with ``_reserved_``
filtered out) and supplemented by ``desc.params`` as fallback.
Message-level metadata comes from ``meta.extra``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# The ``_reserved_`` key in base entries is populated by the encoder
# with tensor info (ndim, shape, strides, dtype).  It must be excluded
# from application-level metadata used for grouping and variable naming.
RESERVED_KEY = "_reserved_"


@dataclass
class ObjectInfo:
    """Metadata for a single data object within a message."""

    msg_index: int
    obj_index: int
    ndim: int
    shape: tuple[int, ...]
    dtype: str
    descriptor: Any  # tensogram.DataObjectDescriptor
    per_object_meta: dict[str, Any]  # from meta.base[i] (filtered)
    common_meta: dict[str, Any]  # from meta.extra

    @property
    def merged_meta(self) -> dict[str, Any]:
        """Return common + per-object metadata merged (per-object wins)."""
        merged: dict[str, Any] = {}
        merged.update(self.common_meta)
        merged.update(self.per_object_meta)
        return merged


@dataclass
class FileIndex:
    """Index over all messages and objects in a ``.tgm`` file."""

    file_path: str
    objects: list[ObjectInfo] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        if not self.objects:
            return 0
        return max(o.msg_index for o in self.objects) + 1


def _desc_params(desc: Any) -> dict[str, Any]:
    """Extract per-object metadata from a descriptor's params."""
    params = getattr(desc, "params", None)
    if params and isinstance(params, dict):
        return dict(params)
    return {}


def _base_entry_from_meta(meta: Any, obj_index: int) -> dict[str, Any]:
    """Extract per-object metadata from ``meta.base[obj_index]``.

    The ``_reserved_`` key is filtered out since it contains
    encoder-populated tensor info (ndim, shape, strides, dtype).
    """
    base = getattr(meta, "base", None)
    if base and isinstance(base, list) and obj_index < len(base):
        entry = base[obj_index]
        if isinstance(entry, dict):
            return {k: v for k, v in entry.items() if k != RESERVED_KEY}
    return {}


def _merge_per_object_meta(meta: Any, obj_index: int, desc: Any) -> dict[str, Any]:
    """Build per-object metadata from base entry + descriptor params.

    Base entry takes priority; descriptor params fill in any missing keys.
    """
    result = _base_entry_from_meta(meta, obj_index)
    for k, v in _desc_params(desc).items():
        if k not in result:
            result[k] = v
    return result


def _extra_from_meta(meta: Any) -> dict[str, Any]:
    """Extract message-level metadata from ``meta.extra``."""
    result: dict[str, Any] = {}
    extra = getattr(meta, "extra", None)
    if extra and isinstance(extra, dict):
        result.update(extra)
    return result


def scan_file(
    file_path: str,
    storage_options: dict[str, str] | None = None,
) -> FileIndex:
    """Scan a ``.tgm`` file and return a :class:`FileIndex`.

    Decodes each message to get descriptors, per-object metadata
    (from ``meta.base`` and ``desc.params``), and extra metadata.
    """
    import tensogram

    is_remote = tensogram.is_remote_url(file_path)
    resolved = file_path if is_remote else os.path.abspath(file_path)
    index = FileIndex(file_path=resolved)

    if is_remote:
        f = tensogram.TensogramFile.open_remote(resolved, storage_options or {})
    else:
        f = tensogram.TensogramFile.open(resolved)

    with f:
        n_messages = len(f)
        for msg_idx in range(n_messages):
            if is_remote:
                result = f.file_decode_descriptors(msg_idx)
                meta = result["metadata"]
                descriptors = result["descriptors"]
            else:
                raw = f.read_message(msg_idx)
                meta = tensogram.decode_metadata(raw)
                _, descriptors = tensogram.decode_descriptors(raw)

            extra = _extra_from_meta(meta)

            for obj_idx, desc in enumerate(descriptors):
                per_obj = _merge_per_object_meta(meta, obj_idx, desc)
                shape = tuple(desc.shape)
                info = ObjectInfo(
                    msg_index=msg_idx,
                    obj_index=obj_idx,
                    ndim=desc.ndim,
                    shape=shape,
                    dtype=desc.dtype,
                    descriptor=desc,
                    per_object_meta=per_obj,
                    common_meta=dict(extra),
                )
                index.objects.append(info)

    return index


def scan_message(raw_msg: bytes) -> list[ObjectInfo]:
    """Scan a single in-memory message and return :class:`ObjectInfo` list."""
    import tensogram

    meta = tensogram.decode_metadata(raw_msg)
    extra = _extra_from_meta(meta)

    _, descriptors = tensogram.decode_descriptors(raw_msg)

    result: list[ObjectInfo] = []
    for obj_idx, desc in enumerate(descriptors):
        per_obj = _merge_per_object_meta(meta, obj_idx, desc)
        shape = tuple(desc.shape)
        info = ObjectInfo(
            msg_index=0,
            obj_index=obj_idx,
            ndim=desc.ndim,
            shape=shape,
            dtype=desc.dtype,
            descriptor=desc,
            per_object_meta=per_obj,
            common_meta=dict(extra),
        )
        result.append(info)

    return result
