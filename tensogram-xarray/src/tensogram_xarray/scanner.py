"""File scanner: extract metadata from all messages/objects without decoding payloads.

The scanner opens a ``.tgm`` file via ``tensogram.TensogramFile``, reads each
message's metadata, and builds an index of per-object metadata dicts that
downstream merge/split logic can consume.

Per-object metadata is read from ``meta.payload[i]`` (primary) with
``desc.params`` as fallback.  Message-level metadata comes from
``meta.common`` and ``meta.extra``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObjectInfo:
    """Metadata for a single data object within a message."""

    msg_index: int
    obj_index: int
    ndim: int
    shape: tuple[int, ...]
    dtype: str
    descriptor: Any  # tensogram.DataObjectDescriptor
    per_object_meta: dict[str, Any]  # from meta.payload[i] + desc.params
    common_meta: dict[str, Any]  # from meta.common + meta.extra

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


# Keys auto-populated by the encoder into each payload entry.
# These duplicate descriptor fields and should be excluded from
# per-object metadata used for grouping and variable naming.
AUTO_PAYLOAD_KEYS = frozenset({"ndim", "shape", "strides", "dtype"})


def _payload_from_meta(meta: Any, obj_index: int) -> dict[str, Any]:
    """Extract per-object metadata from meta.payload[obj_index].

    Auto-populated keys (ndim, shape, strides, dtype) are filtered out.
    """
    payload = getattr(meta, "payload", None)
    if payload and isinstance(payload, list) and obj_index < len(payload):
        p = payload[obj_index]
        if isinstance(p, dict):
            return {k: v for k, v in p.items() if k not in AUTO_PAYLOAD_KEYS}
    return {}


def _merge_per_object_meta(meta: Any, obj_index: int, desc: Any) -> dict[str, Any]:
    """Build per-object metadata by merging payload + descriptor params.

    Payload takes priority; descriptor params fill in any missing keys.
    """
    result = _payload_from_meta(meta, obj_index)
    for k, v in _desc_params(desc).items():
        if k not in result:
            result[k] = v
    return result


def _common_from_meta(meta: Any) -> dict[str, Any]:
    """Extract message-level metadata from meta.common + meta.extra."""
    common: dict[str, Any] = {}
    c = getattr(meta, "common", None)
    if c and isinstance(c, dict):
        common.update(c)
    extra = getattr(meta, "extra", None)
    if extra and isinstance(extra, dict):
        common.update(extra)
    return common


def scan_file(file_path: str) -> FileIndex:
    """Scan a ``.tgm`` file and return a :class:`FileIndex`.

    Decodes each message to get descriptors, per-object metadata
    (from ``meta.payload`` and ``desc.params``), and common metadata.
    """
    import tensogram

    file_path = os.path.abspath(file_path)
    index = FileIndex(file_path=file_path)

    with tensogram.TensogramFile.open(file_path) as f:
        n_messages = len(f)
        for msg_idx in range(n_messages):
            raw = f.read_message(msg_idx)
            meta = tensogram.decode_metadata(raw)
            common = _common_from_meta(meta)

            # Decode descriptors only (no payload decode).
            _, descriptors = tensogram.decode_descriptors(raw)

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
                    common_meta=dict(common),
                )
                index.objects.append(info)

    return index


def scan_message(raw_msg: bytes) -> list[ObjectInfo]:
    """Scan a single in-memory message and return :class:`ObjectInfo` list."""
    import tensogram

    meta = tensogram.decode_metadata(raw_msg)
    common = _common_from_meta(meta)

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
            common_meta=dict(common),
        )
        result.append(info)

    return result
