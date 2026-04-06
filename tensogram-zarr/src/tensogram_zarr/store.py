"""Zarr v3 Store backed by Tensogram .tgm files.

``TensogramStore`` implements the ``zarr.abc.store.Store`` abstract base class,
allowing Zarr to read from and write to ``.tgm`` files transparently.

Virtual filesystem layout (read path)::

    zarr.json                     # root group — from GlobalMetadata
    <variable>/zarr.json          # array metadata — from DataObjectDescriptor
    <variable>/c/0/0              # single chunk — decoded object data (2-D example)

Write path: chunk data is buffered in memory and flushed to a ``.tgm`` file
when the store is closed.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Iterable
from typing import Any

import numpy as np

from tensogram_zarr.mapping import (
    build_array_zarr_json,
    build_group_zarr_json,
    deserialize_zarr_json,
    parse_array_zarr_json,
    resolve_variable_name,
    serialize_zarr_json,
    tgm_dtype_to_numpy,
)

try:
    from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, SuffixByteRequest
    from zarr.abc.store import Store as ZarrStore
    from zarr.core.buffer import Buffer, BufferPrototype
except ImportError as exc:
    raise ImportError(
        "tensogram-zarr requires zarr>=3.0. Install with: pip install 'zarr>=3.0'"
    ) from exc

_log = logging.getLogger(__name__)

# Characters forbidden in Zarr key path segments.  If a resolved variable
# name contains any of these, they are replaced with "_" to prevent
# the virtual key space from creating spurious nested directories.
_FORBIDDEN_KEY_CHARS = frozenset("/\\")


class TensogramStore(ZarrStore):
    """Zarr v3 Store backed by a Tensogram ``.tgm`` file.

    Supports read and write modes:

    - **Read** (``mode="r"``): opens an existing ``.tgm`` file and exposes
      each data object as a Zarr array with a single chunk.
    - **Write** (``mode="w"``): creates a new ``.tgm`` file. Chunk data is
      buffered in memory and flushed when the store is closed.

    Parameters
    ----------
    path : str
        Path to the ``.tgm`` file.
    mode : str
        ``"r"`` for read-only, ``"w"`` for write, ``"a"`` for append.
    message_index : int
        Which message to read (default ``0``). Only used in read mode.
    variable_key : str | None
        Dotted metadata path for variable naming (e.g. ``"mars.param"``).
    """

    def __init__(
        self,
        path: str,
        *,
        mode: str = "r",
        message_index: int = 0,
        variable_key: str | None = None,
    ) -> None:
        if mode not in ("r", "w", "a"):
            raise ValueError(f"invalid mode {mode!r}; expected 'r', 'w', or 'a'")
        if message_index < 0:
            raise ValueError(f"message_index must be >= 0, got {message_index}")
        if not isinstance(path, str) or not path:
            raise ValueError(f"path must be a non-empty string, got {path!r}")
        read_only = mode == "r"
        super().__init__(read_only=read_only)
        self._path = path
        self._mode = mode
        self._message_index = message_index
        self._variable_key = variable_key

        # Virtual key space — maps Zarr keys to raw bytes content
        self._keys: dict[str, bytes] = {}
        self._variable_names: list[str] = []

        # Write-path state
        self._write_arrays: dict[str, dict[str, Any]] = {}  # var → zarr_meta
        self._write_chunks: dict[str, bytes] = {}  # "var/c/0/0" → raw bytes
        self._write_group_attrs: dict[str, Any] = {}
        self._dirty = False

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def open_tgm(
        cls,
        path: str,
        *,
        message_index: int = 0,
        variable_key: str | None = None,
    ) -> TensogramStore:
        """Open a ``.tgm`` file as a read-only Zarr store (synchronous).

        This is a convenience wrapper that creates the store, scans the
        file, and returns a ready-to-use instance.
        """
        store = cls(path, mode="r", message_index=message_index, variable_key=variable_key)
        store._open_sync()
        return store

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _open(self) -> None:
        """Open the store and scan the TGM file if in read mode."""
        self._do_open()

    def _open_sync(self) -> None:
        """Synchronous open — avoids async overhead for simple use cases."""
        self._do_open()

    def _do_open(self) -> None:
        """Shared open logic for async and sync paths."""
        if self._is_open:
            raise ValueError("store is already open")
        if self._mode in ("r", "a"):
            self._scan_tgm_file()
        self._is_open = True

    def close(self) -> None:
        """Close the store. Flushes pending writes in write/append mode."""
        try:
            if self._dirty and self._mode in ("w", "a"):
                self._flush_to_tgm()
        finally:
            self._is_open = False

    def __enter__(self):
        if not self._is_open:
            self._open_sync()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An exception is already in flight.  Attempt flush but log
            # rather than replace the original exception.
            try:
                if self._dirty and self._mode in ("w", "a"):
                    self._flush_to_tgm()
            except Exception:
                _log.warning(
                    "flush to %r failed during exception handling; data may be lost",
                    self._path,
                    exc_info=True,
                )
            finally:
                self._is_open = False
        else:
            self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def supports_writes(self) -> bool:
        return self._mode in ("w", "a")

    @property
    def supports_deletes(self) -> bool:
        return self._mode in ("w", "a")

    @property
    def supports_listing(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensogramStore):
            return NotImplemented
        return self._path == other._path and self._mode == other._mode

    def __repr__(self) -> str:
        return f"TensogramStore({self._path!r}, mode={self._mode!r})"

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        await self._ensure_open()
        return self._get_sync(key, prototype, byte_range)

    def _get_sync(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Synchronous get — the actual implementation."""
        data = self._keys.get(key)
        if data is None:
            return None
        if byte_range is not None:
            data = _apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        await self._ensure_open()
        return [self._get_sync(key, prototype, br) for key, br in key_ranges]

    async def exists(self, key: str) -> bool:
        await self._ensure_open()
        return key in self._keys

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        await self._ensure_open()
        data = value.to_bytes()
        self._keys[key] = data
        self._dirty = True

        # Track array metadata and chunk data for TGM flush
        if key == "zarr.json":
            meta = deserialize_zarr_json(data)
            self._write_group_attrs = meta.get("attributes", {})
        elif key.endswith("/zarr.json"):
            var_name = key[: -len("/zarr.json")]
            if var_name:
                meta = deserialize_zarr_json(data)
                if meta.get("node_type") == "array":
                    self._write_arrays[var_name] = meta
        elif "/c/" in key:
            self._write_chunks[key] = data

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        if not await self.exists(key):
            await self.set(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        for key, value in values:
            await self.set(key, value)

    async def delete(self, key: str) -> None:
        self._check_writable()
        await self._ensure_open()
        self._keys.pop(key, None)
        if key == "zarr.json":
            # Clear stale group attributes to keep flush state consistent
            self._write_group_attrs = {}
        elif key.endswith("/zarr.json"):
            self._write_arrays.pop(key[: -len("/zarr.json")], None)
        self._write_chunks.pop(key, None)
        self._dirty = True

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    async def list(self) -> AsyncIterator[str]:
        await self._ensure_open()
        for key in list(self._keys.keys()):
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        await self._ensure_open()
        for key in list(self._keys.keys()):
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        await self._ensure_open()
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        seen: set[str] = set()
        for key in list(self._keys.keys()):
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            slash_pos = suffix.find("/")
            if slash_pos == -1:
                # Leaf entry (file)
                if suffix and suffix not in seen:
                    seen.add(suffix)
                    yield suffix
            else:
                # Directory entry
                dir_name = suffix[: slash_pos + 1]
                if dir_name not in seen:
                    seen.add(dir_name)
                    yield dir_name

    # ------------------------------------------------------------------
    # Internal: TGM scanning (read path)
    # ------------------------------------------------------------------

    def _scan_tgm_file(self) -> None:
        """Scan the .tgm file and build the virtual key space."""
        import tensogram

        try:
            f = tensogram.TensogramFile.open(self._path)
        except Exception as exc:
            raise OSError(f"failed to open TGM file {self._path!r}: {exc}") from exc

        with f:
            msg_count = len(f)
            if msg_count == 0:
                self._keys["zarr.json"] = serialize_zarr_json(
                    {
                        "zarr_format": 3,
                        "node_type": "group",
                        "attributes": {},
                    }
                )
                return

            if self._message_index >= msg_count:
                raise IndexError(
                    f"message_index {self._message_index} out of range "
                    f"(file has {msg_count} message(s))"
                )

            raw_msg = f.read_message(self._message_index)

        try:
            meta, descriptors = tensogram.decode_descriptors(raw_msg)
        except Exception as exc:
            raise ValueError(
                f"failed to decode message {self._message_index} in {self._path!r}: {exc}"
            ) from exc

        # Determine variable names, deduplicating and sanitizing
        base = meta.base if hasattr(meta, "base") else []
        names: list[str] = []
        name_counts: dict[str, int] = {}
        for i, _desc in enumerate(descriptors):
            per_obj = _filter_reserved(base[i]) if i < len(base) else {}
            extra = meta.extra if hasattr(meta, "extra") else {}
            raw_name = resolve_variable_name(i, per_obj, extra, self._variable_key)
            name = _sanitize_key_segment(raw_name)
            # Deduplicate
            if name in name_counts:
                name_counts[name] += 1
                name = f"{name}_{name_counts[name]}"
            else:
                name_counts[name] = 0
            names.append(name)

        self._variable_names = names

        # Root group zarr.json
        group_meta = build_group_zarr_json(meta, names)
        self._keys["zarr.json"] = serialize_zarr_json(group_meta)

        # Per-array zarr.json + chunk data
        for i, (desc, name) in enumerate(zip(descriptors, names, strict=True)):
            per_obj = _filter_reserved(base[i]) if i < len(base) else {}
            array_meta = build_array_zarr_json(desc, per_obj)
            self._keys[f"{name}/zarr.json"] = serialize_zarr_json(array_meta)

            try:
                _, _, arr = tensogram.decode_object(raw_msg, i)
            except Exception as exc:
                raise ValueError(
                    f"failed to decode object {i} ({name!r}) "
                    f"in message {self._message_index} of {self._path!r}: {exc}"
                ) from exc
            # Store as little-endian raw bytes (matching the bytes codec).
            # Use .view() instead of the deprecated ndarray.newbyteorder()
            # (removed in NumPy 2.x).
            if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and _native_is_big()):
                arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
            chunk_key = _chunk_key_for_shape(desc.shape)
            self._keys[f"{name}/{chunk_key}"] = arr.tobytes()

    # ------------------------------------------------------------------
    # Internal: TGM writing (write path)
    # ------------------------------------------------------------------

    def _flush_to_tgm(self) -> None:
        """Assemble buffered data into a TGM message and write to file."""
        import tensogram

        # Build global metadata from group attributes.
        # In the new metadata model, message-level annotations go into
        # ``extra`` (any unknown top-level keys), and per-object metadata
        # goes into ``base`` (list of dicts, one per object).
        global_meta: dict[str, Any] = {"version": 2}
        # Reserved top-level keys that must not be overwritten by group attrs.
        _reserved_top = {"version", "base", "_extra_", "extra", "_reserved_"}
        clean_attrs = {
            k: v
            for k, v in self._write_group_attrs.items()
            if not k.startswith("_tensogram_") and k not in _reserved_top
        }
        if clean_attrs:
            # Put message-level metadata as extra (unknown keys at top level)
            global_meta.update(clean_attrs)

        # Collect arrays ordered by name
        array_names = sorted(self._write_arrays.keys())
        if not array_names:
            _log.warning("flush to %r skipped: no arrays registered", self._path)
            return

        descriptors_and_data: list[tuple[dict[str, Any], np.ndarray | bytes]] = []
        base_entries: list[dict[str, Any]] = []

        for var_name in array_names:
            zarr_meta = self._write_arrays[var_name]
            parsed = parse_array_zarr_json(dict(zarr_meta))  # copy to avoid mutation

            chunk_data = _find_chunk_data(var_name, self._write_chunks)
            if chunk_data is None:
                _log.warning("array %r has metadata but no chunk data; skipping", var_name)
                continue

            shape = parsed["shape"]
            tgm_dtype = parsed["dtype"]
            byte_order = parsed.get("byte_order", "little")

            # Use the proper TGM→NumPy dtype mapping (handles bfloat16 etc.)
            try:
                np_dtype = tgm_dtype_to_numpy(tgm_dtype)
            except ValueError as exc:
                raise ValueError(f"unsupported dtype for variable {var_name!r}: {exc}") from exc

            # Validate byte count matches expected size
            n_elements = int(np.prod(shape)) if shape else 1
            expected_bytes = n_elements * np_dtype.itemsize
            if len(chunk_data) != expected_bytes:
                raise ValueError(
                    f"chunk data for {var_name!r}: expected {expected_bytes} bytes "
                    f"({n_elements} elements x {np_dtype.itemsize} bytes/element), "
                    f"got {len(chunk_data)}"
                )

            arr = np.frombuffer(chunk_data, dtype=np_dtype).reshape(shape)

            # Honor byte order from Zarr metadata — swap if big-endian
            if byte_order == "big" and np_dtype.itemsize > 1:
                arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
            desc_dict: dict[str, Any] = {
                "type": "ntensor",
                "shape": list(arr.shape),
                "dtype": tgm_dtype,
                "byte_order": "little",
                "encoding": parsed.get("encoding", "none"),
                "filter": parsed.get("filter", "none"),
                "compression": parsed.get("compression", "none"),
            }

            descriptors_and_data.append((desc_dict, arr))

            base_entry: dict[str, Any] = {}
            if parsed.get("attrs"):
                # Filter out _reserved_ — the encoder auto-populates it
                # and user-set values would collide.
                base_entry.update({k: v for k, v in parsed["attrs"].items() if k != "_reserved_"})
            base_entries.append(base_entry)

        if not descriptors_and_data:
            _log.warning("flush to %r produced no data (all arrays skipped)", self._path)
            return

        if base_entries:
            global_meta["base"] = base_entries

        if self._mode == "a":
            with tensogram.TensogramFile.open(self._path) as f:
                f.append(global_meta, descriptors_and_data)
        else:
            with tensogram.TensogramFile.create(self._path) as f:
                f.append(global_meta, descriptors_and_data)

        self._dirty = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_reserved(entry: dict) -> dict:
    """Return a base entry dict with the ``_reserved_`` key removed."""
    if not isinstance(entry, dict):
        return {}
    return {k: v for k, v in entry.items() if k != "_reserved_"}


def _apply_byte_range(data: bytes, byte_range: ByteRequest) -> bytes:
    """Apply a Zarr ByteRequest to raw bytes."""
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        if byte_range.suffix <= 0:
            return b""
        return data[-byte_range.suffix :]
    raise TypeError(f"unsupported ByteRequest type: {type(byte_range).__name__}")


def _find_chunk_data(var_name: str, chunks: dict[str, bytes]) -> bytes | None:
    """Find the single chunk data for a variable in the write buffer.

    Matches any key that starts with ``<var_name>/c/`` — since we only
    support single-chunk arrays, there must be at most one match.

    Raises
    ------
    ValueError
        If multiple chunk keys are found for the same variable (Tensogram
        only supports single-chunk arrays).
    """
    prefix = f"{var_name}/c/"
    matches = [(key, data) for key, data in chunks.items() if key.startswith(prefix)]
    if not matches:
        return None
    if len(matches) > 1:
        keys = [m[0] for m in matches]
        raise ValueError(
            f"variable {var_name!r} has {len(matches)} chunk keys {keys}; "
            f"TensogramStore only supports single-chunk arrays"
        )
    return matches[0][1]


def _chunk_key_for_shape(shape: list[int] | tuple[int, ...]) -> str:
    """Build the Zarr v3 chunk key for a single-chunk array.

    For a 2-D array the key is ``c/0/0``, for 3-D ``c/0/0/0``, etc.
    Scalar arrays (ndim=0) use ``c/0``.
    """
    ndim = len(shape) if shape else 0
    if ndim == 0:
        return "c/0"
    return "c/" + "/".join("0" for _ in range(ndim))


def _sanitize_key_segment(name: str) -> str:
    """Replace forbidden characters in a Zarr key segment.

    Slashes and backslashes would create unintended directory nesting
    in the virtual key space.  Replace them with underscores.  Empty
    names fall back to ``_``.
    """
    if not name:
        return "_"
    return "".join("_" if ch in _FORBIDDEN_KEY_CHARS else ch for ch in name)


def _native_is_big() -> bool:
    """Check if the native byte order is big-endian."""
    return sys.byteorder == "big"
