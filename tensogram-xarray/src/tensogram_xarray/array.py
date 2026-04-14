"""Lazy-loading backend array for tensogram data objects.

``TensogramBackendArray`` implements :class:`xarray.backends.BackendArray` so
that tensor payloads are decoded on demand.  For compressors that support
random access (``none``, ``szip``, ``blosc2``, ``zfp`` fixed-rate) and have no
``shuffle`` filter, N-dimensional slice requests are mapped to flat element
ranges and decoded via ``tensogram.decode_range()``.  Otherwise the full
object is decoded and sliced in-memory via ``tensogram.decode_object()``.

A ratio-based heuristic controls when partial reads are used: if the
fraction of requested elements exceeds ``range_threshold`` (default 0.5),
the backend falls back to a full decode.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from itertools import product as iterproduct
from typing import Any

import numpy as np
from xarray.backends import BackendArray
from xarray.core import indexing

logger = logging.getLogger(__name__)

# Compressor values that support partial decode via decode_range().
_RANDOM_ACCESS_COMPRESSORS = frozenset({"none", "szip", "blosc2", "zfp"})

# Filters that break contiguous byte ranges (shuffle rearranges bytes).
_RANGE_BLOCKING_FILTERS = frozenset({"shuffle"})

# Default ratio threshold: use decode_range when the requested fraction of
# total elements is at or below this value.
DEFAULT_RANGE_THRESHOLD = 0.5


def _supports_range_decode(descriptor: Any) -> bool:
    """Return *True* if the object's pipeline allows ``decode_range()``."""
    compression = getattr(descriptor, "compression", "none")
    filt = getattr(descriptor, "filter", "none")

    if filt in _RANGE_BLOCKING_FILTERS:
        return False

    if compression not in _RANDOM_ACCESS_COMPRESSORS:
        return False

    # zfp supports range decode only in fixed_rate mode.
    if compression == "zfp":
        params = getattr(descriptor, "params", {}) or {}
        if params.get("zfp_mode") != "fixed_rate":
            return False

    return True


def _is_contiguous_slice(key: tuple) -> bool:
    """Return *True* when *key* is a tuple of unit-stride slices."""
    for k in key:
        if not isinstance(k, slice):
            return False
        # Reject non-unit strides (step != 1 and step != None).
        if k.step is not None and k.step != 1:
            return False
    return True


# ---------------------------------------------------------------------------
# N-D slice -> flat element ranges
# ---------------------------------------------------------------------------


def _nd_slice_to_flat_ranges(
    shape: tuple[int, ...],
    key: tuple[slice, ...],
) -> tuple[list[tuple[int, int]], tuple[int, ...]]:
    """Map an N-dimensional slice to flat ``(start, count)`` ranges.

    For a C-contiguous (row-major) array the elements of a hyper-rectangular
    slice are **not** contiguous in general.  Contiguous runs exist only
    along the innermost (rightmost) axis.  This function decomposes the
    N-D slice into the minimal set of flat ranges that cover exactly the
    requested elements, then merges adjacent ranges.

    Parameters
    ----------
    shape
        Shape of the full tensor.
    key
        Tuple of ``slice`` objects (one per dimension, unit-stride only).

    Returns
    -------
    flat_ranges
        List of ``(element_offset, element_count)`` in the flattened array.
    output_shape
        Shape of the result after slicing.
    """
    ndim = len(shape)

    # Parse each slice into (start, count).
    dim_ranges: list[tuple[int, int]] = []
    output_dims: list[int] = []
    for slc, d in zip(key, shape):
        s, e, _ = slc.indices(d)
        count = e - s
        dim_ranges.append((s, count))
        output_dims.append(count)
    output_shape = tuple(output_dims)

    total_output = math.prod(output_dims)
    if total_output == 0:
        return [], output_shape

    # Compute C-contiguous strides (in elements).
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    # Find the *split point* k: the innermost dimension whose slice is
    # NOT a full slice.  All dimensions k+1 .. n-1 are full slices, so
    # their elements form a contiguous block.
    split = -1  # -1 means all dims are full
    for i in range(ndim - 1, -1, -1):
        start_i, count_i = dim_ranges[i]
        if start_i != 0 or count_i != shape[i]:
            split = i
            break

    if split == -1:
        # Every dimension is a full slice -- one range covering everything.
        return [(0, math.prod(shape))], output_shape

    # Contiguous block size: count at split dim * product of trailing dims.
    block_size = dim_ranges[split][1]
    for i in range(split + 1, ndim):
        block_size *= shape[i]

    block_start_within_row = dim_ranges[split][0] * strides[split]

    if split == 0:
        # No outer dimensions to iterate.
        return [(dim_ranges[0][0] * strides[0], block_size)], output_shape

    # Generate one range per combination of outer-dimension indices.
    outer_index_ranges = [
        range(dim_ranges[i][0], dim_ranges[i][0] + dim_ranges[i][1]) for i in range(split)
    ]

    flat_ranges: list[tuple[int, int]] = []
    for idx in iterproduct(*outer_index_ranges):
        base = sum(idx[j] * strides[j] for j in range(split))
        flat_ranges.append((base + block_start_within_row, block_size))

    # Merge adjacent ranges (consecutive with no gap).
    if len(flat_ranges) > 1:
        merged: list[tuple[int, int]] = [flat_ranges[0]]
        for start, count in flat_ranges[1:]:
            prev_start, prev_count = merged[-1]
            if start == prev_start + prev_count:
                merged[-1] = (prev_start, prev_count + count)
            else:
                merged.append((start, count))
        flat_ranges = merged

    return flat_ranges, output_shape


# ---------------------------------------------------------------------------
# Backend array
# ---------------------------------------------------------------------------


class TensogramBackendArray(BackendArray):
    """Lazy array backed by a tensogram file.

    Stores the file path (or remote URL) and optionally a shared file handle.
    The handle is dropped on pickle for dask multiprocessing compatibility
    and lazily reopened on the worker.
    """

    def __init__(
        self,
        file_path: str,
        msg_index: int,
        obj_index: int,
        shape: tuple[int, ...],
        dtype: np.dtype,
        supports_range: bool,
        *,
        verify_hash: bool = False,
        range_threshold: float = DEFAULT_RANGE_THRESHOLD,
        lock: threading.Lock | None = None,
        storage_options: dict[str, Any] | None = None,
        shared_file: Any | None = None,
    ):
        import tensogram

        self._is_remote = tensogram.is_remote_url(file_path)
        self.file_path = file_path if self._is_remote else os.path.abspath(file_path)
        self.msg_index = msg_index
        self.obj_index = obj_index
        self.shape = shape
        self.dtype = dtype
        self.supports_range = supports_range
        self.verify_hash = verify_hash
        self.range_threshold = range_threshold
        self.storage_options = storage_options
        self._shared_file = shared_file

    # -- pickle support (no open handles stored) ----------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_shared_file"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._shared_file = None

    # -- BackendArray interface ---------------------------------------------

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _get_file(self):
        if self._shared_file is not None:
            return self._shared_file
        import tensogram

        if self._is_remote:
            return tensogram.TensogramFile.open_remote(self.file_path, self.storage_options or {})
        return tensogram.TensogramFile.open(self.file_path)

    def _raw_indexing_method(self, key: tuple) -> np.ndarray:
        import tensogram

        if self._shared_file is not None:
            return self._read_from_file(self._shared_file, key, tensogram)

        with self._get_file() as f:
            return self._read_from_file(f, key, tensogram)

    def _read_from_file(self, f, key: tuple, tensogram) -> np.ndarray:
        if self.supports_range and _is_contiguous_slice(key):
            try:
                flat_ranges, out_shape = _nd_slice_to_flat_ranges(self.shape, key)
                total_requested = sum(c for _, c in flat_ranges)
                total_elements = math.prod(self.shape)

                if total_elements > 0 and total_requested / total_elements <= self.range_threshold:
                    arr = f.file_decode_range(
                        self.msg_index,
                        obj_index=self.obj_index,
                        ranges=flat_ranges,
                        join=True,
                        verify_hash=self.verify_hash,
                        native_byte_order=True,
                    )
                    return np.asarray(arr).reshape(out_shape)
            except (ValueError, RuntimeError, OSError) as exc:
                logger.debug(
                    "decode_range failed for %s msg=%d obj=%d, falling back to full decode: %s",
                    self.file_path,
                    self.msg_index,
                    self.obj_index,
                    exc,
                )

        if self._is_remote:
            result = f.file_decode_object(
                self.msg_index,
                self.obj_index,
                verify_hash=self.verify_hash,
            )
            return np.asarray(result["data"][key])

        raw_msg = f.read_message(self.msg_index)
        _meta, _desc, arr = tensogram.decode_object(
            raw_msg,
            self.obj_index,
            verify_hash=self.verify_hash,
        )
        return np.asarray(arr[key])


# ---------------------------------------------------------------------------
# Stacked backend array (lazy hypercube)
# ---------------------------------------------------------------------------


class StackedBackendArray(BackendArray):
    """Lazy stacked array composed of multiple :class:`TensogramBackendArray`.

    Each position along the outer dimensions maps to a separate backing
    array.  Indexing dispatches to the correct backing array(s) and
    assembles the result, so no data is decoded until actually accessed.
    """

    def __init__(
        self,
        arrays: list[TensogramBackendArray],
        outer_shape: tuple[int, ...],
        inner_shape: tuple[int, ...],
        dtype: np.dtype,
    ):
        if len(arrays) != math.prod(outer_shape):
            msg = (
                f"StackedBackendArray: expected {math.prod(outer_shape)} "
                f"backing arrays for outer_shape={outer_shape}, "
                f"got {len(arrays)}"
            )
            raise ValueError(msg)

        self._arrays = arrays
        self._outer_shape = outer_shape
        self._inner_shape = inner_shape
        self.shape = outer_shape + inner_shape
        self.dtype = dtype

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.ndarray:
        n_outer = len(self._outer_shape)

        # Split key into outer and inner parts.
        outer_key = key[:n_outer]
        inner_key = key[n_outer:]

        # Compute which backing arrays are needed.
        outer_indices = _expand_key_to_indices(outer_key, self._outer_shape)

        # Determine output shape for outer dimensions.
        outer_out_shape = tuple(len(idx) for idx in outer_indices)

        # Compute inner output shape from inner_key: slices preserve the
        # dimension (with the slice length), integer keys drop it -- matching
        # numpy's basic-indexing semantics.
        inner_out_shape = tuple(
            len(range(*k.indices(s)))
            for k, s in zip(inner_key, self._inner_shape)
            if isinstance(k, slice)
        )

        result = np.empty(outer_out_shape + inner_out_shape, dtype=self.dtype)

        for flat_pos, combo in enumerate(iterproduct(*outer_indices)):
            # Map N-D outer index to flat backing-array index (row-major).
            flat_idx = 0
            for dim, idx_val in enumerate(combo):
                stride = 1
                for d2 in range(dim + 1, n_outer):
                    stride *= self._outer_shape[d2]
                flat_idx += idx_val * stride

            backing = self._arrays[flat_idx]
            inner_data = backing._raw_indexing_method(inner_key)

            # Unravel flat_pos into N-D output position (row-major / C order).
            # iterproduct iterates in row-major order (rightmost index varies
            # fastest), so unraveling must go right-to-left.
            out_idx: list[int] = []
            remainder = flat_pos
            for size in reversed(outer_out_shape):
                out_idx.append(remainder % size)
                remainder //= size
            out_idx.reverse()
            result[tuple(out_idx)] = inner_data

        # Apply outer slicing to produce correct output shape.
        return result


def _expand_key_to_indices(key: tuple, shape: tuple[int, ...]) -> list[list[int]]:
    """Expand a tuple of slices/ints into lists of concrete indices."""
    result: list[list[int]] = []
    for k, size in zip(key, shape):
        if isinstance(k, slice):
            result.append(list(range(*k.indices(size))))
        elif isinstance(k, int):
            result.append([k])
        else:
            result.append(list(range(size)))
    return result
