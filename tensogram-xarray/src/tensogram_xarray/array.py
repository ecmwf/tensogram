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

    total_output = math.prod(output_dims) if output_dims else 0
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

    This class stores only the file path (no open handles) so that it can be
    safely pickled for dask multiprocessing / distributed execution.
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
        range_threshold: float = DEFAULT_RANGE_THRESHOLD,
        lock: threading.Lock | None = None,
    ):
        self.file_path = file_path
        self.msg_index = msg_index
        self.obj_index = obj_index
        self.shape = shape
        self.dtype = dtype
        self.supports_range = supports_range
        self.range_threshold = range_threshold
        self.lock = lock or threading.Lock()

    # -- pickle support (no open handles stored) ----------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["lock"] = None  # locks are not picklable
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.lock = threading.Lock()

    # -- BackendArray interface ---------------------------------------------

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.ndarray:
        """Thread-safe read from the tensogram file."""
        import tensogram

        with self.lock:
            with tensogram.TensogramFile.open(self.file_path) as f:
                raw_msg = f.read_message(self.msg_index)

            # Try partial decode for contiguous slices when random access
            # is available and the requested fraction is below threshold.
            if self.supports_range and _is_contiguous_slice(key):
                try:
                    flat_ranges, out_shape = _nd_slice_to_flat_ranges(self.shape, key)
                    total_requested = sum(c for _, c in flat_ranges)
                    total_elements = math.prod(self.shape) if self.shape else 0

                    if (
                        total_elements > 0
                        and total_requested / total_elements <= self.range_threshold
                    ):
                        arr = tensogram.decode_range(
                            raw_msg,
                            object_index=self.obj_index,
                            ranges=flat_ranges,
                            join=True,
                        )
                        return np.asarray(arr).reshape(out_shape)
                except (ValueError, RuntimeError, OSError) as exc:
                    logger.debug(
                        "decode_range failed for %s msg=%d obj=%d, "
                        "falling back to full decode: %s",
                        self.file_path,
                        self.msg_index,
                        self.obj_index,
                        exc,
                    )

            # Full object decode + in-memory slice.
            _meta, _desc, arr = tensogram.decode_object(raw_msg, self.obj_index)
            return np.asarray(arr[key])
