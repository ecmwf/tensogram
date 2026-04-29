# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""verify_hash threading through the xarray backend.

The xarray backend re-exposes the upstream
``DecodeOptions::verify_hash`` flag via the ``open_dataset(...,
verify_hash=True)`` keyword.  When the lazy backing arrays
materialise data via ``file_decode_object`` /
``decode_object``, the kwarg propagates and integrity errors
(``MissingHashError`` / ``HashMismatchError``) bubble up to
the caller's first read.

Per Q6 in ``PLAN_DECODE_HASH_VERIFICATION.md``: the partial-
range fast path silently does *not* verify (range decode
does not accept ``verify_hash``).  Set ``range_threshold=0``
to force every read through the full-decode path if you
need uniform coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensogram
import xarray as xr


def _build_unhashed_message(tmp_path) -> str:
    """Encode a 1-object f32 message with hashing off + write to disk.

    Returns the file path.  The unhashed encoding is what makes
    cell C (`verify_hash=True` → MissingHashError) testable.
    """
    meta = {"version": 3}
    desc = {
        "type": "ntensor",
        "ndim": 1,
        "shape": [4],
        "strides": [1],
        "dtype": "float32",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    msg = bytes(tensogram.encode(meta, [(desc, data)], hash=None))
    path = tmp_path / "unhashed.tgm"
    path.write_bytes(msg)
    return str(path)


def _build_hashed_message(tmp_path) -> str:
    """Encode a 1-object f32 message with hashing on + write to disk."""
    meta = {"version": 3}
    desc = {
        "type": "ntensor",
        "ndim": 1,
        "shape": [4],
        "strides": [1],
        "dtype": "float32",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    msg = bytes(tensogram.encode(meta, [(desc, data)], hash="xxh3"))
    path = tmp_path / "hashed.tgm"
    path.write_bytes(msg)
    return str(path)


class TestOpenDatasetVerifyHash:
    def test_verify_hash_default_is_false(self, tmp_path):
        """Default ``verify_hash=False`` decodes both hashed and
        unhashed fixtures cleanly."""
        ds_hashed = xr.open_dataset(_build_hashed_message(tmp_path), engine="tensogram")
        # Force materialisation.
        np.asarray(ds_hashed[next(iter(ds_hashed.data_vars))].values)
        ds_hashed.close()

        ds_unhashed = xr.open_dataset(_build_unhashed_message(tmp_path), engine="tensogram")
        np.asarray(ds_unhashed[next(iter(ds_unhashed.data_vars))].values)
        ds_unhashed.close()

    def test_verify_hash_true_succeeds_on_hashed_dataset(self, tmp_path):
        """Cell B equivalent on the xarray surface: opening a
        hashed file with ``verify_hash=True`` and pulling data
        materialises cleanly."""
        ds = xr.open_dataset(
            _build_hashed_message(tmp_path),
            engine="tensogram",
            verify_hash=True,
            # Force the full-decode path so the verification fires.
            range_threshold=0.0,
        )
        arr = np.asarray(ds[next(iter(ds.data_vars))].values)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0])
        ds.close()

    def test_verify_hash_true_raises_missing_hash_on_unhashed(self, tmp_path):
        """Cell C on xarray: open a hashless file with
        ``verify_hash=True`` and the first read raises
        ``MissingHashError`` from the underlying
        :mod:`tensogram` bindings."""
        ds = xr.open_dataset(
            _build_unhashed_message(tmp_path),
            engine="tensogram",
            verify_hash=True,
            # Force the full-decode path so the verification fires.
            range_threshold=0.0,
        )
        try:
            with pytest.raises(tensogram.MissingHashError) as excinfo:
                _ = np.asarray(ds[next(iter(ds.data_vars))].values)
            assert excinfo.value.object_index == 0
        finally:
            ds.close()
