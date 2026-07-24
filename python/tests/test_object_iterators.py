# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the lazy per-object iterators tensogram.objects() and
tensogram.objects_metadata()."""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

import numpy as np
import pytest
import tensogram

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multi_object_msg():
    arrays = [
        np.arange(10, dtype=np.float32),
        np.arange(6, dtype=np.float64).reshape(2, 3),
        np.arange(4, dtype=np.int32),
    ]
    pairs = []
    for arr in arrays:
        desc = {
            "type": "ntensor",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        pairs.append((desc, arr))
    msg = tensogram.encode({"base": [{} for _ in arrays]}, pairs)
    return msg, arrays


# ---------------------------------------------------------------------------
# objects() — lazy decode iterator
# ---------------------------------------------------------------------------


class TestObjects:
    def test_yields_all_objects(self):
        msg, arrays = _multi_object_msg()
        it = tensogram.objects(msg)
        collected = list(it)
        assert len(collected) == len(arrays)

    def test_values_match_decode(self):
        msg, arrays = _multi_object_msg()
        for (_desc, arr), expected in zip(tensogram.objects(msg), arrays):
            np.testing.assert_array_equal(arr.reshape(expected.shape), expected)

    def test_descriptor_shape(self):
        msg, arrays = _multi_object_msg()
        for (desc, _arr), expected in zip(tensogram.objects(msg), arrays):
            assert desc.shape == list(expected.shape)

    def test_matches_eager_decode(self):
        msg, _ = _multi_object_msg()
        eager = tensogram.decode(msg).objects
        lazy = list(tensogram.objects(msg))
        assert len(eager) == len(lazy)
        for (ed, ea), (ld, la) in zip(eager, lazy):
            assert ed.shape == ld.shape
            np.testing.assert_array_equal(ea, la)

    def test_is_iterator(self):
        msg, _ = _multi_object_msg()
        it = tensogram.objects(msg)
        assert iter(it) is it

    def test_len_decrements(self):
        msg, arrays = _multi_object_msg()
        it = tensogram.objects(msg)
        assert len(it) == len(arrays)
        next(it)
        assert len(it) == len(arrays) - 1

    def test_stop_iteration(self):
        msg, arrays = _multi_object_msg()
        it = tensogram.objects(msg)
        for _ in range(len(arrays)):
            next(it)
        with pytest.raises(StopIteration):
            next(it)

    def test_repr(self):
        msg, _ = _multi_object_msg()
        it = tensogram.objects(msg)
        assert "ObjectIter(remaining=" in repr(it)

    def test_single_object(self):
        arr = np.arange(5, dtype=np.float32)
        msg = tensogram.encode(
            {"base": [{}]},
            [({"type": "ntensor", "shape": [5], "dtype": "float32"}, arr)],
        )
        result = list(tensogram.objects(msg))
        assert len(result) == 1
        np.testing.assert_array_equal(result[0][1], arr)

    def test_wire_byte_order(self):
        arr = np.arange(4, dtype=np.float64)
        msg = tensogram.encode(
            {"base": [{}]},
            [({"type": "ntensor", "shape": [4], "dtype": "float64"}, arr)],
        )
        # native_byte_order=True (default) returns readable native values.
        (_desc, out) = next(tensogram.objects(msg, native_byte_order=True))
        np.testing.assert_array_equal(out, arr)

    def test_invalid_buffer(self):
        with pytest.raises(ValueError, match="buffer too short"):
            list(tensogram.objects(b"not a message"))


# ---------------------------------------------------------------------------
# objects_metadata() — lazy descriptor iterator
# ---------------------------------------------------------------------------


class TestObjectsMetadata:
    def test_yields_descriptors(self):
        msg, arrays = _multi_object_msg()
        descs = list(tensogram.objects_metadata(msg))
        assert len(descs) == len(arrays)

    def test_shapes_and_dtypes(self):
        msg, arrays = _multi_object_msg()
        for desc, expected in zip(tensogram.objects_metadata(msg), arrays):
            assert desc.shape == list(expected.shape)
            assert desc.dtype == str(expected.dtype)

    def test_matches_decode_descriptors(self):
        msg, _ = _multi_object_msg()
        _gm, eager = tensogram.decode_descriptors(msg)
        lazy = list(tensogram.objects_metadata(msg))
        assert [d.shape for d in eager] == [d.shape for d in lazy]
        assert [d.dtype for d in eager] == [d.dtype for d in lazy]

    def test_is_iterator(self):
        msg, _ = _multi_object_msg()
        it = tensogram.objects_metadata(msg)
        assert iter(it) is it

    def test_len_decrements(self):
        msg, arrays = _multi_object_msg()
        it = tensogram.objects_metadata(msg)
        assert len(it) == len(arrays)
        next(it)
        assert len(it) == len(arrays) - 1

    def test_stop_iteration(self):
        msg, arrays = _multi_object_msg()
        it = tensogram.objects_metadata(msg)
        for _ in range(len(arrays)):
            next(it)
        with pytest.raises(StopIteration):
            next(it)

    def test_repr(self):
        msg, _ = _multi_object_msg()
        it = tensogram.objects_metadata(msg)
        assert "ObjectMetadataIter(" in repr(it)

    def test_no_payload_decode_needed(self):
        # objects_metadata reads descriptors only; it works even for a
        # payload that decode() would decompress. A simple round-trip check.
        msg, _arrays = _multi_object_msg()
        descs = list(tensogram.objects_metadata(msg))
        assert descs[0].encoding == "none"

    def test_invalid_buffer(self):
        with pytest.raises(ValueError, match="buffer too short"):
            list(tensogram.objects_metadata(b"not a message"))
