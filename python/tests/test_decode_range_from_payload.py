# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for tensogram.decode_range_from_payload() — payload-relative
range decode."""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

import numpy as np
import pytest
import tensogram

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_object(arr: np.ndarray):
    """Encode a single-object message with encoding=none (so the on-wire
    payload is just the raw native-endian bytes) and return the decoded
    descriptor plus that raw payload."""
    desc = {
        "type": "ntensor",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }
    msg = tensogram.encode({"base": [{}]}, [(desc, arr)])
    _gm, descs = tensogram.decode_descriptors(msg)
    payload = arr.tobytes()
    return descs[0], payload, msg


# ---------------------------------------------------------------------------
# Basic decode
# ---------------------------------------------------------------------------


class TestDecodeRangeFromPayload:
    def test_single_range(self):
        arr = np.arange(20, dtype=np.float32)
        desc, payload, _ = _raw_object(arr)
        out = tensogram.decode_range_from_payload(desc, payload, [(0, 5)])
        assert len(out) == 1
        np.testing.assert_array_equal(out[0], arr[0:5])

    def test_multiple_ranges(self):
        arr = np.arange(20, dtype=np.float64)
        desc, payload, _ = _raw_object(arr)
        out = tensogram.decode_range_from_payload(desc, payload, [(0, 3), (10, 4)])
        assert len(out) == 2
        np.testing.assert_array_equal(out[0], arr[0:3])
        np.testing.assert_array_equal(out[1], arr[10:14])

    def test_join(self):
        arr = np.arange(20, dtype=np.float32)
        desc, payload, _ = _raw_object(arr)
        out = tensogram.decode_range_from_payload(desc, payload, [(0, 3), (10, 2)], join=True)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.concatenate([arr[0:3], arr[10:12]]))

    def test_int_dtype(self):
        arr = np.arange(16, dtype=np.int32)
        desc, payload, _ = _raw_object(arr)
        out = tensogram.decode_range_from_payload(desc, payload, [(4, 4)])
        np.testing.assert_array_equal(out[0], arr[4:8])

    def test_full_range_equals_array(self):
        arr = np.arange(10, dtype=np.float64)
        desc, payload, _ = _raw_object(arr)
        out = tensogram.decode_range_from_payload(desc, payload, [(0, 10)], join=True)
        np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# Parity with the message-level decode_range
# ---------------------------------------------------------------------------


class TestParityWithDecodeRange:
    def test_matches_decode_range(self):
        arr = np.arange(30, dtype=np.float32)
        desc, payload, msg = _raw_object(arr)
        ranges = [(0, 5), (12, 3), (25, 5)]
        via_payload = tensogram.decode_range_from_payload(desc, payload, ranges)
        via_message = tensogram.decode_range(msg, 0, ranges)
        assert len(via_payload) == len(via_message)
        for a, b in zip(via_payload, via_message):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_filter_rejected(self):
        # A shuffle filter is unsupported for range decode; the check
        # fires before the payload is inspected.
        arr = np.arange(16, dtype=np.float32)
        desc_dict = {
            "type": "ntensor",
            "shape": [16],
            "dtype": "float32",
            "filter": "shuffle",
            "shuffle_element_size": 4,
        }
        msg = tensogram.encode({"base": [{}]}, [(desc_dict, arr)])
        _gm, descs = tensogram.decode_descriptors(msg)
        with pytest.raises(ValueError, match="filter"):
            tensogram.decode_range_from_payload(descs[0], b"\x00" * 64, [(0, 4)])

    def test_out_of_bounds_range(self):
        arr = np.arange(10, dtype=np.float32)
        desc, payload, _ = _raw_object(arr)
        with pytest.raises(ValueError, match="exceeds payload size"):
            tensogram.decode_range_from_payload(desc, payload, [(0, 1000)])

    def test_rejects_non_descriptor(self):
        arr = np.arange(10, dtype=np.float32)
        _desc, payload, _ = _raw_object(arr)
        with pytest.raises((TypeError, ValueError)):
            tensogram.decode_range_from_payload({"not": "a descriptor"}, payload, [(0, 1)])
