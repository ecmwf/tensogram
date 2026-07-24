# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for ``StreamingEncoder`` introspection + ``write_preceder``.

Mirrors the Rust ``StreamingEncoder::object_count`` /
``bytes_written`` / ``write_preceder`` accessors.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensogram

_DESCRIPTOR = {
    "type": "ntensor",
    "shape": [4],
    "dtype": "float32",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}
_PAYLOAD = np.arange(4, dtype=np.float32)
_GLOBAL_META = {"base": [{"name": "test"}]}


def test_object_count_increments_per_write() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    assert enc.object_count() == 0
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    assert enc.object_count() == 1
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    assert enc.object_count() == 2


def test_bytes_written_grows_monotonically() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    after_header = enc.bytes_written()
    # The preamble + header frame were written by the constructor.
    assert after_header > 0
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    after_object = enc.bytes_written()
    assert after_object > after_header


def test_write_preceder_metadata_lands_in_base() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_preceder({"step": 42, "param": "2t"})
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish()

    decoded = tensogram.decode(msg)
    assert decoded.metadata.base[0].get("step") == 42
    assert decoded.metadata.base[0].get("param") == "2t"


def test_write_preceder_twice_raises() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_preceder({"step": 1})
    with pytest.raises(ValueError, match=r"(?i)preceder"):
        enc.write_preceder({"step": 2})


def test_write_preceder_rejects_reserved_key() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    with pytest.raises(ValueError, match="_reserved_"):
        enc.write_preceder({"_reserved_": {"tensor": {}}})


def test_introspection_after_finish_raises() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.finish()
    with pytest.raises(RuntimeError, match="finished"):
        enc.object_count()
    with pytest.raises(RuntimeError, match="finished"):
        enc.bytes_written()


def test_write_preceder_after_finish_raises() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.finish()
    with pytest.raises(RuntimeError, match="finished"):
        enc.write_preceder({"step": 1})


def test_object_count_matches_multi_object_message() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    for i in range(3):
        enc.write_preceder({"idx": i})
        enc.write_object(_DESCRIPTOR, _PAYLOAD + float(i))
    assert enc.object_count() == 3
    msg = enc.finish()
    assert len(tensogram.decode(msg).objects) == 3
