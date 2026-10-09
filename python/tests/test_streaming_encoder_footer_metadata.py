# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for ``StreamingEncoder.set_footer_metadata()``.

Message-level annotations a producer only learns while encoding are merged
into the footer metadata frame, where they override same-named keys from the
constructor metadata.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensogram

_DESCRIPTOR = {
    "type": "ntensor",
    "ndim": 1,
    "shape": [4],
    "dtype": "float32",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}
_PAYLOAD = np.arange(4, dtype=np.float32)


def _extra(message: bytes) -> dict:
    """The decoded ``_extra_`` map of a message."""
    return dict(tensogram.decode(message).metadata.extra)


def test_annotations_reach_the_decoded_extra_map() -> None:
    enc = tensogram.StreamingEncoder({})
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.set_footer_metadata({"written_params": ["2t", "10u"], "count": 2})
    msg = enc.finish()

    extra = _extra(msg)
    assert extra["written_params"] == ["2t", "10u"]
    assert extra["count"] == 2


def test_footer_value_overrides_header_value() -> None:
    enc = tensogram.StreamingEncoder({"_extra_": {"status": "provisional", "centre": "ecmwf"}})
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.set_footer_metadata({"status": "complete"})
    msg = enc.finish()

    extra = _extra(msg)
    assert extra["status"] == "complete"
    assert extra["centre"] == "ecmwf"


def test_repeated_calls_accumulate() -> None:
    enc = tensogram.StreamingEncoder({})
    enc.set_footer_metadata({"first": 1, "shared": "old"})
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.set_footer_metadata({"second": 2, "shared": "new"})
    msg = enc.finish()

    extra = _extra(msg)
    assert extra["first"] == 1
    assert extra["second"] == 2
    assert extra["shared"] == "new"


def test_no_annotations_leaves_the_message_unchanged() -> None:
    def encode(annotate: bool) -> bytes:
        enc = tensogram.StreamingEncoder({"_extra_": {"centre": "ecmwf"}})
        enc.write_object(_DESCRIPTOR, _PAYLOAD)
        if annotate:
            enc.set_footer_metadata({})
        return enc.finish()

    plain = encode(annotate=False)
    annotated = encode(annotate=True)

    assert len(plain) == len(annotated)
    assert _extra(plain) == _extra(annotated)


@pytest.mark.parametrize("backfilled", [False, True])
def test_annotations_survive_the_backfilled_finish(backfilled: bool) -> None:
    enc = tensogram.StreamingEncoder({})
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.set_footer_metadata({"status": "complete"})
    msg = enc.finish_backfilled() if backfilled else enc.finish()

    assert _extra(msg)["status"] == "complete"


def test_annotations_are_per_message_across_resets() -> None:
    enc = tensogram.StreamingEncoder({})

    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.set_footer_metadata({"coverage": "first"})
    first = enc.finish_and_reset()

    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    second = enc.finish_and_reset_backfilled()

    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.set_footer_metadata({"coverage": "third"})
    third = enc.finish()

    assert _extra(first)["coverage"] == "first"
    assert "coverage" not in _extra(second)
    assert _extra(third)["coverage"] == "third"


def test_set_after_finish_raises() -> None:
    enc = tensogram.StreamingEncoder({})
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.finish()

    with pytest.raises(RuntimeError, match="already finished"):
        enc.set_footer_metadata({"status": "complete"})


def test_non_dict_annotations_rejected() -> None:
    enc = tensogram.StreamingEncoder({})
    with pytest.raises(TypeError):
        enc.set_footer_metadata([("status", "complete")])
