# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for ``tensogram.compute_common()``.

Direct binding of the Rust core ``tensogram::compute_common`` — extracts
keys shared (with identical values) across every ``base[i]`` entry.
Commonalities are a post-decode software convenience and are **never**
encoded on the wire.  Before this binding existed, ``examples/jupyter/01``
reimplemented the logic in pure Python.
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


def test_returns_common_and_remaining_tuple() -> None:
    base = [
        {"source": "syn", "experiment": "demo", "mars": {"param": "2t"}},
        {"source": "syn", "experiment": "demo", "mars": {"param": "lsm"}},
    ]
    common, remaining = tensogram.compute_common(base)

    assert common == {"source": "syn", "experiment": "demo"}
    # `mars` differs (nested `param`) so it is per-object, not common.
    assert remaining == [{"mars": {"param": "2t"}}, {"mars": {"param": "lsm"}}]


def test_empty_base() -> None:
    common, remaining = tensogram.compute_common([])
    assert common == {}
    assert remaining == []


def test_single_entry_all_keys_common() -> None:
    common, remaining = tensogram.compute_common([{"a": 1, "b": 2}])
    assert common == {"a": 1, "b": 2}
    assert remaining == [{}]


def test_no_common_keys() -> None:
    base = [{"a": 1}, {"b": 2}]
    common, remaining = tensogram.compute_common(base)
    assert common == {}
    assert remaining == [{"a": 1}, {"b": 2}]


def test_all_identical() -> None:
    base = [{"class": "od", "stream": "oper"}, {"class": "od", "stream": "oper"}]
    common, remaining = tensogram.compute_common(base)
    assert common == {"class": "od", "stream": "oper"}
    assert remaining == [{}, {}]


def test_accepts_metadata_object() -> None:
    base = [
        {"source": "syn", "experiment": "demo"},
        {"source": "syn", "experiment": "demo"},
    ]
    msg = tensogram.encode({"base": base}, [(_DESCRIPTOR, _PAYLOAD), (_DESCRIPTOR, _PAYLOAD)])
    meta = tensogram.decode_metadata(msg)

    common, remaining = tensogram.compute_common(meta)
    assert common == {"source": "syn", "experiment": "demo"}
    assert len(remaining) == 2


def test_reserved_key_excluded_from_common() -> None:
    # The encoder auto-populates `_reserved_.tensor` in every base entry
    # (ndim/shape/strides/dtype), which differs per object.  It must never
    # appear in the common set nor leak into `remaining`.
    base = [{"shared": "yes"}, {"shared": "yes"}]
    msg = tensogram.encode({"base": base}, [(_DESCRIPTOR, _PAYLOAD), (_DESCRIPTOR, _PAYLOAD)])
    meta = tensogram.decode_metadata(msg)
    assert "_reserved_" in meta.base[0]  # provenance is present per object

    common, remaining = tensogram.compute_common(meta)
    assert common == {"shared": "yes"}
    assert "_reserved_" not in common
    for entry in remaining:
        assert "_reserved_" not in entry


def test_nested_map_equality() -> None:
    # Identical nested maps ARE common; a single differing nested field is not.
    base = [
        {"mars": {"class": "od", "param": "2t"}},
        {"mars": {"class": "od", "param": "2t"}},
    ]
    common, _ = tensogram.compute_common(base)
    assert common == {"mars": {"class": "od", "param": "2t"}}


def test_invalid_argument_type_raises() -> None:
    with pytest.raises(ValueError, match=r"(?i)Metadata|list"):
        tensogram.compute_common("not a base")
