# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Generate the shared metadata-access parity fixture.

`metadata_parity.tgm` is a deterministic two-object message whose CBOR
metadata frame exercises every capability the cross-language parity tests
assert (Rust, C, C++, Python, TypeScript, Fortran). It is the single fixture
behind `plans/METADATA_ACCESS_PARITY.md` §7.

Run from the repo root, with the `tensogram` Python extension installed:

    python tests/fixtures/gen_metadata_parity.py

The canonical contents (asserted identically by every binding):

  base[0] = {
    shortName: "2t"        # string, first-match wins at message level
    level:     0           # int == 0  (proves absent != default)
    note:      ""          # empty string (proves absent != empty)
    count:     -5          # negative int (as_u64 -> None)
    big:       2147483648  # int within JS safe-integer range
    ratio:     0.5         # float (as_i64 -> None; as_f64 -> 0.5)
    flag:      true        # bool
    mars:      {class:"od", stream:"oper"}   # nested map
    levels:    [1000, 850, 500]              # array
    only0:     "x"         # present in obj0, absent in obj1
    _reserved_.tensor      # auto-populated by the encoder (shape/dtype)
  }
  base[1] = { shortName: "msl", mars: {class:"ea"} }
  _extra_ = { source: "parity-fixture" }
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensogram

OUT = Path(__file__).with_name("metadata_parity.tgm")


def _desc(shape: list[int], dtype: str) -> dict:
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": dtype,
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


def main() -> None:
    global_meta = {
        "base": [
            {
                "shortName": "2t",
                "level": 0,
                "note": "",
                "count": -5,
                "big": 2147483648,
                "ratio": 0.5,
                "flag": True,
                "mars": {"class": "od", "stream": "oper"},
                "levels": [1000, 850, 500],
                "only0": "x",
            },
            {
                "shortName": "msl",
                "mars": {"class": "ea"},
            },
        ],
        # Free-form top-level key -> flows into `_extra_`.
        "source": "parity-fixture",
    }

    obj0 = np.zeros((2, 3), dtype=np.float32)
    obj1 = np.zeros((4,), dtype=np.float64)
    msg = tensogram.encode(
        global_meta,
        [
            (_desc([2, 3], "float32"), obj0.tobytes()),
            (_desc([4], "float64"), obj1.tobytes()),
        ],
    )

    OUT.write_bytes(msg)
    print(f"wrote {OUT} ({len(msg)} bytes)")

    # Self-check: decode and confirm the contract holds through a round-trip.
    meta = tensogram.decode_metadata(msg)
    assert meta.num_objects == 2, meta.num_objects
    assert meta.get_path("shortName") == "2t"
    assert meta.get_path("note") == ""
    assert meta.get_path("level") == 0
    assert meta.has_path("note") and not meta.has_path("missing")
    assert meta.get_path_at(1, "shortName") == "msl"
    assert meta.get_path_at(0, "only0") == "x"
    assert meta.get_path_at(1, "only0") is None
    assert meta.get_path("mars.class") == "od"
    assert meta.get_path("source") == "parity-fixture"
    assert not meta.has_path("_reserved_.tensor")
    assert "_reserved_" in meta.base[0]
    print("self-check OK")


if __name__ == "__main__":
    main()
