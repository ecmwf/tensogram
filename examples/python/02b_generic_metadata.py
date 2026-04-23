# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 02b — Per-object metadata with a generic application namespace (Python)

Tensogram is vocabulary-agnostic: the library never interprets metadata
keys.  This example attaches two parallel per-object namespaces — a
made-up ``"product"`` namespace plus an ``"instrument"`` namespace — to a
2-D field, to show how any domain can model its semantics inside
``metadata["base"][i]``.

The same pattern fits any domain:
    - CF conventions (``"cf"``) for climate / atmospheric data
    - BIDS (``"bids"``) for neuroimaging datasets
    - DICOM (``"dicom"``) for medical imaging
    - Custom (``"experiment"``, ``"run"``, ``"device"``, ...)

The library simply stores and returns the keys you supply; meaning is
assigned by the application layer.
"""

import numpy as np
import tensogram

# ── 1. Build metadata with application-defined namespaces ─────────────────────
#
# Two parallel namespaces coexist freely in the same base[i] entry.  Any
# top-level key inside a base entry becomes a namespace; there is no
# library-imposed schema.
metadata = {
        "base": [
        {
            "product": {
                "name": "intensity",
                "units": "counts",
                "device": "detector_A",
                "run_id": 42,
                "acquired_at": "2026-04-18T10:30:00Z",
            },
            "instrument": {
                "serial": "XYZ-001",
                "firmware": "v3.1.2",
            },
        },
    ],
}

descriptor = {
    "type": "ntensor",
    "shape": [512, 512],
    "dtype": "float32",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}

data = np.zeros((512, 512), dtype=np.float32)
message = bytes(tensogram.encode(metadata, [(descriptor, data)]))

# ── 2. Read back metadata only (no payload decode) ────────────────────────────
meta = tensogram.decode_metadata(message)

print("Object 0 (product namespace):")
print(f"  name   = {meta.base[0]['product']['name']}")
print(f"  units  = {meta.base[0]['product']['units']}")
print(f"  device = {meta.base[0]['product']['device']}")
print(f"  run_id = {meta.base[0]['product']['run_id']}")

print("\nObject 0 (instrument namespace):")
print(f"  serial   = {meta.base[0]['instrument']['serial']}")
print(f"  firmware = {meta.base[0]['instrument']['firmware']}")

assert meta.base[0]["product"]["name"] == "intensity"
assert meta.base[0]["product"]["run_id"] == 42
assert meta.base[0]["instrument"]["serial"] == "XYZ-001"

# Dictionary-style access (meta["key"]) searches base entries first (first
# match across entries) then the message-level _extra_ map:
print(f"\nDictionary access: meta['product'] = {meta['product']}")

# ── 3. Full decode + check ────────────────────────────────────────────────────
msg = tensogram.decode(message)
desc, arr = msg.objects[0]
print(f"\nFull decode: shape={arr.shape}  dtype={arr.dtype}")
np.testing.assert_array_equal(arr, data)

print("\nGeneric-namespace round-trip OK.")
