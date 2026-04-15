# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 01 — Basic encode / decode round-trip (Python)

Encodes a 100x200 float32 temperature grid into a Tensogram message,
then decodes it back, recovering a numpy array with identical values.

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import numpy as np
import tensogram

# ── 1. Source data ────────────────────────────────────────────────────────────
#
# Any numpy array is accepted. The dtype is read automatically.
temps = np.linspace(273.15, 283.15, 100 * 200, dtype=np.float32).reshape(100, 200)
print(f"Input: shape={temps.shape}  dtype={temps.dtype}  size={temps.nbytes} bytes")

# ── 2. Describe the message ───────────────────────────────────────────────────
#
# Metadata and descriptors are plain dicts.
metadata = {"version": 2}
descriptor = {
    "type": "ntensor",
    "shape": list(temps.shape),
    "dtype": "float32",
    "byte_order": "little",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}

# ── 3. Encode ─────────────────────────────────────────────────────────────────
#
# Returns a bytes object containing the complete wire-format message.
# hash="xxh3" (default) appends an integrity hash to each payload.
message = bytes(tensogram.encode(metadata, [(descriptor, temps)], hash="xxh3"))

print(f"Message: {len(message)} bytes")
print(f"  magic:      {message[:8]}")
print(f"  terminator: {message[-16:-8].hex()}")

# ── 4. Decode ─────────────────────────────────────────────────────────────────
#
# Returns a Message namedtuple: msg.metadata and msg.objects.
# Tuple unpacking also works: meta, objects = tensogram.decode(...)
msg = tensogram.decode(message)
meta, objects = msg  # unpack

print(f"\nDecoded: {len(objects)} object(s)")
desc, arr = objects[0]
print(f"  shape={arr.shape}  dtype={arr.dtype}")

assert arr.shape == temps.shape
assert arr.dtype == temps.dtype
np.testing.assert_array_equal(arr, temps)
print("Round-trip OK: identical values.")

# ── 5. Inspect metadata ───────────────────────────────────────────────────────
print(f"\nMetadata:")
print(f"  version = {meta.version}")
print(f"  descriptor shape  = {desc.shape}")
print(f"  descriptor dtype  = {desc.dtype}")
print(f"  descriptor hash   = {desc.hash}")
