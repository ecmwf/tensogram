# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 06 — Hash verification and error handling (Python)

Shows xxh3 hash algorithm, verify_hash on decode, and the
error conditions for common mistakes.

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import numpy as np
import tensogram

data = np.ones(100, dtype=np.float32)
metadata = {"version": 2}
descriptor = {
    "type": "ntensor",
    "shape": [100],
    "dtype": "float32",
    "byte_order": "little",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}

# ── 1. Hash algorithms ────────────────────────────────────────────────────────
for algo in ["xxh3", None]:
    msg = bytes(tensogram.encode(metadata, [(descriptor, data)], hash=algo))
    meta = tensogram.decode_metadata(msg)
    h = algo or "none"
    print(f"hash={h}: message size={len(msg)} bytes")

# ── 2. Hash verification on decode ───────────────────────────────────────────
msg = bytes(tensogram.encode(metadata, [(descriptor, data)], hash="xxh3"))
result = tensogram.decode(msg, verify_hash=True)  # raises on mismatch
print("\nverify_hash=True on clean message: OK")

# No hash → silently skipped with verify_hash=True
msg_no_hash = bytes(tensogram.encode(metadata, [(descriptor, data)], hash=None))
_ = tensogram.decode(msg_no_hash, verify_hash=True)
print("verify_hash=True on no-hash message: silently OK")

# ── 3. Corruption detection ───────────────────────────────────────────────────
msg = bytes(tensogram.encode(metadata, [(descriptor, data)], hash="xxh3"))
corrupted = bytearray(msg)
# Flip a byte in the payload area (past headers)
corrupted[len(corrupted) // 2] ^= 0xFF

try:
    tensogram.decode(bytes(corrupted), verify_hash=True)
    raise AssertionError("should have raised")
except Exception as e:
    print(f"\nCorruption detected: {type(e).__name__}: {e}")

# ── 4. Error handling examples ────────────────────────────────────────────────
print("\nError handling:")

# Garbage data
try:
    tensogram.decode(b"GARBAGE!")
except Exception as e:
    print(f"  Garbage input: {type(e).__name__}")

# Empty data
try:
    tensogram.decode(b"")
except Exception as e:
    print(f"  Empty input: {type(e).__name__}")

# Object index out of range
msg = bytes(tensogram.encode(metadata, [(descriptor, data)]))
try:
    tensogram.decode_object(msg, index=99)
except Exception as e:
    print(f"  Bad index: {type(e).__name__}")

# NaN in compute_packing_params
nan_data = np.array([1.0, float("nan"), 3.0])
try:
    tensogram.compute_packing_params(nan_data, bits_per_value=16, decimal_scale_factor=0)
except ValueError as e:
    print(f"  NaN rejected: {type(e).__name__}")

print("\nAll error checks passed.")
