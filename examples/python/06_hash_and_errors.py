"""
Example 06 — Hash verification and error handling (Python)

NOTE: Requires tensogram Python bindings (not yet implemented).
      This file documents the intended API.

Shows all three hash algorithms, verify_hash on decode, and the
exception hierarchy for common error conditions.
"""

import numpy as np
import tensogram
from tensogram import (
    TensogramError,
    FramingError,
    MetadataError,
    EncodingError,
    HashMismatchError,
    ObjectError,
)

data = np.ones(100, dtype=np.float32)
metadata = tensogram.Metadata(
    version=1,
    objects=[tensogram.ObjectDescriptor(
        type="ntensor", shape=[100], dtype="float32", byte_order="big"
    )],
)

# ── 1. Hash algorithms ────────────────────────────────────────────────────────
for algo in ["xxh3", "sha1", "md5", None]:
    msg = tensogram.encode(metadata, data, hash=algo)
    meta = tensogram.decode_metadata(msg)
    h = meta.payload[0].hash
    if h:
        print(f"{algo:6}: type={h['type']}  value={h['value'][:16]}...")
    else:
        print(f"{algo}: no hash stored")

# ── 2. Hash verification on decode ───────────────────────────────────────────
msg = tensogram.encode(metadata, data, hash="xxh3")
_, arrays = tensogram.decode(msg, verify_hash=True)   # raises on mismatch
print("\nverify_hash=True on clean message: OK")

# No hash → silently skipped with verify_hash=True
msg_no_hash = tensogram.encode(metadata, data, hash=None)
_, _ = tensogram.decode(msg_no_hash, verify_hash=True)
print("verify_hash=True on no-hash message: silently OK")

# ── 3. Corruption detection ───────────────────────────────────────────────────
msg = tensogram.encode(metadata, data, hash="xxh3")
corrupted = bytearray(msg)
# Flip a byte in the payload area (past the CBOR section)
objs_pos = corrupted.find(b"OBJS")
corrupted[objs_pos + 10] ^= 0xFF

try:
    tensogram.decode(bytes(corrupted), verify_hash=True)
    assert False, "should have raised"
except HashMismatchError as e:
    print(f"\nCorruption detected: {e}")
    print(f"  expected: {e.expected[:16]}...")
    print(f"  actual:   {e.actual[:16]}...")

# ── 4. Error hierarchy ────────────────────────────────────────────────────────
print("\nError hierarchy:")

# FramingError — invalid magic bytes
try:
    tensogram.decode(b"GARBAGE!")
except FramingError as e:
    print(f"  FramingError: {e}")

# ObjectError — index out of range
msg = tensogram.encode(metadata, data)
try:
    tensogram.decode_object(msg, index=99)
except ObjectError as e:
    print(f"  ObjectError: {e}")

# EncodingError — NaN in simple_packing
import tensogram.simple_packing as sp
nan_data = np.array([1.0, float("nan"), 3.0])
try:
    sp.compute_params(nan_data, bits_per_value=16)
except EncodingError as e:
    print(f"  EncodingError (NaN): {e}")

# MetadataError — mismatched array count
meta2 = tensogram.Metadata(
    version=1,
    objects=[
        tensogram.ObjectDescriptor(type="ntensor", shape=[10], dtype="float32"),
        tensogram.ObjectDescriptor(type="ntensor", shape=[10], dtype="float32"),
    ],
)
try:
    tensogram.encode(meta2, data)  # only 1 array for 2 objects
except MetadataError as e:
    print(f"  MetadataError (length mismatch): {e}")

# All errors are subclasses of TensogramError
assert issubclass(HashMismatchError, TensogramError)
assert issubclass(FramingError,      TensogramError)
assert issubclass(ObjectError,       TensogramError)
assert issubclass(EncodingError,     TensogramError)
assert issubclass(MetadataError,     TensogramError)
print("\nAll error checks passed.")
