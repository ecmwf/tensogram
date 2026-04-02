"""
Example 01 — Basic encode / decode round-trip (Python)

NOTE: Requires tensogram Python bindings (not yet implemented).
      This file documents the intended API.

Encodes a 100×200 float32 temperature grid into a Tensogram message,
then decodes it back, recovering a numpy array with identical values.
"""

import numpy as np
import tensogram  # planned package

# ── 1. Source data ────────────────────────────────────────────────────────────
#
# Any numpy array is accepted. The dtype is read automatically.
temps = np.linspace(273.15, 283.15, 100 * 200, dtype=np.float32).reshape(100, 200)
print(f"Input: shape={temps.shape}  dtype={temps.dtype}  "
      f"size={temps.nbytes} bytes")

# ── 2. Describe the message ───────────────────────────────────────────────────
#
# Metadata can be a plain dict. The library maps it to CBOR.
metadata = tensogram.Metadata(
    version=1,
    objects=[
        tensogram.ObjectDescriptor(
            type="ntensor",
            shape=list(temps.shape),
            dtype="float32",
            byte_order="big",
        )
    ],
    # payload descriptors are optional — defaults to encoding="none"
)

# ── 3. Encode ─────────────────────────────────────────────────────────────────
#
# Returns a bytes object containing the complete wire-format message.
# hash="xxh3" (default) appends an integrity hash to each payload.
message: bytes = tensogram.encode(metadata, temps, hash="xxh3")

print(f"Message: {len(message)} bytes")
print(f"  magic:      {message[:8]}")
print(f"  terminator: {message[-8:]}")

# ── 4. Decode ─────────────────────────────────────────────────────────────────
#
# Returns (Metadata, list[numpy.ndarray]).
# Each array has the correct dtype and shape already applied.
meta, arrays = tensogram.decode(message)

print(f"\nDecoded: {len(arrays)} object(s)")
print(f"  shape={arrays[0].shape}  dtype={arrays[0].dtype}")

assert arrays[0].shape == temps.shape
assert arrays[0].dtype == temps.dtype
np.testing.assert_array_equal(arrays[0], temps)
print("Round-trip OK: identical values.")

# ── 5. Inspect metadata ───────────────────────────────────────────────────────
print(f"\nMetadata:")
print(f"  version = {meta.version}")
print(f"  objects[0].shape  = {meta.objects[0].shape}")
print(f"  objects[0].dtype  = {meta.objects[0].dtype}")
print(f"  payload[0].hash   = {meta.payload[0].hash}")
