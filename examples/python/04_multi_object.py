"""
Example 04 — Multiple objects in one message (Python)

NOTE: Requires tensogram Python bindings (not yet implemented).
      This file documents the intended API.

One message carries several tensors. Each has its own shape, dtype,
encoding pipeline, and per-object metadata. Objects are decoded by index.
"""

import numpy as np
import tensogram

nlat, nlon, nfreq = 30, 60, 25

# ── Object 0: wave spectrum (float32, lat × lon × freq) ───────────────────────
spectrum = np.random.default_rng(0).random((nlat, nlon, nfreq), dtype=np.float32)

# ── Object 1: land/sea mask (uint8, lat × lon) ────────────────────────────────
mask = np.zeros((nlat, nlon), dtype=np.uint8)
mask[::2] = 1  # alternate rows are land

# ── Shared forecast context + per-object param ────────────────────────────────
metadata = tensogram.Metadata(
    version=1,
    objects=[
        tensogram.ObjectDescriptor(
            type="ntensor",
            shape=list(spectrum.shape),
            dtype="float32",
            byte_order="big",
            mars={"param": "wave_spectra", "levtype": "sfc"},
        ),
        tensogram.ObjectDescriptor(
            type="ntensor",
            shape=list(mask.shape),
            dtype="uint8",
            byte_order="big",
            mars={"param": "lsm", "levtype": "sfc"},
        ),
    ],
    mars={
        "class": "od",
        "date": "20260401",
        "step": 6,
        "type": "fc",
    },
)

# ── Encode both arrays in one call ────────────────────────────────────────────
#
# Pass arrays positionally, one per object descriptor.
message: bytes = tensogram.encode(metadata, spectrum, mask)
print(f"Message: {len(message)} bytes")
print(f"  spectrum:  {spectrum.nbytes} bytes")
print(f"  mask:      {mask.nbytes} bytes")

# ── Decode all objects ─────────────────────────────────────────────────────────
meta, arrays = tensogram.decode(message)

print(f"\ndecode() — {len(arrays)} objects:")
for i, (arr, desc) in enumerate(zip(arrays, meta.objects)):
    print(
        f"  [{i}] shape={arr.shape}  dtype={arr.dtype}  "
        f"param={desc.extra.get('mars', {}).get('param', '?')}"
    )

np.testing.assert_array_equal(arrays[0], spectrum)
np.testing.assert_array_equal(arrays[1], mask)

# ── Decode a single object by index (O(1) via binary header) ──────────────────
#
# Only the binary header and the requested object's payload are read.
desc, mask_decoded = tensogram.decode_object(message, index=1)

print(f"\ndecode_object(index=1):")
print(f"  shape={mask_decoded.shape}  dtype={mask_decoded.dtype}")
np.testing.assert_array_equal(mask_decoded, mask)

# ── decode_range: partial slice from object 0 ──────────────────────────────────
#
# Extract elements [100 .. 149] of the flattened spectrum array.
# Default: returns a list of arrays (one per range).
parts = tensogram.decode_range(message, object_index=0, ranges=[(100, 50)])
assert isinstance(parts, list) and len(parts) == 1
expected = spectrum.ravel()[100:150]
np.testing.assert_array_equal(parts[0], expected)
print(
    f"\ndecode_range(obj=0, 100..150) [split]: {len(parts)} part, shape={parts[0].shape}  OK"
)

# join=True: returns a single concatenated array (pre-0.6 behaviour).
joined = tensogram.decode_range(message, object_index=0, ranges=[(100, 50)], join=True)
np.testing.assert_array_equal(joined, expected)
print(f"decode_range(obj=0, 100..150) [join]:  shape={joined.shape}  OK")

# Multiple ranges: split returns one array per range.
parts2 = tensogram.decode_range(message, object_index=0, ranges=[(0, 10), (200, 5)])
assert len(parts2) == 2
np.testing.assert_array_equal(parts2[0], spectrum.ravel()[:10])
np.testing.assert_array_equal(parts2[1], spectrum.ravel()[200:205])
print(f"decode_range(obj=0, multi) [split]:  {len(parts2)} parts  OK")

print("\nAll assertions passed.")
