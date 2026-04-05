"""
Example 04 -- Multiple objects in one message (Python)

One message carries several tensors. Each has its own shape, dtype,
encoding pipeline, and per-object metadata. Objects are decoded by index.
"""

import numpy as np
import tensogram

nlat, nlon, nfreq = 30, 60, 25

# -- Object 0: wave spectrum (float32, lat x lon x freq) -----------------------
spectrum = np.random.default_rng(0).random((nlat, nlon, nfreq), dtype=np.float32)

# -- Object 1: land/sea mask (uint8, lat x lon) --------------------------------
mask = np.zeros((nlat, nlon), dtype=np.uint8)
mask[::2] = 1  # alternate rows are land

# -- Metadata: shared context + per-object descriptors -------------------------
metadata = {
    "version": 2,
    "common": {
        "mars": {"class": "od", "date": "20260401", "step": 6, "type": "fc"},
    },
    "payload": [
        {"mars": {"param": "wave_spectra", "levtype": "sfc"}},
        {"mars": {"param": "lsm", "levtype": "sfc"}},
    ],
}

desc_spectrum = {
    "type": "ntensor",
    "shape": list(spectrum.shape),
    "dtype": "float32",
    "byte_order": "little",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}

desc_mask = {
    "type": "ntensor",
    "shape": list(mask.shape),
    "dtype": "uint8",
    "byte_order": "little",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}

# -- Encode both arrays in one call --------------------------------------------
message = bytes(
    tensogram.encode(metadata, [(desc_spectrum, spectrum), (desc_mask, mask)])
)
print(f"Message: {len(message)} bytes")
print(f"  spectrum:  {spectrum.nbytes} bytes")
print(f"  mask:      {mask.nbytes} bytes")

# -- Decode all objects --------------------------------------------------------
meta, objects = tensogram.decode(message)

print(f"\ndecode() -- {len(objects)} objects:")
for i, (desc, arr) in enumerate(objects):
    print(
        f"  [{i}] shape={arr.shape}  dtype={arr.dtype}  "
        f"params_keys={list(desc.params.keys())}"
    )

np.testing.assert_array_equal(objects[0][1], spectrum)
np.testing.assert_array_equal(objects[1][1], mask)

# -- Decode a single object by index (O(1) via index frame) --------------------
_, desc_decoded, mask_decoded = tensogram.decode_object(message, index=1)

print(f"\ndecode_object(index=1):")
print(f"  shape={mask_decoded.shape}  dtype={mask_decoded.dtype}")
np.testing.assert_array_equal(mask_decoded, mask)

# -- decode_descriptors: metadata + descriptors without payload decode ----------
meta_d, descriptors = tensogram.decode_descriptors(message)

print(f"\ndecode_descriptors() -- {len(descriptors)} descriptors (no payload decode):")
for i, d in enumerate(descriptors):
    print(f"  [{i}] shape={d.shape}  dtype={d.dtype}")

# -- decode_range: partial slice from object 0 ---------------------------------
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
