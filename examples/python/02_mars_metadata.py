"""Example 02 — MARS-namespaced metadata (Python)

Shows how to attach ECMWF MARS vocabulary keys at message level and
per-object level, then read them back with decode_metadata().

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import numpy as np
import tensogram

# ── 1. Build metadata with MARS keys ──────────────────────────────────────────
#
# Any dict key is valid — the library does not interpret vocabulary.
# Convention: group ECMWF keys under "mars".
metadata = {
    "version": 2,
    "common": {
        "mars": {
            "class": "od",
            "date": "20260401",
            "step": 6,
            "time": "0000",
            "type": "fc",
        },
    },
    "payload": [
        {"mars": {"param": "2t", "levtype": "sfc"}},
    ],
}

descriptor = {
    "type": "ntensor",
    "shape": [721, 1440],
    "dtype": "float32",
    "byte_order": "little",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}

data = np.zeros((721, 1440), dtype=np.float32)
message = bytes(tensogram.encode(metadata, [(descriptor, data)]))

# ── 2. Read metadata without decoding payload ─────────────────────────────────
#
# decode_metadata() is faster than decode() because it reads only the
# CBOR section. Use it for filtering/listing large files.
meta = tensogram.decode_metadata(message)

print("Message-level (common):")
print(f"  mars.class = {meta.common['mars']['class']}")
print(f"  mars.date  = {meta.common['mars']['date']}")
print(f"  mars.step  = {meta.common['mars']['step']}")
print(f"  mars.type  = {meta.common['mars']['type']}")
print("Object 0 (payload):")
print(f"  mars.param   = {meta.payload[0]['mars']['param']}")
print(f"  mars.levtype = {meta.payload[0]['mars']['levtype']}")

assert meta.common["mars"]["class"] == "od"
assert meta.common["mars"]["step"] == 6
assert meta.payload[0]["mars"]["param"] == "2t"

# ── 3. Full decode + check ────────────────────────────────────────────────────
msg = tensogram.decode(message)
desc, arr = msg.objects[0]
print(f"\nFull decode: shape={arr.shape}  dtype={arr.dtype}")
np.testing.assert_array_equal(arr, data)

print("\nAll assertions passed.")
