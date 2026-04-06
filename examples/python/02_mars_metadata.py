"""Example 02 — MARS-namespaced metadata (Python)

Shows how to attach ECMWF MARS vocabulary keys as per-object metadata
and at message level, then read them back with decode_metadata().

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import numpy as np
import tensogram

# ── 1. Build metadata with MARS keys ──────────────────────────────────────────
#
# Per-object metadata lives in `base[i]`. Each entry holds ALL metadata
# for one object independently — shared keys are simply repeated per entry.
# Message-level extra metadata can be added as top-level keys in the dict
# (anything besides "version", "base", "extra" goes into `extra`).
metadata = {
    "version": 2,
    "base": [
        {
            "mars": {
                "class": "od",
                "date": "20260401",
                "step": 6,
                "time": "0000",
                "type": "fc",
                "param": "2t",
                "levtype": "sfc",
            },
        },
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

print("Object 0 (base):")
print(f"  mars.class   = {meta.base[0]['mars']['class']}")
print(f"  mars.date    = {meta.base[0]['mars']['date']}")
print(f"  mars.step    = {meta.base[0]['mars']['step']}")
print(f"  mars.type    = {meta.base[0]['mars']['type']}")
print(f"  mars.param   = {meta.base[0]['mars']['param']}")
print(f"  mars.levtype = {meta.base[0]['mars']['levtype']}")

assert meta.base[0]["mars"]["class"] == "od"
assert meta.base[0]["mars"]["step"] == 6
assert meta.base[0]["mars"]["param"] == "2t"

# Dictionary-style access searches base entries then extra:
print(f"\nDictionary access: meta['mars'] = {meta['mars']}")

# ── 3. Full decode + check ────────────────────────────────────────────────────
msg = tensogram.decode(message)
desc, arr = msg.objects[0]
print(f"\nFull decode: shape={arr.shape}  dtype={arr.dtype}")
np.testing.assert_array_equal(arr, data)

print("\nAll assertions passed.")
