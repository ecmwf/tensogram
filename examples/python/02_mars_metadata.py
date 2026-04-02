"""
Example 02 — MARS-namespaced metadata (Python)

NOTE: Requires tensogram Python bindings (not yet implemented).
      This file documents the intended API.

Shows how to attach ECMWF MARS vocabulary keys at message level and
per-object level, then filter and read them back.
"""

import numpy as np
import tensogram

# ── 1. Build metadata with MARS keys ──────────────────────────────────────────
#
# Any dict key is valid — the library does not interpret vocabulary.
# Convention: group ECMWF keys under "mars".
metadata = tensogram.Metadata(
    version=1,
    objects=[
        tensogram.ObjectDescriptor(
            type="ntensor",
            shape=[721, 1440],      # 0.25-degree global grid
            dtype="float32",
            byte_order="big",
            mars={                  # per-object: parameter identity
                "param": "2t",
                "levtype": "sfc",
            }
        )
    ],
    mars={                          # message-level: forecast context
        "class": "od",
        "date": "20260401",
        "step": 6,
        "time": "0000",
        "type": "fc",
    }
)

data = np.zeros((721, 1440), dtype=np.float32)
message = tensogram.encode(metadata, data)

# ── 2. Read metadata without decoding payload ──────────────────────────────────
#
# decode_metadata() is faster than decode() because it reads only the
# CBOR section. Use it for filtering/listing large files.
meta = tensogram.decode_metadata(message)

# Dot-notation access for namespaced keys
print("Message-level:")
print(f"  mars.class = {meta['mars']['class']}")
print(f"  mars.date  = {meta['mars']['date']}")
print(f"  mars.step  = {meta['mars']['step']}")
print(f"  mars.type  = {meta['mars']['type']}")
print("Object 0:")
print(f"  mars.param   = {meta.objects[0].extra['mars']['param']}")
print(f"  mars.levtype = {meta.objects[0].extra['mars']['levtype']}")
print(f"  shape        = {meta.objects[0].shape}")

assert meta["mars"]["class"] == "od"
assert meta["mars"]["step"] == 6
assert meta.objects[0].extra["mars"]["param"] == "2t"

# ── 3. Where-clause filtering ──────────────────────────────────────────────────
#
# match() checks a message buffer against a filter expression — same syntax
# as the CLI's -w flag. Useful for streaming filtering without a file.
assert tensogram.match(message, "mars.type=fc"),    "should match fc"
assert tensogram.match(message, "mars.step=6"),     "should match step=6"
assert not tensogram.match(message, "mars.step=12"), "should not match step=12"
assert tensogram.match(message, "mars.param=2t"),   "should match param=2t"
assert tensogram.match(message, "mars.param=2t/10u"), "OR syntax should match"
assert not tensogram.match(message, "mars.param!=2t"), "neq should fail"

print("\nAll filter checks passed.")
