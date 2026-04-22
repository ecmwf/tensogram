# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 05 — File API (Python)

Shows the full TensogramFile lifecycle: create, append, open, iterate,
random access by index, and low-level scan.

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import pathlib
import tempfile

import numpy as np
import tensogram


def make_field(param: str, step: int):
    """Return (metadata_dict, descriptor_dict, data) for one forecast field."""
    metadata = {
                "base": [
            {
                "mars": {
                    "class": "od",
                    "date": "20260401",
                    "step": step,
                    "type": "fc",
                    "param": param,
                    "levtype": "sfc",
                },
            },
        ],
    }
    descriptor = {
        "type": "ntensor",
        "shape": [72, 144],
        "dtype": "float32",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }
    data = np.random.default_rng(hash(f"{param}{step}") % 2**32).random(
        (72, 144), dtype=np.float32
    )
    return metadata, descriptor, data


with tempfile.TemporaryDirectory() as tmpdir:
    path = pathlib.Path(tmpdir) / "forecast.tgm"

    # ── 1. Create and append ──────────────────────────────────────────────────
    with tensogram.TensogramFile.create(str(path)) as f:
        params = ["2t", "10u", "10v", "msl"]
        steps = [0, 6]
        for step in steps:
            for param in params:
                meta, desc, data = make_field(param, step)
                f.append(meta, [(desc, data)])

        count = f.message_count()
        print(f"Written {count} messages")  # 8

    # ── 2. Open and inspect ───────────────────────────────────────────────────
    with tensogram.TensogramFile.open(str(path)) as f:
        count = f.message_count()
        print(f"\nOpened: {count} messages")
        assert count == 8

        # Random access by index
        print("\nRandom access:")
        for idx in [0, 3, 7]:
            msg = f.decode_message(idx)
            param = msg.metadata.base[0]["mars"]["param"]
            step = msg.metadata.base[0]["mars"]["step"]
            desc, arr = msg.objects[0]
            print(f"  [{idx}] param={param:<5}  step={step}  shape={arr.shape}")

        # Read raw bytes
        raw = f.read_message(0)
        print(f"\nread_message(0): {len(raw)} bytes  magic={raw[:8]}")

        # Iterate all messages using for loop (PR #11 feature)
        print("\nAll messages:")
        for i, (meta, objects) in enumerate(f):
            param = meta.base[0]["mars"]["param"]
            step = meta.base[0]["mars"]["step"]
            desc, arr = objects[0]
            print(f"  [{i}] param={param:<5}  step={step}  dtype={arr.dtype}")

        # Indexing and slicing (PR #11 feature)
        first = f[0]
        last = f[-1]
        subset = f[::2]
        print(f"\nf[0]: param={first.metadata.base[0]['mars']['param']}")
        print(f"f[-1]: param={last.metadata.base[0]['mars']['param']}")
        print(f"f[::2]: {len(subset)} messages")

        # messages() — all raw buffers
        raw_msgs = f.messages()
        print(f"\nmessages(): {len(raw_msgs)} raw buffers")

    # ── 3. Low-level scan ─────────────────────────────────────────────────────
    buf = path.read_bytes()
    offsets = tensogram.scan(buf)
    print(f"\nscan() on raw bytes: {len(offsets)} messages")

    # Decode message 3 from the raw buffer
    start, length = offsets[3]
    meta3 = tensogram.decode_metadata(buf[start : start + length])
    print(f"  message[3]: param={meta3.base[0]['mars']['param']}")

    # Buffer iteration (PR #11 feature)
    print("\niter_messages() on raw bytes:")
    for msg in tensogram.iter_messages(buf):
        desc, arr = msg.objects[0]
        print(f"  param={msg.metadata.base[0]['mars']['param']}  shape={arr.shape}")

print("\nFile API example complete.")
