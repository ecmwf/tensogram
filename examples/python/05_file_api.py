"""
Example 05 — File API (Python)

NOTE: Requires tensogram Python bindings (not yet implemented).
      This file documents the intended API.

Shows the full TensogramFile lifecycle: create, append, open, iterate,
random access by index, and low-level scan.
"""

import pathlib
import tempfile

import numpy as np
import tensogram


def make_field(param: str, step: int) -> tuple:
    """Return (Metadata, ndarray) for one forecast field."""
    metadata = tensogram.Metadata(
        version=1,
        objects=[
            tensogram.ObjectDescriptor(
                type="ntensor",
                shape=[721, 1440],
                dtype="float32",
                byte_order="big",
                mars={"param": param, "levtype": "sfc"},
            )
        ],
        mars={"class": "od", "date": "20260401", "step": step, "type": "fc"},
    )
    data = np.zeros((721, 1440), dtype=np.float32)
    return metadata, data


with tempfile.TemporaryDirectory() as tmpdir:
    path = pathlib.Path(tmpdir) / "forecast.tgm"

    # ── 1. Create and append ──────────────────────────────────────────────────
    with tensogram.TensogramFile.create(path) as f:
        params = ["2t", "10u", "10v", "msl"]
        steps  = [0, 6, 12]
        for step in steps:
            for param in params:
                meta, data = make_field(param, step)
                f.append(meta, data)   # one array per object

        count = f.message_count()
        print(f"Written {count} messages")   # 12

    # ── 2. Open and inspect ───────────────────────────────────────────────────
    with tensogram.TensogramFile.open(path) as f:
        # Lazy scan: happens on first access, not on open()
        count = f.message_count()
        print(f"\nOpened: {count} messages")
        assert count == 12

        # ── Random access by index ────────────────────────────────────────────
        print("\nRandom access:")
        for idx in [0, 5, 11]:
            meta, arrays = f.decode_message(idx)
            param   = meta.objects[0].extra["mars"]["param"]
            step    = meta["mars"]["step"]
            print(f"  [{idx:2}] param={param:<5}  step={step:2}  "
                  f"shape={arrays[0].shape}")

        # ── Read raw bytes ─────────────────────────────────────────────────────
        raw: bytes = f.read_message(0)
        print(f"\nread_message(0): {len(raw)} bytes  "
              f"magic={raw[:8]}  term={raw[-8:]}")

        # ── Iterate over all messages ─────────────────────────────────────────
        print("\nAll messages:")
        for i, (meta, arrays) in enumerate(f):
            param = meta.objects[0].extra["mars"]["param"]
            step  = meta["mars"]["step"]
            print(f"  [{i:2}] param={param:<5}  step={step:2}  "
                  f"dtype={arrays[0].dtype}")

        # ── messages() — all raw buffers ─────────────────────────────────────
        raw_msgs: list[bytes] = f.messages()
        print(f"\nmessages(): {len(raw_msgs)} raw buffers")

    # ── 3. Low-level scan ─────────────────────────────────────────────────────
    #
    # scan() is the primitive under TensogramFile. Use it when you have an
    # in-memory buffer (e.g. from a network socket or memory-mapped file).
    buf: bytes = path.read_bytes()
    offsets: list[tuple[int, int]] = tensogram.scan(buf)
    print(f"\nscan() on raw bytes: {len(offsets)} messages")

    # Decode message 7 from the raw buffer
    start, length = offsets[7]
    meta7 = tensogram.decode_metadata(buf[start : start + length])
    print(f"  message[7]: param={meta7.objects[0].extra['mars']['param']}")

print("\nFile API example complete.")
