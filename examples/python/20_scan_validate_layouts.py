# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""
Example 20 — Scan, validate, and layout a multi-message file
============================================================

Demonstrates the Wave-D symmetry additions to the Python binding — the
reader-side primitives that mirror the Rust core directly:

  1. ``tensogram.scan_file(path)`` — discover message boundaries with
     streaming I/O (never loads the whole file), plus the options-taking
     ``scan_with_options`` / ``scan_file_with_options`` and the
     ``ScanOptions`` knob object.

  2. ``tensogram.validate_buffer(buf)`` — validate every message in an
     in-memory multi-message buffer (the sibling of ``validate_file``,
     distinct from the single-message ``validate``).

  3. ``TensogramFile.message_layouts()`` — the per-message
     ``(offset, length)`` layout, in scan order, as ``MessageLayout``
     objects.

  4. ``tensogram.objects`` / ``objects_metadata`` — lazy per-object
     iterators, and ``TensogramFile.open_mmap`` for zero-copy reads.

Run:  python 20_scan_validate_layouts.py
"""

import tempfile
from pathlib import Path

import numpy as np
import tensogram

PARAMS = ["2t", "10u", "10v", "msl"]
SHAPE = (3, 4)


def build_file(path: str) -> None:
    """Write a 4-message .tgm — one forecast field per message."""
    with tensogram.TensogramFile.create(path) as f:
        for i, param in enumerate(PARAMS):
            meta = {"base": [{"param": param, "step": i}]}
            desc = {"type": "ntensor", "shape": list(SHAPE), "dtype": "float32"}
            data = np.random.default_rng(seed=i).random(SHAPE, dtype=np.float32)
            f.append(meta, [(desc, data)])


def show_scan(path: str) -> None:
    """Discover message boundaries with streaming I/O."""
    spans = tensogram.scan_file(path)
    print(f"1. scan_file → {len(spans)} messages:")
    for i, (offset, length) in enumerate(spans):
        print(f"   message[{i}]: offset={offset:>5}  length={length}")

    # A forward-only walk yields identical spans — bidirectional only
    # changes *how* the boundaries are discovered, not what they are.
    forward = tensogram.scan_file_with_options(path, tensogram.ScanOptions(bidirectional=False))
    assert forward == spans
    print("   forward-only scan matches the bidirectional walk ✓")


def show_validate(path: str) -> None:
    """Validate every message in the buffer at once."""
    buf = Path(path).read_bytes()

    # In-memory sibling of validate_file: one report over the whole buffer.
    report = tensogram.validate_buffer(buf, level="checksum")
    print(f"\n2. validate_buffer → {len(report['messages'])} message reports:")
    print(f"   file-level issues: {report['file_issues']}")
    all_ok = all(m["hash_verified"] for m in report["messages"])
    print(f"   every message hash-verified: {all_ok}")

    # Trailing garbage is surfaced as a file-level issue.
    corrupt = tensogram.validate_buffer(buf + b"TRAILING_GARBAGE")
    print(f"   with trailing garbage → file_issues: {len(corrupt['file_issues'])}")


def show_layouts(path: str) -> None:
    """Read the per-message byte layout, then slice one message out."""
    buf = Path(path).read_bytes()
    with tensogram.TensogramFile.open(path) as f:
        layouts = f.message_layouts()
        print(f"\n3. message_layouts → {len(layouts)} layouts:")
        for i, layout in enumerate(layouts):
            # MessageLayout carries .offset / .length and also unpacks
            # like a 2-tuple.
            offset, length = layout
            print(f"   layout[{i}] = {layout!r}  (offset={offset}, length={length})")

        # Use a layout to carve a single message out of the raw bytes and
        # decode it independently.
        first = layouts[0]
        msg_bytes = buf[first.offset : first.offset + first.length]
        decoded = tensogram.decode(msg_bytes)
        print(f"   sliced message[0].param = {decoded.metadata.base[0]['param']}")


def show_lazy_objects(path: str) -> None:
    """Lazily walk objects and their descriptors; use mmap if available."""
    buf = Path(path).read_bytes()
    first_msg_len = tensogram.scan(buf)[0][1]
    msg = buf[:first_msg_len]

    print("\n4. lazy per-object iterators:")
    for desc in tensogram.objects_metadata(msg):
        print(f"   objects_metadata → shape={desc.shape} dtype={desc.dtype}")
    for _desc, arr in tensogram.objects(msg):
        print(f"   objects → decoded {arr.shape} mean={arr.mean():.4f}")

    if tensogram.__has_mmap__:
        mm = tensogram.TensogramFile.open_mmap(path)
        print(f"   open_mmap → {mm.message_count()} messages (zero-copy scan)")
    else:
        print("   open_mmap → skipped (built without the mmap feature)")


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "forecast.tgm")
        build_file(path)
        show_scan(path)
        show_validate(path)
        show_layouts(path)
        show_lazy_objects(path)
    print("\nExample 20 complete.")


if __name__ == "__main__":
    main()
