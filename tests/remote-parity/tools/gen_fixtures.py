# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Regenerate the checked-in ``.tgm`` fixtures for the remote-parity harness.

**Fixtures are binary committed artefacts.** Re-running this script
produces **different bytes** each time because ``tensogram.encode``
stamps a fresh UTC timestamp and RFC-4122 v4 UUID into each message's
reserved provenance, and the encoder version string is baked in too —
so any tensogram version bump can shift offsets slightly.

The parity harness relies on the committed bytes being stable across
a CI run (both drivers observe the same file), not across
regenerations. The pytest suite computes expected message offsets
live from the fixture via ``tensogram.scan``, so it survives the
intentional regenerations. Re-run this tool only when the wire format
changes, the encoder version bumps, or the fixture set expands;
review the diff and commit.

Current scope: header-indexed, non-streaming fixtures only. See
README.md for the rationale and deferral list.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import sys

import numpy as np
import tensogram

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_FIXTURES_DIR = _THIS_DIR.parent / "fixtures"


def _descriptor(shape: tuple[int, ...]) -> dict:
    return {
        "type": "ntensor",
        "ndim": len(shape),
        "shape": list(shape),
        "dtype": "float32",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


def _encode_one(msg_index: int, shape: tuple[int, ...]) -> bytes:
    """Encode one message with a deterministic payload (ramp seeded by msg_index).

    The encoded *bytes* are NOT reproducible across runs because
    `tensogram.encode` stamps fresh timestamp + UUID into provenance;
    the deterministic part is the payload shape and ramp values.
    """
    count = math.prod(shape)
    payload = np.arange(count, dtype=np.float32) + float(msg_index)
    descriptor = _descriptor(shape)
    metadata = {"base": [{"fixture": {"msg_index": msg_index, "shape": list(shape)}}]}
    return tensogram.encode(metadata, [(descriptor, payload)])


def _build_multi_message(n: int, shape: tuple[int, ...]) -> bytes:
    return b"".join(_encode_one(i, shape) for i in range(n))


_FIXTURE_SPECS: dict[str, tuple[int, tuple[int, ...]]] = {
    "single-msg": (1, (4,)),
    "two-msg": (2, (4,)),
    "ten-msg": (10, (4,)),
    "hundred-msg": (100, (4,)),
}


def generate_all(out_dir: pathlib.Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name, (n, shape) in _FIXTURE_SPECS.items():
        data = _build_multi_message(n, shape)
        target = out_dir / f"{name}.tgm"
        target.write_bytes(data)
        sizes[name] = len(data)
    return sizes


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_FIXTURES_DIR,
        help=f"Output directory (default: {_FIXTURES_DIR})",
    )
    args = parser.parse_args(argv)

    sizes = generate_all(args.out_dir)
    print(f"Generated {len(sizes)} fixtures in {args.out_dir}:")
    for name, size in sizes.items():
        target = args.out_dir / f"{name}.tgm"
        print(f"  {name:16s} {size:>8d} bytes  sha256={digest(target)[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
