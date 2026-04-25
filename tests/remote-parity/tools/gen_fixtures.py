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
and ``StreamingEncoder`` stamp a fresh UTC timestamp and RFC-4122 v4
UUID into each message's reserved provenance, and the encoder version
string is baked in too — so any tensogram version bump can shift
offsets slightly.

The parity harness relies on the committed bytes being stable across
a CI run (both drivers observe the same file), not across
regenerations. The pytest suite computes expected message offsets
live from the fixture via ``tensogram.scan``, so it survives the
intentional regenerations. Re-run this tool only when the wire format
changes, the encoder version bumps, or the fixture set expands;
review the diff and commit.

Current scope: header-indexed (via ``tensogram.encode``) and
backfilled footer-indexed (via ``StreamingEncoder.finish_backfilled``)
fixtures.  Streaming-mode (``total_length = 0``) fixtures stay deferred
because the lazy walker bails to eager on streaming-tail messages,
making cross-language event parity non-comparable.
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


_HEADER_INDEXED = "header"
_FOOTER_INDEXED = "footer"


def _encode_header_indexed(msg_index: int, shape: tuple[int, ...]) -> bytes:
    """Encode one header-indexed message with a deterministic payload.

    The encoded *bytes* are NOT reproducible across runs because
    `tensogram.encode` stamps fresh timestamp + UUID into provenance;
    the deterministic part is the payload shape and ramp values.
    """
    count = math.prod(shape)
    payload = np.arange(count, dtype=np.float32) + float(msg_index)
    descriptor = _descriptor(shape)
    metadata = {"base": [{"fixture": {"msg_index": msg_index, "shape": list(shape)}}]}
    return tensogram.encode(metadata, [(descriptor, payload)])


def _encode_footer_indexed(msg_index: int, shape: tuple[int, ...]) -> bytes:
    """Encode one footer-indexed message with mirrored ``total_length``.

    Uses ``StreamingEncoder.finish_backfilled`` so the produced bytes
    satisfy the backward-locatability invariant from wire-format §7
    (both preamble and postamble carry the real length).  Required for
    fixtures exercising the bidirectional walker's eager-footer path.
    """
    count = math.prod(shape)
    payload = np.arange(count, dtype=np.float32) + float(msg_index)
    descriptor = _descriptor(shape)
    metadata = {"base": [{"fixture": {"msg_index": msg_index, "shape": list(shape)}}]}
    enc = tensogram.StreamingEncoder(metadata)
    enc.write_object(descriptor, payload)
    return enc.finish_backfilled()


def _build_multi_message(n: int, shape: tuple[int, ...], kind: str) -> bytes:
    encoder = _encode_header_indexed if kind == _HEADER_INDEXED else _encode_footer_indexed
    return b"".join(encoder(i, shape) for i in range(n))


_FIXTURE_SPECS: dict[str, tuple[int, tuple[int, ...], str]] = {
    "single-msg": (1, (4,), _HEADER_INDEXED),
    "two-msg": (2, (4,), _HEADER_INDEXED),
    "ten-msg": (10, (4,), _HEADER_INDEXED),
    "hundred-msg": (100, (4,), _HEADER_INDEXED),
    "single-msg-footer": (1, (4,), _FOOTER_INDEXED),
    "ten-msg-footer": (10, (4,), _FOOTER_INDEXED),
}


def _assert_fixture_well_formed(path: pathlib.Path, kind: str, expected_count: int) -> None:
    """Sanity-check a generated fixture before committing it.

    Asserts that every message scans cleanly (postamble + preamble
    agree on ``total_length``) and that the index-location flag
    matches the requested kind.  Without this guard a silent encoder
    regression could ship fixtures the bidirectional walker can never
    discover backward.
    """
    data = path.read_bytes()
    layouts = list(tensogram.scan(data))
    if len(layouts) != expected_count:
        raise RuntimeError(
            f"{path}: expected {expected_count} messages, scan found {len(layouts)}"
        )
    for offset, length in layouts:
        if length == 0:
            raise RuntimeError(
                f"{path}: streaming-mode (total_length=0) message at offset {offset}; "
                "fixtures must use finish_backfilled for the parity harness"
            )
        # Preamble flags are u16 BE at offset 10 (after magic 8 + version 2).
        # MessageFlags bit positions per wire.rs: HEADER_METADATA=1<<0,
        # FOOTER_METADATA=1<<1, HEADER_INDEX=1<<2, FOOTER_INDEX=1<<3.
        flags_u16 = int.from_bytes(data[offset + 10 : offset + 12], "big")
        has_header_index = bool(flags_u16 & (1 << 2))
        has_footer_index = bool(flags_u16 & (1 << 3))
        if kind == _HEADER_INDEXED and not has_header_index:
            raise RuntimeError(
                f"{path}: message at offset {offset} missing HEADER_INDEX flag "
                f"(flags=0x{flags_u16:04x})"
            )
        if kind == _FOOTER_INDEXED and not has_footer_index:
            raise RuntimeError(
                f"{path}: message at offset {offset} missing FOOTER_INDEX flag "
                f"(flags=0x{flags_u16:04x})"
            )


def generate_all(out_dir: pathlib.Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name, (n, shape, kind) in _FIXTURE_SPECS.items():
        data = _build_multi_message(n, shape, kind)
        target = out_dir / f"{name}.tgm"
        target.write_bytes(data)
        _assert_fixture_well_formed(target, kind, n)
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
