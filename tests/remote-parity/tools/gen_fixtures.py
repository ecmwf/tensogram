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

Current scope: header-indexed (via ``tensogram.encode``), backfilled
footer-indexed (via ``StreamingEncoder.finish_backfilled``), and a
streaming-tail fixture whose final message uses ``StreamingEncoder.finish``
(``total_length = 0``).  The streaming-tail fixture is excluded from
cross-language event parity (the lazy walker bails to eager on its
final message) but is exercised by the bench harness so the
bidirectional walker's forward-fallback path stays measured.
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
_STREAMING_TAIL = "streaming-tail"


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


def _encode_streaming_tail_msg(msg_index: int, shape: tuple[int, ...]) -> bytes:
    """Encode one streaming-mode message (``total_length = 0`` in both
    preamble and postamble).  Forces the bidirectional walker to fall
    back to forward scanning for this segment."""
    count = math.prod(shape)
    payload = np.arange(count, dtype=np.float32) + float(msg_index)
    descriptor = _descriptor(shape)
    metadata = {"base": [{"fixture": {"msg_index": msg_index, "shape": list(shape)}}]}
    enc = tensogram.StreamingEncoder(metadata)
    enc.write_object(descriptor, payload)
    return enc.finish()


def _build_multi_message(n: int, shape: tuple[int, ...], kind: str) -> bytes:
    if kind == _HEADER_INDEXED:
        return b"".join(_encode_header_indexed(i, shape) for i in range(n))
    if kind == _FOOTER_INDEXED:
        return b"".join(_encode_footer_indexed(i, shape) for i in range(n))
    if kind == _STREAMING_TAIL:
        head = b"".join(_encode_header_indexed(i, shape) for i in range(n - 1))
        tail = _encode_streaming_tail_msg(n - 1, shape)
        return head + tail
    raise ValueError(f"unknown fixture kind: {kind}")


_FIXTURE_SPECS: dict[str, tuple[int, tuple[int, ...], str]] = {
    "single-msg": (1, (4,), _HEADER_INDEXED),
    "two-msg": (2, (4,), _HEADER_INDEXED),
    "ten-msg": (10, (4,), _HEADER_INDEXED),
    "hundred-msg": (100, (4,), _HEADER_INDEXED),
    "thousand-msg": (1000, (4,), _HEADER_INDEXED),
    "single-msg-footer": (1, (4,), _FOOTER_INDEXED),
    "ten-msg-footer": (10, (4,), _FOOTER_INDEXED),
    "hundred-msg-footer": (100, (4,), _FOOTER_INDEXED),
    "thousand-msg-footer": (1000, (4,), _FOOTER_INDEXED),
    "streaming-tail": (10, (4,), _STREAMING_TAIL),
}


def _assert_fixture_well_formed(
    path: pathlib.Path, kind: str, expected_count: int
) -> None:
    """Sanity-check a generated fixture before committing it.

    Asserts that every message scans cleanly, that preamble and
    postamble agree on ``total_length``, and that the index-location
    flag matches the requested kind.  Without these guards a silent
    encoder regression could ship fixtures the bidirectional walker
    can never discover backward.

    For ``_STREAMING_TAIL`` fixtures the final message is allowed
    (and required) to carry ``total_length = 0`` in both preamble
    and postamble; the leading ``n - 1`` messages must remain
    header-indexed with mirrored non-zero lengths so only the tail
    forces forward fallback.
    """
    data = path.read_bytes()
    layouts = list(tensogram.scan(data))
    if len(layouts) != expected_count:
        raise RuntimeError(
            f"{path}: expected {expected_count} messages, scan found {len(layouts)}"
        )
    for i, (offset, scanned_length) in enumerate(layouts):
        is_streaming_tail_msg = kind == _STREAMING_TAIL and i == expected_count - 1
        preamble_total_length = int.from_bytes(data[offset + 16 : offset + 24], "big")
        # Postamble layout (24 B): first_footer_offset (8) + total_length (8) +
        # END_MAGIC (8); total_length lives at bytes [scanned_length - 16, -8).
        postamble_total_length = int.from_bytes(
            data[offset + scanned_length - 16 : offset + scanned_length - 8], "big"
        )
        if preamble_total_length != postamble_total_length:
            raise RuntimeError(
                f"{path}: preamble.total_length={preamble_total_length} disagrees with "
                f"postamble.total_length={postamble_total_length} at offset {offset}"
            )
        if preamble_total_length == 0 and not is_streaming_tail_msg:
            raise RuntimeError(
                f"{path}: unexpected total_length=0 at offset {offset}; "
                "non-streaming fixtures must use finish_backfilled"
            )
        if is_streaming_tail_msg:
            if preamble_total_length != 0:
                raise RuntimeError(
                    f"{path}: streaming-tail message must have total_length=0, "
                    f"got {preamble_total_length}"
                )
            continue
        flags_u16 = int.from_bytes(data[offset + 10 : offset + 12], "big")
        has_header_index = bool(flags_u16 & (1 << 2))
        has_footer_index = bool(flags_u16 & (1 << 3))
        expects_header = kind in (_HEADER_INDEXED, _STREAMING_TAIL)
        if expects_header and not has_header_index:
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
