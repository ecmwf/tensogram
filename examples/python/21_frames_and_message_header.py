# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""
Example 21 — Walk a message's frames and read its header
========================================================

Structural introspection: what is *inside* a message, without decoding
any payload or CBOR.  Two entry points, both direct bindings of the Rust
core:

  1. ``tensogram.message_header(msg)`` — the 24-byte preamble as typed
     values: ``version``, ``total_length``, and eight ``has_*``
     predicates that say whether the metadata / index / hash frames live
     in the *header* (random-access) or the *footer* (streaming).

  2. ``tensogram.frames(msg)`` — a lazy iterator over the message's
     frames.  Each ``Frame`` carries ``frame_type`` (a readable name such
     as ``"NTensorFrame"``), ``frame_type_code`` (its numeric wire code),
     ``version``, ``flags``, ``offset``, ``length``, ``payload`` and
     ``has_hash``.  The preamble and postamble are *not* frames.

Both take **one** message.  For a multi-message buffer or file, slice it
with ``tensogram.scan()`` first — shown in ``walk_a_multi_message_buffer``.

Run:  python 21_frames_and_message_header.py
"""

import numpy as np
import tensogram

DESCRIPTOR = {"type": "ntensor", "shape": [3, 4], "dtype": "float32"}


def make_message(n_objects: int = 2, **encode_kwargs) -> bytes:
    """Buffered encode → a random-access message (header-side index)."""
    meta = {"base": [{"param": "2t", "step": i} for i in range(n_objects)]}
    pairs = [
        (DESCRIPTOR, np.random.default_rng(seed=i).random((3, 4), dtype=np.float32))
        for i in range(n_objects)
    ]
    return tensogram.encode(meta, pairs, **encode_kwargs)


def show_message_header(msg: bytes) -> None:
    """Read the envelope — no frame is touched."""
    header = tensogram.message_header(msg)
    print("1. message_header:")
    print(f"   {header!r}")
    print(f"   version={header.version}  total_length={header.total_length}")
    print(f"   raw flags=0x{header.flags:04x}")
    # The eight predicates keep the core's `has_` prefix.
    layout = "random-access" if header.has_header_index else "streaming"
    print(f"   layout: {layout}")
    for name in (
        "has_header_metadata",
        "has_footer_metadata",
        "has_header_index",
        "has_footer_index",
        "has_header_hashes",
        "has_footer_hashes",
        "has_preceder_metadata",
        "has_hashes_present",
    ):
        print(f"     {name:<24} {getattr(header, name)}")


def walk_frames(msg: bytes) -> None:
    """Walk every frame: type, span, and content size."""
    print("\n2. frames:")
    print(f"   {'type':<18} {'code':>4} {'offset':>7} {'length':>7} {'payload':>8}  hash")
    for frame in tensogram.frames(msg):
        print(
            f"   {frame.frame_type:<18} {frame.frame_type_code:>4} "
            f"{frame.offset:>7} {frame.length:>7} {len(frame.payload):>8}  {frame.has_hash}"
        )
        # `offset` / `length` span the whole frame — header through ENDF.
        span = msg[frame.offset : frame.offset + frame.length]
        assert span[:2] == b"FR"
        assert span[-4:] == b"ENDF"
        # `payload` is the *content*: the 16-byte frame header and the
        # type-specific footer are excluded.
        assert frame.payload == span[16 : 16 + len(frame.payload)]

    # The preamble (24 bytes) and the postamble are not frames.
    first = next(tensogram.frames(msg))
    print(f"   first frame starts at byte {first.offset} (right after the preamble)")


def lazy_and_cheap(msg: bytes) -> None:
    """`frames` is an iterator: pull one frame, keep the rest for later."""
    print("\n3. lazy iteration:")
    walker = tensogram.frames(msg)
    print(f"   {walker!r}")
    head = next(walker)
    print(f"   next() → {head!r}")
    print(f"   {len(walker)} frames still to come: {[f.frame_type for f in walker]}")

    # Data-object frames are the ones carrying tensors.
    n_objects = sum(1 for f in tensogram.frames(msg) if f.is_data_object)
    print(f"   data-object frames: {n_objects}")


def compare_buffered_and_streaming() -> None:
    """The same content, two layouts — visible in the header and the walk."""
    print("\n4. buffered vs streaming:")

    buffered = make_message(1)
    encoder = tensogram.StreamingEncoder({"base": [{"param": "2t"}]})
    encoder.write_preceder({"step": 0})
    encoder.write_object(DESCRIPTOR, np.zeros((3, 4), dtype=np.float32))
    streamed = encoder.finish()

    for label, msg in (("buffered", buffered), ("streaming", streamed)):
        header = tensogram.message_header(msg)
        types = [f.frame_type for f in tensogram.frames(msg)]
        print(f"   {label:<10} {types}")
        print(
            f"   {'':<10} header_index={header.has_header_index} "
            f"footer_index={header.has_footer_index} "
            f"total_length={header.total_length}"
        )
    # A streaming message reports total_length = 0 until it is back-filled;
    # the walker still stops cleanly at the postamble.


def see_the_aggregate_hash_knob() -> None:
    """`aggregate_hash=` decides where the hash frame lands — see it."""
    print("\n5. aggregate_hash placement, verified with the walker:")
    for policy in ("auto", "none", "header", "footer", "both"):
        types = [f.frame_type for f in tensogram.frames(make_message(1, aggregate_hash=policy))]
        placed = [t for t in types if t.endswith("Hash")] or ["(none)"]
        print(f"   aggregate_hash={policy:<8} → {placed}")

    # `compression_backend=` picks the szip / zstd implementation when both
    # an FFI and a pure-Rust one are compiled in ("auto", "ffi", "pure").
    msg = make_message(1, compression_backend="pure")
    print(f"   compression_backend='pure' → {len(tensogram.decode(msg).objects)} object decoded")


def walk_a_multi_message_buffer() -> None:
    """Slice with `scan` first: both walkers take exactly one message."""
    print("\n6. multi-message buffer:")
    buf = make_message(1) + make_message(3)
    for i, (offset, length) in enumerate(tensogram.scan(buf)):
        msg = buf[offset : offset + length]
        types = [f.frame_type for f in tensogram.frames(msg)]
        print(f"   message[{i}] at {offset:>5} (+{length}): {types}")


def report_a_broken_chain() -> None:
    """A truncated chain yields the frames that parsed, then raises."""
    print("\n7. a truncated message:")
    msg = make_message(2)
    last = list(tensogram.frames(msg))[-1]
    truncated = msg[: last.offset + 8]  # cut inside the final frame

    walker = tensogram.frames(truncated)  # the preamble still parses
    seen = []
    try:
        for frame in walker:
            seen.append(frame.frame_type)
    except ValueError as exc:
        print(f"   walked {len(seen)} frames, then: {exc}")
    print(f"   frames recovered before the break: {seen}")


def main() -> None:
    msg = make_message(2)
    show_message_header(msg)
    walk_frames(msg)
    lazy_and_cheap(msg)
    compare_buffered_and_streaming()
    see_the_aggregate_hash_knob()
    walk_a_multi_message_buffer()
    report_a_broken_chain()
    print("\nExample 21 complete.")


if __name__ == "__main__":
    main()
