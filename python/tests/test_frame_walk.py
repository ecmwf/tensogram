# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the structural walkers ``tensogram.frames()`` and
``tensogram.message_header()``.

Both mirror the Rust core (``tensogram::frames`` / ``tensogram::message_header``)
and operate on **one** message: the preamble/postamble are not frames, and a
multi-message buffer must be sliced with :func:`tensogram.scan` first.
"""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

from __future__ import annotations

import numpy as np
import pytest
import tensogram

# Wire-format constants (plans/WIRE_FORMAT.md §2, §3) — spelled out here so the
# assertions below read as statements about the format, not about the binding.
PREAMBLE_SIZE = 24
FRAME_HEADER_SIZE = 16
COMMON_FOOTER_SIZE = 12  # [hash u64][ENDF]
DATA_OBJECT_FOOTER_SIZE = 20  # [cbor_offset u64][hash u64][ENDF]

# Frame-type name → numeric code (plans/WIRE_FORMAT.md §4).  Type 4 is
# reserved (obsolete v2 tensor frame) and never appears.
FRAME_TYPE_CODES = {
    "HeaderMetadata": 1,
    "HeaderIndex": 2,
    "HeaderHash": 3,
    "FooterHash": 5,
    "FooterIndex": 6,
    "FooterMetadata": 7,
    "PrecederMetadata": 8,
    "NTensorFrame": 9,
}

_DESCRIPTOR = {
    "type": "ntensor",
    "shape": [4],
    "dtype": "float32",
    "byte_order": "little",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}


def _payload(fill: float = 1.0) -> np.ndarray:
    return np.full(4, fill, dtype=np.float32)


def _buffered(n_objects: int = 2, **kwargs) -> bytes:
    """Buffered (random-access) encode: metadata/index/hash live in the header."""
    pairs = [(_DESCRIPTOR, _payload(float(i))) for i in range(n_objects)]
    return tensogram.encode({"base": [{} for _ in range(n_objects)]}, pairs, **kwargs)


def _streamed(n_objects: int = 1, *, preceders: bool = False, **kwargs) -> bytes:
    """Streaming encode: index/hash land in the footer."""
    enc = tensogram.StreamingEncoder({"base": [{}]}, **kwargs)
    for i in range(n_objects):
        if preceders:
            enc.write_preceder({"index": i})
        enc.write_object(_DESCRIPTOR, _payload(float(i)))
    return enc.finish()


def _types(msg: bytes) -> list[str]:
    return [f.frame_type for f in tensogram.frames(msg)]


def _walk_until_error(walker) -> tuple[list[str], Exception | None]:
    """Drain a frame walker, returning what it yielded before it raised."""
    seen: list[str] = []
    try:
        for frame in walker:
            seen.append(frame.frame_type)
    except ValueError as exc:
        return seen, exc
    return seen, None


# ---------------------------------------------------------------------------
# frames() — the frame walker
# ---------------------------------------------------------------------------


class TestFrames:
    def test_buffered_frame_type_sequence(self):
        # Buffered mode with default options: metadata, index and the
        # aggregate hash frame in the header, then one frame per object.
        assert _types(_buffered(2)) == [
            "HeaderMetadata",
            "HeaderIndex",
            "HeaderHash",
            "NTensorFrame",
            "NTensorFrame",
        ]

    def test_one_data_object_frame_per_encoded_object(self):
        for n in (1, 3, 5):
            assert _types(_buffered(n)).count("NTensorFrame") == n

    def test_frame_type_code_pairs_with_the_name(self):
        for f in tensogram.frames(_buffered(2)):
            assert f.frame_type_code == FRAME_TYPE_CODES[f.frame_type]

    def test_is_data_object_only_for_ntensor_frames(self):
        for f in tensogram.frames(_buffered(2)):
            assert f.is_data_object == (f.frame_type == "NTensorFrame")

    def test_preamble_is_not_a_frame(self):
        first = next(tensogram.frames(_buffered(1)))
        assert first.offset == PREAMBLE_SIZE

    def test_postamble_is_not_a_frame(self):
        msg = _buffered(1)
        last = list(tensogram.frames(msg))[-1]
        # The walk stops before the 24-byte postamble.
        assert last.offset + last.length <= len(msg) - 24

    def test_offsets_and_lengths_are_in_bounds_and_do_not_overlap(self):
        msg = _buffered(3)
        prev_end = PREAMBLE_SIZE
        for f in tensogram.frames(msg):
            assert f.offset >= prev_end
            assert f.length >= FRAME_HEADER_SIZE
            assert f.offset + f.length <= len(msg)
            prev_end = f.offset + f.length

    def test_frame_span_starts_with_the_frame_magic_and_ends_with_endf(self):
        msg = _buffered(2)
        for f in tensogram.frames(msg):
            span = msg[f.offset : f.offset + f.length]
            assert span[:2] == b"FR"
            assert span[-4:] == b"ENDF"

    def test_payload_is_the_content_slice_of_the_frame(self):
        msg = _buffered(2)
        for f in tensogram.frames(msg):
            span = msg[f.offset : f.offset + f.length]
            assert isinstance(f.payload, bytes)
            assert f.payload == span[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + len(f.payload)]

    def test_payload_excludes_the_header_and_the_type_specific_footer(self):
        for f in tensogram.frames(_buffered(2)):
            footer = DATA_OBJECT_FOOTER_SIZE if f.is_data_object else COMMON_FOOTER_SIZE
            assert len(f.payload) == f.length - FRAME_HEADER_SIZE - footer

    def test_uncompressed_object_payload_starts_with_the_raw_values(self):
        # encoding/filter/compression are all "none", so the data-object
        # frame content begins with the little-endian input bytes (the CBOR
        # descriptor follows).
        msg = _buffered(1)
        obj = next(f for f in tensogram.frames(msg) if f.is_data_object)
        assert obj.payload[:16] == _payload(0.0).tobytes()

    def test_version_and_flags_are_ints(self):
        for f in tensogram.frames(_buffered(1)):
            assert isinstance(f.version, int)
            assert isinstance(f.flags, int)
            assert f.version == 1

    def test_has_hash_set_for_every_frame_by_default(self):
        assert all(f.has_hash for f in tensogram.frames(_buffered(2)))

    def test_has_hash_clear_when_hashing_is_disabled(self):
        assert not any(f.has_hash for f in tensogram.frames(_buffered(2, hash=None)))

    def test_has_hash_mirrors_the_per_frame_flag_bit(self):
        for f in tensogram.frames(_buffered(1)):
            assert f.has_hash == bool(f.flags & (1 << 1))

    def test_is_an_iterator(self):
        it = tensogram.frames(_buffered(1))
        assert iter(it) is it

    def test_iteration_is_lazy_and_resumable(self):
        msg = _buffered(3)
        total = len(_types(msg))
        it = tensogram.frames(msg)
        first = next(it)
        assert first.frame_type == "HeaderMetadata"
        # The remainder is still there — pulling one frame does not consume
        # the rest of the walk.
        assert len(list(it)) == total - 1

    def test_stops_cleanly(self):
        it = tensogram.frames(_buffered(1))
        for _ in range(len(_types(_buffered(1)))):
            next(it)
        with pytest.raises(StopIteration):
            next(it)

    def test_len_reports_the_frames_left_to_yield(self):
        msg = _buffered(2)
        it = tensogram.frames(msg)
        assert len(it) == len(_types(msg))

    def test_len_decrements(self):
        it = tensogram.frames(_buffered(2))
        before = len(it)
        next(it)
        assert len(it) == before - 1

    def test_len_is_zero_when_exhausted(self):
        it = tensogram.frames(_buffered(1))
        list(it)
        assert len(it) == 0

    def test_frame_repr(self):
        f = next(tensogram.frames(_buffered(1)))
        text = repr(f)
        assert "Frame(" in text
        assert "HeaderMetadata" in text
        assert "offset=" in text

    def test_iterator_repr(self):
        assert "FrameIter(" in repr(tensogram.frames(_buffered(1)))

    def test_streaming_frame_type_sequence(self):
        # Streaming mode writes the metadata header first, then objects, and
        # defers metadata/hash/index to the footer.
        assert _types(_streamed(2)) == [
            "HeaderMetadata",
            "NTensorFrame",
            "NTensorFrame",
            "FooterMetadata",
            "FooterHash",
            "FooterIndex",
        ]

    def test_streaming_preceder_metadata_frames_are_walked(self):
        types = _types(_streamed(2, preceders=True))
        assert types.count("PrecederMetadata") == 2
        # Each preceder immediately precedes its data-object frame.
        for i, name in enumerate(types):
            if name == "PrecederMetadata":
                assert types[i + 1] == "NTensorFrame"

    def test_streaming_message_without_backfilled_length_still_walks(self):
        # finish() leaves total_length = 0; the walk must still stop at the
        # postamble rather than mistaking it for a frame.
        msg = _streamed(1)
        assert tensogram.message_header(msg).total_length == 0
        assert _types(msg)[-1] == "FooterIndex"

    def test_garbage_buffer_raises(self):
        with pytest.raises(ValueError, match=r"(?i)framing|magic|too short"):
            tensogram.frames(b"not a tensogram message at all")

    def test_empty_buffer_raises(self):
        with pytest.raises(ValueError, match=r"(?i)framing|magic|too short"):
            tensogram.frames(b"")

    def test_truncated_frame_chain_raises_during_iteration_then_stops(self):
        msg = _buffered(2)
        last = list(tensogram.frames(msg))[-1]
        truncated = msg[: last.offset + 8]  # cut inside the final frame
        it = tensogram.frames(truncated)  # the preamble still parses
        with pytest.raises(ValueError, match=r"(?i)framing|truncated|does not fit"):
            list(it)
        # Iteration stops after the error is surfaced.
        with pytest.raises(StopIteration):
            next(it)

    def test_frames_before_the_truncation_are_still_yielded(self):
        msg = _buffered(2)
        last = list(tensogram.frames(msg))[-1]
        seen, error = _walk_until_error(tensogram.frames(msg[: last.offset + 8]))
        assert seen == ["HeaderMetadata", "HeaderIndex", "HeaderHash", "NTensorFrame"]
        assert error is not None
        assert "framing" in str(error).lower()

    def test_multi_message_buffer_is_sliced_with_scan(self):
        buf = _buffered(1) + _buffered(3)
        spans = tensogram.scan(buf)
        assert len(spans) == 2
        second_off, second_len = spans[1]
        second = _types(buf[second_off : second_off + second_len])
        assert second.count("NTensorFrame") == 3
        # Offsets are relative to the start of the message that was passed in.
        assert next(tensogram.frames(buf[second_off : second_off + second_len])).offset == (
            PREAMBLE_SIZE
        )

    def test_unsliced_multi_message_buffer_walks_only_the_first_message(self):
        buf = _buffered(1) + _buffered(3)
        assert _types(buf).count("NTensorFrame") == 1

    def test_accepts_bytearray(self):
        msg = _buffered(1)
        types = [f.frame_type for f in tensogram.frames(bytearray(msg))]
        assert types == _types(msg)

    def test_message_without_objects_walks_to_its_metadata_frame(self):
        # No objects → the encoder builds neither an index nor an aggregate
        # hash frame, so the walk is a single metadata frame.
        msg = tensogram.encode({"base": []}, [])
        assert _types(msg) == ["HeaderMetadata"]
        header = tensogram.message_header(msg)
        assert header.has_header_metadata
        assert not header.has_header_index
        assert not header.has_header_hashes

    def test_a_frame_outlives_the_iterator_that_produced_it(self):
        # Each Frame keeps its own reference to the source buffer, so its
        # payload stays readable after the walker is gone.
        walker = tensogram.frames(_buffered(1))
        frame = next(walker)
        del walker
        assert len(frame.payload) == frame.length - FRAME_HEADER_SIZE - COMMON_FOOTER_SIZE


# ---------------------------------------------------------------------------
# message_header() — the typed preamble
# ---------------------------------------------------------------------------


class TestMessageHeader:
    def test_version_and_total_length(self):
        msg = _buffered(2)
        header = tensogram.message_header(msg)
        assert header.version == tensogram.WIRE_VERSION
        assert header.total_length == len(msg)

    def test_flags_raw_bits(self):
        header = tensogram.message_header(_buffered(1))
        assert isinstance(header.flags, int)
        assert header.flags & (1 << 0)  # HEADER_METADATA

    def test_buffered_flags_match_the_frames_exactly(self):
        msg = _buffered(2)
        header = tensogram.message_header(msg)
        present = set(_types(msg))
        assert header.has_header_metadata == ("HeaderMetadata" in present)
        assert header.has_footer_metadata == ("FooterMetadata" in present)
        assert header.has_header_index == ("HeaderIndex" in present)
        assert header.has_footer_index == ("FooterIndex" in present)
        assert header.has_header_hashes == ("HeaderHash" in present)
        assert header.has_footer_hashes == ("FooterHash" in present)
        assert header.has_preceder_metadata == ("PrecederMetadata" in present)

    def test_buffered_predicates_are_bools(self):
        header = tensogram.message_header(_buffered(1))
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
            assert isinstance(getattr(header, name), bool), name

    def test_hashes_present_tracks_the_hashing_option(self):
        assert tensogram.message_header(_buffered(1)).has_hashes_present is True
        assert tensogram.message_header(_buffered(1, hash=None)).has_hashes_present is False

    def test_streaming_flags_never_understate_the_frames_present(self):
        # The streaming preamble is written before any object, so it is
        # advisory: PRECEDER_METADATA is set optimistically.  The invariant
        # that always holds is "frame present ⇒ flag set".
        msg = _streamed(1)
        header = tensogram.message_header(msg)
        present = set(_types(msg))
        implications = {
            "HeaderMetadata": header.has_header_metadata,
            "FooterMetadata": header.has_footer_metadata,
            "HeaderIndex": header.has_header_index,
            "FooterIndex": header.has_footer_index,
            "HeaderHash": header.has_header_hashes,
            "FooterHash": header.has_footer_hashes,
            "PrecederMetadata": header.has_preceder_metadata,
        }
        for name, flag in implications.items():
            assert (name not in present) or flag, name
        assert "FooterIndex" in present

    def test_streaming_total_length_is_zero_until_backfilled(self):
        enc = tensogram.StreamingEncoder({"base": [{}]})
        enc.write_object(_DESCRIPTOR, _payload())
        msg = enc.finish()
        assert tensogram.message_header(msg).total_length == 0

    def test_backfilled_streaming_total_length_matches(self):
        enc = tensogram.StreamingEncoder({"base": [{}]})
        enc.write_object(_DESCRIPTOR, _payload())
        msg = enc.finish_backfilled()
        assert tensogram.message_header(msg).total_length == len(msg)

    def test_repr(self):
        text = repr(tensogram.message_header(_buffered(1)))
        assert "MessageHeader(" in text
        assert "version=" in text
        assert "total_length=" in text

    def test_multi_message_buffer_is_sliced_with_scan(self):
        buf = _buffered(1) + _buffered(3)
        off, length = tensogram.scan(buf)[1]
        assert tensogram.message_header(buf[off : off + length]).total_length == length
        # Reading the unsliced buffer describes the *first* message.
        assert tensogram.message_header(buf).total_length == tensogram.scan(buf)[0][1]

    def test_garbage_buffer_raises(self):
        with pytest.raises(ValueError, match=r"(?i)framing|magic|too short"):
            tensogram.message_header(b"not a tensogram message at all")

    def test_empty_buffer_raises(self):
        with pytest.raises(ValueError, match=r"(?i)framing|magic|too short"):
            tensogram.message_header(b"")

    def test_accepts_bytearray(self):
        msg = _buffered(1)
        assert tensogram.message_header(bytearray(msg)).total_length == len(msg)
