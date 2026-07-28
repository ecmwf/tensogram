# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the encode-side policy knobs ``aggregate_hash=`` and
``compression_backend=``.

Both map onto the Rust core ``EncodeOptions`` fields of the same name and
are accepted by every encode entry point: :func:`tensogram.encode`,
:class:`tensogram.StreamingEncoder`, :meth:`tensogram.TensogramFile.append`
and :meth:`tensogram.AsyncStreamingEncoder.create`.
"""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

from __future__ import annotations

import numpy as np
import pytest
import tensogram

_DESCRIPTOR = {
    "type": "ntensor",
    "shape": [16],
    "dtype": "float32",
    "byte_order": "little",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}
_ZSTD_DESCRIPTOR = {**_DESCRIPTOR, "compression": "zstd"}
_PAYLOAD = np.arange(16, dtype=np.float32)
_META = {"base": [{}]}

AGGREGATE_HASH_VALUES = ["auto", "none", "header", "footer", "both"]
COMPRESSION_BACKENDS = ["auto", "ffi", "pure"]


def _frame_types(msg: bytes) -> list[str]:
    return [f.frame_type for f in tensogram.frames(msg)]


# ---------------------------------------------------------------------------
# aggregate_hash= — aggregate hash frame placement
# ---------------------------------------------------------------------------


class TestAggregateHash:
    @pytest.mark.parametrize("policy", AGGREGATE_HASH_VALUES)
    def test_value_round_trips(self, policy):
        msg = tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], aggregate_hash=policy)
        decoded = tensogram.decode(msg)
        np.testing.assert_array_equal(decoded.objects[0][1], _PAYLOAD)

    @pytest.mark.parametrize(
        ("policy", "header_hash", "footer_hash"),
        [
            ("auto", True, False),  # buffered default → header
            ("none", False, False),
            ("header", True, False),
            ("footer", False, True),
            ("both", True, True),
        ],
    )
    def test_placement_matches_the_policy(self, policy, header_hash, footer_hash):
        types = _frame_types(
            tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], aggregate_hash=policy)
        )
        assert ("HeaderHash" in types) is header_hash
        assert ("FooterHash" in types) is footer_hash

    def test_both_places_a_hash_frame_in_the_header_and_the_footer(self):
        # The closed loop: the encode knob is verified with the frame walker.
        types = _frame_types(
            tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], aggregate_hash="both")
        )
        assert "HeaderHash" in types
        assert "FooterHash" in types
        # ... and they sit on the correct side of the data objects.
        assert types.index("HeaderHash") < types.index("NTensorFrame")
        assert types.index("FooterHash") > types.index("NTensorFrame")

    @pytest.mark.parametrize("policy", AGGREGATE_HASH_VALUES)
    def test_message_header_flags_track_the_placement(self, policy):
        msg = tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], aggregate_hash=policy)
        types = _frame_types(msg)
        header = tensogram.message_header(msg)
        assert header.has_header_hashes == ("HeaderHash" in types)
        assert header.has_footer_hashes == ("FooterHash" in types)

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match=r"unknown aggregate_hash policy"):
            tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], aggregate_hash="sometimes")

    def test_error_lists_the_accepted_values(self):
        with pytest.raises(ValueError, match=r"'auto'.*'none'.*'header'.*'footer'.*'both'"):
            tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], aggregate_hash="")

    @pytest.mark.parametrize("policy", ["auto", "none", "footer"])
    def test_streaming_accepts_footer_side_policies(self, policy):
        enc = tensogram.StreamingEncoder(_META, aggregate_hash=policy)
        enc.write_object(_DESCRIPTOR, _PAYLOAD)
        msg = enc.finish()
        types = _frame_types(msg)
        assert ("FooterHash" in types) is (policy != "none")
        assert "HeaderHash" not in types

    @pytest.mark.parametrize("policy", ["header", "both"])
    def test_streaming_rejects_header_side_policies(self, policy):
        with pytest.raises(ValueError, match=r"(?i)streaming|header"):
            tensogram.StreamingEncoder(_META, aggregate_hash=policy)

    def test_file_append_accepts_the_policy(self, tmp_path):
        path = str(tmp_path / "aggregate.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(_META, [(_DESCRIPTOR, _PAYLOAD)], aggregate_hash="both")
        with tensogram.TensogramFile.open(path) as f:
            types = _frame_types(f.read_message(0))
        assert "HeaderHash" in types
        assert "FooterHash" in types

    def test_file_append_rejects_an_unknown_policy(self, tmp_path):
        path = str(tmp_path / "bad.tgm")
        with (
            tensogram.TensogramFile.create(path) as f,
            pytest.raises(ValueError, match=r"unknown aggregate_hash policy"),
        ):
            f.append(_META, [(_DESCRIPTOR, _PAYLOAD)], aggregate_hash="nope")

    @pytest.mark.asyncio
    async def test_async_streaming_accepts_the_policy(self):
        enc = await tensogram.AsyncStreamingEncoder.create(_META, aggregate_hash="footer")
        await enc.write_object(_DESCRIPTOR, _PAYLOAD)
        msg = await enc.finish()
        assert "FooterHash" in _frame_types(msg)


# ---------------------------------------------------------------------------
# compression_backend= — FFI vs pure-Rust codec selection
# ---------------------------------------------------------------------------


class TestCompressionBackend:
    @pytest.mark.parametrize("backend", COMPRESSION_BACKENDS)
    def test_value_round_trips(self, backend):
        msg = tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], compression_backend=backend)
        decoded = tensogram.decode(msg)
        np.testing.assert_array_equal(decoded.objects[0][1], _PAYLOAD)

    @pytest.mark.parametrize("backend", COMPRESSION_BACKENDS)
    def test_value_round_trips_through_a_compressed_codec(self, backend):
        # zstd is one of the two codecs (with szip) that has both an FFI and
        # a pure-Rust implementation, so this is where the knob bites.
        msg = tensogram.encode(_META, [(_ZSTD_DESCRIPTOR, _PAYLOAD)], compression_backend=backend)
        decoded = tensogram.decode(msg)
        descriptor, values = decoded.objects[0]
        assert descriptor.compression == "zstd"
        np.testing.assert_array_equal(values, _PAYLOAD)

    def test_backends_agree_on_the_decoded_values(self):
        decoded = [
            tensogram.decode(
                tensogram.encode(_META, [(_ZSTD_DESCRIPTOR, _PAYLOAD)], compression_backend=b)
            ).objects[0][1]
            for b in COMPRESSION_BACKENDS
        ]
        for values in decoded[1:]:
            np.testing.assert_array_equal(values, decoded[0])

    def test_default_matches_auto(self):
        default = tensogram.decode(tensogram.encode(_META, [(_ZSTD_DESCRIPTOR, _PAYLOAD)]))
        explicit = tensogram.decode(
            tensogram.encode(_META, [(_ZSTD_DESCRIPTOR, _PAYLOAD)], compression_backend="auto")
        )
        np.testing.assert_array_equal(default.objects[0][1], explicit.objects[0][1])

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match=r"unknown compression_backend"):
            tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], compression_backend="cuda")

    def test_error_lists_the_accepted_values(self):
        with pytest.raises(ValueError, match=r"'auto'.*'ffi'.*'pure'"):
            tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], compression_backend="")

    def test_value_is_case_sensitive(self):
        with pytest.raises(ValueError, match=r"unknown compression_backend"):
            tensogram.encode(_META, [(_DESCRIPTOR, _PAYLOAD)], compression_backend="PURE")

    @pytest.mark.parametrize("backend", COMPRESSION_BACKENDS)
    def test_streaming_encoder_accepts_the_backend(self, backend):
        enc = tensogram.StreamingEncoder(_META, compression_backend=backend)
        enc.write_object(_ZSTD_DESCRIPTOR, _PAYLOAD)
        msg = enc.finish()
        np.testing.assert_array_equal(tensogram.decode(msg).objects[0][1], _PAYLOAD)

    def test_streaming_encoder_rejects_an_unknown_backend(self):
        with pytest.raises(ValueError, match=r"unknown compression_backend"):
            tensogram.StreamingEncoder(_META, compression_backend="turbo")

    @pytest.mark.parametrize("backend", COMPRESSION_BACKENDS)
    def test_file_append_accepts_the_backend(self, backend, tmp_path):
        path = str(tmp_path / f"backend_{backend}.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(_META, [(_ZSTD_DESCRIPTOR, _PAYLOAD)], compression_backend=backend)
        with tensogram.TensogramFile.open(path) as f:
            _meta, objects = f.decode_message(0)
        np.testing.assert_array_equal(objects[0][1], _PAYLOAD)

    def test_file_append_rejects_an_unknown_backend(self, tmp_path):
        path = str(tmp_path / "bad_backend.tgm")
        with (
            tensogram.TensogramFile.create(path) as f,
            pytest.raises(ValueError, match=r"unknown compression_backend"),
        ):
            f.append(_META, [(_DESCRIPTOR, _PAYLOAD)], compression_backend="turbo")

    @pytest.mark.asyncio
    async def test_async_streaming_accepts_the_backend(self):
        enc = await tensogram.AsyncStreamingEncoder.create(_META, compression_backend="pure")
        await enc.write_object(_ZSTD_DESCRIPTOR, _PAYLOAD)
        msg = await enc.finish()
        np.testing.assert_array_equal(tensogram.decode(msg).objects[0][1], _PAYLOAD)

    @pytest.mark.asyncio
    async def test_async_streaming_rejects_an_unknown_backend(self):
        with pytest.raises(ValueError, match=r"unknown compression_backend"):
            await tensogram.AsyncStreamingEncoder.create(_META, compression_backend="turbo")

    def test_both_knobs_combine(self):
        msg = tensogram.encode(
            _META,
            [(_ZSTD_DESCRIPTOR, _PAYLOAD)],
            aggregate_hash="both",
            compression_backend="pure",
        )
        types = _frame_types(msg)
        assert "HeaderHash" in types
        assert "FooterHash" in types
        np.testing.assert_array_equal(tensogram.decode(msg).objects[0][1], _PAYLOAD)
