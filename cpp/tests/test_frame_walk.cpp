// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF
//
// Tests for the lazy frame walker and the typed message header:
//   * tensogram::frame_type / frame        (tgm_frame_type / TgmFrame)
//   * tensogram::frames -> frame_range     (tgm_frame_iter_create/_next/_free)
//   * tensogram::read_message_header       (tgm_message_header)

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using test_helpers::TempFile;

namespace {

// Structural constants from plans/WIRE_FORMAT.md, spelled out here so the
// assertions below read as statements about the wire format rather than
// about the walker that produced them.
constexpr std::size_t kPreambleSize   = 24;  // §3
constexpr std::size_t kPostambleSize  = 24;  // §7
constexpr std::size_t kFrameHeader    = 16;  // §2.1
constexpr std::size_t kFooterCommon   = 12;  // §2.2 [hash][ENDF]
constexpr std::size_t kFooterNtensor  = 20;  // §2.2 [cbor_offset][hash][ENDF]
constexpr std::uint16_t kHashPresent  = 1u << 1;  // §2.5 frame flag bit 1

/// One float32 descriptor entry for a 4-element 1-D tensor.
const char* const kDescriptor =
    R"({"type":"ndarray","ndim":1,"shape":[4],"strides":[4],"dtype":"float32",)"
    R"("byte_order":"little","encoding":"none","filter":"none","compression":"none"})";

/// Metadata JSON describing @p n identical float32 objects.
std::string multi_object_json(std::size_t n) {
    std::string json = R"({"descriptors":[)";
    for (std::size_t i = 0; i < n; ++i) {
        if (i != 0) json += ",";
        json += kDescriptor;
    }
    return json + "]}";
}

/// A buffered (random-access) message holding @p n objects: metadata, index
/// and hash frames all live in the header.
std::vector<std::uint8_t> buffered_message(std::size_t n, bool hashed = true) {
    const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects;
    for (std::size_t i = 0; i < n; ++i) {
        objects.emplace_back(reinterpret_cast<const std::uint8_t*>(values.data()),
                             values.size() * sizeof(float));
    }
    tensogram::encode_options opts;
    if (!hashed) opts.hash_algo = "";
    return tensogram::encode(multi_object_json(n), objects, opts);
}

/// A streaming message: metadata / index / hashes land in the footer and the
/// preamble's `total_length` is never back-filled.
std::vector<std::uint8_t> streamed_message(bool with_preceder) {
    TempFile tmp;
    const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
    {
        tensogram::streaming_encoder enc(tmp.path, "{}");
        if (with_preceder) enc.write_preceder(R"({"units":"K"})");
        enc.write_object(kDescriptor,
                         reinterpret_cast<const std::uint8_t*>(values.data()),
                         values.size() * sizeof(float));
        enc.finish();
    }
    auto f = tensogram::file::open(tmp.path);
    return f.read_message(0);
}

/// Walk @p msg to exhaustion and collect every frame.
std::vector<tensogram::frame> collect(const std::vector<std::uint8_t>& msg) {
    std::vector<tensogram::frame> out;
    for (const auto& f : tensogram::frames(msg.data(), msg.size())) {
        out.push_back(f);
    }
    return out;
}

std::vector<tensogram::frame_type> types_of(const std::vector<tensogram::frame>& frames) {
    std::vector<tensogram::frame_type> types;
    types.reserve(frames.size());
    for (const auto& f : frames) types.push_back(f.type());
    return types;
}

bool contains(const std::vector<tensogram::frame_type>& types, tensogram::frame_type t) {
    return std::find(types.begin(), types.end(), t) != types.end();
}

}  // namespace

// ---------------------------------------------------------------------------
// frame_type — the C++ enum carries the wire's frame-type numbers
// ---------------------------------------------------------------------------

TEST(FrameTypeTest, MirrorsTheWireNumbers) {
    using tensogram::frame_type;
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::header_metadata), 1u);
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::header_index), 2u);
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::header_hash), 3u);
    // 4 is reserved (the obsolete v2 data-object frame) — no enumerator.
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::footer_hash), 5u);
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::footer_index), 6u);
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::footer_metadata), 7u);
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::preceder_metadata), 8u);
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::ntensor), 9u);
}

TEST(FrameTypeTest, MatchesTheCEnum) {
    using tensogram::frame_type;
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::header_metadata),
              static_cast<std::uint16_t>(TGM_FRAME_TYPE_HEADER_METADATA));
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::footer_metadata),
              static_cast<std::uint16_t>(TGM_FRAME_TYPE_FOOTER_METADATA));
    EXPECT_EQ(static_cast<std::uint16_t>(frame_type::ntensor),
              static_cast<std::uint16_t>(TGM_FRAME_TYPE_NTENSOR));
}

// ---------------------------------------------------------------------------
// frames() — the expected frame sequence
// ---------------------------------------------------------------------------

TEST(FrameWalkTest, BufferedMessageYieldsHeaderFramesThenOneFramePerObject) {
    using tensogram::frame_type;
    auto msg = buffered_message(2);
    const std::vector<frame_type> expected{
        frame_type::header_metadata,
        frame_type::header_index,
        frame_type::header_hash,
        frame_type::ntensor,
        frame_type::ntensor,
    };
    EXPECT_EQ(types_of(collect(msg)), expected);
}

TEST(FrameWalkTest, UnhashedMessageHasNoAggregateHashFrame) {
    using tensogram::frame_type;
    auto msg = buffered_message(1, /*hashed=*/false);
    const std::vector<frame_type> expected{
        frame_type::header_metadata,
        frame_type::header_index,
        frame_type::ntensor,
    };
    EXPECT_EQ(types_of(collect(msg)), expected);
}

TEST(FrameWalkTest, StreamingMessagePutsMetadataAndIndexInTheFooter) {
    using tensogram::frame_type;
    auto msg = streamed_message(/*with_preceder=*/false);
    const std::vector<frame_type> expected{
        frame_type::header_metadata,
        frame_type::ntensor,
        frame_type::footer_metadata,
        frame_type::footer_hash,
        frame_type::footer_index,
    };
    EXPECT_EQ(types_of(collect(msg)), expected);
}

TEST(FrameWalkTest, PrecederMetadataPrecedesItsDataObject) {
    using tensogram::frame_type;
    auto msg = streamed_message(/*with_preceder=*/true);
    auto types = types_of(collect(msg));
    ASSERT_TRUE(contains(types, frame_type::preceder_metadata));
    const auto it = std::find(types.begin(), types.end(), frame_type::preceder_metadata);
    ASSERT_NE(it + 1, types.end());
    EXPECT_EQ(*(it + 1), frame_type::ntensor);
}

TEST(FrameWalkTest, RangeForVisitsEveryFrame) {
    auto msg = buffered_message(3);
    std::size_t seen = 0;
    std::size_t data_objects = 0;
    for (const auto& f : tensogram::frames(msg.data(), msg.size())) {
        ++seen;
        if (f.type() == tensogram::frame_type::ntensor) ++data_objects;
    }
    EXPECT_EQ(seen, 6u);  // metadata + index + hash + 3 data objects
    EXPECT_EQ(data_objects, 3u);
}

TEST(FrameWalkTest, IsDataObjectAgreesWithTheFrameType) {
    auto msg = buffered_message(2);
    std::size_t data_objects = 0;
    for (const auto& f : collect(msg)) {
        EXPECT_EQ(f.is_data_object(), f.type() == tensogram::frame_type::ntensor);
        if (f.is_data_object()) ++data_objects;
    }
    EXPECT_EQ(data_objects, 2u);
}

// ---------------------------------------------------------------------------
// Structural invariants: offsets tile the message and stay in bounds
// ---------------------------------------------------------------------------

TEST(FrameWalkTest, OffsetsTileTheMessageAndStayInBounds) {
    auto msg = buffered_message(2);
    std::size_t prev_end = kPreambleSize;
    std::size_t walked = 0;
    for (const auto& f : collect(msg)) {
        EXPECT_GE(f.offset(), prev_end) << "frames must not overlap";
        EXPECT_LE(f.offset() + f.length(), msg.size()) << "frame stays in bounds";
        EXPECT_GE(f.length(), kFrameHeader);
        // The whole-frame span always ends on the ENDF sentinel.
        const std::size_t end = f.offset() + f.length();
        ASSERT_GE(end, 4u);
        EXPECT_EQ(0, std::memcmp(msg.data() + end - 4, "ENDF", 4))
            << "length() must span the frame footer";
        prev_end = end;
        ++walked;
    }
    ASSERT_GT(walked, 0u);
    EXPECT_LE(prev_end, msg.size() - kPostambleSize)
        << "the postamble is not a frame";
}

TEST(FrameWalkTest, FirstFrameStartsAfterThePreamble) {
    auto msg = buffered_message(1);
    auto frames = collect(msg);
    ASSERT_FALSE(frames.empty());
    EXPECT_GE(frames.front().offset(), kPreambleSize);
}

TEST(FrameWalkTest, PayloadExcludesTheFrameHeaderAndFooter) {
    auto msg = buffered_message(1);
    auto frames = collect(msg);
    ASSERT_FALSE(frames.empty());
    for (const auto& f : frames) {
        const std::size_t footer =
            f.type() == tensogram::frame_type::ntensor ? kFooterNtensor : kFooterCommon;
        EXPECT_EQ(f.payload_size(), f.length() - kFrameHeader - footer);
        EXPECT_EQ(f.payload_data(), msg.data() + f.offset() + kFrameHeader);
        EXPECT_EQ(f.payload().size(), f.payload_size());
        EXPECT_EQ(static_cast<const void*>(f.payload().data()),
                  static_cast<const void*>(f.payload_data()));
    }
}

TEST(FrameWalkTest, PayloadBorrowsTheCallersBufferAndOutlivesTheRange) {
    // Documented lifetime contract: payload() points INTO the caller's message
    // buffer, so it stays valid after the range (and its C cursor) are gone.
    auto msg = buffered_message(1);
    std::vector<tensogram::frame> frames;
    {
        auto range = tensogram::frames(msg.data(), msg.size());
        for (const auto& f : range) frames.push_back(f);
    }  // range destroyed here — tgm_frame_iter_free has run
    ASSERT_FALSE(frames.empty());
    for (const auto& f : frames) {
        ASSERT_GT(f.payload_size(), 0u);
        const std::size_t start = f.offset() + kFrameHeader;
        EXPECT_EQ(f.payload_data(), msg.data() + start);
        EXPECT_EQ(0, std::memcmp(f.payload_data(), msg.data() + start, f.payload_size()))
            << "payload must still read the caller's bytes";
    }
}

TEST(FrameWalkTest, HasHashMirrorsThePerFrameFlagBit) {
    auto hashed = buffered_message(1);
    auto frames = collect(hashed);
    ASSERT_FALSE(frames.empty());
    for (const auto& f : frames) {
        EXPECT_TRUE(f.has_hash());
        EXPECT_EQ(f.has_hash(), (f.flags() & kHashPresent) != 0);
        EXPECT_EQ(f.version(), 1u);
    }

    auto plain = buffered_message(1, /*hashed=*/false);
    for (const auto& f : collect(plain)) {
        EXPECT_FALSE(f.has_hash());
        EXPECT_EQ(f.has_hash(), (f.flags() & kHashPresent) != 0);
    }
}

// ---------------------------------------------------------------------------
// End vs malformed: a clean end stops, a broken chain throws
// ---------------------------------------------------------------------------

TEST(FrameWalkTest, CleanEndDoesNotThrowAndStaysAtEnd) {
    auto msg = buffered_message(1);
    auto range = tensogram::frames(msg.data(), msg.size());
    auto it = range.begin();
    std::size_t n = 0;
    for (; it != range.end(); ++it) ++n;
    EXPECT_GT(n, 0u);
    EXPECT_TRUE(it == range.end());
    // Advancing an exhausted cursor stays safely at the end.
    EXPECT_NO_THROW(++it);
    EXPECT_TRUE(it == range.end());
}

TEST(FrameWalkTest, IteratorsSurviveAMoveOfTheRange) {
    // The range owns the C cursor through a unique_ptr, so a move transfers
    // the same pointer value — an iterator taken before the move keeps
    // walking the same cursor.
    auto msg = buffered_message(2);
    const std::size_t total = collect(msg).size();

    auto range = tensogram::frames(msg.data(), msg.size());
    auto it = range.begin();
    ASSERT_TRUE(it != range.end());
    const auto first_type = it->type();

    auto moved = std::move(range);
    std::size_t seen = 1;  // the frame already pulled above
    std::size_t prev_offset = it->offset();
    for (++it; it != moved.end(); ++it) {
        EXPECT_GT(it->offset(), prev_offset) << "the walk keeps moving forward";
        prev_offset = it->offset();
        ++seen;
    }
    EXPECT_EQ(seen, total) << "the moved-into range finishes the same walk";
    EXPECT_EQ(first_type, tensogram::frame_type::header_metadata);
}

TEST(FrameWalkTest, TruncatedFrameChainThrowsAfterYieldingTheIntactFrames) {
    auto msg = buffered_message(2);
    auto frames = collect(msg);
    ASSERT_FALSE(frames.empty());
    // Cut *inside* the last frame: the preamble still parses, the chain does not.
    const std::size_t cut = frames.back().offset() + 8;
    std::vector<std::uint8_t> truncated(msg.begin(), msg.begin() + static_cast<long>(cut));

    std::size_t yielded = 0;
    EXPECT_THROW(
        {
            for (const auto& f : tensogram::frames(truncated.data(), truncated.size())) {
                (void)f;
                ++yielded;
            }
        },
        tensogram::framing_error);
    EXPECT_GT(yielded, 0u) << "the frames before the cut are still yielded";
}

TEST(FrameWalkTest, NonMessageBufferThrows) {
    const std::string junk = "not a tensogram message at all!!!!!!!!!!";
    EXPECT_THROW((void)tensogram::frames(reinterpret_cast<const std::uint8_t*>(junk.data()),
                                         junk.size()),
                 tensogram::framing_error);
}

TEST(FrameWalkTest, TruncatedPreambleThrows) {
    auto msg = buffered_message(1);
    EXPECT_THROW((void)tensogram::frames(msg.data(), 8), tensogram::framing_error);
}

TEST(FrameWalkTest, NullBufferIsRejected) {
    EXPECT_THROW((void)tensogram::frames(nullptr, 0), tensogram::invalid_arg_error);
}

// ---------------------------------------------------------------------------
// read_message_header
// ---------------------------------------------------------------------------

TEST(MessageHeaderTest, BufferedPreambleDescribesARandomAccessMessage) {
    auto msg = buffered_message(2);
    const auto h = tensogram::read_message_header(msg.data(), msg.size());
    EXPECT_EQ(h.version(), tensogram::wire_version);
    EXPECT_EQ(h.total_length(), msg.size());
    EXPECT_TRUE(h.has_header_metadata());
    EXPECT_TRUE(h.has_header_index());
    EXPECT_FALSE(h.has_footer_metadata());
    EXPECT_FALSE(h.has_footer_index());
    EXPECT_FALSE(h.has_preceder_metadata());
    EXPECT_TRUE(h.has_hashes_present()) << "the default encode hashes every frame";
}

TEST(MessageHeaderTest, BufferedFlagsMatchTheFramesExactly) {
    auto msg = buffered_message(2);
    const auto h = tensogram::read_message_header(msg.data(), msg.size());
    const auto types = types_of(collect(msg));
    using tensogram::frame_type;
    EXPECT_EQ(h.has_header_metadata(), contains(types, frame_type::header_metadata));
    EXPECT_EQ(h.has_footer_metadata(), contains(types, frame_type::footer_metadata));
    EXPECT_EQ(h.has_header_index(), contains(types, frame_type::header_index));
    EXPECT_EQ(h.has_footer_index(), contains(types, frame_type::footer_index));
    EXPECT_EQ(h.has_header_hashes(), contains(types, frame_type::header_hash));
    EXPECT_EQ(h.has_footer_hashes(), contains(types, frame_type::footer_hash));
    EXPECT_EQ(h.has_preceder_metadata(), contains(types, frame_type::preceder_metadata));
}

TEST(MessageHeaderTest, StreamingFlagsNeverUnderstateTheFramesPresent) {
    // The streaming encoder writes its preamble before it knows what follows,
    // so PRECEDER_METADATA is set advisorily: only "frame present => flag set"
    // holds.
    for (bool with_preceder : {false, true}) {
        auto msg = streamed_message(with_preceder);
        const auto h = tensogram::read_message_header(msg.data(), msg.size());
        const auto types = types_of(collect(msg));
        using tensogram::frame_type;
        const auto implies = [&](frame_type t, bool flag) { return !contains(types, t) || flag; };
        EXPECT_TRUE(implies(frame_type::header_metadata, h.has_header_metadata()));
        EXPECT_TRUE(implies(frame_type::footer_metadata, h.has_footer_metadata()));
        EXPECT_TRUE(implies(frame_type::header_index, h.has_header_index()));
        EXPECT_TRUE(implies(frame_type::footer_index, h.has_footer_index()));
        EXPECT_TRUE(implies(frame_type::header_hash, h.has_header_hashes()));
        EXPECT_TRUE(implies(frame_type::footer_hash, h.has_footer_hashes()));
        EXPECT_TRUE(implies(frame_type::preceder_metadata, h.has_preceder_metadata()));
        // Streaming never back-fills the length through the non-seeking sink.
        EXPECT_EQ(h.total_length(), 0u);
        EXPECT_TRUE(h.has_footer_metadata());
        EXPECT_TRUE(h.has_footer_index());
    }
}

TEST(MessageHeaderTest, RejectsBuffersThatAreNotMessages) {
    const std::string junk = "not a tensogram message at all!!!!!!!!!!";
    EXPECT_THROW((void)tensogram::read_message_header(
                     reinterpret_cast<const std::uint8_t*>(junk.data()), junk.size()),
                 tensogram::framing_error);

    auto msg = buffered_message(1);
    EXPECT_THROW((void)tensogram::read_message_header(msg.data(), 8),
                 tensogram::framing_error);
    EXPECT_THROW((void)tensogram::read_message_header(nullptr, 0),
                 tensogram::invalid_arg_error);
}
