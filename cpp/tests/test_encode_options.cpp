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
// Tests for the full encode-side option set (TgmEncodeOptions):
//   * tensogram::aggregate_hash_policy — where the aggregate hash frame goes
//   * tensogram::compression_backend   — which codec implementation to prefer
// wired through encode(), file::append() and streaming_encoder.
//
// The aggregate-hash assertions close the loop with this binding's own frame
// walker: the placement knob is only meaningful if the frames actually land
// where it says, so every case is verified by walking the encoded message.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using test_helpers::TempFile;

namespace {

const char* const kDescriptor =
    R"({"type":"ndarray","ndim":1,"shape":[4],"strides":[4],"dtype":"float32",)"
    R"("byte_order":"little","encoding":"none","filter":"none","compression":"none"})";

const std::vector<float>& sample_values() {
    static const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
    return values;
}

std::vector<std::pair<const std::uint8_t*, std::size_t>> sample_objects() {
    return {{reinterpret_cast<const std::uint8_t*>(sample_values().data()),
             sample_values().size() * sizeof(float)}};
}

std::string one_object_json() {
    return std::string(R"({"descriptors":[)") + kDescriptor + "]}";
}

/// Collect the frame types of an encoded message.
std::vector<tensogram::frame_type> frame_types(const std::vector<std::uint8_t>& msg) {
    std::vector<tensogram::frame_type> types;
    for (const auto& f : tensogram::frames(msg.data(), msg.size())) {
        types.push_back(f.type());
    }
    return types;
}

bool contains(const std::vector<tensogram::frame_type>& types, tensogram::frame_type t) {
    return std::find(types.begin(), types.end(), t) != types.end();
}

std::size_t count_of(const std::vector<tensogram::frame_type>& types, tensogram::frame_type t) {
    return static_cast<std::size_t>(std::count(types.begin(), types.end(), t));
}

}  // namespace

// ---------------------------------------------------------------------------
// The policy enums mirror the C ABI
// ---------------------------------------------------------------------------

TEST(EncodeOptionsTest, PolicyEnumsMirrorTheCValues) {
    using tensogram::aggregate_hash_policy;
    EXPECT_EQ(static_cast<int>(aggregate_hash_policy::automatic),
              static_cast<int>(TGM_AGGREGATE_HASH_POLICY_AUTO));
    EXPECT_EQ(static_cast<int>(aggregate_hash_policy::none),
              static_cast<int>(TGM_AGGREGATE_HASH_POLICY_NONE));
    EXPECT_EQ(static_cast<int>(aggregate_hash_policy::header),
              static_cast<int>(TGM_AGGREGATE_HASH_POLICY_HEADER));
    EXPECT_EQ(static_cast<int>(aggregate_hash_policy::footer),
              static_cast<int>(TGM_AGGREGATE_HASH_POLICY_FOOTER));
    EXPECT_EQ(static_cast<int>(aggregate_hash_policy::both),
              static_cast<int>(TGM_AGGREGATE_HASH_POLICY_BOTH));

    using tensogram::compression_backend;
    EXPECT_EQ(static_cast<int>(compression_backend::automatic),
              static_cast<int>(TGM_COMPRESSION_BACKEND_AUTO));
    EXPECT_EQ(static_cast<int>(compression_backend::ffi),
              static_cast<int>(TGM_COMPRESSION_BACKEND_FFI));
    EXPECT_EQ(static_cast<int>(compression_backend::pure),
              static_cast<int>(TGM_COMPRESSION_BACKEND_PURE));
}

TEST(EncodeOptionsTest, DefaultsSelectTheLibraryBehaviour) {
    const tensogram::encode_options opts;
    EXPECT_EQ(opts.aggregate_hash, tensogram::aggregate_hash_policy::automatic);
    EXPECT_EQ(opts.codec_backend, tensogram::compression_backend::automatic);
}

// ---------------------------------------------------------------------------
// encode() — aggregate hash placement, verified by walking the frames
// ---------------------------------------------------------------------------

TEST(EncodeOptionsTest, DefaultPlacementPutsTheAggregateHashInTheHeader) {
    // AUTO resolves to the header when buffering, matching what plain
    // encode() has always produced.
    auto types = frame_types(tensogram::encode(one_object_json(), sample_objects()));
    EXPECT_TRUE(contains(types, tensogram::frame_type::header_hash));
    EXPECT_FALSE(contains(types, tensogram::frame_type::footer_hash));
}

TEST(EncodeOptionsTest, AggregateHashBothWritesHeaderAndFooterHashFrames) {
    tensogram::encode_options opts;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::both;
    auto msg = tensogram::encode(one_object_json(), sample_objects(), opts);

    auto types = frame_types(msg);
    EXPECT_EQ(count_of(types, tensogram::frame_type::header_hash), 1u);
    EXPECT_EQ(count_of(types, tensogram::frame_type::footer_hash), 1u);

    const auto header = tensogram::read_message_header(msg.data(), msg.size());
    EXPECT_TRUE(header.has_header_hashes());
    EXPECT_TRUE(header.has_footer_hashes());
}

TEST(EncodeOptionsTest, AggregateHashFooterMovesTheFrameOutOfTheHeader) {
    tensogram::encode_options opts;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::footer;
    auto msg = tensogram::encode(one_object_json(), sample_objects(), opts);

    auto types = frame_types(msg);
    EXPECT_FALSE(contains(types, tensogram::frame_type::header_hash));
    EXPECT_EQ(count_of(types, tensogram::frame_type::footer_hash), 1u);
    // The footer hash frame really is after the data object.
    const auto data_pos = std::find(types.begin(), types.end(), tensogram::frame_type::ntensor);
    const auto hash_pos = std::find(types.begin(), types.end(),
                                    tensogram::frame_type::footer_hash);
    EXPECT_LT(data_pos - types.begin(), hash_pos - types.begin());
}

TEST(EncodeOptionsTest, AggregateHashHeaderIsExplicitlySelectable) {
    tensogram::encode_options opts;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::header;
    auto types = frame_types(tensogram::encode(one_object_json(), sample_objects(), opts));
    EXPECT_EQ(count_of(types, tensogram::frame_type::header_hash), 1u);
    EXPECT_FALSE(contains(types, tensogram::frame_type::footer_hash));
}

TEST(EncodeOptionsTest, AggregateHashNoneKeepsThePerFrameInlineHashes) {
    tensogram::encode_options opts;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::none;
    auto msg = tensogram::encode(one_object_json(), sample_objects(), opts);

    auto types = frame_types(msg);
    EXPECT_FALSE(contains(types, tensogram::frame_type::header_hash));
    EXPECT_FALSE(contains(types, tensogram::frame_type::footer_hash));
    // "No aggregate frame" is not "no hashing": the per-frame slots stay.
    for (const auto& f : tensogram::frames(msg.data(), msg.size())) {
        EXPECT_TRUE(f.has_hash());
    }
    EXPECT_TRUE(tensogram::read_message_header(msg.data(), msg.size()).has_hashes_present());
}

TEST(EncodeOptionsTest, PlacementIsMootWhenHashingIsOff) {
    tensogram::encode_options opts;
    opts.hash_algo = "";
    opts.aggregate_hash = tensogram::aggregate_hash_policy::both;
    auto msg = tensogram::encode(one_object_json(), sample_objects(), opts);

    auto types = frame_types(msg);
    EXPECT_FALSE(contains(types, tensogram::frame_type::header_hash));
    EXPECT_FALSE(contains(types, tensogram::frame_type::footer_hash));
    EXPECT_FALSE(tensogram::read_message_header(msg.data(), msg.size()).has_hashes_present());
}

TEST(EncodeOptionsTest, MessagesStillDecodeUnderEveryPlacement) {
    using tensogram::aggregate_hash_policy;
    for (auto policy : {aggregate_hash_policy::automatic, aggregate_hash_policy::none,
                        aggregate_hash_policy::header, aggregate_hash_policy::footer,
                        aggregate_hash_policy::both}) {
        tensogram::encode_options opts;
        opts.aggregate_hash = policy;
        auto msg = tensogram::encode(one_object_json(), sample_objects(), opts);
        auto decoded = tensogram::decode(msg.data(), msg.size());
        ASSERT_EQ(decoded.num_objects(), 1u);
        const auto obj = decoded.object(0);
        ASSERT_EQ(obj.element_count<float>(), sample_values().size());
        EXPECT_EQ(obj.data_as<float>()[3], 4.0f);
    }
}

// ---------------------------------------------------------------------------
// compression_backend
// ---------------------------------------------------------------------------

TEST(EncodeOptionsTest, EveryCodecBackendRoundTripsLosslessly) {
    // The compressed bytes may differ between implementations; the decoded
    // payload must not.
    const std::string json =
        R"({"descriptors":[{"type":"ndarray","ndim":1,"shape":[64],"strides":[4],)"
        R"("dtype":"float32","byte_order":"little","encoding":"none","filter":"none",)"
        R"("compression":"zstd"}]})";
    std::vector<float> values(64);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects{
        {reinterpret_cast<const std::uint8_t*>(values.data()), values.size() * sizeof(float)}};

    using tensogram::compression_backend;
    for (auto backend : {compression_backend::automatic, compression_backend::ffi,
                         compression_backend::pure}) {
        tensogram::encode_options opts;
        opts.codec_backend = backend;
        auto msg = tensogram::encode(json, objects, opts);
        auto decoded = tensogram::decode(msg.data(), msg.size());
        ASSERT_EQ(decoded.num_objects(), 1u);
        const auto obj = decoded.object(0);
        ASSERT_EQ(obj.element_count<float>(), values.size());
        EXPECT_EQ(0, std::memcmp(obj.data(), values.data(), values.size() * sizeof(float)))
            << "backend " << static_cast<int>(backend);
        EXPECT_EQ(obj.compression(), "zstd");
    }
}

// ---------------------------------------------------------------------------
// file::append
// ---------------------------------------------------------------------------

TEST(EncodeOptionsTest, FileAppendHonoursTheAggregateHashPolicy) {
    TempFile tmp;
    tensogram::encode_options opts;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::both;
    {
        auto f = tensogram::file::create(tmp.path);
        f.append(one_object_json(), sample_objects(), opts);
    }
    auto f = tensogram::file::open(tmp.path);
    ASSERT_EQ(f.message_count(), 1u);
    auto raw = f.read_message(0);
    auto types = frame_types(raw);
    EXPECT_EQ(count_of(types, tensogram::frame_type::header_hash), 1u);
    EXPECT_EQ(count_of(types, tensogram::frame_type::footer_hash), 1u);
}

TEST(EncodeOptionsTest, FileAppendHonoursTheCodecBackend) {
    TempFile tmp;
    tensogram::encode_options opts;
    opts.codec_backend = tensogram::compression_backend::pure;
    {
        auto f = tensogram::file::create(tmp.path);
        f.append(one_object_json(), sample_objects(), opts);
    }
    auto f = tensogram::file::open(tmp.path);
    auto decoded = f.decode_message(0);
    ASSERT_EQ(decoded.num_objects(), 1u);
    EXPECT_EQ(decoded.object(0).data_as<float>()[0], 1.0f);
}

// ---------------------------------------------------------------------------
// streaming_encoder
// ---------------------------------------------------------------------------

TEST(EncodeOptionsTest, StreamingRejectsHeaderSideAggregateHashPlacement) {
    // A streaming writer emits its header before any data object exists, so
    // the per-object hashes are not yet known.
    using tensogram::aggregate_hash_policy;
    for (auto policy : {aggregate_hash_policy::header, aggregate_hash_policy::both}) {
        TempFile tmp;
        tensogram::encode_options opts;
        opts.aggregate_hash = policy;
        EXPECT_THROW(tensogram::streaming_encoder(tmp.path, "{}", opts),
                     tensogram::encoding_error);
    }
}

TEST(EncodeOptionsTest, StreamingFooterPlacementWritesAFooterHashFrame) {
    TempFile tmp;
    tensogram::encode_options opts;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::footer;
    {
        tensogram::streaming_encoder enc(tmp.path, "{}", opts);
        enc.write_object(kDescriptor,
                         reinterpret_cast<const std::uint8_t*>(sample_values().data()),
                         sample_values().size() * sizeof(float));
        enc.finish();
    }
    auto f = tensogram::file::open(tmp.path);
    auto raw = f.read_message(0);
    auto types = frame_types(raw);
    EXPECT_EQ(count_of(types, tensogram::frame_type::footer_hash), 1u);
    EXPECT_FALSE(contains(types, tensogram::frame_type::header_hash));
}

TEST(EncodeOptionsTest, StreamingAggregateHashNoneOmitsTheFrame) {
    TempFile tmp;
    tensogram::encode_options opts;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::none;
    {
        tensogram::streaming_encoder enc(tmp.path, "{}", opts);
        enc.write_object(kDescriptor,
                         reinterpret_cast<const std::uint8_t*>(sample_values().data()),
                         sample_values().size() * sizeof(float));
        enc.finish();
    }
    auto f = tensogram::file::open(tmp.path);
    auto raw = f.read_message(0);
    auto types = frame_types(raw);
    EXPECT_FALSE(contains(types, tensogram::frame_type::footer_hash));
    EXPECT_FALSE(contains(types, tensogram::frame_type::header_hash));
    EXPECT_TRUE(contains(types, tensogram::frame_type::footer_index));
}

TEST(EncodeOptionsTest, StreamingHonoursTheCodecBackend) {
    TempFile tmp;
    tensogram::encode_options opts;
    opts.codec_backend = tensogram::compression_backend::ffi;
    {
        tensogram::streaming_encoder enc(tmp.path, "{}", opts);
        enc.write_object(kDescriptor,
                         reinterpret_cast<const std::uint8_t*>(sample_values().data()),
                         sample_values().size() * sizeof(float));
        enc.finish();
    }
    auto f = tensogram::file::open(tmp.path);
    auto decoded = f.decode_message(0);
    ASSERT_EQ(decoded.num_objects(), 1u);
    EXPECT_EQ(decoded.object(0).data_as<float>()[2], 3.0f);
}

// ---------------------------------------------------------------------------
// encode_pre_encoded — the one entry point the C ABI cannot carry these to
// ---------------------------------------------------------------------------

TEST(EncodeOptionsTest, PreEncodedRejectsOptionsItCannotHonour) {
    // There is no `tgm_encode_pre_encoded_with_encode_options`, so the
    // placement / backend knobs have no route to the encoder here.  Silently
    // dropping them would produce a message that contradicts what the caller
    // asked for, so this reports the mismatch instead.
    const std::string json =
        R"({"descriptors":[{"type":"ndarray","ndim":1,"shape":[4],"strides":[4],)"
        R"("dtype":"float32","byte_order":"little","encoding":"none","filter":"none",)"
        R"("compression":"none"}]})";

    tensogram::encode_options placement;
    placement.aggregate_hash = tensogram::aggregate_hash_policy::footer;
    EXPECT_THROW((void)tensogram::encode_pre_encoded(json, sample_objects(), placement),
                 tensogram::invalid_arg_error);

    tensogram::encode_options backend;
    backend.codec_backend = tensogram::compression_backend::pure;
    EXPECT_THROW((void)tensogram::encode_pre_encoded(json, sample_objects(), backend),
                 tensogram::invalid_arg_error);

    // The default options still encode pre-encoded bytes as before.
    auto msg = tensogram::encode_pre_encoded(json, sample_objects());
    auto decoded = tensogram::decode(msg.data(), msg.size());
    ASSERT_EQ(decoded.num_objects(), 1u);
    EXPECT_EQ(decoded.object(0).data_as<float>()[0], 1.0f);
}

// ---------------------------------------------------------------------------
// The mask knobs keep working alongside the new fields
// ---------------------------------------------------------------------------

TEST(EncodeOptionsTest, MaskOptionsCombineWithTheAggregateHashPolicy) {
    std::vector<float> values{1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects{
        {reinterpret_cast<const std::uint8_t*>(values.data()), values.size() * sizeof(float)}};
    tensogram::encode_options opts;
    opts.allow_nan = true;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::footer;
    auto msg = tensogram::encode(one_object_json(), objects, opts);

    auto types = frame_types(msg);
    EXPECT_TRUE(contains(types, tensogram::frame_type::footer_hash));
    auto decoded = tensogram::decode(msg.data(), msg.size());
    ASSERT_EQ(decoded.num_objects(), 1u);
    EXPECT_TRUE(std::isnan(decoded.object(0).data_as<float>()[1]));
}
