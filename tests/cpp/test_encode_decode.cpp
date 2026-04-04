// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF
//
// Basic encode/decode round-trip tests for the C++ wrapper.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstring>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Encode / Decode round-trip
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, BasicFloat32RoundTrip) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};

    std::string json = R"({
        "version": 2,
        "descriptors": [{
            "type": "ndarray",
            "ndim": 1,
            "shape": [4],
            "strides": [4],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none"
        }]
    })";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    auto encoded = tensogram::encode(json, objects);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.version(), 2u);
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.ndim(), 1u);
    EXPECT_EQ(obj.shape(), std::vector<std::uint64_t>{4});
    EXPECT_EQ(obj.dtype_string(), "float32");
    EXPECT_EQ(obj.data_size(), values.size() * sizeof(float));

    const float* decoded = obj.data_as<float>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(decoded[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Helper-based round-trip
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, HelperRoundTrip) {
    std::vector<float> values = {10.0f, 20.0f, 30.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.element_count<float>(), 3u);
    const float* p = obj.data_as<float>();
    EXPECT_FLOAT_EQ(p[0], 10.0f);
    EXPECT_FLOAT_EQ(p[1], 20.0f);
    EXPECT_FLOAT_EQ(p[2], 30.0f);
}

// ---------------------------------------------------------------------------
// Metadata access
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, MetadataAccess) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto meta = msg.get_metadata();
    EXPECT_EQ(meta.version(), 2u);
}

// ---------------------------------------------------------------------------
// Decode metadata only
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, DecodeMetadataOnly) {
    std::vector<float> values = {1.0f, 2.0f};
    auto encoded = test_helpers::encode_simple_f32(values);

    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());
    EXPECT_EQ(meta.version(), 2u);
}

// ---------------------------------------------------------------------------
// Decode single object
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, DecodeSingleObject) {
    std::vector<float> values = {5.0f, 6.0f, 7.0f, 8.0f};
    auto encoded = test_helpers::encode_simple_f32(values);

    auto msg = tensogram::decode_object(
        encoded.data(), encoded.size(), 0);
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    const float* p = obj.data_as<float>();
    EXPECT_FLOAT_EQ(p[0], 5.0f);
    EXPECT_FLOAT_EQ(p[3], 8.0f);
}

// ---------------------------------------------------------------------------
// Object descriptor fields
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, ObjectDescriptorFields) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);

    EXPECT_EQ(obj.object_type(), "ndarray");
    EXPECT_EQ(obj.byte_order_string(), "little");
    EXPECT_EQ(obj.encoding(), "none");
    EXPECT_EQ(obj.filter(), "none");
    EXPECT_EQ(obj.compression(), "none");
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, ScanBuffer) {
    std::vector<float> v1 = {1.0f, 2.0f};
    std::vector<float> v2 = {3.0f, 4.0f, 5.0f};
    auto e1 = test_helpers::encode_simple_f32(v1);
    auto e2 = test_helpers::encode_simple_f32(v2);

    // Concatenate two messages
    std::vector<std::uint8_t> combined;
    combined.insert(combined.end(), e1.begin(), e1.end());
    combined.insert(combined.end(), e2.begin(), e2.end());

    auto entries = tensogram::scan(combined.data(), combined.size());
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].offset, 0u);
    EXPECT_EQ(entries[0].length, e1.size());
    EXPECT_EQ(entries[1].offset, e1.size());
    EXPECT_EQ(entries[1].length, e2.size());
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, ComputeHash) {
    std::vector<std::uint8_t> data = {0x01, 0x02, 0x03, 0x04};
    auto hash = tensogram::compute_hash(data.data(), data.size());
    EXPECT_FALSE(hash.empty());
}

// ---------------------------------------------------------------------------
// Encode with hash, decode with verification
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, HashVerification) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};

    std::string json = R"({
        "version": 2,
        "descriptors": [{
            "type": "ndarray",
            "ndim": 1,
            "shape": [3],
            "strides": [4],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none"
        }]
    })";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    tensogram::encode_options enc_opts;
    enc_opts.hash_algo = "xxh3";
    auto encoded = tensogram::encode(json, objects, enc_opts);

    // Decode with hash verification enabled
    tensogram::decode_options dec_opts;
    dec_opts.verify_hash = true;
    auto msg = tensogram::decode(encoded.data(), encoded.size(), dec_opts);
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_TRUE(obj.has_hash());
    EXPECT_EQ(obj.hash_type(), "xxh3");
    EXPECT_FALSE(obj.hash_value().empty());
}

// ---------------------------------------------------------------------------
// Message iterator (range-based for)
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, MessageIterator) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());

    std::size_t count = 0;
    for (const auto& obj : msg) {
        EXPECT_EQ(obj.dtype_string(), "float32");
        ++count;
    }
    EXPECT_EQ(count, 1u);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, InvalidBufferThrows) {
    std::vector<std::uint8_t> garbage = {0x00, 0x01, 0x02};
    EXPECT_THROW(
        (void)tensogram::decode(garbage.data(), garbage.size()),
        tensogram::framing_error);
}

TEST(EncodeDecodeTest, ErrorCodePreserved) {
    std::vector<std::uint8_t> garbage = {0x00, 0x01, 0x02};
    try {
        (void)tensogram::decode(garbage.data(), garbage.size());
        FAIL() << "Expected exception";
    } catch (const tensogram::error& e) {
        EXPECT_EQ(e.code(), TGM_ERROR_FRAMING);
    }
}

// ---------------------------------------------------------------------------
// decode_range: extract a sub-range from an uncompressed object
// ---------------------------------------------------------------------------

TEST(EncodeDecodeTest, DecodeRangePartial) {
    // Encode 8 floats: 0.0, 1.0, ..., 7.0
    constexpr std::size_t N = 8;
    std::vector<float> values(N);
    for (std::size_t i = 0; i < N; ++i) values[i] = static_cast<float>(i);

    auto encoded = test_helpers::encode_simple_f32(values);

    // Request elements [2, 3, 4] — offset=2, count=3
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {{2, 3}};
    auto raw = tensogram::decode_range(
        encoded.data(), encoded.size(), 0, ranges);

    ASSERT_EQ(raw.size(), 3 * sizeof(float));
    const float* p = reinterpret_cast<const float*>(raw.data());
    EXPECT_FLOAT_EQ(p[0], 2.0f);
    EXPECT_FLOAT_EQ(p[1], 3.0f);
    EXPECT_FLOAT_EQ(p[2], 4.0f);
}

// decode_range: full range returns entire payload
TEST(EncodeDecodeTest, DecodeRangeFull) {
    std::vector<float> values = {10.0f, 20.0f, 30.0f};
    auto encoded = test_helpers::encode_simple_f32(values);

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {{0, 3}};
    auto raw = tensogram::decode_range(
        encoded.data(), encoded.size(), 0, ranges);

    ASSERT_EQ(raw.size(), 3 * sizeof(float));
    const float* p = reinterpret_cast<const float*>(raw.data());
    EXPECT_FLOAT_EQ(p[0], 10.0f);
    EXPECT_FLOAT_EQ(p[1], 20.0f);
    EXPECT_FLOAT_EQ(p[2], 30.0f);
}
