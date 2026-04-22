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
// Edge cases and regression tests.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using test_helpers::TempFile;

// ---------------------------------------------------------------------------
// Helper: dtype JSON
// ---------------------------------------------------------------------------

namespace {

/// Build JSON for a 1-D array with given dtype and element size.
std::string dtype_json(const std::string& dtype, std::size_t elem_size,
                       std::size_t count)
{
    return R"({"version":3,"descriptors":[{"type":"ndarray","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[)" + std::to_string(elem_size) +
           R"(],"dtype":")" + dtype +
           R"(","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Large array (1000+ elements)
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, LargeArray) {
    constexpr std::size_t N = 2000;
    std::vector<float> values(N);
    std::iota(values.begin(), values.end(), 0.0f);

    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);

    EXPECT_EQ(obj.element_count<float>(), N);
    const float* p = obj.data_as<float>();
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[N - 1], static_cast<float>(N - 1));
}

// ---------------------------------------------------------------------------
// Float64 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Float64RoundTrip) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    auto json = dtype_json("float64", sizeof(double), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(double)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "float64");
    EXPECT_EQ(obj.element_count<double>(), 4u);
    const double* p = obj.data_as<double>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_DOUBLE_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Int32 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Int32RoundTrip) {
    std::vector<std::int32_t> values = {-100, 0, 100, 2147483647};
    auto json = dtype_json("int32", sizeof(std::int32_t), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(std::int32_t)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "int32");
    const auto* p = obj.data_as<std::int32_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Int64 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Int64RoundTrip) {
    std::vector<std::int64_t> values = {-1, 0, 1, 9223372036854775807LL};
    auto json = dtype_json("int64", sizeof(std::int64_t), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(std::int64_t)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "int64");
    const auto* p = obj.data_as<std::int64_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Uint8 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Uint8RoundTrip) {
    std::vector<std::uint8_t> values = {0, 1, 127, 255};
    auto json = dtype_json("uint8", sizeof(std::uint8_t), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {values.data(), values.size() * sizeof(std::uint8_t)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "uint8");
    const auto* p = obj.data_as<std::uint8_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Int8 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Int8RoundTrip) {
    std::vector<std::int8_t> values = {-128, -1, 0, 127};
    auto json = dtype_json("int8", sizeof(std::int8_t), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(std::int8_t)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "int8");
    const auto* p = obj.data_as<std::int8_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Int16 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Int16RoundTrip) {
    std::vector<std::int16_t> values = {-32768, 0, 32767};
    auto json = dtype_json("int16", sizeof(std::int16_t), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(std::int16_t)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "int16");
    const auto* p = obj.data_as<std::int16_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Uint16 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Uint16RoundTrip) {
    std::vector<std::uint16_t> values = {0, 1, 65535};
    auto json = dtype_json("uint16", sizeof(std::uint16_t), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(std::uint16_t)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "uint16");
    const auto* p = obj.data_as<std::uint16_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Uint32 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Uint32RoundTrip) {
    std::vector<std::uint32_t> values = {0, 1, 4294967295u};
    auto json = dtype_json("uint32", sizeof(std::uint32_t), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(std::uint32_t)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "uint32");
    const auto* p = obj.data_as<std::uint32_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Uint64 round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, Uint64RoundTrip) {
    std::vector<std::uint64_t> values = {0, 1, 18446744073709551615ULL};
    auto json = dtype_json("uint64", sizeof(std::uint64_t), values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(std::uint64_t)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "uint64");
    const auto* p = obj.data_as<std::uint64_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// ---------------------------------------------------------------------------
// Big-endian round-trip
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, BigEndianRoundTrip) {
    // Encode as big-endian.  Decode returns native-endian by default.
    std::string json = R"({"version":3,"descriptors":[{"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32","byte_order":"big","encoding":"none","filter":"none","compression":"none"}]})";

    // Provide big-endian bytes for 1.0f and 2.0f
    // IEEE 754 big-endian: 1.0f = 3F800000, 2.0f = 40000000
    std::uint8_t be_data[] = {
        0x3F, 0x80, 0x00, 0x00,  // 1.0f big-endian
        0x40, 0x00, 0x00, 0x00   // 2.0f big-endian
    };
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {be_data, sizeof(be_data)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);

    EXPECT_EQ(obj.byte_order_string(), "big");
    EXPECT_EQ(obj.data_size(), sizeof(be_data));
    // Decoded bytes are in native byte order — verify the float values.
    auto* floats = obj.data_as<float>();
    EXPECT_FLOAT_EQ(floats[0], 1.0f);
    EXPECT_FLOAT_EQ(floats[1], 2.0f);
}

// ---------------------------------------------------------------------------
// Idempotent decode (decode same buffer twice gives same result)
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, IdempotentDecode) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    auto encoded = test_helpers::encode_simple_f32(values);

    auto msg1 = tensogram::decode(encoded.data(), encoded.size());
    auto msg2 = tensogram::decode(encoded.data(), encoded.size());

    EXPECT_EQ(msg1.version(), msg2.version());
    EXPECT_EQ(msg1.num_objects(), msg2.num_objects());

    auto obj1 = msg1.object(0);
    auto obj2 = msg2.object(0);

    EXPECT_EQ(obj1.dtype_string(), obj2.dtype_string());
    EXPECT_EQ(obj1.data_size(), obj2.data_size());
    EXPECT_EQ(std::memcmp(obj1.data(), obj2.data(), obj1.data_size()), 0);
}

// ---------------------------------------------------------------------------
// Hash verification on decode (verify_hash=true)
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, HashVerificationPasses) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    std::string json = test_helpers::simple_f32_json(values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    tensogram::encode_options enc_opts;
    enc_opts.hash_algo = "xxh3";
    auto encoded = tensogram::encode(json, objects, enc_opts);

    tensogram::decode_options dec_opts;
    dec_opts.verify_hash = true;
    EXPECT_NO_THROW((void)tensogram::decode(encoded.data(), encoded.size(), dec_opts));
}

// ---------------------------------------------------------------------------
// Hash mismatch detection (tamper with data)
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, HashMismatchDetection) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    std::string json = test_helpers::simple_f32_json(values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    tensogram::encode_options enc_opts;
    enc_opts.hash_algo = "xxh3";
    auto encoded = tensogram::encode(json, objects, enc_opts);

    // Tamper with the last few bytes (payload data area)
    if (encoded.size() > 10) {
        encoded[encoded.size() - 5] ^= 0xFF;
        encoded[encoded.size() - 6] ^= 0xFF;
    }

    tensogram::decode_options dec_opts;
    dec_opts.verify_hash = true;
    // Should throw due to hash mismatch or framing error
    EXPECT_THROW(
        (void)tensogram::decode(encoded.data(), encoded.size(), dec_opts),
        tensogram::error);
}

// ---------------------------------------------------------------------------
// Multi-object message with mixed dtypes
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, MultiObjectMixedDtypes) {
    std::string json = R"({"version":3,"descriptors":[)"
        R"({"type":"ndarray","ndim":1,"shape":[3],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"},)"
        R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"int32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<float> floats = {1.0f, 2.0f, 3.0f};
    std::vector<std::int32_t> ints = {42, -7};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(floats.data()),
         floats.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(ints.data()),
         ints.size() * sizeof(std::int32_t)}
    };

    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    ASSERT_EQ(msg.num_objects(), 2u);

    auto obj0 = msg.object(0);
    EXPECT_EQ(obj0.dtype_string(), "float32");
    EXPECT_FLOAT_EQ(obj0.data_as<float>()[0], 1.0f);

    auto obj1 = msg.object(1);
    EXPECT_EQ(obj1.dtype_string(), "int32");
    EXPECT_EQ(obj1.data_as<std::int32_t>()[0], 42);
    EXPECT_EQ(obj1.data_as<std::int32_t>()[1], -7);
}

// ---------------------------------------------------------------------------
// Scan then decode each message
// ---------------------------------------------------------------------------

TEST(EdgeCaseTest, ScanThenDecodeEach) {
    auto e1 = test_helpers::encode_simple_f32({1.0f});
    auto e2 = test_helpers::encode_simple_f32({2.0f, 3.0f});
    auto e3 = test_helpers::encode_simple_f32({4.0f, 5.0f, 6.0f});

    std::vector<std::uint8_t> combined;
    combined.insert(combined.end(), e1.begin(), e1.end());
    combined.insert(combined.end(), e2.begin(), e2.end());
    combined.insert(combined.end(), e3.begin(), e3.end());

    auto entries = tensogram::scan(combined.data(), combined.size());
    ASSERT_EQ(entries.size(), 3u);

    // Decode first message
    auto msg0 = tensogram::decode(
        combined.data() + entries[0].offset, entries[0].length);
    EXPECT_EQ(msg0.object(0).element_count<float>(), 1u);

    // Decode second message
    auto msg1 = tensogram::decode(
        combined.data() + entries[1].offset, entries[1].length);
    EXPECT_EQ(msg1.object(0).element_count<float>(), 2u);

    // Decode third message
    auto msg2 = tensogram::decode(
        combined.data() + entries[2].offset, entries[2].length);
    EXPECT_EQ(msg2.object(0).element_count<float>(), 3u);
}
