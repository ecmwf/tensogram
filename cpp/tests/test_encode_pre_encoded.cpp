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
// GoogleTest coverage for encode_pre_encoded() and
// streaming_encoder::write_object_pre_encoded().

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using test_helpers::TempFile;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Build a bare descriptor JSON for a 1-D encoding=none object (no envelope).
static std::string descriptor_json(std::size_t count,
                                   const std::string& dtype = "float32",
                                   const std::string& byte_order = "little") {
    return R"({"type":"ndarray","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[)" +
           std::to_string(dtype == "float64" ? 8
                          : dtype == "int64" ? 8
                          : dtype == "int32" ? 4
                          : dtype == "uint8" ? 1
                          : 4) +
           R"(],"dtype":")" + dtype +
           R"(","byte_order":")" + byte_order +
           R"(","encoding":"none","filter":"none","compression":"none"})";
}

/// Build a v2 JSON metadata+descriptor string for a 1-D encoding=none message.
static std::string encoding_none_json(std::size_t count,
                                      const std::string& dtype = "float32",
                                      const std::string& byte_order = "little") {
    return R"({"version":2,"descriptors":[)" + descriptor_json(count, dtype, byte_order) + R"(]})";
}

/// Build a v2 JSON string for simple_packing.
static std::string simple_packing_json(std::size_t count,
                                       double reference_value,
                                       int binary_scale_factor,
                                       int decimal_scale_factor,
                                       int bits_per_value) {
    return R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[8],"dtype":"float64","byte_order":"little",)"
           R"("encoding":"simple_packing","filter":"none","compression":"none",)"
           R"("reference_value":)" + std::to_string(reference_value) +
           R"(,"binary_scale_factor":)" + std::to_string(binary_scale_factor) +
           R"(,"decimal_scale_factor":)" + std::to_string(decimal_scale_factor) +
           R"(,"bits_per_value":)" + std::to_string(bits_per_value) +
           R"(}]})";
}

/// Manually bit-pack integers (MSB-first, big-endian bit order).
static std::vector<std::uint8_t> bit_pack(const std::vector<std::uint64_t>& values,
                                          int bits_per_value) {
    std::size_t total_bits = values.size() * static_cast<std::size_t>(bits_per_value);
    std::size_t total_bytes = (total_bits + 7) / 8;
    std::vector<std::uint8_t> buf(total_bytes, 0);

    std::size_t bit_pos = 0;
    for (auto val : values) {
        int remaining = bits_per_value;
        while (remaining > 0) {
            std::size_t byte_idx = bit_pos / 8;
            int bit_offset = static_cast<int>(bit_pos % 8);
            int space = 8 - bit_offset;
            int write_bits = std::min(remaining, space);

            int shift = remaining - write_bits;
            auto bits = static_cast<std::uint8_t>((val >> shift) & ((1ULL << write_bits) - 1));

            buf[byte_idx] |= static_cast<std::uint8_t>(bits << (space - write_bits));
            bit_pos += static_cast<std::size_t>(write_bits);
            remaining -= write_bits;
        }
    }
    return buf;
}

// -----------------------------------------------------------------------
// Test 1: encoding=none round-trip
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, EncodingNoneFloat32RoundTrip) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto json = encoding_none_json(values.size());

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    auto encoded = tensogram::encode_pre_encoded(json, objects);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "float32");
    EXPECT_EQ(obj.element_count<float>(), values.size());

    const float* p = obj.data_as<float>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(p[i], values[i]);
    }
}

// -----------------------------------------------------------------------
// Test 2: encoding=none int32 round-trip
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, EncodingNoneInt32RoundTrip) {
    std::vector<std::int32_t> values = {-100, 0, 42, 1000, -1};
    auto json = encoding_none_json(values.size(), "int32");

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(std::int32_t)}
    };
    auto encoded = tensogram::encode_pre_encoded(json, objects);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "int32");

    const auto* p = obj.data_as<std::int32_t>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(p[i], values[i]);
    }
}

// -----------------------------------------------------------------------
// Test 3: simple_packing round-trip
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, SimplePackingRoundTrip) {
    // Generate 100 temperature-like values
    constexpr std::size_t N = 100;
    std::vector<double> values(N);
    for (std::size_t i = 0; i < N; ++i) {
        values[i] = 250.0 + static_cast<double>(i) * 1.0;
    }

    // Packing parameters (manually chosen for these values)
    double ref_val = 250.0;
    int bsf = -9;   // binary_scale_factor
    int dsf = 0;    // decimal_scale_factor
    int bpv = 16;   // bits_per_value

    // Quantise
    double scale = std::pow(10.0, dsf) / std::pow(2.0, bsf);
    std::vector<std::uint64_t> packed_ints(N);
    for (std::size_t i = 0; i < N; ++i) {
        packed_ints[i] = static_cast<std::uint64_t>(
            std::round((values[i] - ref_val) * scale));
    }

    auto packed_bytes = bit_pack(packed_ints, bpv);
    auto json = simple_packing_json(N, ref_val, bsf, dsf, bpv);

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {packed_bytes.data(), packed_bytes.size()}
    };
    auto encoded = tensogram::encode_pre_encoded(json, objects);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "float64");

    const double* p = obj.data_as<double>();
    double max_err = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        max_err = std::max(max_err, std::abs(p[i] - values[i]));
    }
    EXPECT_LT(max_err, 0.01) << "Max quantisation error: " << max_err;
}

// -----------------------------------------------------------------------
// Test 4: multiple objects
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, MultipleObjects) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {10.0f, 20.0f};

    std::string json = R"({"version":2,"descriptors":[)"
        R"({"type":"ndarray","ndim":1,"shape":[3],"strides":[4],)"
        R"("dtype":"float32","byte_order":"little",)"
        R"("encoding":"none","filter":"none","compression":"none"},)"
        R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],)"
        R"("dtype":"float32","byte_order":"little",)"
        R"("encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(a.data()), a.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(b.data()), b.size() * sizeof(float)},
    };
    auto encoded = tensogram::encode_pre_encoded(json, objects);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.num_objects(), 2u);

    auto obj0 = msg.object(0);
    EXPECT_EQ(obj0.element_count<float>(), 3u);
    const float* p0 = obj0.data_as<float>();
    EXPECT_FLOAT_EQ(p0[0], 1.0f);
    EXPECT_FLOAT_EQ(p0[1], 2.0f);
    EXPECT_FLOAT_EQ(p0[2], 3.0f);

    auto obj1 = msg.object(1);
    EXPECT_EQ(obj1.element_count<float>(), 2u);
    const float* p1 = obj1.data_as<float>();
    EXPECT_FLOAT_EQ(p1[0], 10.0f);
    EXPECT_FLOAT_EQ(p1[1], 20.0f);
}

// -----------------------------------------------------------------------
// Test 5: invalid descriptor rejection
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, RejectsUnknownEncoding) {
    std::string json = R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[4],)"
        R"("strides":[4],"dtype":"float32","byte_order":"little",)"
        R"("encoding":"bogus","filter":"none","compression":"none"}]})";

    std::vector<std::uint8_t> data(16, 0);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {data.data(), data.size()}
    };

    EXPECT_THROW(tensogram::encode_pre_encoded(json, objects), tensogram::error);
}

// -----------------------------------------------------------------------
// Test 6: zero objects
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, ZeroObjects) {
    std::string json = R"({"version":2,"descriptors":[]})";
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects;

    auto encoded = tensogram::encode_pre_encoded(json, objects);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.num_objects(), 0u);
}

// -----------------------------------------------------------------------
// Test 7: streaming encoder with write_object_pre_encoded
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, StreamingMixedMode) {
    TempFile tmp;
    std::string meta = R"({"version":2})";

    {
        tensogram::streaming_encoder enc(tmp.path, meta);

        // Object 0: normal encode
        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::string desc_a = R"({"type":"ndarray","ndim":1,"shape":[3],"strides":[4],)"
            R"("dtype":"float32","byte_order":"little","encoding":"none",)"
            R"("filter":"none","compression":"none"})";
        enc.write_object(desc_a,
                         reinterpret_cast<const std::uint8_t*>(a.data()),
                         a.size() * sizeof(float));

        // Object 1: pre-encoded (raw bytes, encoding=none)
        std::vector<float> b = {10.0f, 20.0f};
        std::string desc_b = R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],)"
            R"("dtype":"float32","byte_order":"little","encoding":"none",)"
            R"("filter":"none","compression":"none"})";
        enc.write_object_pre_encoded(desc_b,
                                     reinterpret_cast<const std::uint8_t*>(b.data()),
                                     b.size() * sizeof(float));

        enc.finish();
    }

    // Open the file and decode
    auto f = tensogram::file::open(tmp.path);
    ASSERT_EQ(f.message_count(), 1u);

    auto msg = f.decode_message(0);
    ASSERT_EQ(msg.num_objects(), 2u);

    auto obj0 = msg.object(0);
    EXPECT_EQ(obj0.element_count<float>(), 3u);
    const float* p0 = obj0.data_as<float>();
    EXPECT_FLOAT_EQ(p0[0], 1.0f);
    EXPECT_FLOAT_EQ(p0[1], 2.0f);
    EXPECT_FLOAT_EQ(p0[2], 3.0f);

    auto obj1 = msg.object(1);
    EXPECT_EQ(obj1.element_count<float>(), 2u);
    const float* p1 = obj1.data_as<float>();
    EXPECT_FLOAT_EQ(p1[0], 10.0f);
    EXPECT_FLOAT_EQ(p1[1], 20.0f);
}

// -----------------------------------------------------------------------
// Test 8: hash is recomputed (not from caller)
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, HashIsRecomputed) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    auto json = encoding_none_json(values.size());

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };

    // Encode with hash
    auto encoded = tensogram::encode_pre_encoded(json, objects);
    ASSERT_FALSE(encoded.empty());

    // Verify hash succeeds on decode
    tensogram::decode_options opts;
    opts.verify_hash = true;
    EXPECT_NO_THROW(tensogram::decode(encoded.data(), encoded.size(), opts));
}

// -----------------------------------------------------------------------
// Additional edge-case tests
// -----------------------------------------------------------------------

TEST(EncodePreEncodedTest, MalformedJsonDescriptor) {
    std::string json = R"({"version":2,"descriptors":[MALFORMED_JSON]})";
    std::vector<float> values = {1.0f, 2.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    EXPECT_ANY_THROW(tensogram::encode_pre_encoded(json, objects));
}

TEST(EncodePreEncodedTest, EmptyJsonDescriptor) {
    std::string json = R"({})";
    std::vector<float> values = {1.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    EXPECT_ANY_THROW(tensogram::encode_pre_encoded(json, objects));
}

TEST(EncodePreEncodedTest, DataSizeMismatch) {
    // encoding=none, shape=[10] float32 = 40 bytes expected, but pass only 20.
    std::string json = encoding_none_json(10, "float32");
    std::vector<std::uint8_t> short_data(20, 0);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {short_data.data(), short_data.size()}
    };
    EXPECT_ANY_THROW(tensogram::encode_pre_encoded(json, objects));
}

TEST(EncodePreEncodedTest, RoundTrip2D) {
    // 2D array [2, 3] of float32 (24 bytes)
    std::string json =
        R"({"version":2,"descriptors":[{"type":"ndarray","ndim":2,"shape":[2,3],)"
        R"("strides":[12,4],"dtype":"float32","byte_order":"little",)"
        R"("encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };

    auto encoded = tensogram::encode_pre_encoded(json, objects);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    ASSERT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.element_count<float>(), 6u);
    const float* p = obj.data_as<float>();
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[5], 6.0f);
}

TEST(EncodePreEncodedTest, StreamingPreEncodedOnly) {
    // Streaming encoder with ONLY pre-encoded writes (no write_object).
    TempFile tmp;
    std::string meta = R"({"version":2})";

    {
        tensogram::streaming_encoder enc(tmp.path, meta);

        std::vector<float> a = {10.0f, 20.0f, 30.0f, 40.0f};
        std::string desc_a = descriptor_json(a.size());

        enc.write_object_pre_encoded(desc_a,
                                     reinterpret_cast<const std::uint8_t*>(a.data()),
                                     a.size() * sizeof(float));
        enc.finish();
    }

    auto f = tensogram::file::open(tmp.path);
    ASSERT_EQ(f.message_count(), 1u);
    auto msg = f.decode_message(0);
    ASSERT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.element_count<float>(), 4u);
    const float* p = obj.data_as<float>();
    EXPECT_FLOAT_EQ(p[0], 10.0f);
    EXPECT_FLOAT_EQ(p[3], 40.0f);
}

TEST(EncodePreEncodedTest, StreamingMultiplePreEncoded) {
    // Streaming encoder with multiple pre-encoded objects.
    TempFile tmp;
    std::string meta = R"({"version":2})";

    {
        tensogram::streaming_encoder enc(tmp.path, meta);

        std::vector<float> a = {1.0f, 2.0f};
        std::vector<double> b = {100.0, 200.0, 300.0};

        std::string desc_a = descriptor_json(a.size(), "float32");
        std::string desc_b = descriptor_json(b.size(), "float64");

        enc.write_object_pre_encoded(desc_a,
                                     reinterpret_cast<const std::uint8_t*>(a.data()),
                                     a.size() * sizeof(float));
        enc.write_object_pre_encoded(desc_b,
                                     reinterpret_cast<const std::uint8_t*>(b.data()),
                                     b.size() * sizeof(double));
        enc.finish();
    }

    auto f = tensogram::file::open(tmp.path);
    ASSERT_EQ(f.message_count(), 1u);
    auto msg = f.decode_message(0);
    ASSERT_EQ(msg.num_objects(), 2u);

    auto obj0 = msg.object(0);
    EXPECT_EQ(obj0.element_count<float>(), 2u);
    EXPECT_FLOAT_EQ(obj0.data_as<float>()[0], 1.0f);

    auto obj1 = msg.object(1);
    EXPECT_EQ(obj1.element_count<double>(), 3u);
    EXPECT_DOUBLE_EQ(obj1.data_as<double>()[2], 300.0);
}

TEST(EncodePreEncodedTest, SingleElement) {
    // Single-element array.
    std::string json = encoding_none_json(1, "float32");
    float val = 42.0f;
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(&val), sizeof(float)}
    };

    auto encoded = tensogram::encode_pre_encoded(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    ASSERT_EQ(msg.num_objects(), 1u);
    EXPECT_FLOAT_EQ(msg.object(0).data_as<float>()[0], 42.0f);
}
