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
// Tests for tensogram::decoded_object properties.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// ndim
// ---------------------------------------------------------------------------

TEST(DescriptorTest, Ndim1D) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.ndim(), 1u);
}

TEST(DescriptorTest, Ndim2D) {
    // 2x3 float32 array
    std::string json = R"({"version":2,"descriptors":[{"type":"ndarray","ndim":2,"shape":[2,3],"strides":[12,4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.ndim(), 2u);
}

// ---------------------------------------------------------------------------
// shape and strides
// ---------------------------------------------------------------------------

TEST(DescriptorTest, ShapeAndStrides) {
    std::string json = R"({"version":2,"descriptors":[{"type":"ndarray","ndim":2,"shape":[2,3],"strides":[12,4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);

    auto shape = obj.shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 2u);
    EXPECT_EQ(shape[1], 3u);

    auto strides = obj.strides();
    ASSERT_EQ(strides.size(), 2u);
    EXPECT_EQ(strides[0], 12u);
    EXPECT_EQ(strides[1], 4u);
}

// ---------------------------------------------------------------------------
// dtype_string
// ---------------------------------------------------------------------------

TEST(DescriptorTest, DtypeStringFloat32) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.object(0).dtype_string(), "float32");
}

TEST(DescriptorTest, DtypeStringFloat64) {
    std::string json = R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[2],"strides":[8],"dtype":"float64","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
    std::vector<double> values = {1.0, 2.0};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(double)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.object(0).dtype_string(), "float64");
}

// ---------------------------------------------------------------------------
// object_type
// ---------------------------------------------------------------------------

TEST(DescriptorTest, ObjectTypeNdarray) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.object(0).object_type(), "ndarray");
}

// ---------------------------------------------------------------------------
// byte_order_string
// ---------------------------------------------------------------------------

TEST(DescriptorTest, ByteOrderLittle) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.object(0).byte_order_string(), "little");
}

// ---------------------------------------------------------------------------
// encoding, filter, compression
// ---------------------------------------------------------------------------

TEST(DescriptorTest, EncodingFilterCompression) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.encoding(), "none");
    EXPECT_EQ(obj.filter(), "none");
    EXPECT_EQ(obj.compression(), "none");
}

// ---------------------------------------------------------------------------
// has_hash, hash_type, hash_value (no hash)
// ---------------------------------------------------------------------------

TEST(DescriptorTest, NoHash) {
    tensogram::encode_options opts;
    opts.hash_algo = "";
    std::string json = test_helpers::simple_f32_json(1);
    std::vector<float> values = {1.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    auto encoded = tensogram::encode(json, objects, opts);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_FALSE(obj.has_hash());
}

// ---------------------------------------------------------------------------
// data_as<T> typed access
// ---------------------------------------------------------------------------

TEST(DescriptorTest, DataAsFloat) {
    std::vector<float> values = {42.0f, 99.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    const float* p = obj.data_as<float>();
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p[0], 42.0f);
    EXPECT_FLOAT_EQ(p[1], 99.0f);
}

// ---------------------------------------------------------------------------
// element_count<T>
// ---------------------------------------------------------------------------

TEST(DescriptorTest, ElementCount) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.element_count<float>(), 5u);
    // data_size should be 5 * 4 = 20
    EXPECT_EQ(obj.data_size(), 20u);
}
