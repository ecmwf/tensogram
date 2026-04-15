// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF
//
// Parametrized tests for all supported dtypes.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: generic dtype encode/decode round-trip
// ---------------------------------------------------------------------------

namespace {

/// Build JSON for a 1-D array of given dtype/elem_size/count.
std::string make_json(const std::string& dtype, std::size_t elem_size,
                      std::size_t count) {
    return R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[)" + std::to_string(elem_size) +
           R"(],"dtype":")" + dtype +
           R"(","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
}

/// Encode raw bytes with given dtype metadata, then decode and verify dtype/bytes.
void verify_dtype_roundtrip(const std::string& dtype, std::size_t elem_size,
                            const std::uint8_t* data, std::size_t count)
{
    auto json = make_json(dtype, elem_size, count);
    std::size_t total_bytes = count * elem_size;

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {data, total_bytes}
    };
    auto encoded = tensogram::encode(json, objects);
    ASSERT_FALSE(encoded.empty()) << "Encode failed for dtype=" << dtype;

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    ASSERT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), dtype);
    EXPECT_EQ(obj.data_size(), total_bytes);
    EXPECT_EQ(std::memcmp(obj.data(), data, total_bytes), 0)
        << "Data mismatch for dtype=" << dtype;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// float32
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Float32) {
    std::vector<float> values = {-1.5f, 0.0f, 1.5f, 3.14f};
    verify_dtype_roundtrip("float32", sizeof(float),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}

// ---------------------------------------------------------------------------
// float64
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Float64) {
    std::vector<double> values = {-1.5, 0.0, 1.5, 3.14159265};
    verify_dtype_roundtrip("float64", sizeof(double),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}

// ---------------------------------------------------------------------------
// int8
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Int8) {
    std::vector<std::int8_t> values = {-128, -1, 0, 1, 127};
    verify_dtype_roundtrip("int8", sizeof(std::int8_t),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}

// ---------------------------------------------------------------------------
// int16
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Int16) {
    std::vector<std::int16_t> values = {-32768, 0, 32767};
    verify_dtype_roundtrip("int16", sizeof(std::int16_t),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}

// ---------------------------------------------------------------------------
// int32
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Int32) {
    std::vector<std::int32_t> values = {-2147483648, 0, 2147483647};
    verify_dtype_roundtrip("int32", sizeof(std::int32_t),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}

// ---------------------------------------------------------------------------
// int64
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Int64) {
    std::vector<std::int64_t> values = {
        std::numeric_limits<std::int64_t>::min(),
        0,
        std::numeric_limits<std::int64_t>::max()
    };
    verify_dtype_roundtrip("int64", sizeof(std::int64_t),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}

// ---------------------------------------------------------------------------
// uint8
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Uint8) {
    std::vector<std::uint8_t> values = {0, 1, 128, 255};
    verify_dtype_roundtrip("uint8", sizeof(std::uint8_t),
        values.data(), values.size());
}

// ---------------------------------------------------------------------------
// uint16
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Uint16) {
    std::vector<std::uint16_t> values = {0, 1, 32768, 65535};
    verify_dtype_roundtrip("uint16", sizeof(std::uint16_t),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}

// ---------------------------------------------------------------------------
// uint32
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Uint32) {
    std::vector<std::uint32_t> values = {0, 1, 2147483648u, 4294967295u};
    verify_dtype_roundtrip("uint32", sizeof(std::uint32_t),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}

// ---------------------------------------------------------------------------
// uint64
// ---------------------------------------------------------------------------

TEST(MultiDtypeTest, Uint64) {
    std::vector<std::uint64_t> values = {
        0, 1,
        std::numeric_limits<std::uint64_t>::max()
    };
    verify_dtype_roundtrip("uint64", sizeof(std::uint64_t),
        reinterpret_cast<const std::uint8_t*>(values.data()), values.size());
}
