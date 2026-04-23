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
// Tests for all iterator types.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using test_helpers::TempFile;

// ---------------------------------------------------------------------------
// buffer_iterator: count
// ---------------------------------------------------------------------------

TEST(IteratorTest, BufferIteratorCount) {
    auto e1 = test_helpers::encode_simple_f32({1.0f, 2.0f});
    auto e2 = test_helpers::encode_simple_f32({3.0f, 4.0f, 5.0f});

    std::vector<std::uint8_t> combined;
    combined.insert(combined.end(), e1.begin(), e1.end());
    combined.insert(combined.end(), e2.begin(), e2.end());

    tensogram::buffer_iterator iter(combined.data(), combined.size());
    EXPECT_EQ(iter.count(), 2u);
}

// ---------------------------------------------------------------------------
// buffer_iterator: next loop
// ---------------------------------------------------------------------------

TEST(IteratorTest, BufferIteratorNextLoop) {
    auto e1 = test_helpers::encode_simple_f32({10.0f});
    auto e2 = test_helpers::encode_simple_f32({20.0f});
    auto e3 = test_helpers::encode_simple_f32({30.0f});

    std::vector<std::uint8_t> combined;
    combined.insert(combined.end(), e1.begin(), e1.end());
    combined.insert(combined.end(), e2.begin(), e2.end());
    combined.insert(combined.end(), e3.begin(), e3.end());

    tensogram::buffer_iterator iter(combined.data(), combined.size());

    const std::uint8_t* buf = nullptr;
    std::size_t len = 0;
    std::vector<float> first_values;
    std::size_t count = 0;

    while (iter.next(buf, len)) {
        auto msg = tensogram::decode(buf, len);
        auto obj = msg.object(0);
        first_values.push_back(obj.data_as<float>()[0]);
        ++count;
    }

    EXPECT_EQ(count, 3u);
    ASSERT_EQ(first_values.size(), 3u);
    EXPECT_FLOAT_EQ(first_values[0], 10.0f);
    EXPECT_FLOAT_EQ(first_values[1], 20.0f);
    EXPECT_FLOAT_EQ(first_values[2], 30.0f);
}

// ---------------------------------------------------------------------------
// buffer_iterator: single message
// ---------------------------------------------------------------------------

TEST(IteratorTest, BufferIteratorSingleMessage) {
    auto encoded = test_helpers::encode_simple_f32({99.0f});
    tensogram::buffer_iterator iter(encoded.data(), encoded.size());
    EXPECT_EQ(iter.count(), 1u);

    const std::uint8_t* buf = nullptr;
    std::size_t len = 0;
    ASSERT_TRUE(iter.next(buf, len));
    EXPECT_EQ(len, encoded.size());
    EXPECT_FALSE(iter.next(buf, len));
}

// ---------------------------------------------------------------------------
// file_iterator: iterate file messages
// ---------------------------------------------------------------------------

TEST(IteratorTest, FileIteratorBasic) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);

    for (int i = 0; i < 4; ++i) {
        auto encoded = test_helpers::encode_simple_f32(
            {static_cast<float>(i)});
        f.append_raw(encoded);
    }

    tensogram::file_iterator iter(f);
    std::vector<std::uint8_t> raw;
    std::size_t count = 0;
    while (iter.next(raw)) {
        auto msg = tensogram::decode(raw.data(), raw.size());
        EXPECT_EQ(msg.num_objects(), 1u);
        ++count;
    }
    EXPECT_EQ(count, 4u);
}

// ---------------------------------------------------------------------------
// file_iterator: empty file
// ---------------------------------------------------------------------------

TEST(IteratorTest, FileIteratorEmptyFile) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);

    tensogram::file_iterator iter(f);
    std::vector<std::uint8_t> raw;
    EXPECT_FALSE(iter.next(raw));
}

// ---------------------------------------------------------------------------
// object_iterator: iterate objects in a message
// ---------------------------------------------------------------------------

TEST(IteratorTest, ObjectIteratorSingleObject) {
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f, 3.0f});

    // object_iterator::next() requires a pre-constructed message as out-param.
    auto dummy = tensogram::decode(encoded.data(), encoded.size());

    tensogram::object_iterator iter(encoded.data(), encoded.size());
    std::size_t count = 0;
    while (iter.next(dummy)) {
        EXPECT_EQ(dummy.num_objects(), 1u);
        auto obj = dummy.object(0);
        EXPECT_EQ(obj.dtype_string(), "float32");
        ++count;
    }
    EXPECT_EQ(count, 1u);
}

// ---------------------------------------------------------------------------
// object_iterator: multi-object message
// ---------------------------------------------------------------------------

TEST(IteratorTest, ObjectIteratorMultiObject) {
    // Build a 2-object message
    std::string json = R"({"version":3,"descriptors":[)"
        R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"},)"
        R"({"type":"ndarray","ndim":1,"shape":[3],"strides":[8],"dtype":"float64","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<float> f_values = {1.0f, 2.0f};
    std::vector<double> d_values = {10.0, 20.0, 30.0};

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(f_values.data()),
         f_values.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(d_values.data()),
         d_values.size() * sizeof(double)}
    };

    auto encoded = tensogram::encode(json, objects);

    // Need a message to receive each iteration
    auto dummy = tensogram::decode(encoded.data(), encoded.size());
    tensogram::object_iterator iter(encoded.data(), encoded.size());
    std::size_t count = 0;
    std::vector<std::string> dtypes;
    while (iter.next(dummy)) {
        dtypes.push_back(dummy.object(0).dtype_string());
        ++count;
    }
    EXPECT_EQ(count, 2u);
    ASSERT_EQ(dtypes.size(), 2u);
    EXPECT_EQ(dtypes[0], "float32");
    EXPECT_EQ(dtypes[1], "float64");
}

// ---------------------------------------------------------------------------
// message::begin/end range-based for loop
// ---------------------------------------------------------------------------

TEST(IteratorTest, MessageRangeBasedFor) {
    // Multi-object message
    std::string json = R"({"version":3,"descriptors":[)"
        R"({"type":"ndarray","ndim":1,"shape":[1],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"},)"
        R"({"type":"ndarray","ndim":1,"shape":[1],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<float> v1 = {42.0f};
    std::vector<float> v2 = {99.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(v1.data()), sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(v2.data()), sizeof(float)}
    };

    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());

    std::size_t count = 0;
    for (const auto& obj : msg) {
        EXPECT_EQ(obj.dtype_string(), "float32");
        ++count;
    }
    EXPECT_EQ(count, 2u);
}

// ---------------------------------------------------------------------------
// Iterator RAII (no leaks)
// ---------------------------------------------------------------------------

TEST(IteratorTest, IteratorRAII) {
    auto encoded = test_helpers::encode_simple_f32({1.0f});

    // Create and immediately destroy — should not leak
    {
        tensogram::buffer_iterator iter(encoded.data(), encoded.size());
    }

    {
        tensogram::object_iterator iter(encoded.data(), encoded.size());
    }

    // If we get here without crash/ASAN errors, RAII is working
    SUCCEED();
}
