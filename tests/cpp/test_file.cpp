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
// Tests for tensogram::file class.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using test_helpers::TempFile;

// ---------------------------------------------------------------------------
// create + path() returns correct path
// ---------------------------------------------------------------------------

TEST(FileTest, CreateAndPath) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);
    EXPECT_EQ(f.path(), tmp.path);
}

// ---------------------------------------------------------------------------
// append_raw + message_count
// ---------------------------------------------------------------------------

TEST(FileTest, AppendRawAndMessageCount) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);

    std::vector<float> values = {1.0f, 2.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    f.append_raw(encoded);

    EXPECT_EQ(f.message_count(), 1u);
}

// ---------------------------------------------------------------------------
// decode_message from file
// ---------------------------------------------------------------------------

TEST(FileTest, DecodeMessageFromFile) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);

    std::vector<float> values = {10.0f, 20.0f, 30.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    f.append_raw(encoded);

    auto msg = f.decode_message(0);
    EXPECT_EQ(msg.num_objects(), 1u);
    auto obj = msg.object(0);
    const float* p = obj.data_as<float>();
    EXPECT_FLOAT_EQ(p[0], 10.0f);
    EXPECT_FLOAT_EQ(p[2], 30.0f);
}

// ---------------------------------------------------------------------------
// read_message returns raw bytes that can be decoded
// ---------------------------------------------------------------------------

TEST(FileTest, ReadMessageRawBytes) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);

    std::vector<float> values = {5.0f, 6.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    f.append_raw(encoded);

    auto raw = f.read_message(0);
    ASSERT_FALSE(raw.empty());

    // Decode the raw bytes independently
    auto msg = tensogram::decode(raw.data(), raw.size());
    EXPECT_EQ(msg.num_objects(), 1u);
    auto obj = msg.object(0);
    EXPECT_FLOAT_EQ(obj.data_as<float>()[0], 5.0f);
}

// ---------------------------------------------------------------------------
// open existing file
// ---------------------------------------------------------------------------

TEST(FileTest, OpenExistingFile) {
    TempFile tmp;

    // Create and populate
    {
        auto f = tensogram::file::create(tmp.path);
        auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f});
        f.append_raw(encoded);
    }

    // Re-open
    auto f = tensogram::file::open(tmp.path);
    EXPECT_EQ(f.message_count(), 1u);
    auto msg = f.decode_message(0);
    EXPECT_EQ(msg.num_objects(), 1u);
}

// ---------------------------------------------------------------------------
// Multiple messages in a file
// ---------------------------------------------------------------------------

TEST(FileTest, MultipleMessages) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);

    for (int i = 0; i < 5; ++i) {
        std::vector<float> values = {static_cast<float>(i)};
        auto encoded = test_helpers::encode_simple_f32(values);
        f.append_raw(encoded);
    }

    EXPECT_EQ(f.message_count(), 5u);

    for (std::size_t i = 0; i < 5; ++i) {
        auto msg = f.decode_message(i);
        auto obj = msg.object(0);
        EXPECT_FLOAT_EQ(obj.data_as<float>()[0], static_cast<float>(i));
    }
}

// ---------------------------------------------------------------------------
// File not found throws io_error
// ---------------------------------------------------------------------------

TEST(FileTest, FileNotFoundThrows) {
    EXPECT_THROW(
        (void)tensogram::file::open("/tmp/nonexistent_tensogram_file_XXXXXX.tgm"),
        tensogram::io_error);
}

// ---------------------------------------------------------------------------
// RAII cleanup (file is closed on destruction)
// ---------------------------------------------------------------------------

TEST(FileTest, RAIICleanup) {
    TempFile tmp;
    {
        auto f = tensogram::file::create(tmp.path);
        auto encoded = test_helpers::encode_simple_f32({1.0f});
        f.append_raw(encoded);
        // f goes out of scope here — file should be closed
    }
    // Re-open should succeed (file was properly closed)
    auto f = tensogram::file::open(tmp.path);
    EXPECT_EQ(f.message_count(), 1u);
}

// ---------------------------------------------------------------------------
// append() with encoding (encode-and-append)
// ---------------------------------------------------------------------------

TEST(FileTest, AppendEncodeAndAppend) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);

    std::vector<float> values = {100.0f, 200.0f, 300.0f};
    std::string json = test_helpers::simple_f32_json(values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    f.append(json, objects);

    EXPECT_EQ(f.message_count(), 1u);

    auto msg = f.decode_message(0);
    auto obj = msg.object(0);
    const float* p = obj.data_as<float>();
    EXPECT_FLOAT_EQ(p[0], 100.0f);
    EXPECT_FLOAT_EQ(p[1], 200.0f);
    EXPECT_FLOAT_EQ(p[2], 300.0f);
}

// ---------------------------------------------------------------------------
// file_iterator: iterate all messages
// ---------------------------------------------------------------------------

TEST(FileTest, FileIteratorIterateAll) {
    TempFile tmp;
    auto f = tensogram::file::create(tmp.path);

    for (int i = 0; i < 3; ++i) {
        std::vector<float> values = {static_cast<float>(i * 10)};
        auto encoded = test_helpers::encode_simple_f32(values);
        f.append_raw(encoded);
    }

    tensogram::file_iterator iter(f);
    std::vector<std::uint8_t> raw;
    std::size_t count = 0;
    while (iter.next(raw)) {
        auto msg = tensogram::decode(raw.data(), raw.size());
        auto obj = msg.object(0);
        EXPECT_FLOAT_EQ(obj.data_as<float>()[0],
                        static_cast<float>(count * 10));
        ++count;
    }
    EXPECT_EQ(count, 3u);
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

TEST(FileTest, MoveSemantics) {
    TempFile tmp;
    auto f1 = tensogram::file::create(tmp.path);
    auto encoded = test_helpers::encode_simple_f32({7.0f});
    f1.append_raw(encoded);

    // Move construct
    auto f2 = std::move(f1);
    EXPECT_EQ(f2.message_count(), 1u);

    auto msg = f2.decode_message(0);
    EXPECT_FLOAT_EQ(msg.object(0).data_as<float>()[0], 7.0f);
}
