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
// Tests for tensogram::streaming_encoder.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using test_helpers::TempFile;

// ---------------------------------------------------------------------------
// Helper: descriptor JSON
// ---------------------------------------------------------------------------

namespace {

/// Per-object descriptor JSON for a 1-D float32 array.
std::string f32_descriptor(std::size_t count) {
    return R"({"type":"ndarray","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"})";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Create, write objects, finish, then open and decode
// ---------------------------------------------------------------------------

TEST(StreamingTest, CreateWriteFinishDecode) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    {
        tensogram::streaming_encoder enc(tmp.path, meta_json);

        std::vector<float> values = {1.0f, 2.0f, 3.0f};
        enc.write_object(
            f32_descriptor(values.size()),
            reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size() * sizeof(float));

        enc.finish();
    }

    // Open the file and decode
    auto f = tensogram::file::open(tmp.path);
    EXPECT_EQ(f.message_count(), 1u);

    auto msg = f.decode_message(0);
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "float32");
    EXPECT_EQ(obj.element_count<float>(), 3u);
    const float* p = obj.data_as<float>();
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 3.0f);
}

// ---------------------------------------------------------------------------
// object_count increments after each write
// ---------------------------------------------------------------------------

TEST(StreamingTest, ObjectCountIncrements) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    tensogram::streaming_encoder enc(tmp.path, meta_json);

    EXPECT_EQ(enc.object_count(), 0u);

    std::vector<float> v1 = {1.0f};
    enc.write_object(
        f32_descriptor(v1.size()),
        reinterpret_cast<const std::uint8_t*>(v1.data()),
        v1.size() * sizeof(float));
    EXPECT_EQ(enc.object_count(), 1u);

    std::vector<float> v2 = {2.0f, 3.0f};
    enc.write_object(
        f32_descriptor(v2.size()),
        reinterpret_cast<const std::uint8_t*>(v2.data()),
        v2.size() * sizeof(float));
    EXPECT_EQ(enc.object_count(), 2u);

    enc.finish();
}

// ---------------------------------------------------------------------------
// Multi-object streaming write
// ---------------------------------------------------------------------------

TEST(StreamingTest, MultiObjectStreaming) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    tensogram::streaming_encoder enc(tmp.path, meta_json);

    for (int i = 0; i < 5; ++i) {
        std::vector<float> values = {static_cast<float>(i * 10)};
        enc.write_object(
            f32_descriptor(values.size()),
            reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size() * sizeof(float));
    }
    enc.finish();

    auto f = tensogram::file::open(tmp.path);
    auto msg = f.decode_message(0);
    ASSERT_EQ(msg.num_objects(), 5u);

    for (std::size_t i = 0; i < 5; ++i) {
        auto obj = msg.object(i);
        EXPECT_FLOAT_EQ(obj.data_as<float>()[0],
                        static_cast<float>(i * 10));
    }
}

// ---------------------------------------------------------------------------
// Read back the streamed file and verify data
// ---------------------------------------------------------------------------

TEST(StreamingTest, ReadBackVerifyData) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    {
        tensogram::streaming_encoder enc(tmp.path, meta_json);

        std::vector<float> v1 = {42.0f, 43.0f};
        std::vector<float> v2 = {99.0f};
        enc.write_object(
            f32_descriptor(v1.size()),
            reinterpret_cast<const std::uint8_t*>(v1.data()),
            v1.size() * sizeof(float));
        enc.write_object(
            f32_descriptor(v2.size()),
            reinterpret_cast<const std::uint8_t*>(v2.data()),
            v2.size() * sizeof(float));
        enc.finish();
    }

    // Read raw bytes, then decode
    auto f = tensogram::file::open(tmp.path);
    auto raw = f.read_message(0);
    ASSERT_FALSE(raw.empty());

    auto msg = tensogram::decode(raw.data(), raw.size());
    ASSERT_EQ(msg.num_objects(), 2u);

    auto obj0 = msg.object(0);
    EXPECT_EQ(obj0.element_count<float>(), 2u);
    EXPECT_FLOAT_EQ(obj0.data_as<float>()[0], 42.0f);

    auto obj1 = msg.object(1);
    EXPECT_EQ(obj1.element_count<float>(), 1u);
    EXPECT_FLOAT_EQ(obj1.data_as<float>()[0], 99.0f);
}

// ---------------------------------------------------------------------------
// Streaming encoder RAII (destructor frees without crash)
// ---------------------------------------------------------------------------

TEST(StreamingTest, StreamingEncoderRAII) {
    TempFile tmp;

    // Create and immediately destroy WITHOUT calling finish()
    {
        std::string meta_json = R"({"version":2})";
        tensogram::streaming_encoder enc(tmp.path, meta_json);

        std::vector<float> values = {1.0f};
        enc.write_object(
            f32_descriptor(values.size()),
            reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size() * sizeof(float));
        // enc goes out of scope without finish() — destructor should free
    }

    // If we get here without crash, RAII is working
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Streaming with hash algorithm
// ---------------------------------------------------------------------------

TEST(StreamingTest, StreamingWithHash) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    tensogram::encode_options opts;
    opts.hash_algo = "xxh3";

    {
        tensogram::streaming_encoder enc(tmp.path, meta_json, opts);

        std::vector<float> values = {1.0f, 2.0f};
        enc.write_object(
            f32_descriptor(values.size()),
            reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size() * sizeof(float));
        enc.finish();
    }

    auto f = tensogram::file::open(tmp.path);
    tensogram::decode_options dec_opts;
    dec_opts.verify_hash = true;
    auto msg = f.decode_message(0, dec_opts);
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_TRUE(obj.has_hash());
    EXPECT_EQ(obj.hash_type(), "xxh3");
    EXPECT_FALSE(obj.hash_value().empty());
}

// ---------------------------------------------------------------------------
// Preceder metadata: write_preceder + write_object round-trip
// ---------------------------------------------------------------------------

TEST(StreamingTest, PrecederRoundTrip) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    {
        tensogram::streaming_encoder enc(tmp.path, meta_json);

        // Write preceder for first object
        enc.write_preceder(R"({"mars":{"param":"2t"},"units":"K"})");

        std::vector<float> values = {1.0f, 2.0f, 3.0f};
        enc.write_object(
            f32_descriptor(values.size()),
            reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size() * sizeof(float));

        enc.finish();
    }

    // Open and decode — verify data survives the preceder + data object path.
    // (Preceder metadata merges into base[0] on the Rust side;
    // the C++ metadata API accesses per-object keys via base entries.)
    auto f = tensogram::file::open(tmp.path);
    auto msg = f.decode_message(0);
    EXPECT_EQ(msg.num_objects(), 1u);

    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "float32");
    EXPECT_EQ(obj.element_count<float>(), 3u);
    EXPECT_FLOAT_EQ(obj.data_as<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(obj.data_as<float>()[1], 2.0f);
    EXPECT_FLOAT_EQ(obj.data_as<float>()[2], 3.0f);
}

// ---------------------------------------------------------------------------
// Preceder: mixed objects (some with, some without)
// ---------------------------------------------------------------------------

TEST(StreamingTest, PrecederMixedObjects) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    {
        tensogram::streaming_encoder enc(tmp.path, meta_json);

        // Object 0: with preceder
        enc.write_preceder(R"({"note":"obj0"})");
        std::vector<float> v0 = {10.0f};
        enc.write_object(
            f32_descriptor(v0.size()),
            reinterpret_cast<const std::uint8_t*>(v0.data()),
            v0.size() * sizeof(float));

        // Object 1: no preceder
        std::vector<float> v1 = {20.0f};
        enc.write_object(
            f32_descriptor(v1.size()),
            reinterpret_cast<const std::uint8_t*>(v1.data()),
            v1.size() * sizeof(float));

        enc.finish();
    }

    auto f = tensogram::file::open(tmp.path);
    auto msg = f.decode_message(0);
    ASSERT_EQ(msg.num_objects(), 2u);
    EXPECT_FLOAT_EQ(msg.object(0).data_as<float>()[0], 10.0f);
    EXPECT_FLOAT_EQ(msg.object(1).data_as<float>()[0], 20.0f);
}

// ---------------------------------------------------------------------------
// Preceder: consecutive write_preceder without write_object throws
// ---------------------------------------------------------------------------

TEST(StreamingTest, PrecederDoubleWriteThrows) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    tensogram::streaming_encoder enc(tmp.path, meta_json);

    enc.write_preceder(R"({})");
    EXPECT_THROW(enc.write_preceder(R"({})"), tensogram::framing_error);
}

// ---------------------------------------------------------------------------
// Preceder: non-object JSON throws
// ---------------------------------------------------------------------------

TEST(StreamingTest, PrecederNonObjectJsonThrows) {
    TempFile tmp;

    std::string meta_json = R"({"version":2})";
    tensogram::streaming_encoder enc(tmp.path, meta_json);

    EXPECT_THROW(enc.write_preceder(R"([1,2,3])"), tensogram::metadata_error);
}
