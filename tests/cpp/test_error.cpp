// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF
//
// Tests for error handling.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Garbage data throws framing_error
// ---------------------------------------------------------------------------

TEST(ErrorTest, GarbageDataThrowsFramingError) {
    std::vector<std::uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF, 0x00};
    EXPECT_THROW(
        (void)tensogram::decode(garbage.data(), garbage.size()),
        tensogram::framing_error);
}

// ---------------------------------------------------------------------------
// Empty data throws framing_error
// ---------------------------------------------------------------------------

TEST(ErrorTest, EmptyDataThrows) {
    std::vector<std::uint8_t> empty;
    // Empty vector's data() may be null, which triggers invalid_arg_error.
    EXPECT_THROW(
        (void)tensogram::decode(empty.data(), 0),
        tensogram::error);
}

// ---------------------------------------------------------------------------
// Truncated data throws framing_error
// ---------------------------------------------------------------------------

TEST(ErrorTest, TruncatedDataThrowsFramingError) {
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f, 3.0f});
    // Truncate to half the size
    const std::size_t half = encoded.size() / 2;
    EXPECT_THROW(
        (void)tensogram::decode(encoded.data(), half),
        tensogram::framing_error);
}

// ---------------------------------------------------------------------------
// error.code() returns correct enum value
// ---------------------------------------------------------------------------

TEST(ErrorTest, ErrorCodePreserved) {
    std::vector<std::uint8_t> garbage = {0x00, 0x01};
    try {
        (void)tensogram::decode(garbage.data(), garbage.size());
        FAIL() << "Expected framing_error";
    } catch (const tensogram::error& e) {
        EXPECT_EQ(e.code(), TGM_ERROR_FRAMING);
    }
}

// ---------------------------------------------------------------------------
// error.what() returns meaningful message
// ---------------------------------------------------------------------------

TEST(ErrorTest, ErrorWhatHasContent) {
    std::vector<std::uint8_t> garbage = {0x00};
    try {
        (void)tensogram::decode(garbage.data(), garbage.size());
        FAIL() << "Expected exception";
    } catch (const tensogram::error& e) {
        std::string msg = e.what();
        EXPECT_FALSE(msg.empty());
        // Should contain some descriptive text
        EXPECT_GT(msg.size(), 3u);
    }
}

// ---------------------------------------------------------------------------
// Error hierarchy: framing_error is-a error is-a runtime_error
// ---------------------------------------------------------------------------

TEST(ErrorTest, ErrorHierarchy) {
    std::vector<std::uint8_t> garbage = {0xFF};

    // Catch as framing_error
    bool caught_framing = false;
    try {
        (void)tensogram::decode(garbage.data(), garbage.size());
    } catch (const tensogram::framing_error&) {
        caught_framing = true;
    }
    EXPECT_TRUE(caught_framing);

    // Catch as tensogram::error
    bool caught_error = false;
    try {
        (void)tensogram::decode(garbage.data(), garbage.size());
    } catch (const tensogram::error&) {
        caught_error = true;
    }
    EXPECT_TRUE(caught_error);

    // Catch as std::runtime_error
    bool caught_runtime = false;
    try {
        (void)tensogram::decode(garbage.data(), garbage.size());
    } catch (const std::runtime_error&) {
        caught_runtime = true;
    }
    EXPECT_TRUE(caught_runtime);
}

// ---------------------------------------------------------------------------
// tgm_error_string returns non-null for all codes
// ---------------------------------------------------------------------------

TEST(ErrorTest, ErrorStringNonNull) {
    // Test all known error codes
    EXPECT_NE(tgm_error_string(TGM_ERROR_OK), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_FRAMING), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_METADATA), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_ENCODING), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_COMPRESSION), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_OBJECT), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_IO), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_HASH_MISMATCH), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_INVALID_ARG), nullptr);
    EXPECT_NE(tgm_error_string(TGM_ERROR_END_OF_ITER), nullptr);

    // Each should be non-empty
    EXPECT_GT(std::strlen(tgm_error_string(TGM_ERROR_FRAMING)), 0u);
}

// ---------------------------------------------------------------------------
// Invalid file path throws io_error
// ---------------------------------------------------------------------------

TEST(ErrorTest, InvalidFilePathThrowsIoError) {
    EXPECT_THROW(
        (void)tensogram::file::open("/nonexistent/path/to/file.tgm"),
        tensogram::io_error);
}

// ---------------------------------------------------------------------------
// Hash mismatch throws hash_mismatch_error (specific type, not just error)
// ---------------------------------------------------------------------------

TEST(ErrorTest, HashMismatchThrowsHashMismatchError) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    std::string json = test_helpers::simple_f32_json(values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    tensogram::encode_options enc_opts;
    enc_opts.hash_algo = "xxh3";
    auto encoded = tensogram::encode(json, objects, enc_opts);

    // The wire format stores payload data inside the last data object frame.
    // Corrupt a few bytes that are guaranteed to be in the payload area
    // (around 53% into the message, well past all header/index/hash frames).
    const std::size_t payload_offset = (encoded.size() * 53) / 100;
    ASSERT_GT(encoded.size(), 20u);
    encoded[payload_offset]     ^= 0xFF;
    encoded[payload_offset + 1] ^= 0xFF;

    tensogram::decode_options dec_opts;
    dec_opts.verify_hash = true;
    EXPECT_THROW(
        (void)tensogram::decode(encoded.data(), encoded.size(), dec_opts),
        tensogram::hash_mismatch_error);
}
