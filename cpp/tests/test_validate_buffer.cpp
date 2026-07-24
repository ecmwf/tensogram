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
// Tests for tensogram::validate_buffer() — the multi-message, in-memory
// companion to validate_file().  Binds the C ABI tgm_validate_buffer.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace {

// Count non-overlapping occurrences of a needle in a haystack.
std::size_t count_substr(const std::string& hay, const std::string& needle) {
    std::size_t count = 0;
    for (std::size_t pos = hay.find(needle); pos != std::string::npos;
         pos = hay.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

}  // namespace

TEST(ValidateBufferTest, SingleMessageShapesReport) {
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f, 3.0f});
    auto json = tensogram::validate_buffer(encoded.data(), encoded.size());
    // FileValidationReport shape: file-level issues + per-message reports.
    EXPECT_NE(json.find("\"file_issues\""), std::string::npos);
    EXPECT_NE(json.find("\"messages\""), std::string::npos);
    // One clean message: no file-level issues, one per-message report.
    EXPECT_NE(json.find("\"file_issues\":[]"), std::string::npos);
    EXPECT_EQ(count_substr(json, "\"object_count\""), 1u);
}

TEST(ValidateBufferTest, MultiMessageValidatesEach) {
    auto a = test_helpers::encode_simple_f32({1.0f, 2.0f});
    auto b = test_helpers::encode_simple_f32({3.0f, 4.0f, 5.0f});
    auto c = test_helpers::encode_simple_f32({6.0f});
    std::vector<std::uint8_t> concat;
    concat.insert(concat.end(), a.begin(), a.end());
    concat.insert(concat.end(), b.begin(), b.end());
    concat.insert(concat.end(), c.begin(), c.end());

    auto json = tensogram::validate_buffer(concat.data(), concat.size());
    EXPECT_NE(json.find("\"file_issues\":[]"), std::string::npos);
    // Three clean messages → three per-message "object_count" fields.
    EXPECT_EQ(count_substr(json, "\"object_count\""), 3u);
}

TEST(ValidateBufferTest, TrailingJunkFlaggedAtFileLevel) {
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f});
    std::vector<std::uint8_t> buf(encoded.begin(), encoded.end());
    // Append bytes that do not form a message — a file-level anomaly.
    for (int i = 0; i < 32; ++i) buf.push_back(0xEE);

    auto json = tensogram::validate_buffer(buf.data(), buf.size());
    // The single real message still validates; the tail is a file issue,
    // so file_issues is non-empty.
    EXPECT_EQ(json.find("\"file_issues\":[]"), std::string::npos);
    EXPECT_NE(json.find("\"file_issues\""), std::string::npos);
}

TEST(ValidateBufferTest, EmptyBufferHasNoMessages) {
    auto json = tensogram::validate_buffer(nullptr, 0);
    EXPECT_NE(json.find("\"messages\":[]"), std::string::npos);
}

TEST(ValidateBufferTest, LevelsAccepted) {
    auto encoded = test_helpers::encode_simple_f32({1.0f});
    for (const char* level : {"quick", "default", "checksum", "full"}) {
        auto json = tensogram::validate_buffer(encoded.data(), encoded.size(), level);
        EXPECT_NE(json.find("\"messages\""), std::string::npos) << level;
    }
}

TEST(ValidateBufferTest, StringViewLevelIsNulTerminatedSafely) {
    auto encoded = test_helpers::encode_simple_f32({1.0f});
    // Build a level from a substring of a longer string to make sure the
    // std::string_view -> C string conversion does not read past the level.
    const std::string levels = "defaultXXXX";
    std::string_view level(levels.data(), 7);  // "default"
    auto json = tensogram::validate_buffer(encoded.data(), encoded.size(), level);
    EXPECT_NE(json.find("\"messages\""), std::string::npos);
}

TEST(ValidateBufferTest, InvalidLevelThrows) {
    auto encoded = test_helpers::encode_simple_f32({1.0f});
    EXPECT_THROW(
        tensogram::validate_buffer(encoded.data(), encoded.size(), "bogus"),
        tensogram::invalid_arg_error);
}

TEST(ValidateBufferTest, NullBufferNonZeroLenThrows) {
    EXPECT_THROW(
        tensogram::validate_buffer(nullptr, 42),
        tensogram::invalid_arg_error);
}

TEST(ValidateBufferTest, CheckCanonicalRuns) {
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f});
    auto json = tensogram::validate_buffer(encoded.data(), encoded.size(),
                                           "default", /*check_canonical=*/true);
    EXPECT_NE(json.find("\"file_issues\":[]"), std::string::npos);
}
