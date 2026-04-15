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
// Tests for tensogram::validate() and tensogram::validate_file().

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using test_helpers::TempFile;

// ---------------------------------------------------------------------------
// validate() — single message
// ---------------------------------------------------------------------------

TEST(ValidateTest, ValidMessage) {
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f, 3.0f});
    auto json = tensogram::validate(encoded.data(), encoded.size());
    EXPECT_NE(json.find("\"issues\""), std::string::npos);
    EXPECT_NE(json.find("\"object_count\""), std::string::npos);
    EXPECT_NE(json.find("\"hash_verified\""), std::string::npos);
    EXPECT_NE(json.find("\"issues\":[]"), std::string::npos);
}

TEST(ValidateTest, EmptyBuffer) {
    auto json = tensogram::validate(nullptr, 0);
    EXPECT_NE(json.find("\"buffer_too_short\""), std::string::npos);
}

TEST(ValidateTest, CorruptedMagic) {
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f});
    std::memcpy(encoded.data(), "WRONGMAG", 8);
    auto json = tensogram::validate(encoded.data(), encoded.size());
    EXPECT_NE(json.find("\"invalid_magic\""), std::string::npos);
}

TEST(ValidateTest, QuickLevel) {
    auto encoded = test_helpers::encode_simple_f32({1.0f});
    auto json = tensogram::validate(encoded.data(), encoded.size(), "quick");
    EXPECT_NE(json.find("\"hash_verified\":false"), std::string::npos);
}

TEST(ValidateTest, FullLevel) {
    auto encoded = test_helpers::encode_simple_f32({1.0f});
    auto json = tensogram::validate(encoded.data(), encoded.size(), "full");
    EXPECT_NE(json.find("\"issues\":[]"), std::string::npos);
}

TEST(ValidateTest, InvalidLevelThrows) {
    auto encoded = test_helpers::encode_simple_f32({1.0f});
    EXPECT_THROW(
        tensogram::validate(encoded.data(), encoded.size(), "bogus"),
        tensogram::invalid_arg_error);
}

TEST(ValidateTest, CheckCanonical) {
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f});
    auto json = tensogram::validate(encoded.data(), encoded.size(), "default", true);
    EXPECT_NE(json.find("\"issues\":[]"), std::string::npos);
}

// ---------------------------------------------------------------------------
// validate_file()
// ---------------------------------------------------------------------------

TEST(ValidateFileTest, ValidFile) {
    TempFile tmp;
    {
        auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f, 3.0f});
        auto f = tensogram::file::create(tmp.path);
        f.append_raw(encoded);
    }
    auto json = tensogram::validate_file(tmp.path.c_str());
    EXPECT_NE(json.find("\"file_issues\""), std::string::npos);
    EXPECT_NE(json.find("\"messages\""), std::string::npos);
}

TEST(ValidateFileTest, NonexistentFileThrows) {
    EXPECT_THROW(
        tensogram::validate_file("/nonexistent/path/to/file.tgm"),
        tensogram::io_error);
}

TEST(ValidateFileTest, InvalidLevelThrows) {
    TempFile tmp;
    {
        auto encoded = test_helpers::encode_simple_f32({1.0f});
        auto f = tensogram::file::create(tmp.path);
        f.append_raw(encoded);
    }
    EXPECT_THROW(
        tensogram::validate_file(tmp.path.c_str(), "bogus"),
        tensogram::invalid_arg_error);
}

TEST(ValidateFileTest, EmptyFile) {
    TempFile tmp;
    { std::ofstream(tmp.path, std::ios::binary); }
    auto json = tensogram::validate_file(tmp.path.c_str());
    EXPECT_NE(json.find("\"messages\":[]"), std::string::npos);
}
