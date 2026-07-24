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
// Tests for the reader-side scan family:
//   * tensogram::scan_with_options   (tgm_scan_with_options)
//   * tensogram::scan_file           (tgm_scan_file)
//   * tensogram::scan_file_with_options (tgm_scan_file_with_options)
//   * tensogram::data_object_inline_hashes (tgm_data_object_inline_hashes)

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using test_helpers::TempFile;

namespace {

std::vector<std::uint8_t> concat(const std::vector<std::uint8_t>& a,
                                 const std::vector<std::uint8_t>& b) {
    std::vector<std::uint8_t> out(a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

bool is_lower_hex(const std::string& s) {
    for (char c : s) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!ok) return false;
    }
    return true;
}

// Encode a single 1-D float32 message, optionally without an inline hash.
std::vector<std::uint8_t> encode_one(const std::vector<float>& values,
                                     bool with_hash = true) {
    auto json = test_helpers::simple_f32_json(values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}};
    tensogram::encode_options opts;
    if (!with_hash) opts.hash_algo = "";
    return tensogram::encode(json, objects, opts);
}

}  // namespace

// ---------------------------------------------------------------------------
// scan_with_options
// ---------------------------------------------------------------------------

TEST(ScanWithOptionsTest, DefaultMatchesScan) {
    auto a = encode_one({1.0f});
    auto b = encode_one({2.0f, 3.0f});
    auto buf = concat(a, b);

    auto plain = tensogram::scan(buf.data(), buf.size());
    auto opted = tensogram::scan_with_options(buf.data(), buf.size(), {});
    ASSERT_EQ(opted.size(), 2u);
    EXPECT_EQ(opted[0].offset, 0u);
    EXPECT_EQ(opted[1].offset, a.size());
    ASSERT_EQ(plain.size(), opted.size());
    EXPECT_EQ(plain[1].offset, opted[1].offset);
    EXPECT_EQ(plain[1].length, opted[1].length);
}

TEST(ScanWithOptionsTest, ForwardOnlyStillFindsBoth) {
    auto a = encode_one({1.0f});
    auto b = encode_one({2.0f});
    auto buf = concat(a, b);

    tensogram::scan_options opts;
    opts.bidirectional = false;
    auto entries = tensogram::scan_with_options(buf.data(), buf.size(), opts);
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[1].offset, a.size());
}

TEST(ScanWithOptionsTest, MaxMessageSizeHonoured) {
    auto a = encode_one({1.0f});
    auto b = encode_one({2.0f});
    auto buf = concat(a, b);

    tensogram::scan_options opts;
    opts.max_message_size = 1024 * 1024;  // 1 MiB cap, well above our tiny msgs
    auto entries = tensogram::scan_with_options(buf.data(), buf.size(), opts);
    EXPECT_EQ(entries.size(), 2u);
}

TEST(ScanWithOptionsTest, NonMessageBufferYieldsNothing) {
    std::vector<std::uint8_t> junk(32, 0xAB);
    auto entries = tensogram::scan_with_options(junk.data(), junk.size(), {});
    EXPECT_TRUE(entries.empty());
}

// ---------------------------------------------------------------------------
// scan_file / scan_file_with_options
// ---------------------------------------------------------------------------

TEST(ScanFileTest, Roundtrip) {
    auto a = encode_one({1.0f});
    auto b = encode_one({2.0f, 3.0f});
    TempFile tmp;
    {
        auto f = tensogram::file::create(tmp.path);
        f.append_raw(a);
        f.append_raw(b);
    }

    auto entries = tensogram::scan_file(tmp.path);
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].offset, 0u);
    EXPECT_EQ(entries[1].offset, a.size());
}

TEST(ScanFileTest, WithOptionsForwardOnly) {
    auto a = encode_one({1.0f});
    auto b = encode_one({2.0f});
    TempFile tmp;
    {
        auto f = tensogram::file::create(tmp.path);
        f.append_raw(a);
        f.append_raw(b);
    }

    tensogram::scan_options opts;
    opts.bidirectional = false;
    opts.max_message_size = 1024 * 1024;
    auto entries = tensogram::scan_file_with_options(tmp.path, opts);
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[1].offset, a.size());
}

TEST(ScanFileTest, NonexistentThrows) {
    EXPECT_THROW(
        tensogram::scan_file("/nonexistent/path/wave_c_scan.tgm"),
        tensogram::io_error);
}

// ---------------------------------------------------------------------------
// data_object_inline_hashes
// ---------------------------------------------------------------------------

TEST(InlineHashesTest, HashedObjectReturnsHexDigest) {
    auto encoded = encode_one({1.0f, 2.0f, 3.0f}, /*with_hash=*/true);
    auto hashes = tensogram::data_object_inline_hashes(encoded.data(), encoded.size());
    ASSERT_EQ(hashes.size(), 1u);
    ASSERT_TRUE(hashes[0].has_value());
    EXPECT_EQ(hashes[0]->size(), 16u);
    EXPECT_TRUE(is_lower_hex(*hashes[0]));

    // Parity: the inline digest equals the decoded object's hash_value().
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(*hashes[0], msg.object(0).hash_value());
}

TEST(InlineHashesTest, UnhashedObjectReturnsNullopt) {
    auto encoded = encode_one({1.0f, 2.0f}, /*with_hash=*/false);
    auto hashes = tensogram::data_object_inline_hashes(encoded.data(), encoded.size());
    ASSERT_EQ(hashes.size(), 1u);
    EXPECT_FALSE(hashes[0].has_value());
}

TEST(InlineHashesTest, MultiObjectAllHashed) {
    // Two hashed objects in one message.
    const std::string json =
        R"({"version":3,"descriptors":[)"
        R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32",)"
        R"("byte_order":"little","encoding":"none","filter":"none","compression":"none"},)"
        R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32",)"
        R"("byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {3.0f, 4.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(a.data()), a.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(b.data()), b.size() * sizeof(float)},
    };
    auto encoded = tensogram::encode(json, objects);  // default hash = xxh3

    auto hashes = tensogram::data_object_inline_hashes(encoded.data(), encoded.size());
    ASSERT_EQ(hashes.size(), 2u);
    ASSERT_TRUE(hashes[0].has_value());
    ASSERT_TRUE(hashes[1].has_value());
    EXPECT_EQ(hashes[0]->size(), 16u);
    EXPECT_EQ(hashes[1]->size(), 16u);
}

TEST(InlineHashesTest, GarbageThrows) {
    std::vector<std::uint8_t> garbage(20, 0x00);
    EXPECT_THROW(
        tensogram::data_object_inline_hashes(garbage.data(), garbage.size()),
        tensogram::error);
}

// The mixed (string + null) array shape cannot be produced by encode (the
// HASHES_PRESENT toggle is message-wide), so exercise the JSON scanner
// directly to guard its null / string / whitespace handling.
TEST(InlineHashesTest, ParserHandlesMixedArray) {
    auto parsed = tensogram::detail::parse_inline_hashes_json(
        R"(["1122334455667788",null,"aabbccddeeff0011"])");
    ASSERT_EQ(parsed.size(), 3u);
    ASSERT_TRUE(parsed[0].has_value());
    EXPECT_EQ(*parsed[0], "1122334455667788");
    EXPECT_FALSE(parsed[1].has_value());
    ASSERT_TRUE(parsed[2].has_value());
    EXPECT_EQ(*parsed[2], "aabbccddeeff0011");
}

TEST(InlineHashesTest, ParserHandlesEmptyAndWhitespace) {
    EXPECT_TRUE(tensogram::detail::parse_inline_hashes_json("[]").empty());
    EXPECT_TRUE(tensogram::detail::parse_inline_hashes_json("   [   ]  ").empty());

    auto padded = tensogram::detail::parse_inline_hashes_json(R"([ "aa" , null ])");
    ASSERT_EQ(padded.size(), 2u);
    ASSERT_TRUE(padded[0].has_value());
    EXPECT_EQ(*padded[0], "aa");
    EXPECT_FALSE(padded[1].has_value());
}
