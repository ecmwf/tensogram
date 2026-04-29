// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// Decode-time hash verification on the C++ wrapper surface.  Mirrors
// `python/tests/test_decode_verify_hash.py` and
// `rust/tensogram/tests/decode_verify_hash.rs`.  See
// `PLAN_DECODE_HASH_VERIFICATION.md` §5.2 for the matrix.

#include <gtest/gtest.h>
#include <tensogram.hpp>

#include "test_helpers.hpp"

#include <cstring>
#include <vector>

using namespace test_helpers;

namespace {

/// Encode a 1-D float32 message with hashing toggled on/off.  Mirrors
/// `simple_f32_json` from `test_helpers.hpp` but plumbs the hash
/// flag through `encode_options::hash_algo`.
std::vector<std::uint8_t> encode_simple_with_hash(
    const std::vector<float>& values, bool hashing)
{
    auto json = simple_f32_json(values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    tensogram::encode_options opts{};
    // `encode_options::hash_algo` defaults to "xxh3", so we have
    // to *clear* it (empty string → tgm_encode receives NULL →
    // hashing off) when the caller wants an unhashed message.
    if (!hashing) {
        opts.hash_algo.clear();
    }
    return tensogram::encode(json, objects, opts);
}

}  // namespace

// ── Cells A & B — verify on/off, hashed message ───────────────────────

TEST(VerifyHashTest, CellADecodeNoVerifySucceedsOnHashed) {
    auto bytes = encode_simple_with_hash({1.0f, 2.0f, 3.0f, 4.0f}, /*hashing=*/true);
    tensogram::decode_options opts{};
    opts.verify_hash = false;
    auto msg = tensogram::decode(bytes.data(), bytes.size(), opts);
    EXPECT_EQ(msg.num_objects(), 1u);
}

TEST(VerifyHashTest, CellBDecodeWithVerifySucceedsOnHashed) {
    auto bytes = encode_simple_with_hash({1.0f, 2.0f, 3.0f, 4.0f}, /*hashing=*/true);
    tensogram::decode_options opts{};
    opts.verify_hash = true;
    auto msg = tensogram::decode(bytes.data(), bytes.size(), opts);
    EXPECT_EQ(msg.num_objects(), 1u);
}

TEST(VerifyHashTest, CellBDecodeObjectWithVerifySucceedsOnHashed) {
    auto bytes = encode_simple_with_hash({10.0f, 20.0f}, /*hashing=*/true);
    tensogram::decode_options opts{};
    opts.verify_hash = true;
    auto msg = tensogram::decode_object(bytes.data(), bytes.size(), 0, opts);
    EXPECT_EQ(msg.num_objects(), 1u);
}

// ── Cell C — unhashed message + verify=true → missing_hash_error ──────

TEST(VerifyHashTest, CellCDecodeVerifyOnUnhashedThrowsMissingHash) {
    auto bytes = encode_simple_with_hash({5.0f}, /*hashing=*/false);
    tensogram::decode_options opts{};
    opts.verify_hash = true;
    EXPECT_THROW(
        tensogram::decode(bytes.data(), bytes.size(), opts),
        tensogram::missing_hash_error);
}

TEST(VerifyHashTest, CellCMissingHashIsCatchableViaIntegrityErrorBase) {
    // Documents the hierarchy: callers can catch the family with
    // `tensogram::integrity_error` and discriminate later via
    // dynamic_cast or by re-throwing into a subclass-specific
    // handler.  See `tensogram.hpp` exception comments.
    auto bytes = encode_simple_with_hash({5.0f}, /*hashing=*/false);
    tensogram::decode_options opts{};
    opts.verify_hash = true;
    EXPECT_THROW(
        tensogram::decode(bytes.data(), bytes.size(), opts),
        tensogram::integrity_error);
}

TEST(VerifyHashTest, CellCDecodeObjectVerifyOnUnhashedThrowsMissingHash) {
    auto bytes = encode_simple_with_hash({5.0f}, /*hashing=*/false);
    tensogram::decode_options opts{};
    opts.verify_hash = true;
    EXPECT_THROW(
        tensogram::decode_object(bytes.data(), bytes.size(), 0, opts),
        tensogram::missing_hash_error);
}

TEST(VerifyHashTest, NoVerifySilentlyDecodesUnhashedMessage) {
    auto bytes = encode_simple_with_hash({5.0f, 6.0f}, /*hashing=*/false);
    tensogram::decode_options opts{};  // verify_hash defaults to false
    auto msg = tensogram::decode(bytes.data(), bytes.size(), opts);
    EXPECT_EQ(msg.num_objects(), 1u);
}

// ── Cell D — tampered hash slot → hash_mismatch_error ─────────────────

// ── Cell F — multi-object golden, tamper object 1 ────────────────────

namespace {

/// Read a committed golden file from `rust/tensogram/tests/golden/`.
/// CMake places the test binary in `build/cpp/tests/`, so the
/// path is `../../rust/tensogram/tests/golden/<name>` relative to
/// the test binary's CWD when CTest runs it.
std::vector<std::uint8_t> read_golden(const std::string& name) {
    // Try a couple of likely paths so this works under both CTest
    // (CWD = build dir) and direct invocation from the repo root.
    const char* candidates[] = {
        "../../rust/tensogram/tests/golden/",
        "rust/tensogram/tests/golden/",
        "../../../rust/tensogram/tests/golden/",
    };
    for (const char* prefix : candidates) {
        std::string path = std::string(prefix) + name;
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) continue;
        std::fseek(f, 0, SEEK_END);
        long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<std::uint8_t> buf(static_cast<std::size_t>(size));
        std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        return buf;
    }
    ADD_FAILURE() << "could not locate golden file: " << name;
    return {};
}

/// Walk the message buffer to find the i-th NTensorFrame.  Mirrors
/// the per-binding helper in `python/tests/test_decode_verify_hash.py`
/// and `rust/tensogram/tests/decode_verify_hash.rs`.
std::pair<std::size_t, std::size_t> locate_object_frame(
    const std::vector<std::uint8_t>& buf, std::size_t which)
{
    std::size_t pos = 24;  // past 24-byte preamble
    std::size_t seen = 0;
    while (pos + 16 <= buf.size()) {
        if (std::memcmp(buf.data() + pos, "FR", 2) != 0) {
            ++pos;
            continue;
        }
        const auto frame_type =
            static_cast<std::uint16_t>((buf[pos + 2] << 8) | buf[pos + 3]);
        std::uint64_t total = 0;
        for (int i = 0; i < 8; ++i) {
            total = (total << 8) | buf[pos + 8 + i];
        }
        if (frame_type == 9 /* NTensorFrame */) {
            if (seen == which) {
                return {pos, static_cast<std::size_t>(total)};
            }
            ++seen;
        }
        pos = (pos + total + 7) & ~static_cast<std::size_t>(7);
    }
    ADD_FAILURE() << "object frame index " << which << " not found";
    return {0, 0};
}

}  // namespace

TEST(VerifyHashTest, CellFMultiObjectGoldenVerifiesCleanly) {
    auto bytes = read_golden("multi_object_xxh3.tgm");
    if (bytes.empty()) {
        GTEST_SKIP() << "could not locate golden — running outside the repo?";
    }
    tensogram::decode_options opts{};
    opts.verify_hash = true;
    auto msg = tensogram::decode(bytes.data(), bytes.size(), opts);
    EXPECT_EQ(msg.num_objects(), 3u);
}

TEST(VerifyHashTest, CellFTamperObjectOneSurfacesIndexOne) {
    auto bytes = read_golden("multi_object_xxh3.tgm");
    if (bytes.empty()) {
        GTEST_SKIP() << "could not locate golden — running outside the repo?";
    }
    auto [frame_start, total_length] = locate_object_frame(bytes, 1);
    ASSERT_GT(total_length, 0u);
    bytes[frame_start + total_length - 12] ^= 0xff;

    tensogram::decode_options opts{};
    opts.verify_hash = true;
    try {
        tensogram::decode(bytes.data(), bytes.size(), opts);
        FAIL() << "expected hash_mismatch_error";
    } catch (const tensogram::hash_mismatch_error& e) {
        // The Display form embeds `object Some(N)` (mirrors the
        // Rust core); pin the canonical "object Some(1)" substring
        // so this test fails loudly if the format ever changes.
        EXPECT_NE(std::string(e.what()).find("object Some(1)"), std::string::npos)
            << "expected message to name object 1, got: " << e.what();
    }
}

TEST(VerifyHashTest, CellDDecodeVerifyOnTamperedSlotThrowsHashMismatch) {
    auto bytes = encode_simple_with_hash({1.0f, 2.0f, 3.0f}, /*hashing=*/true);
    // Walk frame-by-frame to find the first NTensorFrame and flip a
    // byte of its inline hash slot (frame_end - 12).
    std::size_t pos = 24;  // past 24-byte preamble
    std::size_t target = 0;
    while (pos + 16 <= bytes.size()) {
        if (std::memcmp(bytes.data() + pos, "FR", 2) != 0) {
            ++pos;
            continue;
        }
        const auto frame_type =
            static_cast<std::uint16_t>((bytes[pos + 2] << 8) | bytes[pos + 3]);
        std::uint64_t total_length = 0;
        for (int i = 0; i < 8; ++i) {
            total_length = (total_length << 8) | bytes[pos + 8 + i];
        }
        if (frame_type == 9 /* NTensorFrame */) {
            target = pos + total_length - 12;
            break;
        }
        pos = (pos + total_length + 7) & ~static_cast<std::size_t>(7);
    }
    ASSERT_GT(target, 0u);
    bytes[target] ^= 0xFF;

    tensogram::decode_options opts{};
    opts.verify_hash = true;
    EXPECT_THROW(
        tensogram::decode(bytes.data(), bytes.size(), opts),
        tensogram::hash_mismatch_error);
}
