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
