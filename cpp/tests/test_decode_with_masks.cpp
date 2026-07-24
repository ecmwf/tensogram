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
// Tests for tensogram::decode_with_masks() plus decoded_object::has_masks()
// and decoded_object::mask().  Binds tgm_decode_with_masks /
// tgm_object_has_masks / tgm_object_mask.
// See docs/src/guide/nan-inf-handling.md for the user contract.

#include <gtest/gtest.h>
#include <tensogram.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string f32_descriptor_json(std::size_t n) {
    const std::string ns = std::to_string(n);
    return std::string(R"({"version":3,"descriptors":[{)")
        + R"("type":"ndarray","ndim":1,)"
        + "\"shape\":[" + ns + "],\"strides\":[4],"
        + R"("dtype":"float32","byte_order":"little",)"
        + R"("encoding":"none","filter":"none","compression":"none"}]})";
}

std::vector<std::uint8_t> encode_masked(const std::vector<float>& values,
                                        bool allow_nan, bool allow_inf) {
    tensogram::encode_options opts;
    opts.allow_nan = allow_nan;
    opts.allow_inf = allow_inf;
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}};
    return tensogram::encode(f32_descriptor_json(values.size()), objects, opts);
}

const float kNan = std::numeric_limits<float>::quiet_NaN();
const float kPosInf = std::numeric_limits<float>::infinity();
const float kNegInf = -std::numeric_limits<float>::infinity();

using bytemask = std::vector<std::uint8_t>;

}  // namespace

TEST(DecodeWithMasksTest, RetainsAllThreeKinds) {
    // [finite, NaN, +Inf, -Inf, finite]
    auto encoded = encode_masked({1.0f, kNan, kPosInf, kNegInf, 2.0f}, true, true);

    auto msg = tensogram::decode_with_masks(encoded.data(), encoded.size());
    ASSERT_EQ(msg.num_objects(), 1u);
    auto obj = msg.object(0);

    EXPECT_TRUE(obj.has_masks());
    EXPECT_EQ(obj.mask(tensogram::mask_kind::nan), (bytemask{0, 1, 0, 0, 0}));
    EXPECT_EQ(obj.mask(tensogram::mask_kind::pos_inf), (bytemask{0, 0, 1, 0, 0}));
    EXPECT_EQ(obj.mask(tensogram::mask_kind::neg_inf), (bytemask{0, 0, 0, 1, 0}));
}

TEST(DecodeWithMasksTest, PayloadHoldsCanonicalZeroAtMaskedPositions) {
    // decode_with_masks forces restore_non_finite = false, so the payload
    // keeps the 0.0 substitution at every masked position.
    auto encoded = encode_masked({1.0f, kNan, kPosInf, kNegInf, 2.0f}, true, true);
    auto msg = tensogram::decode_with_masks(encoded.data(), encoded.size());
    auto obj = msg.object(0);

    ASSERT_EQ(obj.data_size(), 5 * sizeof(float));
    const float* out = obj.data_as<float>();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);  // was NaN
    EXPECT_FLOAT_EQ(out[2], 0.0f);  // was +Inf
    EXPECT_FLOAT_EQ(out[3], 0.0f);  // was -Inf
    EXPECT_FLOAT_EQ(out[4], 2.0f);
}

TEST(DecodeWithMasksTest, NanOnlyLeavesInfKindsEmpty) {
    auto encoded = encode_masked({1.0f, kNan, 2.0f}, true, false);
    auto msg = tensogram::decode_with_masks(encoded.data(), encoded.size());
    auto obj = msg.object(0);

    EXPECT_TRUE(obj.has_masks());
    EXPECT_EQ(obj.mask(tensogram::mask_kind::nan), (bytemask{0, 1, 0}));
    // Absent kinds return an empty vector (not an error).
    EXPECT_TRUE(obj.mask(tensogram::mask_kind::pos_inf).empty());
    EXPECT_TRUE(obj.mask(tensogram::mask_kind::neg_inf).empty());
}

TEST(DecodeWithMasksTest, FiniteDataHasNoMasks) {
    auto encoded = encode_masked({1.0f, 2.0f, 3.0f}, false, false);
    auto msg = tensogram::decode_with_masks(encoded.data(), encoded.size());
    auto obj = msg.object(0);

    EXPECT_FALSE(obj.has_masks());
    EXPECT_TRUE(obj.mask(tensogram::mask_kind::nan).empty());
    EXPECT_TRUE(obj.mask(tensogram::mask_kind::pos_inf).empty());
    EXPECT_TRUE(obj.mask(tensogram::mask_kind::neg_inf).empty());

    // Finite values pass through untouched.
    const float* out = obj.data_as<float>();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST(DecodeWithMasksTest, RegularDecodeRestoresAndKeepsNoMasks) {
    // Contrast with decode_with_masks: the plain decode restores the
    // canonical NaN / ±Inf in place and retains no masks.
    auto encoded = encode_masked({1.0f, kNan, kPosInf, kNegInf, 2.0f}, true, true);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);

    EXPECT_FALSE(obj.has_masks());
    EXPECT_TRUE(obj.mask(tensogram::mask_kind::nan).empty());

    const float* out = obj.data_as<float>();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_TRUE(std::isnan(out[1]));
    EXPECT_TRUE(std::isinf(out[2]) && out[2] > 0);
    EXPECT_TRUE(std::isinf(out[3]) && out[3] < 0);
    EXPECT_FLOAT_EQ(out[4], 2.0f);
}

TEST(DecodeWithMasksTest, HonoursDecodeOptions) {
    // native_byte_order / threads / verify_hash flow through the options;
    // restore_non_finite is deliberately ignored by this entry point.
    auto encoded = encode_masked({1.0f, kNan, 2.0f}, true, false);

    tensogram::decode_options opts;
    opts.threads = 1;
    opts.restore_non_finite = true;  // ignored — masks are still retained
    auto msg = tensogram::decode_with_masks(encoded.data(), encoded.size(), opts);
    auto obj = msg.object(0);

    EXPECT_TRUE(obj.has_masks());
    EXPECT_EQ(obj.mask(tensogram::mask_kind::nan), (bytemask{0, 1, 0}));
    const float* out = obj.data_as<float>();
    EXPECT_FLOAT_EQ(out[1], 0.0f);  // not restored
}

TEST(DecodeWithMasksTest, NullBufferThrows) {
    EXPECT_THROW(
        tensogram::decode_with_masks(nullptr, 0),
        tensogram::invalid_arg_error);
}
