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
// Tests for the allow_nan / allow_inf C++ bindings.
// See docs/src/guide/nan-inf-handling.md for the user contract.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

std::string descriptor_json(std::size_t n, const char* dtype = "float64") {
    const std::string ns = std::to_string(n);
    return std::string("{ \"version\": 2, \"descriptors\": [{")
        + "\"type\": \"ndarray\", \"ndim\": 1, "
        + "\"shape\": [" + ns + "], "
        + "\"strides\": [" + ns + "], "
        + "\"dtype\": \"" + dtype + "\", "
        + "\"byte_order\": \"little\", "
        + "\"encoding\": \"none\", \"filter\": \"none\", "
        + "\"compression\": \"none\" }] }";
}

std::vector<std::pair<const std::uint8_t*, std::size_t>>
one_object(const std::vector<double>& values) {
    return {{reinterpret_cast<const std::uint8_t*>(values.data()),
             values.size() * sizeof(double)}};
}

}  // namespace

TEST(AllowNanInfTest, DefaultRejectsNaNF64) {
    std::vector<double> values = {1.0, std::nan(""), 3.0};
    EXPECT_THROW(
        tensogram::encode(descriptor_json(3), one_object(values)),
        tensogram::encoding_error);
}

TEST(AllowNanInfTest, AllowNanRoundTripRestoresCanonicalNaN) {
    std::vector<double> values = {1.0, std::nan(""), 3.0, std::nan(""), 5.0};
    tensogram::encode_options opts;
    opts.allow_nan = true;
    opts.small_mask_threshold_bytes = 0;  // force requested method
    auto encoded = tensogram::encode(descriptor_json(5), one_object(values), opts);

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    ASSERT_EQ(obj.data_size(), 5 * sizeof(double));
    const double* out = reinterpret_cast<const double*>(obj.data());
    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_TRUE(std::isnan(out[1]));
    EXPECT_DOUBLE_EQ(out[2], 3.0);
    EXPECT_TRUE(std::isnan(out[3]));
    EXPECT_DOUBLE_EQ(out[4], 5.0);
}

TEST(AllowNanInfTest, RestoreNonFiniteFalseKeepsZeroes) {
    std::vector<double> values = {1.0, std::nan(""), 3.0};
    tensogram::encode_options enc_opts;
    enc_opts.allow_nan = true;
    auto encoded = tensogram::encode(descriptor_json(3), one_object(values), enc_opts);

    tensogram::decode_options dec_opts;
    dec_opts.restore_non_finite = false;
    auto msg = tensogram::decode(encoded.data(), encoded.size(), dec_opts);
    auto obj = msg.object(0);
    const double* out = reinterpret_cast<const double*>(obj.data());
    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_DOUBLE_EQ(out[1], 0.0);
    EXPECT_DOUBLE_EQ(out[2], 3.0);
}

TEST(AllowNanInfTest, AllowInfRestoresPosAndNegInf) {
    std::vector<double> values = {
        std::numeric_limits<double>::infinity(), 1.0,
        -std::numeric_limits<double>::infinity(), 2.0,
    };
    tensogram::encode_options opts;
    opts.allow_inf = true;
    opts.small_mask_threshold_bytes = 0;
    auto encoded = tensogram::encode(descriptor_json(4), one_object(values), opts);

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    const double* out = reinterpret_cast<const double*>(obj.data());
    EXPECT_TRUE(std::isinf(out[0]) && out[0] > 0);
    EXPECT_DOUBLE_EQ(out[1], 1.0);
    EXPECT_TRUE(std::isinf(out[2]) && out[2] < 0);
    EXPECT_DOUBLE_EQ(out[3], 2.0);
}

TEST(AllowNanInfTest, UnknownMaskMethodReturnsInvalidArg) {
    // Pass 4 regression: the FFI layer used to silently fall back
    // to "roaring" on unknown method names, which hid user typos
    // and broke cross-language parity with the Python / TS / CLI
    // frontends.  tgm_encode_with_options now returns InvalidArg
    // with a clear error message.
    std::vector<double> values = {1.0, std::nan(""), 3.0};
    tensogram::encode_options opts;
    opts.allow_nan = true;
    opts.nan_mask_method = "totally-bogus";
    EXPECT_THROW(
        tensogram::encode(descriptor_json(3), one_object(values), opts),
        tensogram::invalid_arg_error);
}

TEST(AllowNanInfTest, StreamingEncoderHonoursAllowNan) {
    // Regression: the C++ streaming_encoder used to ignore the
    // allow_nan / allow_inf / mask-method fields on encode_options
    // because it always called tgm_streaming_encoder_create
    // (pre-Pass-5, pre-review-fix).  Must now route through
    // tgm_streaming_encoder_create_with_options when any mask option
    // is set.
    const std::string path = std::tmpnam(nullptr);
    std::vector<double> values = {1.0, std::nan(""), 3.0};
    tensogram::encode_options opts;
    opts.allow_nan = true;
    opts.small_mask_threshold_bytes = 0;
    {
        tensogram::streaming_encoder enc(path, R"({"version":3})", opts);
        const std::string desc_json =
            R"({"type":"ndarray","ndim":1,"shape":[3],"strides":[8],)"
            R"("dtype":"float64","byte_order":"little",)"
            R"("encoding":"none","filter":"none","compression":"none"})";
        enc.write_object(
            desc_json,
            reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size() * sizeof(double));
        enc.finish();
    }
    // Read back via file open and verify NaN is restored.  If the
    // streaming_encoder silently dropped allow_nan, the encode would
    // have errored with 'NaN at element 1' instead.
    tensogram::file f = tensogram::file::open(path);
    ASSERT_EQ(f.message_count(), 1u);
    auto msg = f.decode_message(0);
    const double* out = reinterpret_cast<const double*>(msg.object(0).data());
    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_TRUE(std::isnan(out[1]));
    EXPECT_DOUBLE_EQ(out[2], 3.0);
    std::remove(path.c_str());
}

TEST(AllowNanInfTest, MaskMethodRleRoundTrip) {
    std::vector<double> values(128);
    for (std::size_t i = 0; i < 128; ++i) {
        values[i] = static_cast<double>(i);
    }
    values[10] = std::nan("");
    values[50] = std::nan("");
    values[100] = std::nan("");

    tensogram::encode_options opts;
    opts.allow_nan = true;
    opts.nan_mask_method = "rle";
    opts.small_mask_threshold_bytes = 0;
    auto encoded = tensogram::encode(descriptor_json(128), one_object(values), opts);

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    const double* out = reinterpret_cast<const double*>(obj.data());
    EXPECT_TRUE(std::isnan(out[10]));
    EXPECT_TRUE(std::isnan(out[50]));
    EXPECT_TRUE(std::isnan(out[100]));
    EXPECT_DOUBLE_EQ(out[11], 11.0);
    EXPECT_DOUBLE_EQ(out[99], 99.0);
}
