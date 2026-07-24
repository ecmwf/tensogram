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
// Tests for simple packing functions via C API.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// We call the C API directly for compute_params since the C++ wrapper
// may not wrap this function.
extern "C" {
#include "tensogram.h"
}

// ---------------------------------------------------------------------------
// Basic compute_params call
// ---------------------------------------------------------------------------

TEST(SimplePackingTest, ComputeParamsBasic) {
    std::vector<double> values = {100.0, 200.0, 300.0, 400.0};
    double reference_value = 0.0;
    std::int32_t binary_scale_factor = 0;

    tgm_error err = tgm_simple_packing_compute_params(
        values.data(), values.size(),
        16,   // bits_per_value
        0,    // decimal_scale_factor
        &reference_value, &binary_scale_factor);

    EXPECT_EQ(err, TGM_ERROR_OK);
    // Reference value should be the minimum value
    EXPECT_DOUBLE_EQ(reference_value, 100.0);
}

// ---------------------------------------------------------------------------
// NaN rejection throws encoding error
// ---------------------------------------------------------------------------

TEST(SimplePackingTest, NaNRejection) {
    std::vector<double> values = {1.0, std::nan(""), 3.0};
    double reference_value = 0.0;
    std::int32_t binary_scale_factor = 0;

    tgm_error err = tgm_simple_packing_compute_params(
        values.data(), values.size(),
        16, 0,
        &reference_value, &binary_scale_factor);

    EXPECT_NE(err, TGM_ERROR_OK);
}

// ---------------------------------------------------------------------------
// Round-trip: encode with simple_packing, decode, values approximately match
// ---------------------------------------------------------------------------

TEST(SimplePackingTest, RoundTripApproximate) {
    // Create values and encode with simple_packing encoding
    std::vector<double> values = {100.0, 200.0, 300.0, 400.0, 500.0};

    // Compute packing parameters
    double reference_value = 0.0;
    std::int32_t binary_scale_factor = 0;
    std::uint32_t bits_per_value = 16;
    std::int32_t decimal_scale_factor = 0;

    tgm_error err = tgm_simple_packing_compute_params(
        values.data(), values.size(),
        bits_per_value, decimal_scale_factor,
        &reference_value, &binary_scale_factor);
    ASSERT_EQ(err, TGM_ERROR_OK);

    // Build JSON with simple_packing encoding.
    // Packing params are flattened at the descriptor level (not nested under "packing").
    // Use snprintf to format doubles cleanly.
    char ref_buf[64];
    std::snprintf(ref_buf, sizeof(ref_buf), "%.17g", reference_value);

    std::string json =
        R"({"version":3,"descriptors":[{"type":"ndarray","ndim":1,"shape":[5],"strides":[8],"dtype":"float64","byte_order":"little","encoding":"simple_packing","filter":"none","compression":"none","sp_bits_per_value":)" +
        std::to_string(bits_per_value) +
        R"(,"sp_reference_value":)" + std::string(ref_buf) +
        R"(,"sp_binary_scale_factor":)" + std::to_string(binary_scale_factor) +
        R"(,"sp_decimal_scale_factor":)" + std::to_string(decimal_scale_factor) +
        R"(}]})";

    // For simple_packing, we pass the raw double values which get packed internally
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(double)}
    };

    auto encoded = tensogram::encode(json, objects);
    ASSERT_FALSE(encoded.empty());

    // Decode
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.encoding(), "simple_packing");

    const double* decoded = obj.data_as<double>();
    std::size_t count = obj.element_count<double>();
    ASSERT_EQ(count, values.size());

    // Values should approximately match (simple packing is lossy)
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(decoded[i], values[i], 1.0)
            << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Different bits_per_value settings
// ---------------------------------------------------------------------------

TEST(SimplePackingTest, DifferentBitsPerValue) {
    std::vector<double> values = {10.0, 20.0, 30.0, 40.0};

    for (std::uint32_t bits : {8u, 12u, 16u, 24u}) {
        double reference_value = 0.0;
        std::int32_t binary_scale_factor = 0;

        tgm_error err = tgm_simple_packing_compute_params(
            values.data(), values.size(),
            bits, 0,
            &reference_value, &binary_scale_factor);

        EXPECT_EQ(err, TGM_ERROR_OK)
            << "compute_params failed for bits_per_value=" << bits;
    }
}

// ---------------------------------------------------------------------------
// Constant field (all same values)
// ---------------------------------------------------------------------------

TEST(SimplePackingTest, ConstantField) {
    std::vector<double> values = {42.0, 42.0, 42.0, 42.0};
    double reference_value = 0.0;
    std::int32_t binary_scale_factor = 0;

    tgm_error err = tgm_simple_packing_compute_params(
        values.data(), values.size(),
        16, 0,
        &reference_value, &binary_scale_factor);

    EXPECT_EQ(err, TGM_ERROR_OK);
    EXPECT_DOUBLE_EQ(reference_value, 42.0);
}

// ---------------------------------------------------------------------------
// Typed C++ wrapper: compute_simple_packing_params
// ---------------------------------------------------------------------------

TEST(SimplePackingTest, TypedHelperVectorOverloadMatchesRawCApi) {
    std::vector<double> values = {100.0, 200.0, 300.0, 400.0};

    // Raw C API for comparison.
    double ref_c = 0.0;
    std::int32_t bsf_c = 0;
    ASSERT_EQ(tgm_simple_packing_compute_params(
                  values.data(), values.size(), 16, 0, &ref_c, &bsf_c),
              TGM_ERROR_OK);

    // Typed helper — vector overload.
    auto params = tensogram::compute_simple_packing_params(values, 16);
    EXPECT_DOUBLE_EQ(params.reference_value, ref_c);
    EXPECT_EQ(params.binary_scale_factor, bsf_c);
    EXPECT_DOUBLE_EQ(params.reference_value, 100.0);  // reference == minimum
    EXPECT_EQ(params.decimal_scale_factor, 0);        // echoed back
}

TEST(SimplePackingTest, TypedHelperPointerOverloadEchoesDecimalScale) {
    std::vector<double> values = {270.0, 275.0, 280.0, 285.0, 290.0};
    auto params = tensogram::compute_simple_packing_params(
        values.data(), values.size(), /*bits_per_value=*/12,
        /*decimal_scale_factor=*/2);
    EXPECT_DOUBLE_EQ(params.reference_value, 270.0);
    EXPECT_EQ(params.decimal_scale_factor, 2);
}

TEST(SimplePackingTest, TypedHelperThrowsEncodingErrorOnNaN) {
    std::vector<double> values = {1.0, std::nan(""), 3.0};
    EXPECT_THROW(tensogram::compute_simple_packing_params(values, 16),
                 tensogram::encoding_error);
}

TEST(SimplePackingTest, TypedHelperRoundTripBuildsDescriptor) {
    // End-to-end: use the typed helper to fill in the descriptor's packing
    // params, encode, decode, and verify the values round-trip.
    std::vector<double> values = {100.0, 200.0, 300.0, 400.0, 500.0};
    const int bits_per_value = 16;

    auto params = tensogram::compute_simple_packing_params(values, bits_per_value);

    char ref_buf[64];
    std::snprintf(ref_buf, sizeof(ref_buf), "%.17g", params.reference_value);

    std::string json =
        R"({"version":3,"descriptors":[{"type":"ndarray","ndim":1,"shape":[5],"strides":[8],"dtype":"float64","byte_order":"little","encoding":"simple_packing","filter":"none","compression":"none","sp_bits_per_value":)" +
        std::to_string(bits_per_value) +
        R"(,"sp_reference_value":)" + std::string(ref_buf) +
        R"(,"sp_binary_scale_factor":)" + std::to_string(params.binary_scale_factor) +
        R"(,"sp_decimal_scale_factor":)" + std::to_string(params.decimal_scale_factor) +
        R"(}]})";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(double)}};

    auto encoded = tensogram::encode(json, objects);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.encoding(), "simple_packing");

    const double* decoded = obj.data_as<double>();
    ASSERT_EQ(obj.element_count<double>(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(decoded[i], values[i], 1.0) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Auto-compute path: descriptor with only sp_bits_per_value lets the
// encoder derive sp_reference_value + sp_binary_scale_factor from the data.
// ---------------------------------------------------------------------------

TEST(SimplePackingTest, AutoComputeFromBitsOnly) {
    std::vector<double> values = {270.0, 275.0, 280.0, 285.0, 290.0};

    // Descriptor with only sp_bits_per_value — the encoder fills in
    // sp_reference_value and sp_binary_scale_factor itself.
    std::string json =
        R"({"version":3,"descriptors":[{"type":"ndarray","ndim":1,"shape":[5],"strides":[8],"dtype":"float64","byte_order":"little","encoding":"simple_packing","filter":"none","compression":"none","sp_bits_per_value":16}]})";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(double)}};

    auto encoded = tensogram::encode(json, objects);
    ASSERT_FALSE(encoded.empty());

    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    EXPECT_EQ(obj.encoding(), "simple_packing");

    // Values round-trip within 16-bit quantisation tolerance.
    const double* decoded = obj.data_as<double>();
    std::size_t count = obj.element_count<double>();
    ASSERT_EQ(count, values.size());
    const double tolerance =
        (values.back() - values.front()) / static_cast<double>(1u << 16);
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(decoded[i], values[i], tolerance)
            << "auto-compute mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Auto-compute rejects half-explicit params: providing only
// sp_reference_value (without sp_binary_scale_factor) is ambiguous.
// ---------------------------------------------------------------------------

TEST(SimplePackingTest, HalfExplicitParamsRejected) {
    std::vector<double> values = {270.0, 275.0, 280.0, 285.0};

    // Provide sp_reference_value but NOT sp_binary_scale_factor.  The
    // encoder must error rather than silently auto-compute over our
    // half-explicit input.
    std::string json =
        R"({"version":3,"descriptors":[{"type":"ndarray","ndim":1,"shape":[4],"strides":[8],"dtype":"float64","byte_order":"little","encoding":"simple_packing","filter":"none","compression":"none","sp_bits_per_value":16,"sp_reference_value":200.0}]})";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(double)}};

    EXPECT_THROW(tensogram::encode(json, objects), tensogram::error);
}
