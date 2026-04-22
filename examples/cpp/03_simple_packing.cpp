// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 03_simple_packing.cpp
/// @brief Example 03 — Simple packing (lossy quantization) using the C++ wrapper.
///
/// Demonstrates encoding with simple_packing at a given bits_per_value,
/// then decoding and measuring the quantization error.

#include <tensogram.hpp>

extern "C" {
#include "tensogram.h"
}

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr int N = 1000;

    // Source data: 1000 temperature values as f64
    std::vector<double> temps(N);
    for (int i = 0; i < N; ++i)
        temps[i] = 249.15 + i * 0.1;

    // -- Compute packing parameters --
    double reference_value = 0.0;
    std::int32_t binary_scale_factor = 0;
    constexpr std::uint32_t bits_per_value = 16;
    constexpr std::int32_t decimal_scale_factor = 0;

    [[maybe_unused]] tgm_error err = tgm_simple_packing_compute_params(
        temps.data(), temps.size(),
        bits_per_value, decimal_scale_factor,
        &reference_value, &binary_scale_factor);
    assert(err == TGM_ERROR_OK);

    // -- Build JSON with simple_packing encoding --
    char ref_buf[64];
    std::snprintf(ref_buf, sizeof(ref_buf), "%.17g", reference_value);

    std::string json =
        R"({"version":3,"descriptors":[{"type":"ntensor","ndim":1,"shape":[1000],"strides":[8],"dtype":"float64","byte_order":"little","encoding":"simple_packing","filter":"none","compression":"none","bits_per_value":)" +
        std::to_string(bits_per_value) +
        R"(,"reference_value":)" + std::string(ref_buf) +
        R"(,"binary_scale_factor":)" + std::to_string(binary_scale_factor) +
        R"(,"decimal_scale_factor":)" + std::to_string(decimal_scale_factor) +
        R"(}]})";

    // -- Encode --
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(temps.data()),
         temps.size() * sizeof(double)}
    };

    auto encoded = tensogram::encode(json, objects);
    std::printf("Raw:     %zu bytes\n", temps.size() * sizeof(double));
    std::printf("Encoded: %zu bytes\n", encoded.size());

    // -- Decode --
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto obj = msg.object(0);
    assert(obj.encoding() == "simple_packing");

    const double* decoded = obj.data_as<double>();
    [[maybe_unused]] const std::size_t count = obj.element_count<double>();
    assert(count == static_cast<std::size_t>(N));

    // -- Measure quantization error --
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        const double e = std::abs(temps[i] - decoded[i]);
        if (e > max_err) max_err = e;
    }
    std::printf("Max error: %.6f K\n", max_err);
    assert(max_err < 0.01);
    std::printf("Precision OK (< 0.01 K)\n");

    return 0;
}
