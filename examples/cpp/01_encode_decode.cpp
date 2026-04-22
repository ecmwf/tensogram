// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 01_encode_decode.cpp
/// @brief Example 01 — Basic encode / decode round-trip using the C++ wrapper.
///
/// Encodes a 100x200 float32 temperature grid into a Tensogram message,
/// then decodes it and verifies the round-trip.

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    // -- 1. Prepare data --
    constexpr int ROWS = 100;
    constexpr int COLS = 200;
    constexpr int N = ROWS * COLS;

    std::vector<float> temps(N);
    for (int i = 0; i < N; ++i)
        temps[i] = 273.15f + static_cast<float>(i) * 0.001f;

    // -- 2. Describe the tensor as JSON metadata --
    const std::string metadata_json = R"({
        "version": 3,
        "descriptors": [{
            "type": "ndarray",
            "ndim": 2,
            "shape": [100, 200],
            "strides": [800, 4],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none"
        }],
        "mars": {
            "class": "od",
            "type": "fc",
            "date": "20260401",
            "step": "6"
        }
    })";

    // -- 3. Encode --
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(temps.data()),
         temps.size() * sizeof(float)}
    };

    auto encoded = tensogram::encode(metadata_json, objects);
    std::printf("Encoded: %zu bytes\n", encoded.size());

    // -- 4. Decode --
    auto msg = tensogram::decode(encoded.data(), encoded.size());

    // -- 5. Inspect --
    std::printf("Version: %llu\n", static_cast<unsigned long long>(msg.version()));
    std::printf("Objects: %zu\n", msg.num_objects());
    assert(msg.num_objects() == 1);

    auto obj = msg.object(0);
    std::printf("  ndim=%llu  shape=[%llu, %llu]  dtype=%s\n",
                static_cast<unsigned long long>(obj.ndim()),
                static_cast<unsigned long long>(obj.shape()[0]),
                static_cast<unsigned long long>(obj.shape()[1]),
                obj.dtype_string().c_str());

    // -- 6. Access raw data and verify round-trip --
    assert(obj.data_size() == temps.size() * sizeof(float));
    assert(std::memcmp(obj.data(), temps.data(), obj.data_size()) == 0);
    std::printf("Data: %zu bytes — round-trip OK\n", obj.data_size());

    // -- 7. Access via typed pointer --
    const float* decoded = obj.data_as<float>();
    assert(obj.element_count<float>() == static_cast<std::size_t>(N));
    std::printf("First value: %.3f K\n", decoded[0]);
    std::printf("Last value:  %.3f K\n", decoded[N - 1]);

    return 0;
}
