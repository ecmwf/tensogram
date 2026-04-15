// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 02_mars_metadata.cpp
/// @brief Example 02 — MARS-namespaced metadata using the C++ wrapper.
///
/// Shows how to encode a message with MARS namespace keys and read them
/// back using dot-notation key lookups on the metadata handle.

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    // -- Encode with MARS metadata --
    const std::string metadata_json = R"({
        "version": 2,
        "descriptors": [{
            "type": "ndarray",
            "ndim": 2,
            "shape": [721, 1440],
            "strides": [5760, 4],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none"
        }],
        "mars": {
            "class": "od",
            "date": "20260401",
            "step": "6",
            "time": "0000",
            "type": "fc"
        }
    })";

    std::vector<float> data(721 * 1440, 273.15f);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(data.data()),
         data.size() * sizeof(float)}
    };

    auto encoded = tensogram::encode(metadata_json, objects);

    // -- Decode metadata only (no payload) --
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    // -- Read keys using dot-notation --
    std::printf("version   = %llu\n",
                static_cast<unsigned long long>(meta.version()));
    std::printf("mars.class = %s\n", meta.get_string("mars.class").c_str());
    std::printf("mars.date  = %s\n", meta.get_string("mars.date").c_str());
    std::printf("mars.type  = %s\n", meta.get_string("mars.type").c_str());
    std::printf("mars.step  = %s\n", meta.get_string("mars.step").c_str());

    assert(meta.get_string("mars.class") == "od");
    assert(meta.get_string("mars.date") == "20260401");
    std::printf("All assertions passed.\n");

    // -- Also demonstrate full-message decode with metadata extraction --
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto msg_meta = msg.get_metadata();
    assert(msg_meta.get_string("mars.class") == "od");
    std::printf("Message metadata also works: mars.class = %s\n",
                msg_meta.get_string("mars.class").c_str());

    return 0;
}
