// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 02_mars_metadata.cpp
/// @brief Example 02 — Per-object metadata with MARS (C++ wrapper).
///
/// Shows how to encode a message with MARS namespace keys and read them back
/// using dot-notation key lookups on the metadata handle. MARS is used as a
/// concrete example vocabulary; the same mechanism works with any namespace.
/// See `02b_generic_metadata.cpp` for a non-MARS example.

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    // -- Encode with per-object MARS metadata --
    //
    // `base` is a JSON array with one entry per data object. Each entry
    // holds ALL metadata for that object (first-match lookup semantics,
    // no common/varying split on the wire). Here we have a single object
    // so `base` has one entry.
    //
    // Top-level keys outside `"version"`, `"descriptors"`, and `"base"`
    // are treated as message-level `_extra_` annotations — use those
    // for provenance notes that apply to the whole message, not to an
    // individual object.
    const std::string metadata_json = R"({
        "version": 3,
        "descriptors": [{
            "type": "ntensor",
            "ndim": 2,
            "shape": [721, 1440],
            "strides": [5760, 4],
            "dtype": "float32",
            "encoding": "none",
            "filter": "none",
            "compression": "none"
        }],
        "base": [{
            "mars": {
                "class": "od",
                "date": "20260401",
                "step": "6",
                "time": "0000",
                "type": "fc"
            }
        }],
        "source": "ifs-cycle49r2"
    })";

    std::vector<float> data(721 * 1440, 273.15f);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(data.data()),
         data.size() * sizeof(float)}
    };

    auto encoded = tensogram::encode(metadata_json, objects);

    // -- Decode metadata only (no payload) --
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    // -- Read per-object keys using dot-notation --
    //
    // `get_string("mars.class")` searches base entries first (skipping
    // `_reserved_`), then falls back to the message-level _extra_ map.
    std::printf("version    = %llu\n",
                static_cast<unsigned long long>(meta.version()));
    std::printf("num_objects = %zu\n", meta.num_objects());
    std::printf("mars.class = %s\n", meta.get_string("mars.class").c_str());
    std::printf("mars.date  = %s\n", meta.get_string("mars.date").c_str());
    std::printf("mars.type  = %s\n", meta.get_string("mars.type").c_str());
    std::printf("mars.step  = %s\n", meta.get_string("mars.step").c_str());

    // Message-level _extra_ key — resolves via the fallback branch.
    std::printf("source (extra) = %s\n",
                meta.get_string("source").c_str());

    assert(meta.num_objects() == 1);
    assert(meta.get_string("mars.class") == "od");
    assert(meta.get_string("mars.date") == "20260401");
    assert(meta.get_string("source") == "ifs-cycle49r2");
    std::printf("All assertions passed.\n");

    // -- Also demonstrate full-message decode with metadata extraction --
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto msg_meta = msg.get_metadata();
    assert(msg_meta.get_string("mars.class") == "od");
    std::printf("Message metadata also works: mars.class = %s\n",
                msg_meta.get_string("mars.class").c_str());

    return 0;
}
