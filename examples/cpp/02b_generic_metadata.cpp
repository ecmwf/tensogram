// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 02b_generic_metadata.cpp
/// @brief Example 02b — Per-object metadata with a generic application
///        namespace (C++ wrapper).
///
/// Shows that the metadata mechanism in example 02 is not specific to the
/// MARS vocabulary. Any application namespace works the same way. Here we
/// use a made-up "product" namespace plus an "instrument" namespace to tag
/// a 2-D field with semantic context.
///
/// The library never interprets any of these — it simply stores and returns
/// the keys you supply. Meaning is assigned by the application layer.

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    // -- Encode with two parallel per-object namespaces --
    //
    // Both `product` and `instrument` live inside `base[0]` — this is
    // per-object metadata (attached to data object #0). Multiple
    // namespaces can coexist freely in the same base entry; the library
    // never interprets their contents.
    //
    // For comparison, `pipeline` is placed at the top level: it becomes
    // a message-level _extra_ annotation, covering the whole message.
    const std::string metadata_json = R"({
                "descriptors": [{
            "type": "ndarray",
            "ndim": 2,
            "shape": [512, 512],
            "strides": [2048, 4],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none"
        }],
        "base": [{
            "product": {
                "name": "intensity",
                "units": "counts",
                "device": "detector_A",
                "run_id": 42,
                "acquired_at": "2026-04-18T10:30:00Z"
            },
            "instrument": {
                "serial": "XYZ-001",
                "firmware": "v3.1.2"
            }
        }],
        "pipeline": "detector-v3"
    })";

    std::vector<float> data(512 * 512, 0.0f);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(data.data()),
         data.size() * sizeof(float)}
    };

    auto encoded = tensogram::encode(metadata_json, objects);

    // -- Decode metadata only (no payload) --
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    // -- Read keys using dot-notation over any namespace --
    //
    // Per-object keys resolve through the base[] search.
    std::printf("product.name    = %s\n",
                meta.get_string("product.name").c_str());
    std::printf("product.units   = %s\n",
                meta.get_string("product.units").c_str());
    std::printf("product.device  = %s\n",
                meta.get_string("product.device").c_str());
    std::printf("product.run_id  = %lld\n",
                static_cast<long long>(meta.get_int("product.run_id")));
    std::printf("instrument.serial   = %s\n",
                meta.get_string("instrument.serial").c_str());
    std::printf("instrument.firmware = %s\n",
                meta.get_string("instrument.firmware").c_str());

    // Message-level _extra_ key — resolves via the fallback branch.
    std::printf("pipeline (extra) = %s\n",
                meta.get_string("pipeline").c_str());

    assert(meta.num_objects() == 1);
    assert(meta.get_string("product.name") == "intensity");
    assert(meta.get_string("product.device") == "detector_A");
    assert(meta.get_int("product.run_id") == 42);
    assert(meta.get_string("instrument.firmware") == "v3.1.2");
    assert(meta.get_string("pipeline") == "detector-v3");
    std::printf("Generic-namespace assertions passed.\n");

    return 0;
}
