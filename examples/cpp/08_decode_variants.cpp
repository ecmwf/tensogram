// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 08_decode_variants.cpp
/// @brief Example 08 — All decode variants (C++ wrapper).
///
/// Tensogram provides four decode entry points to match different
/// use-cases:
///
///   decode()          — all objects, full pipeline
///   decode_metadata() — global CBOR only, no payload bytes touched
///   decode_object()   — single object by index (O(1) via binary header)
///   decode_range()    — contiguous sub-slice of an uncompressed object
///
/// This example builds a 3-object message and exercises each one.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/08_decode_variants

#include <tensogram.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    // ── Build a 3-object message ──────────────────────────────────────────
    //
    // Each object is filled with a distinct byte so the decoded payload
    // is trivial to verify visually.
    const std::string metadata_json = R"({
        "version": 3,
        "descriptors": [
            {"type":"ntensor","ndim":2,"shape":[10,20],"strides":[20,1],
             "dtype":"float32",
             "encoding":"none","filter":"none","compression":"none"},
            {"type":"ntensor","ndim":1,"shape":[5],"strides":[1],
             "dtype":"float64",
             "encoding":"none","filter":"none","compression":"none"},
            {"type":"ntensor","ndim":2,"shape":[8,8],"strides":[8,1],
             "dtype":"uint8",
             "encoding":"none","filter":"none","compression":"none"}
        ],
        "base": [{}, {}, {}]
    })";

    std::vector<std::uint8_t> data0(10 * 20 * 4, 0xAAu);  // float32
    std::vector<std::uint8_t> data1(5 * 8,       0xBBu);  // float64
    std::vector<std::uint8_t> data2(8 * 8,       0xCCu);  // uint8

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {data0.data(), data0.size()},
        {data1.data(), data1.size()},
        {data2.data(), data2.size()},
    };
    auto message = tensogram::encode(metadata_json, objects);
    std::printf("Message: %zu bytes  (3 objects)\n\n", message.size());

    // ── decode_metadata() — no payload bytes are touched ──────────────────
    //
    // Use this to filter/list large files quickly: it parses the preamble
    // and the global CBOR only, never reaching into payload regions.
    {
        auto meta = tensogram::decode_metadata(message.data(), message.size());
        std::printf("decode_metadata():\n");
        std::printf("  objects=%zu\n", meta.num_objects());
    }

    // ── decode() — full decode of every object ────────────────────────────
    {
        auto msg = tensogram::decode(message.data(), message.size(), {});
        std::printf("\ndecode() — all objects:\n");
        for (std::size_t i = 0; i < msg.num_objects(); ++i) {
            auto obj = msg.object(i);
            const std::uint8_t* bytes = obj.data_as<std::uint8_t>();
            std::printf(
                "  [%zu] dtype=%s  shape=[%zu,%zu]  %zu bytes  first_byte=0x%02X\n",
                i, obj.dtype_string().c_str(),
                obj.shape()[0], obj.ndim() > 1 ? obj.shape()[1] : 1,
                obj.data_size(), bytes[0]);
        }
    }

    // ── decode_object() — a single object by index, O(1) seek ─────────────
    //
    // The binary header stores every object's byte offset, so the
    // decoder skips directly to the requested object without touching
    // the others.
    {
        auto one = tensogram::decode_object(message.data(), message.size(), 1, {});
        auto obj = one.object(0);
        const double* payload = obj.data_as<double>();
        std::printf("\ndecode_object(index=1):\n");
        std::printf("  %zu bytes, dtype=%s, first_byte=0x%02X\n",
                    obj.data_size(), obj.dtype_string().c_str(),
                    reinterpret_cast<const std::uint8_t*>(payload)[0]);

        // Out-of-range index throws object_error.
        try {
            tensogram::decode_object(message.data(), message.size(), 99, {});
            std::fprintf(stderr, "expected object_error for index=99\n");
            return 1;
        } catch (const tensogram::object_error& e) {
            std::printf("  index=99 → object_error: %s\n", e.what());
        }
    }

    // ── decode_range() — partial sub-slice of an uncompressed object ──────
    //
    // decode_range works on encoding="none" and compression="none"
    // objects.  Ranges are (element_offset, element_count) pairs.
    {
        auto parts = tensogram::decode_range(
            message.data(), message.size(),
            2,                            // object index (uint8, 8×8)
            {{10u, 8u}},                  // one range: offset 10, count 8
            {});
        assert(parts.size() == 1);
        std::printf("\ndecode_range(obj=2, offset=10, count=8) [split]:\n");
        std::printf("  1 part, %zu bytes\n", parts[0].size());

        auto two = tensogram::decode_range(
            message.data(), message.size(),
            2,
            {{0u, 4u}, {60u, 4u}},        // first 4 + last 4 of the 8×8 grid
            {});
        std::printf("  Two ranges [(0,4),(60,4)]: %zu parts\n", two.size());

        auto joined = tensogram::decode_range_joined(
            message.data(), message.size(),
            2,
            {{0u, 4u}, {60u, 4u}},
            {});
        std::printf("  Joined: %zu bytes total\n", joined.size());
    }

    std::printf("\nAll decode variants OK.\n");
    return 0;
}
