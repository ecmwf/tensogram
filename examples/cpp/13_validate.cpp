// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 13_validate.cpp
/// @brief Example 13 — Message validation at different levels (C++ wrapper).
///
/// `tensogram::validate()` runs a tiered check on a message buffer and
/// returns a JSON report with `issues`, `object_count`, and
/// `hash_verified`.  Validation findings are part of the report —
/// only operational failures (invalid arguments, I/O errors) throw.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/13_validate

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main() {
    // ── 1. Encode a clean message ─────────────────────────────────────────
    const std::string metadata_json = R"({
        "version": 3,
        "descriptors": [{
            "type": "ntensor", "ndim": 1, "shape": [4], "strides": [4],
            "dtype": "float32",
            "encoding": "none", "filter": "none", "compression": "none"
        }]
    })";

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(data.data()),
         data.size() * sizeof(float)}
    };
    auto msg = tensogram::encode(metadata_json, objects);

    // ── 2. Default level (integrity — hash slots are verified) ────────────
    std::printf("=== validate(default) ===\n");
    auto report = tensogram::validate(msg.data(), msg.size());
    std::printf("%s\n", report.c_str());

    // ── 3. All four levels ────────────────────────────────────────────────
    std::printf("\n=== validate at every level ===\n");
    for (const char* level : {"quick", "default", "checksum", "full"}) {
        auto r = tensogram::validate(msg.data(), msg.size(), level);
        // Trim to first 80 chars for a compact summary line.
        std::string head = r.size() > 80 ? r.substr(0, 80) + "..." : r;
        std::printf("  %-10s  %s\n", level, head.c_str());
    }

    // ── 4. Corruption detection ───────────────────────────────────────────
    //
    // Overwrite the preamble magic — `tensogram::validate` reports a
    // level-1 structural issue in its JSON rather than throwing.
    std::printf("\n=== validate(corrupted magic) ===\n");
    std::vector<std::uint8_t> corrupted = msg;
    std::memcpy(corrupted.data(), "WRONGMAG", 8);
    auto bad_report = tensogram::validate(corrupted.data(), corrupted.size());
    std::printf("%s\n", bad_report.c_str());

    return 0;
}
