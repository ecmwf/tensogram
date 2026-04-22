// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 07_scan_buffer.cpp
/// @brief Example 07 — Scanning a multi-message buffer (C++ wrapper).
///
/// A `.tgm` file (or any byte buffer) is a flat sequence of independent
/// messages.  `tensogram::scan()` finds each message's `(offset, length)`
/// by locating magic markers and cross-checking the terminator — it
/// tolerates and skips corrupt regions between messages.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/07_scan_buffer

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<std::uint8_t> make_message(const char* param, int step) {
    char json[512];
    std::snprintf(json, sizeof(json),
        R"({"version":3,"descriptors":[{"type":"ntensor","ndim":1,"shape":[10],"strides":[1],"dtype":"float32","encoding":"none","filter":"none","compression":"none"}],"base":[{"mars":{"param":"%s","step":%d}}]})",
        param, step);

    std::vector<float> data(10, 0.0f);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(data.data()),
         data.size() * sizeof(float)}
    };
    return tensogram::encode(json, objects);
}

int main() {
    // ── 1. Build a buffer with 5 messages ─────────────────────────────────
    const std::pair<const char*, int> params_steps[] = {
        {"2t",  0}, {"10u", 0}, {"2t",  6}, {"10u", 6}, {"msl", 0},
    };

    std::vector<std::vector<std::uint8_t>> messages;
    messages.reserve(5);
    std::size_t total_clean = 0;
    for (auto [p, s] : params_steps) {
        auto msg = make_message(p, s);
        total_clean += msg.size();
        messages.push_back(std::move(msg));
    }
    std::printf("5 messages, %zu bytes total (clean)\n", total_clean);

    // ── 2. Inject 256 bytes of garbage between messages 1 and 2 ───────────
    //
    // Garbage in the middle of a file should not prevent reading the
    // messages after it.  scan() recovers by skipping byte-by-byte when
    // the terminator cross-check fails.
    std::vector<std::uint8_t> corrupted;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        corrupted.insert(corrupted.end(), messages[i].begin(), messages[i].end());
        if (i == 1) {
            std::vector<std::uint8_t> garbage(256, 0xDEu);
            corrupted.insert(corrupted.end(), garbage.begin(), garbage.end());
            std::printf("  (injected 256 garbage bytes after message 1)\n");
        }
    }
    std::printf("Buffer size with corruption: %zu bytes\n\n", corrupted.size());

    // ── 3. Scan ───────────────────────────────────────────────────────────
    auto entries = tensogram::scan(corrupted.data(), corrupted.size());
    std::printf("scan() found %zu valid messages:\n", entries.size());
    assert(entries.size() == 5);

    // ── 4. Decode each message from its scanned offset ────────────────────
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        const std::uint8_t* slice = corrupted.data() + e.offset;
        auto meta = tensogram::decode_metadata(slice, e.length);
        std::printf("  [%zu] offset=%6zu  len=%6zu  param=%-5s  step=%lld\n",
                    i, e.offset, e.length,
                    meta.get_string("mars.param").c_str(),
                    static_cast<long long>(meta.get_int("mars.step")));
    }

    std::printf("\nAll 5 messages decoded correctly despite 256 bytes of injected garbage.\n");
    return 0;
}
