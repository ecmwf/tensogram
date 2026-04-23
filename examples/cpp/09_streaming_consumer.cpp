// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 09_streaming_consumer.cpp
/// @brief Example 09 — Streaming consumer: decode tensograms as bytes arrive.
///
/// Demonstrates consumer-side streaming: read a `.tgm` byte stream in
/// small chunks, scan the growing buffer for complete messages on each
/// chunk boundary, decode each completed message immediately, and
/// discard consumed bytes.  Peak buffer is a single message, not the
/// whole file.
///
/// A real-world caller would read the chunks from an HTTP response, a
/// socket, or a pipe.  This example drives the same pattern off a
/// local `.tgm` file to keep the example self-contained.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/09_streaming_consumer

#include <tensogram.hpp>

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

static std::string descriptor_json(const char* param) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        R"({"version":3,"descriptors":[{"type":"ntensor","ndim":2,"shape":[181,360],"strides":[360,1],"dtype":"float32","encoding":"none","filter":"none","compression":"none"}],"base":[{"mars":{"param":"%s","step":0}}]})",
        param);
    return buf;
}

int main() {
    const auto tmp = std::filesystem::temp_directory_path() /
                     ("tensogram_example_09_" + std::to_string(::getpid()) + ".tgm");
    const std::string tgm_path_str = tmp.string();
    const char* tgm_path = tgm_path_str.c_str();
    const char* PARAMS[] = {"2t", "10u", "10v", "msl"};

    // ── 1. Create a multi-message .tgm file on disk ───────────────────────
    {
        auto f = tensogram::file::create(tgm_path);
        std::vector<float> data(181 * 360, 0.0f);  // filled below per param
        for (std::size_t i = 0; i < std::size(PARAMS); ++i) {
            // Distinct sentinel value per param so the consumer can verify
            // the right payload landed against the right metadata.
            for (auto& v : data) v = static_cast<float>(i) + 0.5f;
            std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
                {reinterpret_cast<const std::uint8_t*>(data.data()),
                 data.size() * sizeof(float)}
            };
            f.append(descriptor_json(PARAMS[i]), objects);
        }
    }
    std::ifstream probe(tgm_path, std::ios::binary | std::ios::ate);
    const auto total_bytes = static_cast<std::size_t>(probe.tellg());
    probe.close();
    std::printf("Wrote %s: %zu bytes, %zu messages\n",
                tgm_path, total_bytes, std::size(PARAMS));

    // ── 2. Stream from disk in small chunks, decode as messages complete ──
    //
    // Strategy: grow a rolling buffer, call scan() on each new chunk,
    // decode any messages scan reports, then drop consumed bytes from
    // the head of the buffer.  scan() tolerates a partial message at
    // the buffer tail — it just won't appear in the result until the
    // last byte arrives.
    std::printf("\nStreaming from %s (chunk size = 4096)\n", tgm_path);
    std::printf("%s\n", std::string(60, '-').c_str());

    std::ifstream in(tgm_path, std::ios::binary);
    std::vector<std::uint8_t> buffer;
    std::array<char, 4096> chunk{};
    std::size_t messages_decoded = 0;

    while (in) {
        in.read(chunk.data(), chunk.size());
        const auto n = static_cast<std::size_t>(in.gcount());
        if (n == 0) break;
        buffer.insert(buffer.end(), chunk.begin(), chunk.begin() + n);

        auto entries = tensogram::scan(buffer.data(), buffer.size());
        std::size_t last_end = 0;
        for (const auto& e : entries) {
            const std::uint8_t* slice = buffer.data() + e.offset;
            auto msg = tensogram::decode(slice, e.length, {});
            auto obj = msg.object(0);
            auto meta = msg.get_metadata();
            ++messages_decoded;
            std::printf("  message %zu: param=%-5s  shape=[%zu,%zu]  %zu bytes\n",
                        messages_decoded,
                        meta.get_string("mars.param").c_str(),
                        obj.shape()[0], obj.shape()[1],
                        obj.data_size());
            last_end = e.offset + e.length;
        }
        if (last_end > 0) {
            // Drop consumed bytes; keep any partial-message tail.
            buffer.erase(buffer.begin(), buffer.begin() + last_end);
        }
    }

    std::printf("%s\n", std::string(60, '-').c_str());
    std::printf("Total: %zu messages decoded from stream\n", messages_decoded);
    std::printf("Peak buffer: single message (not whole file)\n");

    assert(messages_decoded == std::size(PARAMS));
    std::remove(tgm_path);
    return 0;
}
