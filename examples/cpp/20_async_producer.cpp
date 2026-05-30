// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 20_async_producer.cpp
/// @brief Example 20 — Asynchronous streaming producer (coroutine frontend).
///
/// Models the HPC producer: a job emits forecast steps as they are
/// computed, streaming each into a `.tgm` without buffering the whole
/// message.  Uses the C++20 coroutine streaming encoder.  The consumer
/// half is example 21.
///
/// Each `write_object` appends one data object to the in-flight
/// message; `finish(backfill=true)` writes the footer and back-fills
/// the preamble/postamble lengths so the result is a fully
/// random-access `.tgm`.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/20_async_producer

#include <tensogram.hpp>
#include <tensogram/async/coro.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace tco = tensogram::coro;

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tensogram_example_20_" + std::to_string(std::random_device{}()) + ".tgm");
    const std::string path_str = path.string();

    constexpr int STEPS = 8;
    constexpr int NX = 16;
    constexpr int NY = 16;

    auto produce = [&]() -> tco::task<void> {
        auto enc = co_await tco::async_streaming_encoder::create(
            path_str, R"({"base":[]})");

        // Byte strides, row-major: inner dim = sizeof(float), outer = NY*4.
        const std::string desc = R"({
            "type":"ntensor","ndim":2,"shape":[16,16],"strides":[64,4],
            "dtype":"float32","encoding":"none","filter":"none",
            "compression":"none","params":{}
        })";

        for (int step = 0; step < STEPS; ++step) {
            std::vector<float> field(static_cast<std::size_t>(NX) * NY);
            for (std::size_t i = 0; i < field.size(); ++i) {
                field[i] = static_cast<float>(step) +
                           0.001f * static_cast<float>(i);
            }
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(field.data());
            co_await enc.write_object(desc, bytes, field.size() * sizeof(float));
            std::printf("  streamed step %d\n", step);
        }
        co_await enc.finish(/*backfill=*/true);
    };

    tco::block_on(produce());
    std::printf("Producer finished: %d steps -> %s\n", STEPS, path_str.c_str());

    // Verify by decoding the finished file with the synchronous API.
    std::ifstream in(path_str, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    auto msg = tensogram::decode(bytes.data(), bytes.size());
    std::printf("Read back: %zu objects in one message\n", msg.num_objects());
    assert(msg.num_objects() == static_cast<std::size_t>(STEPS));

    std::remove(path_str.c_str());
    std::printf("SUCCESS\n");
    return 0;
}
