// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 21_async_consumer.cpp
/// @brief Example 21 — Asynchronous streaming consumer (coroutine frontend).
///
/// Models the HPC consumer: a job reads each message from a `.tgm` as
/// soon as it is available, overlapping decode with the next fetch.
/// `tco::async_for_each` walks every message via C++20 coroutines and
/// hands each decoded `tensogram::message` to your callback.  The
/// producer half is example 20.
///
/// To stay self-contained the example first writes a small
/// multi-message file with the synchronous API, then consumes it
/// asynchronously.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/21_async_consumer

#include <tensogram.hpp>
#include <tensogram/async/coro.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace tco = tensogram::coro;

namespace {

std::filesystem::path make_multi_message_file(int messages) {
    const auto path = std::filesystem::temp_directory_path() /
        ("tensogram_example_21_" + std::to_string(std::random_device{}()) + ".tgm");
    const std::string meta = R"({
        "descriptors":[{"type":"ntensor","ndim":1,"shape":[8],"strides":[4],
        "dtype":"float32","encoding":"none","filter":"none","compression":"none"}],
        "base":[{"mars":{"param":"2t"}}]
    })";
    auto f = tensogram::file::create(path.string());
    for (int m = 0; m < messages; ++m) {
        std::vector<float> values(8);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(m) + 0.1f * static_cast<float>(i);
        }
        std::vector<std::pair<const std::uint8_t*, std::size_t>> objs = {
            {reinterpret_cast<const std::uint8_t*>(values.data()),
             values.size() * sizeof(float)}};
        f.append(meta, objs);
    }
    return path;
}

}  // namespace

int main() {
    constexpr int MESSAGES = 5;
    const auto path = make_multi_message_file(MESSAGES);
    const std::string path_str = path.string();
    std::printf("Wrote %d messages to %s\n", MESSAGES, path_str.c_str());

    auto consume = [&]() -> tco::task<std::size_t> {
        auto file = co_await tco::async_file::open(path_str);
        std::size_t seen = 0;
        co_await tco::async_for_each(file, [&seen](tensogram::message m) {
            auto obj = m.object(0);
            const float* data = obj.data_as<float>();
            std::printf("  message %zu: %zu object(s), first value %.1f\n",
                        seen, m.num_objects(), static_cast<double>(data[0]));
            ++seen;
        });
        co_return seen;
    };

    const std::size_t seen = tco::block_on(consume());
    std::printf("Consumed %zu messages\n", seen);
    assert(seen == static_cast<std::size_t>(MESSAGES));

    std::remove(path_str.c_str());
    std::printf("SUCCESS\n");
    return 0;
}
