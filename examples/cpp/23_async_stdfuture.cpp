// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 23_async_stdfuture.cpp
/// @brief Example 23 — Asynchronous read via the std::future frontend.
///
/// The std::future frontend (`tensogram/async/std_future.hpp`) returns
/// a `std::future<T>` from every operation.  Call `.get()` to block
/// until completion; failures surface as the typed `tensogram::error`
/// hierarchy thrown out of `.get()`.  This is the most concise async
/// surface when you simply want to wait for a result inline.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/23_async_stdfuture

#include <tensogram.hpp>
#include <tensogram/async/std_future.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <future>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace tsf = tensogram::stdfuture;

namespace {

std::filesystem::path make_sample_file() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tensogram_example_23_" + std::to_string(std::random_device{}()) + ".tgm");
    const std::string meta = R"({
        "descriptors":[{"type":"ntensor","ndim":1,"shape":[4],"strides":[4],
        "dtype":"float32","encoding":"none","filter":"none","compression":"none"}],
        "base":[{"mars":{"param":"2t"}}]
    })";
    std::vector<float> values{280.0f, 281.5f, 282.25f, 283.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objs = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}};
    auto f = tensogram::file::create(path.string());
    f.append(meta, objs);
    f.append(meta, objs);
    return path;
}

}  // namespace

int main() {
    const auto path = make_sample_file();
    std::printf("Wrote %s\n", path.string().c_str());

    // -- 1. Open, count, decode — each call returns a std::future<T> --
    auto file = tsf::async_file::open(path.string()).get();
    std::printf("Opened asynchronously\n");

    const std::size_t count = file.message_count().get();
    std::printf("Messages: %zu\n", count);
    assert(count == 2);

    auto msg = file.decode_message(0).get();
    auto obj = msg.object(0);
    const float* data = obj.data_as<float>();
    std::printf("Decoded message 0: %zu object(s), first value %.2f\n",
                msg.num_objects(), static_cast<double>(data[0]));
    assert(msg.num_objects() == 1);

    // -- 2. Failures surface as exceptions through .get() --
    try {
        auto missing = tsf::async_file::open("/nonexistent/missing.tgm").get();
        (void)missing;
        std::fprintf(stderr, "expected an exception for a missing file\n");
        return 1;
    } catch (const tensogram::error& e) {
        std::printf("Open of a missing file threw as expected: %s\n", e.what());
    }

    std::remove(path.string().c_str());
    std::printf("SUCCESS\n");
    return 0;
}
