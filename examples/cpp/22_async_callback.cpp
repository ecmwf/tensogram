// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 22_async_callback.cpp
/// @brief Example 22 — Asynchronous read via the callback frontend.
///
/// The callback frontend (`tensogram/async/callback.hpp`) is the
/// always-available C++17 async surface.  Every operation takes a
/// completion handler that fires exactly once, on the FFI dispatcher
/// pool, when the work resolves.  The contract: do as little as
/// possible inside the handler — signal a promise / condition variable
/// / coroutine handle and return.
///
/// This example opens a local `.tgm`, counts its messages, and decodes
/// the first one, chaining three callbacks and bridging each result
/// back to `main()` through a `std::promise`.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/22_async_callback

#include <tensogram.hpp>
#include <tensogram/async/callback.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <future>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace tac = tensogram::async_callback;

namespace {

// Write a small two-message float32 file with the synchronous API so the
// async reader below has something to open.
std::filesystem::path make_sample_file() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tensogram_example_22_" + std::to_string(std::random_device{}()) + ".tgm");
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
    return path;  // destructor flushes + closes before we return
}

}  // namespace

int main() {
    const auto path = make_sample_file();
    std::printf("Wrote %s\n", path.string().c_str());

    // -- 1. Open asynchronously --
    //
    // The completion handler runs on the dispatcher pool, not this
    // thread.  We hand the result back to main() through a promise and
    // block on its future — the canonical "signal and return" shape.
    std::promise<tac::result<tac::async_file>> open_promise;
    auto open_future = open_promise.get_future();
    tac::async_file::open(path.string(),
        [&open_promise](tac::result<tac::async_file> r) {
            open_promise.set_value(std::move(r));
        });
    auto open_result = open_future.get();
    if (!open_result.ok()) {
        std::fprintf(stderr, "open failed: %s\n", open_result.message().c_str());
        return 1;
    }
    auto file = open_result.take();
    std::printf("Opened asynchronously\n");

    // -- 2. Count messages asynchronously --
    std::promise<tac::result<std::size_t>> count_promise;
    auto count_future = count_promise.get_future();
    file.message_count([&count_promise](tac::result<std::size_t> r) {
        count_promise.set_value(std::move(r));
    });
    auto count_result = count_future.get();
    assert(count_result.ok());
    const std::size_t count = count_result.value();
    std::printf("Messages: %zu\n", count);
    assert(count == 2);

    // -- 3. Decode message 0 asynchronously --
    std::promise<tac::result<tensogram::message>> decode_promise;
    auto decode_future = decode_promise.get_future();
    file.decode_message(0, [&decode_promise](tac::result<tensogram::message> r) {
        decode_promise.set_value(std::move(r));
    });
    auto decode_result = decode_future.get();
    assert(decode_result.ok());
    auto& msg = decode_result.value();
    auto obj = msg.object(0);
    const float* data = obj.data_as<float>();
    std::printf("Decoded message 0: %zu object(s), first value %.2f\n",
                msg.num_objects(), static_cast<double>(data[0]));
    assert(msg.num_objects() == 1);

    std::remove(path.string().c_str());
    std::printf("SUCCESS\n");
    return 0;
}
