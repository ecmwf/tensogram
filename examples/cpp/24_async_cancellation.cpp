// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 24_async_cancellation.cpp
/// @brief Example 24 — Cancellation tokens and timeouts (callback frontend).
///
/// Every async-launching call accepts an optional `cancellation_token*`
/// and a `std::chrono::milliseconds` timeout.  Cancelled or timed-out
/// operations complete with `TGM_ERROR_CANCELLED` / `TGM_ERROR_TIMEOUT`
/// instead of a value; a token may be shared across many in-flight
/// tasks ("cancel all my pending work").
///
/// Cancellation matters most for long-running remote fetches.  For the
/// fast local read used here there is an inherent race — the read may
/// finish before the cancellation is observed — so this example reports
/// whichever outcome occurs rather than asserting a single one.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/24_async_cancellation

#include <tensogram.hpp>
#include <tensogram/async/callback.hpp>

#include <cassert>
#include <chrono>
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

std::filesystem::path make_sample_file() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tensogram_example_24_" + std::to_string(std::random_device{}()) + ".tgm");
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
    return path;
}

// Open synchronously-from-async: launch the async open and block on it.
tac::async_file open_blocking(const std::string& path) {
    std::promise<tac::result<tac::async_file>> p;
    auto fut = p.get_future();
    tac::async_file::open(path,
        [&p](tac::result<tac::async_file> r) { p.set_value(std::move(r)); });
    return fut.get().take();
}

}  // namespace

int main() {
    const auto path = make_sample_file();
    auto file = open_blocking(path.string());

    // -- 1. cancellation_token lifecycle --
    tac::cancellation_token tok;
    assert(!tok.cancelled());
    tok.cancel();
    assert(tok.cancelled());
    std::printf("Token cancelled flag: %s\n", tok.cancelled() ? "true" : "false");

    // -- 2. Launch a decode with the already-cancelled token + a timeout --
    {
        std::promise<tac::result<tensogram::message>> p;
        auto fut = p.get_future();
        file.decode_message(0,
            [&p](tac::result<tensogram::message> r) { p.set_value(std::move(r)); },
            &tok,                                // already cancelled
            std::chrono::milliseconds{5000});    // 5s deadline
        auto r = fut.get();
        if (r.ok()) {
            std::printf("Completed before cancellation was observed "
                        "(expected for fast local reads): %zu object(s)\n",
                        r.value().num_objects());
        } else if (r.code() == TGM_ERROR_CANCELLED) {
            std::printf("Cancelled as requested (TGM_ERROR_CANCELLED)\n");
        } else if (r.code() == TGM_ERROR_TIMEOUT) {
            std::printf("Deadline elapsed (TGM_ERROR_TIMEOUT)\n");
        } else {
            std::fprintf(stderr, "unexpected error: %s\n", r.message().c_str());
            std::remove(path.string().c_str());
            return 1;
        }
    }

    // -- 3. A fresh, uncancelled token with a generous timeout succeeds --
    {
        tac::cancellation_token live;
        std::promise<tac::result<tensogram::message>> p;
        auto fut = p.get_future();
        file.decode_message(0,
            [&p](tac::result<tensogram::message> r) { p.set_value(std::move(r)); },
            &live, std::chrono::milliseconds{5000});
        auto r = fut.get();
        assert(r.ok());
        std::printf("Uncancelled decode succeeded: %zu object(s)\n",
                    r.value().num_objects());
    }

    std::remove(path.string().c_str());
    std::printf("SUCCESS\n");
    return 0;
}
