// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 19_async_decode_remote.cpp
/// @brief Example 19 — Asynchronous remote decode (coroutine frontend).
///
/// Opens a `.tgm` over the object-store backend and decodes a message,
/// all via C++20 coroutines (`tensogram/async/coro.hpp`).  Production
/// callers point `open_remote` at `s3://`, `gs://`, `az://`, or
/// `https://` URLs; this example uses a `file://` URL so it runs
/// offline and deterministically through the very same code path.
///
/// Requires the FFI built with the `async-remote` Cargo feature.
/// Configure CMake with `-DTENSOGRAM_ASYNC_REMOTE=ON`, which adds
/// `--features=async-remote` to the cargo build and enables this
/// example target.
///
/// For a real object store you would instead write, e.g.:
///
///   std::vector<std::pair<std::string, std::string>> opts = {
///       {"aws_region", "eu-west-1"},
///   };
///   auto file = co_await tco::async_file::open_remote(
///       "s3://my-bucket/forecast.tgm", opts, /*bidirectional=*/true);
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DTENSOGRAM_ASYNC_REMOTE=ON
///   cmake --build build -j
///   ./build/bin/19_async_decode_remote

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

std::filesystem::path make_sample_file() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tensogram_example_19_" + std::to_string(std::random_device{}()) + ".tgm");
    const std::string meta = R"({
        "descriptors":[{"type":"ntensor","ndim":1,"shape":[4],"strides":[4],
        "dtype":"float32","encoding":"none","filter":"none","compression":"none"}],
        "base":[{"mars":{"param":"2t"}}]
    })";
    std::vector<float> values{300.0f, 301.0f, 302.0f, 303.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objs = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}};
    auto f = tensogram::file::create(path.string());
    f.append(meta, objs);
    return path;
}

}  // namespace

int main() {
    const auto path = make_sample_file();
    // object_store's local backend expects an absolute `file://` URL.
    const std::string url = "file://" + std::filesystem::absolute(path).string();
    std::printf("Reading %s\n", url.c_str());

    auto run = [&url]() -> tco::task<int> {
        // No storage options needed for file://; for s3:// etc. pass
        // credentials/region here or rely on ambient configuration.
        auto file = co_await tco::async_file::open_remote(
            url, /*storage_options=*/{}, /*bidirectional=*/false);
        const std::size_t count = co_await file.message_count();
        auto msg = co_await file.decode_message(0);
        auto obj = msg.object(0);
        const float* data = obj.data_as<float>();
        std::printf("Remote open OK: %zu message(s), first value %.1f\n",
                    count, static_cast<double>(data[0]));
        co_return static_cast<int>(msg.num_objects());
    };

    try {
        const int objects = tco::block_on(run());
        assert(objects == 1);
    } catch (const tensogram::error& e) {
        // Most likely cause if you see this: the FFI was built without
        // the `async-remote` feature (configure with
        // -DTENSOGRAM_ASYNC_REMOTE=ON).
        std::fprintf(stderr, "remote decode failed (code %d): %s\n",
                     static_cast<int>(e.code()), e.what());
        std::remove(path.string().c_str());
        return 1;
    }

    std::remove(path.string().c_str());
    std::printf("SUCCESS\n");
    return 0;
}
