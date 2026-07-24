// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 25_async_decode_object_range.cpp
/// @brief Example 25 — Async single-object + partial-range decode
///        (callback frontend), plus path() and the pull-model task handle.
///
/// Builds on 22_async_callback by exercising the finer-grained async read
/// entry points:
///
///   async_file::decode_object(msg, obj)   — one object from a message
///   async_file::decode_range(msg, obj, …) — sub-ranges of an object
///   async_file::path()                    — the opened file's path
///   async_file::decode_object_task(…)     — pull-model handle you can
///                                           poll (ready()), cancel(), join()
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/25_async_decode_object_range

#include <tensogram.hpp>
#include <tensogram/async/callback.hpp>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <future>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace tac = tensogram::async_callback;

namespace {

// A one-message, three-object file.  Object 2 is a byte-addressable
// uint8[64] tensor (byte i == i) so range decode is easy to verify.
std::filesystem::path make_sample_file() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tensogram_example_25_" + std::to_string(std::random_device{}()) + ".tgm");
    const std::string meta = R"({
        "descriptors": [
            {"type":"ntensor","ndim":1,"shape":[4],"strides":[4],
             "dtype":"float32","encoding":"none","filter":"none","compression":"none"},
            {"type":"ntensor","ndim":1,"shape":[64],"strides":[1],
             "dtype":"uint8","encoding":"none","filter":"none","compression":"none"}
        ],
        "base": [{}, {}]
    })";
    std::vector<float> f0{280.0f, 281.5f, 282.25f, 283.0f};
    std::vector<std::uint8_t> u1(64);
    for (std::size_t i = 0; i < u1.size(); ++i) u1[i] = static_cast<std::uint8_t>(i);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objs = {
        {reinterpret_cast<const std::uint8_t*>(f0.data()), f0.size() * sizeof(float)},
        {u1.data(), u1.size()},
    };
    auto f = tensogram::file::create(path.string());
    f.append(meta, objs);
    return path;
}

// Bridge a callback launch to a future so main() can block on each step.
template <typename T>
tac::result<T> await(std::function<void(std::function<void(tac::result<T>)>)> launch) {
    std::promise<tac::result<T>> p;
    auto fut = p.get_future();
    launch([&p](tac::result<T> r) { p.set_value(std::move(r)); });
    return fut.get();
}

}  // namespace

int main() {
    const auto path = make_sample_file();
    std::printf("Wrote %s\n", path.string().c_str());

    // -- Open, and read back the path the handle remembers --
    auto open_r = await<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); });
    if (!open_r.ok()) {
        std::fprintf(stderr, "open failed: %s\n", open_r.message().c_str());
        return 1;
    }
    auto file = open_r.take();
    std::printf("async_file::path() = %s\n", file.path().c_str());

    // -- decode_object: pull just object 1 (the uint8[64] tensor) --
    auto obj_r = await<tensogram::message>(
        [&](auto cb) { file.decode_object(0, 1, std::move(cb)); });
    assert(obj_r.ok());
    {
        auto obj = obj_r.value().object(0);
        std::printf("decode_object(0, 1): dtype=%s, %zu bytes, byte[7]=%u\n",
                    obj.dtype_string().c_str(), obj.data_size(),
                    static_cast<unsigned>(obj.data()[7]));
        assert(obj.data_size() == 64);
        assert(obj.data()[7] == 7);
    }

    // -- decode_range: two sub-slices of object 1, one buffer each --
    {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {{0, 4}, {60, 4}};
        auto rng_r = await<std::vector<std::vector<std::uint8_t>>>(
            [&](auto cb) { file.decode_range(0, 1, ranges, std::move(cb)); });
        assert(rng_r.ok());
        const auto& parts = rng_r.value();
        std::printf("decode_range(0, 1, [(0,4),(60,4)]): %zu buffers, "
                    "first byte %u, last byte %u\n",
                    parts.size(), static_cast<unsigned>(parts.front().front()),
                    static_cast<unsigned>(parts.back().back()));
        assert(parts.size() == 2);
        assert(parts.back().back() == 63);
    }

    // -- Pull-model handle: poll ready(), then join() --
    {
        auto task = file.decode_object_task(0, 0);  // float32[4]
        for (int i = 0; i < 1000 && !task.ready(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::printf("decode_object_task ready() = %s\n",
                    task.ready() ? "true" : "false");
        auto r = task.join();
        assert(r.ok());
        std::printf("  joined: %zu object(s), first value %.2f\n",
                    r.value().num_objects(),
                    static_cast<double>(r.value().object(0).data_as<float>()[0]));
    }

    std::remove(path.string().c_str());
    std::printf("SUCCESS\n");
    return 0;
}
