// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF
//
// Async single-object + partial-range decode across the callback and
// std::future frontends (tgm_async_file_decode_object /
// tgm_async_file_decode_range / tgm_async_task_join_multi_bytes).

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include <tensogram/async/callback.hpp>
#include <tensogram/async/std_future.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <string>
#include <utility>
#include <vector>

namespace tac = tensogram::async_callback;
namespace tsf = tensogram::stdfuture;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    auto tmp = std::filesystem::temp_directory_path();
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return tmp / ("tensogram_objrange_" + std::to_string(t) + suffix);
}

/// Write a one-message, three-object file:
///   obj0 = float32[4], obj1 = float64[3], obj2 = uint8[64] (byte i == i).
/// obj2 is `encoding=none`/`compression=none` so decode_range applies.
std::filesystem::path make_multi_object_file() {
    auto path = make_temp_path(".tgm");
    const std::string meta = R"({
        "version": 3,
        "descriptors": [
            {"type":"ntensor","ndim":1,"shape":[4],"strides":[4],
             "dtype":"float32","encoding":"none","filter":"none","compression":"none"},
            {"type":"ntensor","ndim":1,"shape":[3],"strides":[8],
             "dtype":"float64","encoding":"none","filter":"none","compression":"none"},
            {"type":"ntensor","ndim":1,"shape":[64],"strides":[1],
             "dtype":"uint8","encoding":"none","filter":"none","compression":"none"}
        ],
        "base": [{}, {}, {}]
    })";

    std::vector<float> f0{10.0f, 11.0f, 12.0f, 13.0f};
    std::vector<double> f1{100.0, 200.0, 300.0};
    std::vector<std::uint8_t> u2(64);
    for (std::size_t i = 0; i < u2.size(); ++i) u2[i] = static_cast<std::uint8_t>(i);

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objs = {
        {reinterpret_cast<const std::uint8_t*>(f0.data()), f0.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(f1.data()), f1.size() * sizeof(double)},
        {u2.data(), u2.size()},
    };
    auto f = tensogram::file::create(path.string());
    f.append(meta, objs);
    return path;
}

/// Bridge a callback-style launch into a std::future for terse asserts.
template <typename T>
std::future<tac::result<T>> as_future(
    std::function<void(std::function<void(tac::result<T>)>)> launch) {
    auto promise = std::make_shared<std::promise<tac::result<T>>>();
    auto fut = promise->get_future();
    launch([promise](tac::result<T> r) { promise->set_value(std::move(r)); });
    return fut;
}

using range_buffers = std::vector<std::vector<std::uint8_t>>;

}  // namespace

// ── callback frontend ────────────────────────────────────────────────

TEST(AsyncDecodeObject, CallbackDecodesSingleObject) {
    auto path = make_multi_object_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();

    // Object 2 is the uint8[64] tensor.
    auto r = as_future<tensogram::message>(
        [&](auto cb) { file.decode_object(0, 2, std::move(cb)); }).get();
    ASSERT_TRUE(r.ok()) << r.message();
    auto& msg = r.value();
    ASSERT_EQ(msg.num_objects(), 1u);
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "uint8");
    ASSERT_EQ(obj.data_size(), 64u);
    EXPECT_EQ(obj.data()[7], 7u);

    std::filesystem::remove(path);
}

TEST(AsyncDecodeObject, CallbackOutOfRangeObjectReportsError) {
    auto path = make_multi_object_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();

    auto r = as_future<tensogram::message>(
        [&](auto cb) { file.decode_object(0, 99, std::move(cb)); }).get();
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.code(), TGM_ERROR_OK);

    std::filesystem::remove(path);
}

TEST(AsyncDecodeRange, CallbackSplitRangesReturnOneBufferEach) {
    auto path = make_multi_object_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {{10, 8}};
    auto r = as_future<range_buffers>(
        [&](auto cb) { file.decode_range(0, 2, ranges, std::move(cb)); }).get();
    ASSERT_TRUE(r.ok()) << r.message();
    ASSERT_EQ(r.value().size(), 1u);
    ASSERT_EQ(r.value()[0].size(), 8u);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(r.value()[0][i], static_cast<std::uint8_t>(10 + i));
    }

    std::filesystem::remove(path);
}

TEST(AsyncDecodeRange, CallbackMultipleRangesYieldMultipleBuffers) {
    auto path = make_multi_object_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {{0, 4}, {60, 4}};
    auto r = as_future<range_buffers>(
        [&](auto cb) { file.decode_range(0, 2, ranges, std::move(cb)); }).get();
    ASSERT_TRUE(r.ok()) << r.message();
    ASSERT_EQ(r.value().size(), 2u);
    EXPECT_EQ(r.value()[0].size(), 4u);
    EXPECT_EQ(r.value()[1].size(), 4u);
    EXPECT_EQ(r.value()[1][3], 63u);  // last byte of the object

    std::filesystem::remove(path);
}

// ── std::future frontend ─────────────────────────────────────────────

TEST(AsyncDecodeObject, StdFutureDecodesSingleObject) {
    auto path = make_multi_object_file();
    auto file = tsf::async_file::open(path.string()).get();

    auto msg = file.decode_object(0, 0).get();  // float32[4]
    ASSERT_EQ(msg.num_objects(), 1u);
    auto obj = msg.object(0);
    EXPECT_EQ(obj.dtype_string(), "float32");
    ASSERT_EQ(obj.element_count<float>(), 4u);
    EXPECT_FLOAT_EQ(obj.data_as<float>()[0], 10.0f);

    std::filesystem::remove(path);
}

TEST(AsyncDecodeObject, StdFutureOutOfRangeThrows) {
    auto path = make_multi_object_file();
    auto file = tsf::async_file::open(path.string()).get();
    auto fut = file.decode_object(0, 99);
    EXPECT_THROW(fut.get(), tensogram::error);
    std::filesystem::remove(path);
}

TEST(AsyncDecodeRange, StdFutureSplitRanges) {
    auto path = make_multi_object_file();
    auto file = tsf::async_file::open(path.string()).get();

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {{0, 4}, {60, 4}};
    auto parts = file.decode_range(0, 2, ranges).get();
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0].size(), 4u);
    EXPECT_EQ(parts[0][0], 0u);
    EXPECT_EQ(parts[1][3], 63u);

    std::filesystem::remove(path);
}
