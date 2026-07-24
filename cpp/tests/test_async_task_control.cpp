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
// Async path accessors (tgm_async_file_path /
// tgm_async_streaming_encoder_path) and the pull-model task<T> handle
// (tgm_async_task_is_ready / tgm_async_task_cancel).

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
#include <thread>
#include <utility>
#include <vector>

namespace tac = tensogram::async_callback;
namespace tsf = tensogram::stdfuture;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    auto tmp = std::filesystem::temp_directory_path();
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return tmp / ("tensogram_taskctl_" + std::to_string(t) + suffix);
}

std::filesystem::path make_test_file() {
    auto path = make_temp_path(".tgm");
    const std::string meta = R"({
        "version": 3,
        "descriptors": [
            {"type":"ntensor","ndim":1,"shape":[64],"strides":[1],
             "dtype":"uint8","encoding":"none","filter":"none","compression":"none"}
        ],
        "base": [{}]
    })";
    std::vector<std::uint8_t> u(64);
    for (std::size_t i = 0; i < u.size(); ++i) u[i] = static_cast<std::uint8_t>(i);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objs = {{u.data(), u.size()}};
    auto f = tensogram::file::create(path.string());
    f.append(meta, objs);
    return path;
}

template <typename T>
std::future<tac::result<T>> as_future(
    std::function<void(std::function<void(tac::result<T>)>)> launch) {
    auto promise = std::make_shared<std::promise<tac::result<T>>>();
    auto fut = promise->get_future();
    launch([promise](tac::result<T> r) { promise->set_value(std::move(r)); });
    return fut;
}

}  // namespace

// ── path accessors ───────────────────────────────────────────────────

TEST(AsyncPath, CallbackAsyncFilePath) {
    auto path = make_test_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();
    const std::string reported = file.path();
    EXPECT_FALSE(reported.empty());
    EXPECT_NE(reported.find(path.filename().string()), std::string::npos)
        << "reported=" << reported << " expected to contain " << path.filename();
    std::filesystem::remove(path);
}

TEST(AsyncPath, StdFutureAsyncFilePath) {
    auto path = make_test_file();
    auto file = tsf::async_file::open(path.string()).get();
    EXPECT_NE(file.path().find(path.filename().string()), std::string::npos);
    std::filesystem::remove(path);
}

TEST(AsyncPath, StreamingEncoderPath) {
    auto path = make_temp_path(".tgm");
    auto enc = as_future<tac::async_streaming_encoder>([&](auto cb) {
        tac::async_streaming_encoder::create(path.string(), R"({"base":[]})", std::move(cb));
    }).get().take();
    const std::string reported = enc.path();
    EXPECT_FALSE(reported.empty());
    EXPECT_NE(reported.find(path.filename().string()), std::string::npos);
    // Drive a clean finish so the on-disk file is valid before cleanup.
    ASSERT_TRUE(as_future<void>([&](auto cb) { enc.finish(std::move(cb)); }).get().ok());
    std::filesystem::remove(path);
}

// ── pull-model task<T>: ready() / cancel() / join() ──────────────────

TEST(AsyncTaskControl, PollReadyThenJoin) {
    auto path = make_test_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();

    auto task = file.decode_object_task(0, 0);

    // Non-blocking poll: a local decode resolves well within the budget.
    bool became_ready = false;
    for (int i = 0; i < 5000; ++i) {
        if (task.ready()) { became_ready = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(became_ready);

    auto r = task.join();  // ready → returns immediately
    ASSERT_TRUE(r.ok()) << r.message();
    EXPECT_EQ(r.value().num_objects(), 1u);
    EXPECT_TRUE(task.ready());  // still ready after consumption

    std::filesystem::remove(path);
}

TEST(AsyncTaskControl, DoubleJoinReportsInvalidArg) {
    auto path = make_test_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();

    auto task = file.message_count_task();
    auto first = task.join();
    ASSERT_TRUE(first.ok()) << first.message();
    EXPECT_EQ(first.value(), 1u);

    auto second = task.join();
    EXPECT_FALSE(second.ok());
    EXPECT_EQ(second.code(), TGM_ERROR_INVALID_ARG);

    std::filesystem::remove(path);
}

TEST(AsyncTaskControl, RangeTaskJoins) {
    auto path = make_test_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {{4, 4}};
    auto task = file.decode_range_task(0, 0, ranges);
    auto r = task.join();
    ASSERT_TRUE(r.ok()) << r.message();
    ASSERT_EQ(r.value().size(), 1u);
    ASSERT_EQ(r.value()[0].size(), 4u);
    EXPECT_EQ(r.value()[0][0], 4u);

    std::filesystem::remove(path);
}

TEST(AsyncTaskControl, StdFutureExposesPullTask) {
    // The pull-model handle is reachable from the std::future frontend
    // via decode_object_task, so is_ready / cancel / join are usable
    // there too.
    auto path = make_test_file();
    auto file = tsf::async_file::open(path.string()).get();
    auto task = file.decode_object_task(0, 0);
    (void)task.ready();
    auto r = task.join();
    ASSERT_TRUE(r.ok()) << r.message();
    EXPECT_EQ(r.value().object(0).data_size(), 64u);
    std::filesystem::remove(path);
}

TEST(AsyncTaskControl, CancelIsSafeAndJoinStillResolves) {
    auto path = make_test_file();
    auto file = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }).get().take();

    auto task = file.decode_object_task(0, 0);
    task.cancel();  // request cancellation (idempotent, thread-safe)
    auto r = task.join();
    // Racy for a fast local decode: it may finish before cancellation is
    // observed, or come back cancelled.  Either is a valid, crash-free
    // resolution — the point is that cancel() + join() cooperate.
    EXPECT_TRUE(r.ok() || r.code() == TGM_ERROR_CANCELLED) << r.message();
    // Idempotent: cancelling an already-resolved task is a no-op.
    task.cancel();

    std::filesystem::remove(path);
}
