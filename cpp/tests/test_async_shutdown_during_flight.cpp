// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// Tests for tgm_runtime_shutdown_blocking: the runtime drains in-flight
// work within the supplied timeout and is permanently single-shot
// afterwards.
//
// This lives in its own test executable because the shared async runtime
// is process-global and shutdown is irreversible: running it alongside the
// other async tests in one process would tear the runtime down underneath
// them.  ctest invokes each discovered test in a fresh process, but a
// dedicated binary keeps the isolation guarantee even when the suite is run
// directly.

#include <gtest/gtest.h>
#include <tensogram/async/callback.hpp>
#include <tensogram.hpp>
#include <tensogram.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <future>
#include <string>
#include <vector>

namespace tac = tensogram::async_callback;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    auto tmp = std::filesystem::temp_directory_path();
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return tmp / ("tensogram_shutdown_test_" + std::to_string(t) + suffix);
}

/// Encode a one-message file with the sync API and return its path.
std::filesystem::path make_sync_test_file() {
    auto path = make_temp_path(".tgm");
    std::string meta_json = R"json({
        "base":[],
        "descriptors":[{
            "type":"ntensor","ndim":1,"shape":[4],"strides":[1],
            "dtype":"float32","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none"
        }]
    })json";
    std::vector<std::uint8_t> payload(16);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<std::uint8_t>(i);
    }
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objs;
    objs.emplace_back(payload.data(), payload.size());

    auto bytes = tensogram::encode(meta_json, objs);
    auto* fp = std::fopen(path.c_str(), "wb");
    std::fwrite(bytes.data(), 1, bytes.size(), fp);
    std::fclose(fp);
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

// The whole lifecycle runs in one test because shutdown is single-shot: a
// second test would observe an already-torn-down runtime.
TEST(AsyncShutdown, DrainsThenRejectsSubsequentCalls) {
    auto path = make_sync_test_file();

    // Drive a real async operation so the runtime is actually built and
    // has done work before we shut it down.
    {
        auto open_fut = as_future<tac::async_file>(
            [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); });
        auto open_r = open_fut.get();
        ASSERT_TRUE(open_r.ok()) << open_r.message();
        auto file = open_r.take();

        auto count_fut = as_future<std::size_t>(
            [&](auto cb) { file.message_count(std::move(cb)); });
        auto count_r = count_fut.get();
        ASSERT_TRUE(count_r.ok()) << count_r.message();
        EXPECT_EQ(count_r.value(), 1u);
    }

    // Shut the runtime down.  The completed operations above have already
    // drained, so a generous timeout returns promptly.  The return value is
    // the count of tasks still unfinished at the deadline; with no work in
    // flight it is 0.  (The precise non-zero counting behaviour is covered
    // deterministically by the Rust unit test, which can hold a task open
    // with a controllable sleep.)
    std::uint64_t unfinished = tgm_runtime_shutdown_blocking(1000);
    EXPECT_EQ(unfinished, 0u);

    // The runtime is now single-shot: a subsequent async open must fail
    // fast rather than rebuild the runtime.  The FFI maps the shut-down
    // state to TGM_ERROR_IO.
    auto reopen_fut = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); });
    auto reopen_r = reopen_fut.get();
    EXPECT_FALSE(reopen_r.ok())
        << "async open after shutdown must fail (runtime is single-shot)";

    // A second shutdown is an idempotent no-op returning 0.
    EXPECT_EQ(tgm_runtime_shutdown_blocking(10), 0u);

    std::filesystem::remove(path);
}
