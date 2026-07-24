// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include <tensogram/async/coro.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace tco = tensogram::coro;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    auto tmp = std::filesystem::temp_directory_path();
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return tmp / ("tensogram_coro_" + std::to_string(t) + suffix);
}

std::filesystem::path make_test_file() {
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

/// A one-message file with a single uint8[64] object (byte i == i) so
/// decode_object / decode_range have deterministic, byte-addressable data.
std::filesystem::path make_uint8_file() {
    auto path = make_temp_path(".tgm");
    const std::string meta = R"json({
        "base":[{}],
        "descriptors":[{
            "type":"ntensor","ndim":1,"shape":[64],"strides":[1],
            "dtype":"uint8","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none"
        }]
    })json";
    std::vector<std::uint8_t> u(64);
    for (std::size_t i = 0; i < u.size(); ++i) u[i] = static_cast<std::uint8_t>(i);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objs = {{u.data(), u.size()}};
    auto bytes = tensogram::encode(meta, objs);
    auto* fp = std::fopen(path.c_str(), "wb");
    std::fwrite(bytes.data(), 1, bytes.size(), fp);
    std::fclose(fp);
    return path;
}

}  // namespace

TEST(AsyncCoro, OpenAndMessageCount) {
    auto path = make_test_file();

    auto t = []() -> tco::task<std::size_t> {
        auto file = co_await tco::async_file::open(""); // placeholder
        co_return 0;
    };
    // Use the path-based variant via block_on:
    auto open_and_count = [&path]() -> tco::task<std::size_t> {
        auto file = co_await tco::async_file::open(path.string());
        std::size_t n = co_await file.message_count();
        co_return n;
    };
    auto count = tco::block_on(open_and_count());
    EXPECT_EQ(count, 1u);

    std::filesystem::remove(path);
}

TEST(AsyncCoro, DecodeMessage) {
    auto path = make_test_file();
    auto run = [&path]() -> tco::task<std::size_t> {
        auto file = co_await tco::async_file::open(path.string());
        auto msg = co_await file.decode_message(0);
        co_return msg.num_objects();
    };
    auto n = tco::block_on(run());
    EXPECT_EQ(n, 1u);
    std::filesystem::remove(path);
}

TEST(AsyncCoro, OpenNonexistentThrows) {
    auto run = []() -> tco::task<std::size_t> {
        auto file = co_await tco::async_file::open("/nonexistent/missing.tgm");
        co_return 0;
    };
    EXPECT_THROW(tco::block_on(run()), tensogram::error);
}

TEST(AsyncCoro, StreamingEncoderRoundTrip) {
    auto path = make_temp_path(".tgm");
    std::string meta_json = R"json({"base":[]})json";
    std::string desc = R"json({
        "type":"ntensor","ndim":1,"shape":[4],"strides":[1],
        "dtype":"float32","byte_order":"little","encoding":"none",
        "filter":"none","compression":"none","params":{}
    })json";
    std::vector<std::uint8_t> data(16);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i);

    auto run = [&]() -> tco::task<void> {
        auto enc = co_await tco::async_streaming_encoder::create(path.string(), meta_json);
        co_await enc.write_object(desc, data.data(), data.size());
        co_await enc.finish();
    };
    tco::block_on(run());

    std::ifstream is(path.string(), std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(is)),
                                    std::istreambuf_iterator<char>());
    auto msg = tensogram::decode(bytes.data(), bytes.size());
    EXPECT_EQ(msg.num_objects(), 1u);

    std::filesystem::remove(path);
}

TEST(AsyncCoro, AsyncForEachWalksAllMessages) {
    auto path = make_test_file();
    auto run = [&path]() -> tco::task<std::size_t> {
        auto file = co_await tco::async_file::open(path.string());
        std::size_t count = 0;
        co_await tco::async_for_each(file, [&count](tensogram::message m) {
            (void)m;
            ++count;
        });
        co_return count;
    };
    auto n = tco::block_on(run());
    EXPECT_EQ(n, 1u);
    std::filesystem::remove(path);
}

TEST(AsyncCoro, OpenRemoteSurfacesError) {
    // open_remote is awaitable; the error surfaces as a thrown
    // tensogram::error out of block_on (TGM_ERROR_REMOTE without the
    // async-remote feature, file-not-found with it).
    auto run = []() -> tco::task<int> {
        // Build the storage-option vector before the co_await: constructing
        // this temporary inside the co_await argument list triggers an
        // internal compiler error on some GCC versions.
        const std::vector<std::pair<std::string, std::string>> opts = {
            {"region", "eu-west-1"}};
        auto file = co_await tco::async_file::open_remote(
            "file:///nonexistent/tensogram_open_remote_probe.tgm", opts,
            /*bidirectional=*/false);
        (void)file;
        co_return 0;
    };
    EXPECT_THROW(tco::block_on(run()), tensogram::error);
}

TEST(AsyncCoro, DecodeObject) {
    auto path = make_uint8_file();
    auto run = [&path]() -> tco::task<std::size_t> {
        auto file = co_await tco::async_file::open(path.string());
        auto msg = co_await file.decode_object(0, 0);
        co_return msg.object(0).data_size();
    };
    auto sz = tco::block_on(run());
    EXPECT_EQ(sz, 64u);
    std::filesystem::remove(path);
}

TEST(AsyncCoro, DecodeRange) {
    auto path = make_uint8_file();
    auto run = [&path]() -> tco::task<std::vector<std::vector<std::uint8_t>>> {
        auto file = co_await tco::async_file::open(path.string());
        // Build the ranges vector before the co_await (temporary-in-co_await
        // ICE workaround, matching OpenRemoteSurfacesError above).
        const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {{8, 4}};
        co_return co_await file.decode_range(0, 0, ranges);
    };
    auto parts = tco::block_on(run());
    ASSERT_EQ(parts.size(), 1u);
    ASSERT_EQ(parts[0].size(), 4u);
    EXPECT_EQ(parts[0][0], 8u);
    std::filesystem::remove(path);
}

TEST(AsyncCoro, FilePathAccessor) {
    auto path = make_uint8_file();
    auto run = [&path]() -> tco::task<std::string> {
        auto file = co_await tco::async_file::open(path.string());
        co_return file.path();
    };
    auto reported = tco::block_on(run());
    EXPECT_NE(reported.find(path.filename().string()), std::string::npos);
    std::filesystem::remove(path);
}

TEST(AsyncCoro, DecodeObjectTaskPollAndJoin) {
    // The pull-model handle (tac::task<T>) is reachable from the coro
    // frontend too, for callers that want manual poll / cancel / join
    // instead of co_await.  Open via block_on (the coroutine parameter is
    // by-value so its frame owns the path), then drive the task off the
    // coroutine on this thread.
    auto path = make_uint8_file();
    auto open_coro = [](std::string p) -> tco::task<tco::async_file> {
        co_return co_await tco::async_file::open(p);
    };
    auto file = tco::block_on(open_coro(path.string()));
    auto task = file.decode_object_task(0, 0);
    (void)task.ready();  // non-blocking poll is callable pre-join
    auto r = task.join();
    ASSERT_TRUE(r.ok()) << r.message();
    EXPECT_EQ(r.value().object(0).data_size(), 64u);
    std::filesystem::remove(path);
}
