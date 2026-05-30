// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// Tests for the async/callback.hpp frontend (PR 4 of the cpp-async plan).

#include <gtest/gtest.h>
#include <tensogram/async/callback.hpp>
#include <tensogram.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>

namespace tac = tensogram::async_callback;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    auto tmp = std::filesystem::temp_directory_path();
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return tmp / ("tensogram_test_" + std::to_string(t) + suffix);
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

/// Helper: convert callback API to a future for cleaner test code.
template <typename T>
std::future<tac::result<T>> as_future(
    std::function<void(std::function<void(tac::result<T>)>)> launch) {
    auto promise = std::make_shared<std::promise<tac::result<T>>>();
    auto fut = promise->get_future();
    launch([promise](tac::result<T> r) {
        promise->set_value(std::move(r));
    });
    return fut;
}

}  // namespace

TEST(AsyncCallback, OpenAndMessageCount) {
    auto path = make_sync_test_file();

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

    std::filesystem::remove(path);
}

TEST(AsyncCallback, DecodeMessageRoundTrip) {
    auto path = make_sync_test_file();

    auto open_fut = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); });
    auto file = open_fut.get().take();

    auto dec_fut = as_future<tensogram::message>(
        [&](auto cb) { file.decode_message(0, std::move(cb)); });
    auto dec_r = dec_fut.get();
    ASSERT_TRUE(dec_r.ok()) << dec_r.message();

    auto& msg = dec_r.value();
    EXPECT_EQ(msg.num_objects(), 1u);

    std::filesystem::remove(path);
}

TEST(AsyncCallback, DecodeMetadata) {
    auto path = make_sync_test_file();

    auto open_fut = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); });
    auto file = open_fut.get().take();

    auto md_fut = as_future<tensogram::metadata>(
        [&](auto cb) { file.decode_metadata(0, std::move(cb)); });
    auto md_r = md_fut.get();
    ASSERT_TRUE(md_r.ok()) << md_r.message();
    auto& m = md_r.value();
    EXPECT_EQ(m.version(), tensogram::wire_version);

    std::filesystem::remove(path);
}

TEST(AsyncCallback, ReadMessageRawBytes) {
    auto path = make_sync_test_file();

    auto open_fut = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); });
    auto file = open_fut.get().take();

    auto read_fut = as_future<std::vector<std::uint8_t>>(
        [&](auto cb) { file.read_message(0, std::move(cb)); });
    auto read_r = read_fut.get();
    ASSERT_TRUE(read_r.ok()) << read_r.message();
    EXPECT_GT(read_r.value().size(), 0u);

    std::filesystem::remove(path);
}

TEST(AsyncCallback, OpenNonexistentReportsError) {
    auto open_fut = as_future<tac::async_file>([&](auto cb) {
        tac::async_file::open("/nonexistent/missing.tgm", std::move(cb));
    });
    auto r = open_fut.get();
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.code(), TGM_ERROR_OK);
}

TEST(AsyncCallback, CancellationTokenLifecycle) {
    tac::cancellation_token tok;
    EXPECT_FALSE(tok.cancelled());
    tok.cancel();
    EXPECT_TRUE(tok.cancelled());
}

TEST(AsyncCallback, StreamingEncoderRoundTrip) {
    auto path = make_temp_path(".tgm");
    std::string meta_json = R"json({"base":[]})json";

    auto create_fut = as_future<tac::async_streaming_encoder>([&](auto cb) {
        tac::async_streaming_encoder::create(path.string(), meta_json, std::move(cb));
    });
    auto cr = create_fut.get();
    ASSERT_TRUE(cr.ok()) << cr.message();
    auto enc = cr.take();

    std::string desc = R"json({
        "type":"ntensor","ndim":1,"shape":[4],"strides":[1],
        "dtype":"float32","byte_order":"little","encoding":"none",
        "filter":"none","compression":"none","params":{}
    })json";
    std::vector<std::uint8_t> data(16);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i);

    auto wr_fut = as_future<void>(
        [&](auto cb) { enc.write_object(desc, data.data(), data.size(), std::move(cb)); });
    auto wr_r = wr_fut.get();
    ASSERT_TRUE(wr_r.ok()) << wr_r.message();

    auto fin_fut = as_future<void>([&](auto cb) { enc.finish(std::move(cb)); });
    auto fin_r = fin_fut.get();
    ASSERT_TRUE(fin_r.ok()) << fin_r.message();

    // Verify the file decodes via the sync API.
    std::ifstream is(path.string(), std::ios::binary);
    ASSERT_TRUE(is.good());
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(is)),
                                    std::istreambuf_iterator<char>());
    auto msg = tensogram::decode(bytes.data(), bytes.size());
    EXPECT_EQ(msg.num_objects(), 1u);

    std::filesystem::remove(path);
}

TEST(AsyncCallback, TryObjectCountStateMachine) {
    auto path = make_temp_path(".tgm");
    std::string meta_json = R"json({"base":[]})json";

    auto create_fut = as_future<tac::async_streaming_encoder>([&](auto cb) {
        tac::async_streaming_encoder::create(path.string(), meta_json, std::move(cb));
    });
    auto cr = create_fut.get();
    ASSERT_TRUE(cr.ok()) << cr.message();
    auto enc = cr.take();

    // Fresh encoder, no writes yet, no in-flight ops → ok / 0.
    {
        auto r = enc.try_object_count();
        EXPECT_EQ(r.status,
                  tac::async_streaming_encoder::object_count_status::ok);
        EXPECT_EQ(r.value, 0u);
        EXPECT_EQ(enc.object_count(), 0u);
    }

    // After one successful write → ok / 1.
    std::string desc = R"json({
        "type":"ntensor","ndim":1,"shape":[4],"strides":[1],
        "dtype":"float32","byte_order":"little","encoding":"none",
        "filter":"none","compression":"none","params":{}
    })json";
    std::vector<std::uint8_t> data(16);
    auto wr_fut = as_future<void>(
        [&](auto cb) { enc.write_object(desc, data.data(), data.size(), std::move(cb)); });
    ASSERT_TRUE(wr_fut.get().ok());
    {
        auto r = enc.try_object_count();
        EXPECT_EQ(r.status,
                  tac::async_streaming_encoder::object_count_status::ok);
        EXPECT_EQ(r.value, 1u);
    }

    // After finish → finished / 0 (count value is meaningless here).
    auto fin_fut = as_future<void>([&](auto cb) { enc.finish(std::move(cb)); });
    ASSERT_TRUE(fin_fut.get().ok());
    {
        auto r = enc.try_object_count();
        EXPECT_EQ(r.status,
                  tac::async_streaming_encoder::object_count_status::finished);
        // Convenience accessor collapses to the historical sentinel.
        EXPECT_EQ(enc.object_count(), static_cast<std::size_t>(-1));
    }

    std::filesystem::remove(path);
}

TEST(AsyncCallback, RuntimeConfigureAfterInitErrors) {
    // The runtime is built lazily on first call.  Trigger it.
    auto path = make_sync_test_file();
    auto open_fut = as_future<tac::async_file>(
        [&](auto cb) { tac::async_file::open(path.string(), std::move(cb)); });
    [[maybe_unused]] auto _ = open_fut.get().take();

    // Now configure should throw.
    EXPECT_THROW(tac::runtime_configure(2, 0, 0), tensogram::invalid_arg_error);

    std::filesystem::remove(path);
}

TEST(AsyncCallback, OpenRemoteSurfacesError) {
    // open_remote marshals the storage-option key/value arrays and calls
    // the always-linkable tgm_async_file_open_remote.  Without the
    // async-remote feature it resolves synchronously with
    // TGM_ERROR_REMOTE; with it, a nonexistent file:// URL resolves with
    // a not-found error.  Either way the result is an error, never a
    // value, and the call never crashes.
    auto fut = as_future<tac::async_file>([&](auto cb) {
        tac::async_file::open_remote(
            "file:///nonexistent/tensogram_open_remote_probe.tgm",
            {{"region", "eu-west-1"}},  // exercises the key/value marshalling
            /*bidirectional=*/false, std::move(cb));
    });
    auto r = fut.get();
    EXPECT_FALSE(r.ok());
#ifndef TENSOGRAM_ASYNC_REMOTE
    EXPECT_EQ(r.code(), TGM_ERROR_REMOTE) << r.message();
#endif
}

#ifdef TENSOGRAM_ASYNC_REMOTE
TEST(AsyncCallback, OpenRemoteFileUrlRoundTrip) {
    // With the async-remote feature a file:// URL routes through
    // object_store's LocalFileSystem, so the full remote read path
    // (open_remote -> message_count -> decode) can be exercised offline
    // and deterministically.  This is the success-path counterpart to
    // OpenRemoteSurfacesError.
    auto path = make_sync_test_file();
    const std::string url = "file://" + std::filesystem::absolute(path).string();

    auto open_r = as_future<tac::async_file>([&](auto cb) {
        tac::async_file::open_remote(url, {}, /*bidirectional=*/false, std::move(cb));
    }).get();
    ASSERT_TRUE(open_r.ok()) << open_r.message();
    auto file = open_r.take();

    auto count_r = as_future<std::size_t>(
        [&](auto cb) { file.message_count(std::move(cb)); }).get();
    ASSERT_TRUE(count_r.ok()) << count_r.message();
    EXPECT_EQ(count_r.value(), 1u);

    auto dec_r = as_future<tensogram::message>(
        [&](auto cb) { file.decode_message(0, std::move(cb)); }).get();
    ASSERT_TRUE(dec_r.ok()) << dec_r.message();
    EXPECT_EQ(dec_r.value().num_objects(), 1u);

    std::filesystem::remove(path);
}
#endif  // TENSOGRAM_ASYNC_REMOTE
