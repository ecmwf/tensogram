// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include <tensogram/async/std_future.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace tsf = tensogram::stdfuture;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    auto tmp = std::filesystem::temp_directory_path();
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return tmp / ("tensogram_stdfuture_" + std::to_string(t) + suffix);
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

}  // namespace

TEST(AsyncStdFuture, OpenAndMessageCount) {
    auto path = make_test_file();
    auto file = tsf::async_file::open(path.string()).get();
    auto count = file.message_count().get();
    EXPECT_EQ(count, 1u);
    std::filesystem::remove(path);
}

TEST(AsyncStdFuture, DecodeMessage) {
    auto path = make_test_file();
    auto file = tsf::async_file::open(path.string()).get();
    auto msg = file.decode_message(0).get();
    EXPECT_EQ(msg.num_objects(), 1u);
    std::filesystem::remove(path);
}

TEST(AsyncStdFuture, OpenNonexistentThrows) {
    auto fut = tsf::async_file::open("/nonexistent/missing.tgm");
    EXPECT_THROW(fut.get(), tensogram::error);
}

TEST(AsyncStdFuture, StreamingEncoderRoundTrip) {
    auto path = make_temp_path(".tgm");
    std::string meta_json = R"json({"base":[]})json";
    auto enc = tsf::async_streaming_encoder::create(path.string(), meta_json).get();

    std::string desc = R"json({
        "type":"ntensor","ndim":1,"shape":[4],"strides":[1],
        "dtype":"float32","byte_order":"little","encoding":"none",
        "filter":"none","compression":"none","params":{}
    })json";
    std::vector<std::uint8_t> data(16);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i);

    enc.write_object(desc, data.data(), data.size()).get();
    enc.finish().get();

    std::ifstream is(path.string(), std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(is)),
                                    std::istreambuf_iterator<char>());
    auto msg = tensogram::decode(bytes.data(), bytes.size());
    EXPECT_EQ(msg.num_objects(), 1u);

    std::filesystem::remove(path);
}

TEST(AsyncStdFuture, OpenRemoteSurfacesError) {
    // open_remote returns a std::future; the error (TGM_ERROR_REMOTE
    // without the async-remote feature, file-not-found with it) surfaces
    // as a typed tensogram::error through .get().
    auto fut = tsf::async_file::open_remote(
        "file:///nonexistent/tensogram_open_remote_probe.tgm",
        {{"region", "eu-west-1"}}, /*bidirectional=*/false);
    EXPECT_THROW(fut.get(), tensogram::error);
}
