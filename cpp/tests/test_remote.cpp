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
// Tests for the synchronous remote surface:
//   * tensogram::is_remote_url        (tgm_is_remote_url)
//   * tensogram::remote_scan_options  (TgmRemoteScanOptions)
//   * tensogram::file::open_remote    (tgm_file_open_remote)
//
// The remote backend is an opt-in Cargo feature.  Symbols always link, so
// every build exercises the argument validation and the local-path answers;
// the tests that need a working backend are compiled only when the C API was
// built with `--features remote` (CMake: -DTENSOGRAM_REMOTE=ON), and the
// feature-off build asserts the honest "this build cannot" behaviour instead.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

using test_helpers::TempFile;

#ifdef TENSOGRAM_REMOTE
namespace {

/// Write a two-message `.tgm` and return the `file://` URL addressing it.
///
/// `file://` travels the same object-store code path as `s3://` & friends,
/// so it exercises the real remote backend without a network.
std::string write_fixture(const std::string& path) {
    const std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b{5.0f, 6.0f, 7.0f, 8.0f};
    auto f = tensogram::file::create(path);
    f.append_raw(test_helpers::encode_simple_f32(a));
    f.append_raw(test_helpers::encode_simple_f32(b));
    return "file://" + path;
}

}  // namespace
#endif  // TENSOGRAM_REMOTE

// ---------------------------------------------------------------------------
// remote_scan_options
// ---------------------------------------------------------------------------

TEST(RemoteTest, ScanOptionsDefaultToTheBidirectionalWalk) {
    const tensogram::remote_scan_options opts;
    EXPECT_TRUE(opts.bidirectional);
}

// ---------------------------------------------------------------------------
// is_remote_url
// ---------------------------------------------------------------------------

TEST(RemoteTest, LocalSourcesAreNeverRemote) {
    // True in every build: these belong to the local backend (file::open).
    EXPECT_FALSE(tensogram::is_remote_url("/tmp/data.tgm"));
    EXPECT_FALSE(tensogram::is_remote_url("data.tgm"));
    EXPECT_FALSE(tensogram::is_remote_url("./relative/data.tgm"));
    EXPECT_FALSE(tensogram::is_remote_url("file:///tmp/data.tgm"));
    EXPECT_FALSE(tensogram::is_remote_url(""));
    EXPECT_FALSE(tensogram::is_remote_url("ftp://host/data.tgm"));
}

TEST(RemoteTest, ObjectStoreSchemesMatchThisBuildsCapability) {
    // A remote-capable build recognises every object-store scheme; a build
    // without the feature genuinely cannot open any of them, so "not remote
    // for me" is the honest answer (the C layer records why).
#ifdef TENSOGRAM_REMOTE
    const bool expected = true;
#else
    const bool expected = false;
#endif
    for (const char* url : {"s3://bucket/key.tgm", "s3a://bucket/key.tgm",
                            "gs://bucket/key.tgm", "az://container/key.tgm",
                            "azure://container/key.tgm", "http://host/key.tgm",
                            "https://host/key.tgm"}) {
        EXPECT_EQ(tensogram::is_remote_url(url), expected) << url;
    }
}

TEST(RemoteTest, SchemesAreComparedCaseInsensitively) {
#ifdef TENSOGRAM_REMOTE
    EXPECT_TRUE(tensogram::is_remote_url("S3://bucket/key.tgm"));
    EXPECT_TRUE(tensogram::is_remote_url("HTTPS://host/key.tgm"));
#else
    EXPECT_FALSE(tensogram::is_remote_url("S3://bucket/key.tgm"));
#endif
}

TEST(RemoteTest, IsRemoteUrlAcceptsAStringViewOfASubstring) {
    // The predicate takes a string_view, so it must not rely on the caller's
    // buffer being NUL-terminated at the end of the view.
    const std::string haystack = "s3://bucket/key.tgmXXXX";
    const std::string_view view(haystack.data(), haystack.size() - 4);
#ifdef TENSOGRAM_REMOTE
    EXPECT_TRUE(tensogram::is_remote_url(view));
#else
    EXPECT_FALSE(tensogram::is_remote_url(view));
#endif
}

// ---------------------------------------------------------------------------
// file::open_remote — argument validation (identical in every build)
// ---------------------------------------------------------------------------

TEST(RemoteTest, OpenRemoteRejectsAnEmbeddedNulInTheSource) {
    // A C string cannot carry an interior NUL: silently truncating would open
    // a *different* object than the caller asked for.
    const std::string source("s3://bucket/key.tgm\0extra", 25);
    EXPECT_THROW((void)tensogram::file::open_remote(source), tensogram::invalid_arg_error);
}

TEST(RemoteTest, OpenRemoteRejectsEmbeddedNulsInStorageOptions) {
    const std::map<std::string, std::string> bad_key{
        {std::string("aws_region\0x", 12), "eu-west-1"}};
    EXPECT_THROW((void)tensogram::file::open_remote("s3://bucket/key.tgm", bad_key),
                 tensogram::invalid_arg_error);

    const std::map<std::string, std::string> bad_value{
        {"aws_region", std::string("eu-west-1\0x", 11)}};
    EXPECT_THROW((void)tensogram::file::open_remote("s3://bucket/key.tgm", bad_value),
                 tensogram::invalid_arg_error);
}

// ---------------------------------------------------------------------------
// file::open_remote — behaviour, per build flavour
// ---------------------------------------------------------------------------

#ifndef TENSOGRAM_REMOTE

TEST(RemoteTest, OpenRemoteWithoutTheFeatureExplainsHowToEnableIt) {
    try {
        (void)tensogram::file::open_remote("s3://bucket/key.tgm");
        FAIL() << "expected a remote_error from a build without the feature";
    } catch (const tensogram::remote_error& e) {
        EXPECT_EQ(e.code(), TGM_ERROR_REMOTE);
        const std::string what = e.what();
        EXPECT_NE(what.find("remote"), std::string::npos) << what;
    }
}

TEST(RemoteTest, OpenRemoteWithoutTheFeatureStillValidatesArgumentsFirst) {
    // Argument validation runs before the feature check, so a mistake is
    // reported as a mistake rather than as "no remote support".
    const std::string source("s3://bucket/key.tgm\0extra", 25);
    EXPECT_THROW((void)tensogram::file::open_remote(source), tensogram::invalid_arg_error);
}

#else  // TENSOGRAM_REMOTE

TEST(RemoteTest, OpenRemoteReadsAFileUrlThroughTheOrdinaryFileApi) {
    TempFile tmp;
    const std::string url = write_fixture(tmp.path);

    auto remote = tensogram::file::open_remote(url);
    ASSERT_EQ(remote.message_count(), 2u);
    EXPECT_EQ(remote.path(), url);

    auto raw = remote.read_message(0);
    EXPECT_FALSE(raw.empty());
    auto decoded = remote.decode_message(1);
    ASSERT_EQ(decoded.num_objects(), 1u);
    EXPECT_EQ(decoded.object(0).data_as<float>()[0], 5.0f);

    // The frame walker works on remotely-read bytes like any other message.
    const auto header = tensogram::read_message_header(raw.data(), raw.size());
    EXPECT_EQ(header.version(), tensogram::wire_version);
}

TEST(RemoteTest, OpenRemoteAcceptsForwardOnlyScanOptions) {
    TempFile tmp;
    const std::string url = write_fixture(tmp.path);
    tensogram::remote_scan_options opts;
    opts.bidirectional = false;
    auto remote = tensogram::file::open_remote(url, {}, opts);
    EXPECT_EQ(remote.message_count(), 2u);
}

TEST(RemoteTest, OpenRemoteReportsAMissingObjectAsARemoteError) {
    EXPECT_THROW((void)tensogram::file::open_remote("file:///nonexistent/missing.tgm"),
                 tensogram::remote_error);
}

TEST(RemoteTest, OpenRemoteRejectsAnUnparseableUrl) {
    EXPECT_THROW((void)tensogram::file::open_remote("s3://"), tensogram::remote_error);
}

#endif  // TENSOGRAM_REMOTE
