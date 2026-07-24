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
// Tests for tensogram::metadata::compute_common() — the {common, remaining}
// per-object metadata split.  Binds the C ABI tgm_compute_common.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

// A 1-D float32 descriptor entry (2 elements) for a multi-object message.
constexpr const char* kDescEntry =
    R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],)"
    R"("dtype":"float32","byte_order":"little",)"
    R"("encoding":"none","filter":"none","compression":"none"})";

// Encode a two-object message whose per-object `base` metadata shares a
// "centre" key but differs on "param".
std::vector<std::uint8_t> encode_two_objects() {
    const std::string json =
        std::string(R"({"version":3,"descriptors":[)") + kDescEntry + "," +
        kDescEntry +
        R"(],"base":[{"centre":"ecmwf","param":"2t"},)"
        R"({"centre":"ecmwf","param":"10u"}]})";

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {3.0f, 4.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(a.data()), a.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(b.data()), b.size() * sizeof(float)},
    };
    return tensogram::encode(json, objects);
}

}  // namespace

TEST(ComputeCommonTest, SharedKeyIsCommonVaryingKeyRemains) {
    auto encoded = encode_two_objects();
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    auto json = meta.compute_common();
    EXPECT_NE(json.find("\"common\""), std::string::npos);
    EXPECT_NE(json.find("\"remaining\""), std::string::npos);
    // The identical "centre" is hoisted into `common`.
    EXPECT_NE(json.find("\"centre\":\"ecmwf\""), std::string::npos);
    // The differing "param" values stay in `remaining`.
    EXPECT_NE(json.find("\"param\":\"2t\""), std::string::npos);
    EXPECT_NE(json.find("\"param\":\"10u\""), std::string::npos);
}

TEST(ComputeCommonTest, WorksFromMessageMetadata) {
    auto encoded = encode_two_objects();
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto meta = msg.get_metadata();

    auto json = meta.compute_common();
    EXPECT_NE(json.find("\"centre\":\"ecmwf\""), std::string::npos);
    EXPECT_NE(json.find("\"param\":\"2t\""), std::string::npos);
}

TEST(ComputeCommonTest, SingleObjectWithoutAppMetadataHasEmptyCommon) {
    // A single-object message with no application `base` metadata: the sole
    // per-object entry carries only the library-managed `_reserved_` key,
    // which compute_common excludes from both sides.  So `common` is empty
    // and `remaining` holds one empty object (one per described object).
    auto encoded = test_helpers::encode_simple_f32({1.0f, 2.0f, 3.0f});
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    auto json = meta.compute_common();
    EXPECT_NE(json.find("\"common\":{}"), std::string::npos);
    EXPECT_NE(json.find("\"remaining\":[{}]"), std::string::npos);
}
