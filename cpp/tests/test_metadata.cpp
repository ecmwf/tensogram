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
// Tests for tensogram::metadata class.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: encode a message with extra metadata keys
// ---------------------------------------------------------------------------

namespace {

/// Build JSON with extra top-level metadata keys.
std::string json_with_metadata(std::size_t count) {
    return R"({"version":3,"mars":{"class":"od","type":"an","stream":"oper","expver":"0001"},"custom_int":42,"custom_float":3.14,"descriptors":[{"type":"ndarray","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
}

std::vector<std::uint8_t> encode_with_metadata(
    const std::vector<float>& values)
{
    auto json = json_with_metadata(values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    return tensogram::encode(json, objects);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Version accessor
// ---------------------------------------------------------------------------

TEST(MetadataTest, VersionAccessor) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());
    EXPECT_EQ(meta.version(), 3u);
}

// ---------------------------------------------------------------------------
// String access via dot-notation
// ---------------------------------------------------------------------------

TEST(MetadataTest, StringAccessDotNotation) {
    std::vector<float> values = {1.0f};
    auto encoded = encode_with_metadata(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_EQ(meta.get_string("mars.class"), "od");
    EXPECT_EQ(meta.get_string("mars.type"), "an");
    EXPECT_EQ(meta.get_string("mars.stream"), "oper");
    EXPECT_EQ(meta.get_string("mars.expver"), "0001");
}

// ---------------------------------------------------------------------------
// Integer access with default
// ---------------------------------------------------------------------------

TEST(MetadataTest, IntegerAccessWithDefault) {
    std::vector<float> values = {1.0f};
    auto encoded = encode_with_metadata(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_EQ(meta.get_int("custom_int", 0), 42);
}

// ---------------------------------------------------------------------------
// Integer missing key returns default
// ---------------------------------------------------------------------------

TEST(MetadataTest, IntegerMissingKeyReturnsDefault) {
    std::vector<float> values = {1.0f};
    auto encoded = encode_with_metadata(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_EQ(meta.get_int("nonexistent", -999), -999);
}

// ---------------------------------------------------------------------------
// Float access with default
// ---------------------------------------------------------------------------

TEST(MetadataTest, FloatAccessWithDefault) {
    std::vector<float> values = {1.0f};
    auto encoded = encode_with_metadata(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_DOUBLE_EQ(meta.get_float("custom_float", 0.0), 3.14);
}

// ---------------------------------------------------------------------------
// Float missing key returns default
// ---------------------------------------------------------------------------

TEST(MetadataTest, FloatMissingKeyReturnsDefault) {
    std::vector<float> values = {1.0f};
    auto encoded = encode_with_metadata(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_DOUBLE_EQ(meta.get_float("nonexistent", 99.5), 99.5);
}

// ---------------------------------------------------------------------------
// Missing key returns empty string
// ---------------------------------------------------------------------------

TEST(MetadataTest, MissingKeyReturnsEmptyString) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_EQ(meta.get_string("nonexistent.key"), "");
}

// ---------------------------------------------------------------------------
// Nested metadata (mars namespace)
// ---------------------------------------------------------------------------

TEST(MetadataTest, NestedMetadataNamespace) {
    std::vector<float> values = {1.0f};
    auto encoded = encode_with_metadata(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    // "mars" is a namespace; accessing sub-keys should work
    EXPECT_EQ(meta.get_string("mars.class"), "od");
    // Accessing "mars" itself as string should return empty (it's an object)
    EXPECT_EQ(meta.get_string("mars"), "");
}

// ---------------------------------------------------------------------------
// Metadata from decoded message
// ---------------------------------------------------------------------------

TEST(MetadataTest, MetadataFromDecodedMessage) {
    std::vector<float> values = {1.0f, 2.0f};
    auto encoded = encode_with_metadata(values);
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    auto meta = msg.get_metadata();

    EXPECT_EQ(meta.version(), 3u);
    EXPECT_EQ(meta.get_string("mars.class"), "od");
    EXPECT_EQ(meta.get_int("custom_int", 0), 42);
}

// ---------------------------------------------------------------------------
// Metadata num_objects from decode_metadata
// ---------------------------------------------------------------------------

TEST(MetadataTest, MetadataNumObjects) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());
    // num_objects() returns base.len() from the global metadata CBOR.
    // The encoder auto-populates base[i] with _reserved_.tensor for each
    // object, so a single-object message has base.len() == 1.
    EXPECT_EQ(meta.num_objects(), 1u);
}

// ---------------------------------------------------------------------------
// Empty metadata (no extra keys)
// ---------------------------------------------------------------------------

TEST(MetadataTest, EmptyMetadataNoExtraKeys) {
    std::vector<float> values = {1.0f};
    auto encoded = test_helpers::encode_simple_f32(values);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    // No mars namespace in simple encoding
    EXPECT_EQ(meta.get_string("mars.class"), "");
    EXPECT_EQ(meta.get_int("custom_int", -1), -1);
    EXPECT_DOUBLE_EQ(meta.get_float("custom_float", -1.0), -1.0);
}

// ---------------------------------------------------------------------------
// Multiple metadata sections (several namespace keys)
// ---------------------------------------------------------------------------

TEST(MetadataTest, MultipleMetadataSections) {
    std::string json = R"({"version":3,"mars":{"class":"od","type":"an"},"product":{"name":"temperature","units":"K"},"descriptors":[{"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<float> values = {273.15f, 300.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    auto encoded = tensogram::encode(json, objects);
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_EQ(meta.get_string("mars.class"), "od");
    EXPECT_EQ(meta.get_string("mars.type"), "an");
    EXPECT_EQ(meta.get_string("product.name"), "temperature");
    EXPECT_EQ(meta.get_string("product.units"), "K");
}

// ---------------------------------------------------------------------------
// Per-object metadata (base[i]) — the multi-object walk that previously
// forced clients to reimplement CBOR parsing.
// ---------------------------------------------------------------------------

namespace {

/// Encode a 2-object message whose per-object metadata differs, via the
/// top-level "base" array (base[i] aligns with object i).
std::vector<std::uint8_t> encode_two_objects_with_base() {
    const char* desc =
        R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32",)"
        R"("byte_order":"little","encoding":"none","filter":"none","compression":"none"})";
    std::string json =
        std::string(R"({"descriptors":[)") + desc + "," + desc + "]," +
        R"("base":[)"
        R"({"shortName":"2t","level":0,"units":"K"},)"
        R"({"shortName":"msl","level":500,"scale":0.01,"geometry":{"gridType":"regular_ll"}})"
        "]}";

    std::vector<float> o0 = {273.15f, 300.0f};
    std::vector<float> o1 = {1.0f, 2.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(o0.data()), o0.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(o1.data()), o1.size() * sizeof(float)},
    };
    return tensogram::encode(json, objects);
}

} // anonymous namespace

TEST(MetadataTest, PerObjectStringScopesToIndex) {
    auto encoded = encode_two_objects_with_base();
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());
    ASSERT_EQ(meta.num_objects(), 2u);

    // The message-level getter first-matches object 0…
    EXPECT_EQ(meta.get_string("shortName"), "2t");
    // …the per-object getter reaches each object independently.
    EXPECT_EQ(meta.get_string_at(0, "shortName"), "2t");
    EXPECT_EQ(meta.get_string_at(1, "shortName"), "msl");
}

TEST(MetadataTest, PerObjectIntFloatAndNested) {
    auto encoded = encode_two_objects_with_base();
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_EQ(meta.get_int_at(0, "level", -1), 0);
    EXPECT_EQ(meta.get_int_at(1, "level", -1), 500);
    EXPECT_DOUBLE_EQ(meta.get_float_at(1, "scale", 0.0), 0.01);
    EXPECT_EQ(meta.get_string_at(1, "geometry.gridType"), "regular_ll");
    // object 0 has no geometry / scale → defaults
    EXPECT_EQ(meta.get_string_at(0, "geometry.gridType"), "");
    EXPECT_EQ(meta.get_int_at(0, "scale", 7), 7);
}

TEST(MetadataTest, PerObjectOutOfRange) {
    auto encoded = encode_two_objects_with_base();
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    EXPECT_EQ(meta.get_string_at(9, "shortName"), "");
    EXPECT_EQ(meta.get_int_at(9, "level", 42), 42);
    EXPECT_EQ(meta.object_to_json(9), "");
}

TEST(MetadataTest, ObjectToJsonEnumeratesEverything) {
    auto encoded = encode_two_objects_with_base();
    auto meta = tensogram::decode_metadata(encoded.data(), encoded.size());

    const std::string j1 = meta.object_to_json(1);
    // Full object dumped as JSON — a viewer can display it without knowing keys.
    EXPECT_NE(j1.find(R"("shortName":"msl")"), std::string::npos);
    EXPECT_NE(j1.find(R"("gridType":"regular_ll")"), std::string::npos);
    // Includes the encoder-populated tensor descriptor.
    EXPECT_NE(j1.find("_reserved_"), std::string::npos);

    const std::string all = meta.to_json();
    EXPECT_NE(all.find(R"("base")"), std::string::npos);
    EXPECT_NE(all.find(R"("shortName":"2t")"), std::string::npos);
}
