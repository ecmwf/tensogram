// (C) Copyright 2024- ECMWF and individual contributors.
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
    return R"({"version":2,"mars":{"class":"od","type":"an","stream":"oper","expver":"0001"},"custom_int":42,"custom_float":3.14,"descriptors":[{"type":"ndarray","ndim":1,"shape":[)" +
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
    EXPECT_EQ(meta.version(), 2u);
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

    EXPECT_EQ(meta.version(), 2u);
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
    std::string json = R"({"version":2,"mars":{"class":"od","type":"an"},"product":{"name":"temperature","units":"K"},"descriptors":[{"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";

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
