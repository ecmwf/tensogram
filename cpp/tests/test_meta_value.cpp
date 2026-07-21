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
// Capability-matrix tests for the precise metadata cursor API:
// tensogram::meta_value plus metadata::contains / get / try_get_* /
// object / extra / reserved.  Covers presence vs wrong-type vs
// absent-vs-empty, nested map + array navigation, section enumeration,
// per-object scoping and `_reserved_` visibility rules (see
// plans/METADATA_ACCESS_PARITY.md §5).

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <set>
#include <string>
#include <vector>

using tensogram::value_type;

// ---------------------------------------------------------------------------
// Fixture: a 2-object message whose metadata exercises every value kind.
//
// base[0] carries strings (incl. an empty string), a real `0`, a negative
// int, a float, a bool, a CBOR null, a string array, an int array and a
// nested `mars` map.  base[1] carries a distinct `shortName`, a different
// `level`, and a nested `geometry` map — and deliberately omits `count` so
// per-object scoping is observable.  Top-level `producer` / `revision` flow
// into `_extra_`.  The encoder additionally populates `base[i]._reserved_`.
// ---------------------------------------------------------------------------

namespace {

std::vector<std::uint8_t> encode_capability_fixture() {
    const char* desc =
        R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32",)"
        R"("byte_order":"little","encoding":"none","filter":"none","compression":"none"})";
    std::string json =
        std::string(R"({"producer":"ecmwf","revision":7,"descriptors":[)") +
        desc + "," + desc + R"(],"base":[)"
        R"({"shortName":"2t","level":0,"emptyStr":"","count":42,"neg":-5,)"
        R"("ratio":0.5,"flag":true,"nothing":null,)"
        R"("tags":["a","b","c"],"nums":[10,20,30],)"
        R"("mars":{"class":"od","type":"an"}},)"
        R"({"shortName":"msl","level":500,"geometry":{"gridType":"regular_ll"}})"
        R"(]})";

    std::vector<float> o0 = {273.15f, 300.0f};
    std::vector<float> o1 = {1.0f, 2.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(o0.data()), o0.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(o1.data()), o1.size() * sizeof(float)},
    };
    return tensogram::encode(json, objects);
}

tensogram::metadata decode_fixture(const std::vector<std::uint8_t>& encoded) {
    return tensogram::decode_metadata(encoded.data(), encoded.size());
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// contains(): presence is distinct from value, wrong-type and absence.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, ContainsPresentAndAbsent) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    EXPECT_TRUE(meta.contains("shortName"));
    EXPECT_TRUE(meta.contains("count"));
    EXPECT_TRUE(meta.contains("mars.class"));   // nested via dot-path
    EXPECT_TRUE(meta.contains("producer"));     // _extra_ fallback

    EXPECT_FALSE(meta.contains("nonexistent"));
    EXPECT_FALSE(meta.contains("mars.missing"));
}

// ---------------------------------------------------------------------------
// get(): present -> cursor, absent -> nullopt.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, GetPresentReturnsCursorAbsentIsNullopt) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    const auto v = meta.get("shortName");
    ASSERT_TRUE(v.has_value());
    EXPECT_TRUE(v->valid());
    EXPECT_EQ(v->as_string(), std::string_view("2t"));

    EXPECT_FALSE(meta.get("nonexistent").has_value());
}

// ---------------------------------------------------------------------------
// value_type() maps every CBOR kind (and null is a real, present value).
// ---------------------------------------------------------------------------

TEST(MetaValueTest, ValueTypeMapping) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    EXPECT_EQ(meta.get("shortName")->type(), value_type::string);
    EXPECT_EQ(meta.get("count")->type(),     value_type::integer);
    EXPECT_EQ(meta.get("ratio")->type(),     value_type::floating);
    EXPECT_EQ(meta.get("flag")->type(),      value_type::boolean);
    EXPECT_EQ(meta.get("nothing")->type(),   value_type::null);
    EXPECT_EQ(meta.get("tags")->type(),      value_type::array);
    EXPECT_EQ(meta.get("mars")->type(),      value_type::map);

    EXPECT_TRUE(meta.get("shortName")->is_string());
    EXPECT_TRUE(meta.get("count")->is_int());
    EXPECT_TRUE(meta.get("ratio")->is_float());
    EXPECT_TRUE(meta.get("flag")->is_bool());
    EXPECT_TRUE(meta.get("nothing")->is_null());
    EXPECT_TRUE(meta.get("tags")->is_array());
    EXPECT_TRUE(meta.get("mars")->is_map());
}

// ---------------------------------------------------------------------------
// A present CBOR null is distinct from an absent key.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, PresentNullIsNotAbsent) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    EXPECT_TRUE(meta.contains("nothing"));
    const auto v = meta.get("nothing");
    ASSERT_TRUE(v.has_value());
    EXPECT_TRUE(v->is_null());
    // A null-*value* still yields no typed extraction.
    EXPECT_FALSE(v->as_int().has_value());
    EXPECT_FALSE(v->as_string().has_value());
}

// ---------------------------------------------------------------------------
// try_get_*: right type extracts, wrong type -> nullopt (no coercion).
// ---------------------------------------------------------------------------

TEST(MetaValueTest, TryGetRightType) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    EXPECT_EQ(meta.try_get_string("shortName"), std::string_view("2t"));
    EXPECT_EQ(meta.try_get_int("count"), std::optional<std::int64_t>(42));
    EXPECT_EQ(meta.try_get_uint("count"), std::optional<std::uint64_t>(42));
    EXPECT_EQ(meta.try_get_double("ratio"), std::optional<double>(0.5));
    EXPECT_EQ(meta.try_get_bool("flag"), std::optional<bool>(true));
}

TEST(MetaValueTest, TryGetWrongTypeIsNullopt) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    // string is not an int / double / bool
    EXPECT_FALSE(meta.try_get_int("shortName").has_value());
    EXPECT_FALSE(meta.try_get_double("shortName").has_value());
    EXPECT_FALSE(meta.try_get_bool("shortName").has_value());
    // int is not a string; a float is not an int (no truncation)
    EXPECT_FALSE(meta.try_get_string("count").has_value());
    EXPECT_FALSE(meta.try_get_int("ratio").has_value());
    // a map/array is not a scalar
    EXPECT_FALSE(meta.try_get_string("mars").has_value());
    EXPECT_FALSE(meta.try_get_int("tags").has_value());
}

TEST(MetaValueTest, TryGetAbsentIsNullopt) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    EXPECT_FALSE(meta.try_get_int("missing").has_value());
    EXPECT_FALSE(meta.try_get_string("missing").has_value());
    EXPECT_FALSE(meta.try_get_bool("missing").has_value());
    EXPECT_FALSE(meta.try_get_double("missing").has_value());
}

// ---------------------------------------------------------------------------
// as_double() is the sole widening accessor: an integer widens to double.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, AsDoubleWidensInteger) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    const auto d = meta.try_get_double("count");   // count is integer 42
    ASSERT_TRUE(d.has_value());
    EXPECT_DOUBLE_EQ(*d, 42.0);
    // but the integer does not extract as a float type-wise
    EXPECT_TRUE(meta.get("count")->is_int());
    EXPECT_FALSE(meta.get("count")->is_float());
}

// ---------------------------------------------------------------------------
// Signed/unsigned range reporting.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, SignedUnsignedRange) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    const auto neg = meta.get("neg");
    ASSERT_TRUE(neg.has_value());
    EXPECT_EQ(neg->as_int(), std::optional<std::int64_t>(-5));
    EXPECT_FALSE(neg->as_uint().has_value());   // negative -> not u64
}

// ---------------------------------------------------------------------------
// Absent vs empty / default: a stored "" and a stored 0 are FOUND.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, AbsentVsEmptyString) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    // Stored empty string is present and extracts as "".
    EXPECT_TRUE(meta.contains("emptyStr"));
    const auto s = meta.try_get_string("emptyStr");
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(*s, std::string_view(""));
    EXPECT_TRUE(s->empty());

    // A missing string key is absent, not "".
    EXPECT_FALSE(meta.contains("noStr"));
    EXPECT_FALSE(meta.try_get_string("noStr").has_value());
}

TEST(MetaValueTest, AbsentVsDefaultZero) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    // Stored 0 is present and extracts as 0 — not confused with absence.
    EXPECT_TRUE(meta.contains("level"));
    EXPECT_EQ(meta.try_get_int("level"), std::optional<std::int64_t>(0));

    // A missing int key is absent, not 0.
    EXPECT_FALSE(meta.contains("noInt"));
    EXPECT_FALSE(meta.try_get_int("noInt").has_value());
}

// ---------------------------------------------------------------------------
// Nested map navigation via the cursor.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, NestedMapNavigation) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    // meta.get("mars")->get("class")->as_string()
    const auto mars = meta.get("mars");
    ASSERT_TRUE(mars.has_value());
    ASSERT_TRUE(mars->is_map());
    EXPECT_EQ(mars->size(), 2u);
    EXPECT_TRUE(mars->contains("class"));
    EXPECT_FALSE(mars->contains("missing"));

    const auto cls = mars->get("class");
    ASSERT_TRUE(cls.has_value());
    EXPECT_EQ(cls->as_string(), std::string_view("od"));

    // Absent nested key -> nullopt (does not throw / dereference).
    EXPECT_FALSE(mars->get("missing").has_value());
}

// ---------------------------------------------------------------------------
// Array navigation: size / at / operator[] / range-for.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, ArrayNavigationStrings) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    const auto tags = meta.get("tags");
    ASSERT_TRUE(tags.has_value());
    ASSERT_TRUE(tags->is_array());
    EXPECT_EQ(tags->size(), 3u);

    EXPECT_EQ(tags->at(0)->as_string(), std::string_view("a"));
    EXPECT_EQ((*tags)[1].as_string(), std::string_view("b"));
    EXPECT_EQ(tags->at(2)->as_string(), std::string_view("c"));

    // Range-for yields each element as a meta_value.
    std::vector<std::string> collected;
    for (const auto& v : *tags) {
        const auto s = v.as_string();
        ASSERT_TRUE(s.has_value());
        collected.emplace_back(*s);
    }
    EXPECT_EQ(collected, (std::vector<std::string>{"a", "b", "c"}));
}

TEST(MetaValueTest, ArrayNavigationInts) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    const auto nums = meta.get("nums");
    ASSERT_TRUE(nums.has_value());
    EXPECT_EQ(nums->size(), 3u);

    std::int64_t sum = 0;
    for (const auto& v : *nums) {
        const auto n = v.as_int();
        ASSERT_TRUE(n.has_value());
        sum += *n;
    }
    EXPECT_EQ(sum, 60);
    EXPECT_EQ(nums->at(0)->as_int(), std::optional<std::int64_t>(10));
    EXPECT_EQ((*nums)[2].as_int(), std::optional<std::int64_t>(30));
}

TEST(MetaValueTest, ArrayOutOfRange) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    const auto nums = meta.get("nums");
    ASSERT_TRUE(nums.has_value());

    EXPECT_FALSE(nums->at(3).has_value());     // checked -> nullopt
    EXPECT_FALSE((*nums)[3].valid());          // unchecked -> invalid cursor
    // size() on a non-array is 0; at() on a non-array is nullopt.
    EXPECT_EQ(meta.get("count")->size(), 0u);
    EXPECT_FALSE(meta.get("count")->at(0).has_value());
}

// ---------------------------------------------------------------------------
// as_bytes() on a non-bytes value is nullopt (no coercion).
// ---------------------------------------------------------------------------

TEST(MetaValueTest, AsBytesOnNonBytesIsNullopt) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    EXPECT_FALSE(meta.get("shortName")->as_bytes().has_value());
    EXPECT_FALSE(meta.get("count")->as_bytes().has_value());
}

// ---------------------------------------------------------------------------
// Map enumeration via object(i): keys are enumerable and INCLUDE _reserved_.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, ObjectEnumerationIncludesReserved) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    const auto obj0 = meta.object(0);
    ASSERT_TRUE(obj0.valid());
    ASSERT_TRUE(obj0.is_map());
    ASSERT_GT(obj0.size(), 0u);

    std::set<std::string> keys;
    for (std::size_t i = 0; i < obj0.size(); ++i) {
        const auto k = obj0.key_at(i);
        ASSERT_TRUE(k.has_value());
        keys.emplace(*k);
        // value_at(i) is a valid cursor in the same order as key_at(i).
        EXPECT_TRUE(obj0.value_at(i).has_value());
    }

    EXPECT_TRUE(keys.count("shortName"));
    EXPECT_TRUE(keys.count("count"));
    EXPECT_TRUE(keys.count("mars"));
    // Enumeration exposes the library-managed _reserved_ section.
    EXPECT_TRUE(keys.count("_reserved_"));
    EXPECT_TRUE(obj0.contains("_reserved_"));

    // Nested navigation off a section view.
    EXPECT_EQ(obj0.get("mars")->get("class")->as_string(), std::string_view("od"));
    EXPECT_TRUE(obj0.get("_reserved_")->is_map());
}

// ---------------------------------------------------------------------------
// extra() / reserved() section views.
// ---------------------------------------------------------------------------

TEST(MetaValueTest, ExtraSectionView) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    const auto extra = meta.extra();
    ASSERT_TRUE(extra.valid());
    ASSERT_TRUE(extra.is_map());
    EXPECT_TRUE(extra.contains("producer"));
    EXPECT_EQ(extra.get("producer")->as_string(), std::string_view("ecmwf"));
    EXPECT_EQ(extra.get("revision")->as_int(), std::optional<std::int64_t>(7));
}

TEST(MetaValueTest, ReservedSectionViewIsAlwaysAMap) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    // Message-level reserved view is never null — an empty section is an
    // empty map, distinct from an invalid cursor.
    const auto reserved = meta.reserved();
    EXPECT_TRUE(reserved.valid());
    EXPECT_TRUE(reserved.is_map());
}

// ---------------------------------------------------------------------------
// Per-object scoping: get_at / contains_at reach a single base[i].
// ---------------------------------------------------------------------------

TEST(MetaValueTest, PerObjectScoping) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);
    ASSERT_EQ(meta.num_objects(), 2u);

    EXPECT_EQ(meta.get_at(0, "shortName")->as_string(), std::string_view("2t"));
    EXPECT_EQ(meta.get_at(1, "shortName")->as_string(), std::string_view("msl"));

    EXPECT_EQ(meta.get_at(0, "level")->as_int(), std::optional<std::int64_t>(0));
    EXPECT_EQ(meta.get_at(1, "level")->as_int(), std::optional<std::int64_t>(500));

    // `count` exists only in object 0.
    EXPECT_TRUE(meta.contains_at(0, "count"));
    EXPECT_FALSE(meta.contains_at(1, "count"));
    EXPECT_FALSE(meta.get_at(1, "count").has_value());

    // Nested per-object navigation.
    EXPECT_EQ(meta.get_at(1, "geometry.gridType")->as_string(),
              std::string_view("regular_ll"));
    EXPECT_TRUE(meta.contains_at(1, "geometry.gridType"));
    EXPECT_FALSE(meta.contains_at(0, "geometry.gridType"));
}

TEST(MetaValueTest, PerObjectHasNoExtraFallback) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    // `producer` lives in _extra_: reachable message-level, NOT per-object.
    EXPECT_TRUE(meta.contains("producer"));
    EXPECT_FALSE(meta.contains_at(0, "producer"));
    EXPECT_FALSE(meta.get_at(0, "producer").has_value());
}

TEST(MetaValueTest, PerObjectOutOfRange) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    EXPECT_FALSE(meta.contains_at(9, "shortName"));
    EXPECT_FALSE(meta.get_at(9, "shortName").has_value());
    EXPECT_FALSE(meta.object(9).valid());
    // Every accessor on the invalid cursor yields empty results.
    EXPECT_EQ(meta.object(9).size(), 0u);
    EXPECT_FALSE(meta.object(9).get("shortName").has_value());
    EXPECT_TRUE(meta.object(9).is_null());
}

// ---------------------------------------------------------------------------
// Cross-object first-match at message level (geometry only in object 1).
// ---------------------------------------------------------------------------

TEST(MetaValueTest, MessageLevelCrossObjectFirstMatch) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    // `geometry` is absent from base[0] but present in base[1]; the
    // message-level lookup first-matches across objects.
    EXPECT_TRUE(meta.contains("geometry.gridType"));
    EXPECT_EQ(meta.get("geometry.gridType")->as_string(),
              std::string_view("regular_ll"));
}

// ---------------------------------------------------------------------------
// `_reserved_` is hidden from path getters (message-level and per-object),
// but visible when enumerating a section (see ObjectEnumeration test).
// ---------------------------------------------------------------------------

TEST(MetaValueTest, ReservedHiddenFromPathLookups) {
    auto encoded = encode_capability_fixture();
    auto meta = decode_fixture(encoded);

    // Hidden at the first path segment, both message-level and per-object.
    EXPECT_FALSE(meta.contains("_reserved_"));
    EXPECT_FALSE(meta.get("_reserved_").has_value());
    EXPECT_FALSE(meta.contains("_reserved_.tensor"));

    EXPECT_FALSE(meta.contains_at(0, "_reserved_"));
    EXPECT_FALSE(meta.get_at(0, "_reserved_").has_value());
    EXPECT_FALSE(meta.get_at(0, "_reserved_.tensor").has_value());

    // …but visible via the object() enumeration view.
    EXPECT_TRUE(meta.object(0).contains("_reserved_"));
}
