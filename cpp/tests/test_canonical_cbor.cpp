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
// Tests for tensogram::verify_canonical_cbor() — RFC 8949 §4.2.1 canonical
// form check.  Binds the C ABI tgm_verify_canonical_cbor.

#include <gtest/gtest.h>
#include <tensogram.hpp>

#include <cstdint>
#include <vector>

TEST(VerifyCanonicalCborTest, ScalarIsCanonical) {
    // CBOR unsigned integer 42 → 0x18 0x2A.
    const std::vector<std::uint8_t> bytes = {0x18, 0x2A};
    EXPECT_TRUE(tensogram::verify_canonical_cbor(bytes.data(), bytes.size()));
}

TEST(VerifyCanonicalCborTest, LengthFirstSortedMapIsCanonical) {
    // {"a": 1, "bb": 2} — length-first ordering ("a" len 1 before "bb" len 2).
    const std::vector<std::uint8_t> bytes = {
        0xA2,              // map(2)
        0x61, 0x61,        // "a"
        0x01,              // 1
        0x62, 0x62, 0x62,  // "bb"
        0x02,              // 2
    };
    EXPECT_TRUE(tensogram::verify_canonical_cbor(bytes.data(), bytes.size()));
}

TEST(VerifyCanonicalCborTest, UnsortedMapIsNotCanonical) {
    // {"bb": 2, "a": 1} — longer key first violates length-first ordering.
    const std::vector<std::uint8_t> bytes = {
        0xA2,              // map(2)
        0x62, 0x62, 0x62,  // "bb"
        0x02,              // 2
        0x61, 0x61,        // "a"
        0x01,              // 1
    };
    EXPECT_FALSE(tensogram::verify_canonical_cbor(bytes.data(), bytes.size()));
}

TEST(VerifyCanonicalCborTest, GarbageIsNotCanonical) {
    const std::vector<std::uint8_t> garbage = {0xFF, 0xFE, 0xFD, 0xFC};
    EXPECT_FALSE(tensogram::verify_canonical_cbor(garbage.data(), garbage.size()));
}

TEST(VerifyCanonicalCborTest, EmptyIsNotCanonical) {
    // Empty input is not valid CBOR → false (and never throws).
    EXPECT_FALSE(tensogram::verify_canonical_cbor(nullptr, 0));
}

TEST(VerifyCanonicalCborTest, NullWithNonZeroLenIsFalse) {
    // Arg error surfaces as a false predicate, not an exception.
    EXPECT_FALSE(tensogram::verify_canonical_cbor(nullptr, 8));
}
