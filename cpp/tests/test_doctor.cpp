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
// Tests for tensogram::doctor() — the C++ wrapper over tgm_doctor_to_json.

#include <gtest/gtest.h>
#include <tensogram.hpp>

#include <string>

// The report is a JSON object shaped `{ "build": {...}, "features": [...],
// "self_test": [...] }` (see rust/tensogram/src/doctor/mod.rs).  We assert on
// stable structural substrings rather than pulling in a JSON parser.

TEST(DoctorTest, ReturnsNonEmptyJsonObject) {
    const std::string report = tensogram::doctor();
    ASSERT_FALSE(report.empty());
    EXPECT_EQ(report.front(), '{');
    EXPECT_EQ(report.back(), '}');
}

TEST(DoctorTest, ContainsTopLevelSections) {
    const std::string report = tensogram::doctor();
    EXPECT_NE(report.find("\"build\""), std::string::npos) << report;
    EXPECT_NE(report.find("\"features\""), std::string::npos) << report;
    EXPECT_NE(report.find("\"self_test\""), std::string::npos) << report;
}

TEST(DoctorTest, BuildReportsWireVersion) {
    const std::string report = tensogram::doctor();
    // The build block carries the wire-format version integer.
    EXPECT_NE(report.find("\"wire_version\""), std::string::npos) << report;
    EXPECT_NE(report.find(std::to_string(tensogram::wire_version)), std::string::npos)
        << report;
}

TEST(DoctorTest, StableAcrossRepeatedCalls) {
    // doctor() owns and frees its tgm_bytes_t each call; repeated calls must
    // not leak or diverge in shape.
    const std::string a = tensogram::doctor();
    const std::string b = tensogram::doctor();
    EXPECT_EQ(a, b);
}
