// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 10_doctor.cpp
/// @brief Example 10 — Environment diagnostics via tensogram::doctor().
///
/// `tensogram::doctor()` runs the same diagnostics as the
/// `tensogram doctor` CLI subcommand and the Python / WASM `doctor()`
/// exports, returning a JSON report describing the build, the compiled-in
/// features (codecs, threading, I/O), and an encode/decode self-test.
///
/// This example prints the report and checks the top-level sections are
/// present.  Pipe the output through `jq` to explore the schema:
///
///   ./build/bin/10_doctor | jq .
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/10_doctor

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <string>

int main() {
    const std::string report = tensogram::doctor();

    // The report is a JSON object; print it verbatim for inspection.
    std::printf("%s\n", report.c_str());

    // Sanity-check the documented top-level shape:
    //   { "build": {...}, "features": [...], "self_test": [...] }
    assert(!report.empty());
    assert(report.front() == '{');
    assert(report.find("\"build\"") != std::string::npos);
    assert(report.find("\"features\"") != std::string::npos);
    assert(report.find("\"self_test\"") != std::string::npos);

    std::printf("\nDiagnostics report OK "
                "(build / features / self_test present, wire version %u)\n",
                static_cast<unsigned>(tensogram::wire_version));
    return 0;
}
