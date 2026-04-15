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
// Shared helpers for C++ tests.

#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include <tensogram.hpp>

#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace test_helpers {

/// Build a minimal v2 JSON descriptor for a 1-D float32 message.
inline std::string simple_f32_json(std::size_t count) {
    return R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
}

/// Encode a simple 1-D float32 message.
inline std::vector<std::uint8_t> encode_simple_f32(
    const std::vector<float>& values)
{
    auto json = simple_f32_json(values.size());
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}
    };
    return tensogram::encode(json, objects);
}

/// Create a unique temporary file path with @p suffix (e.g. ".tgm").
///
/// On POSIX: uses mkstemp() to reserve a unique name, then removes the
/// placeholder file and returns the path with the caller-supplied suffix.
/// On Windows: uses GetTempPath/GetTempFileName for a portable equivalent.
inline std::string temp_path(const std::string& suffix = ".tgm") {
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_dir);
    char tmp_file[MAX_PATH];
    GetTempFileNameA(tmp_dir, "tgm", 0, tmp_file);
    std::remove(tmp_file);  // remove placeholder — we want the suffixed path
    return std::string(tmp_file) + suffix;
#else
    char tmpl[] = "/tmp/tensogram_test_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd >= 0) {
        close(fd);
        std::remove(tmpl);   // remove the base file — we want the suffixed path
    }
    return std::string(tmpl) + suffix;
#endif
}

/// RAII guard that creates a unique temp file and removes it on destruction.
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& suffix = ".tgm")
        : path(temp_path(suffix)) {}
    ~TempFile() { std::remove(path.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

} // namespace test_helpers

#endif // TEST_HELPERS_HPP
