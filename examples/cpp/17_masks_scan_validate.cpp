// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 17_masks_scan_validate.cpp
/// @brief Example 17 — Wave C reader helpers: mask-aware decode, whole-buffer
///        validation, and file/buffer scanning (C++ wrapper).
///
/// Demonstrates three of the widened C-ABI reader entry points:
///   * `tensogram::decode_with_masks()` + `decoded_object::has_masks()` /
///     `mask()` — decode without restoring NaN / ±Inf in place and recover
///     the per-element masks instead.
///   * `tensogram::validate_buffer()` — validate every message in a
///     multi-message buffer (the in-memory companion to validate_file()).
///   * `tensogram::scan_file()` — locate message boundaries in a `.tgm`
///     file via streaming seek-and-read I/O.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/17_masks_scan_validate

#include <tensogram.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

// A 1-D float32 descriptor (n elements), no encoding pipeline.
std::string f32_json(std::size_t n) {
    return std::string(R"({"version":3,"descriptors":[{)")
        + R"("type":"ndarray","ndim":1,)"
        + "\"shape\":[" + std::to_string(n) + "],\"strides\":[4],"
        + R"("dtype":"float32","byte_order":"little",)"
        + R"("encoding":"none","filter":"none","compression":"none"}]})";
}

std::vector<std::uint8_t> encode_f32(const std::vector<float>& values,
                                     const tensogram::encode_options& opts) {
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}};
    return tensogram::encode(f32_json(values.size()), objects, opts);
}

const char* kind_name(tensogram::mask_kind k) {
    switch (k) {
        case tensogram::mask_kind::nan:     return "nan";
        case tensogram::mask_kind::pos_inf: return "+inf";
        case tensogram::mask_kind::neg_inf: return "-inf";
    }
    return "?";
}

}  // namespace

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float pinf = std::numeric_limits<float>::infinity();
    const float ninf = -std::numeric_limits<float>::infinity();

    // ── 1. decode_with_masks: keep the non-finite masks ───────────────────
    std::printf("=== decode_with_masks ===\n");
    tensogram::encode_options mopts;
    mopts.allow_nan = true;
    mopts.allow_inf = true;
    auto masked = encode_f32({1.0f, nan, pinf, ninf, 2.0f}, mopts);

    auto msg = tensogram::decode_with_masks(masked.data(), masked.size());
    auto obj = msg.object(0);
    std::printf("object has_masks: %s\n", obj.has_masks() ? "true" : "false");

    // The payload keeps the canonical 0.0 at every masked position.
    const float* raw = obj.data_as<float>();
    std::printf("payload (0.0 where non-finite):");
    for (std::size_t i = 0; i < obj.element_count<float>(); ++i) {
        std::printf(" %.1f", static_cast<double>(raw[i]));
    }
    std::printf("\n");

    for (auto k : {tensogram::mask_kind::nan, tensogram::mask_kind::pos_inf,
                   tensogram::mask_kind::neg_inf}) {
        auto m = obj.mask(k);
        std::printf("  mask[%-4s]:", kind_name(k));
        if (m.empty()) {
            std::printf(" (absent)");
        } else {
            for (std::uint8_t byte : m) std::printf(" %u", byte);
        }
        std::printf("\n");
    }

    // ── 2. validate_buffer: validate a multi-message buffer ───────────────
    std::printf("\n=== validate_buffer (multi-message) ===\n");
    tensogram::encode_options plain;  // default: inline xxh3 hashes
    auto a = encode_f32({1.0f, 2.0f, 3.0f}, plain);
    auto b = encode_f32({4.0f, 5.0f}, plain);
    std::vector<std::uint8_t> buffer;
    buffer.insert(buffer.end(), a.begin(), a.end());
    buffer.insert(buffer.end(), b.begin(), b.end());

    auto report = tensogram::validate_buffer(buffer.data(), buffer.size(), "full");
    std::printf("%s\n", report.c_str());

    // Inline hash slots recovered without a full CBOR decode:
    auto hashes = tensogram::data_object_inline_hashes(a.data(), a.size());
    std::printf("inline hashes of message 0:");
    for (const auto& h : hashes) {
        std::printf(" %s", h ? h->c_str() : "null");
    }
    std::printf("\n");

    // ── 3. scan_file: locate message boundaries on disk ───────────────────
    std::printf("\n=== scan_file ===\n");
    const std::string path = "tensogram_example_17_scan.tgm";
    {
        auto f = tensogram::file::create(path);
        f.append_raw(a);
        f.append_raw(b);
    }
    auto entries = tensogram::scan_file(path);
    std::printf("scan_file found %zu messages:\n", entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        std::printf("  [%zu] offset=%zu length=%zu\n", i,
                    entries[i].offset, entries[i].length);
    }
    std::remove(path.c_str());

    return 0;
}
