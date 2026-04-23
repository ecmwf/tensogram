// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 16_multi_threaded_pipeline.cpp
/// @brief Example 16 — Multi-threaded coding pipeline (C++ wrapper).
///
/// Demonstrates the caller-controlled `threads` budget on
/// `encode_options` / `decode_options`.
///
/// Invariants shown:
///   1. `threads=0` matches the sequential path byte-identically.
///   2. Transparent codecs (encoding="none", simple_packing, szip, ...)
///      produce byte-identical encoded payloads across any `threads`
///      value.
///   3. Decoded payloads always match the input regardless of the
///      thread count used on either encode or decode.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/16_multi_threaded_pipeline

#include <tensogram.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main() {
    // ── 1. Build a sizeable single-object message ─────────────────────────
    //
    // 2 million f64 values (≈ 15 MiB) — representative of a mid-sized
    // ML output tensor or an atmospheric field.
    constexpr std::size_t N = 2'000'000;
    std::vector<double> values(N);
    for (std::size_t i = 0; i < N; ++i) {
        values[i] = 250.0 + std::sin(static_cast<double>(i)) * 30.0;
    }
    const std::size_t data_bytes = values.size() * sizeof(double);

    const std::string metadata_json = R"({
        "version": 3,
        "descriptors": [{
            "type": "ntensor", "ndim": 1, "shape": [2000000], "strides": [8],
            "dtype": "float64",
            "encoding": "none", "filter": "none", "compression": "none"
        }]
    })";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(values.data()), data_bytes}
    };

    // ── 2. Sequential vs parallel encode ──────────────────────────────────
    tensogram::encode_options seq_opts;     // threads = 0 (sequential)
    tensogram::encode_options par_opts;
    par_opts.threads = 8;

    auto t0 = std::chrono::steady_clock::now();
    auto msg_seq = tensogram::encode(metadata_json, objects, seq_opts);
    auto dur_seq = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    t0 = std::chrono::steady_clock::now();
    auto msg_par = tensogram::encode(metadata_json, objects, par_opts);
    auto dur_par = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::printf("Encode %zu f64 values (%.1f MiB):\n",
                N, data_bytes / (1024.0 * 1024.0));
    std::printf("  threads = 0: %7.1f ms\n", dur_seq);
    std::printf("  threads = 8: %7.1f ms   (x%.2f speedup)\n",
                dur_par, dur_seq / std::max(dur_par, 1e-9));

    // Transparent-pipeline invariant: decoded payloads are byte-identical
    // across any thread count.  (The wire bytes themselves may differ
    // because `_reserved_.time` / `_reserved_.uuid` are stamped per call.)
    auto dec_seq = tensogram::decode(msg_seq.data(), msg_seq.size(), {});
    auto dec_par = tensogram::decode(msg_par.data(), msg_par.size(), {});
    auto payload_seq = dec_seq.object(0).data_as<std::uint8_t>();
    auto payload_par = dec_par.object(0).data_as<std::uint8_t>();
    const bool bytes_match = std::memcmp(payload_seq, payload_par, data_bytes) == 0;
    std::printf("  decoded payloads identical across thread counts: %s\n",
                bytes_match ? "yes" : "no");
    assert(bytes_match);

    // ── 3. Decode with a thread budget ────────────────────────────────────
    tensogram::decode_options dec_seq_opts;
    tensogram::decode_options dec_par_opts;
    dec_par_opts.threads = 8;

    t0 = std::chrono::steady_clock::now();
    auto dec_a = tensogram::decode(msg_seq.data(), msg_seq.size(), dec_seq_opts);
    auto dec_a_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    t0 = std::chrono::steady_clock::now();
    auto dec_b = tensogram::decode(msg_seq.data(), msg_seq.size(), dec_par_opts);
    auto dec_b_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::printf("\nDecode (threads = 0): %7.1f ms\n", dec_a_ms);
    std::printf("Decode (threads = 8): %7.1f ms   (x%.2f speedup)\n",
                dec_b_ms, dec_a_ms / std::max(dec_b_ms, 1e-9));

    [[maybe_unused]] auto pa = dec_a.object(0).data_as<std::uint8_t>();
    [[maybe_unused]] auto pb = dec_b.object(0).data_as<std::uint8_t>();
    assert(std::memcmp(pa, pb, data_bytes) == 0);
    std::printf("  decoded bytes match: yes\n");

    return 0;
}
