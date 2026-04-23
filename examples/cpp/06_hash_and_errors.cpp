// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 06_hash_and_errors.cpp
/// @brief Example 06 — Integrity and typed error handling (C++ wrapper).
///
/// Every frame carries an inline xxh3-64 hash slot.  When the preamble's
/// `HASHES_PRESENT` flag is set (the default), those slots are populated
/// at encode time and `tensogram::validate` at the default level
/// recomputes the per-frame body hashes and compares them to the inline
/// values.
///
/// Importantly, `tensogram::decode` is **not** the integrity surface:
/// `decode_options::verify_hash = true` is accepted for API
/// compatibility but does not perform integrity checking — use
/// `tensogram::validate()` for that.
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/06_hash_and_errors

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main() {
    // ── 1. Encode one message with hashing on (default) and one without ───
    //
    // Sizing: 4096 float32 values = 16 KiB payload.  This is large
    // enough that a byte-flip near the message midpoint reliably lands
    // deep inside the data-object frame body (the hashed region).
    const std::string metadata_json = R"({
        "version": 3,
        "descriptors": [{
            "type": "ntensor", "ndim": 1, "shape": [4096], "strides": [4],
            "dtype": "float32",
            "encoding": "none", "filter": "none", "compression": "none"
        }]
    })";

    std::vector<float> data(4096, 42.0f);
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(data.data()),
         data.size() * sizeof(float)}
    };

    tensogram::encode_options hashed_opts;                 // hash_algo="xxh3"
    tensogram::encode_options unhashed_opts;
    unhashed_opts.hash_algo = "";                          // clears HASHES_PRESENT

    auto hashed   = tensogram::encode(metadata_json, objects, hashed_opts);
    auto unhashed = tensogram::encode(metadata_json, objects, unhashed_opts);
    std::printf("Hashed message:   %zu bytes\n", hashed.size());
    std::printf("Unhashed message: %zu bytes\n", unhashed.size());

    // ── 2. decode is hash-agnostic ────────────────────────────────────────
    {
        tensogram::decode_options verify;
        verify.verify_hash = true;
        auto m1 = tensogram::decode(hashed.data(),   hashed.size(),   verify);
        auto m2 = tensogram::decode(unhashed.data(), unhashed.size(), verify);
        std::printf("\nBoth messages decode cleanly (decode is hash-agnostic).\n");
        (void)m1; (void)m2;
    }

    // ── 3. validate() detects corruption ──────────────────────────────────
    //
    // Clean hashed message: every inline slot verifies → no issues,
    // hash_verified=true in the JSON.  The unhashed clean message yields
    // a single `no_hash_available` warning (HASHES_PRESENT=0, nothing to
    // recompute against).  A byte-flip in the body surfaces as a
    // `hash_mismatch` error.
    std::printf("\n=== validate(clean, hashed) ===\n%s\n",
                tensogram::validate(hashed.data(), hashed.size()).c_str());
    std::printf("\n=== validate(clean, unhashed) ===\n%s\n",
                tensogram::validate(unhashed.data(), unhashed.size()).c_str());

    std::vector<std::uint8_t> corrupted = hashed;
    const std::size_t mid = corrupted.size() / 2;
    corrupted[mid] ^= 0xFF;
    std::printf("\n=== validate(corrupted at byte %zu) ===\n%s\n",
                mid,
                tensogram::validate(corrupted.data(), corrupted.size()).c_str());

    // ── 4. Typed exception hierarchy for malformed input ──────────────────
    //
    // All library-raised errors extend tensogram::error (≡ std::runtime_error).
    // Callers can catch the base class for coarse handling or a specific
    // subclass for fine-grained recovery.
    std::printf("\n=== Error handling (typed exceptions) ===\n");
    try {
        const std::uint8_t garbage[] = "GARBAGE!";
        tensogram::decode(garbage, sizeof(garbage) - 1, {});
    } catch (const tensogram::framing_error& e) {
        std::printf("  garbage input  → framing_error: %s\n", e.what());
    }

    try {
        tensogram::decode(nullptr, 0, {});
    } catch (const tensogram::error& e) {
        std::printf("  empty input    → %s\n", e.what());
    }

    try {
        tensogram::decode_object(hashed.data(), hashed.size(), 99, {});
    } catch (const tensogram::object_error& e) {
        std::printf("  bad index (99) → object_error: %s\n", e.what());
    }

    std::printf("\nAll checks passed.\n");
    return 0;
}
