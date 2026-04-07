/// @file cross_language_pre_encoded_helper.cpp
/// @brief Cross-language SHA256 golden test helper (C++ side).
///
/// Generates the SAME deterministic float64[1024] payload as the Rust driver,
/// encodes via tensogram::encode_pre_encoded(), decodes, and prints the
/// SHA-256 hex digest of the decoded payload to stdout.
///
/// Build:
///   g++ -std=c++17 -I include -I crates/tensogram-ffi \
///       tests/cpp/cross_language_pre_encoded_helper.cpp \
///       -L target/release -ltensogram_ffi \
///       -framework CoreFoundation -framework Security \
///       -framework SystemConfiguration -lc++ -lm \
///       -o build/cross_language_pre_encoded_helper

#include <tensogram.hpp>

extern "C" {
#include "tensogram.h"
}

// CommonCrypto for SHA-256 on macOS
#include <CommonCrypto/CommonDigest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string sha256_hex(const std::uint8_t* data, std::size_t len) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, static_cast<CC_LONG>(len), hash);
    char hex[CC_SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    return std::string(hex, CC_SHA256_DIGEST_LENGTH * 2);
}

int main() {
    // Same deterministic input as the Rust driver: 1024 float64 values.
    constexpr std::size_t N = 1024;
    std::vector<double> values(N);
    for (std::size_t i = 0; i < N; ++i) {
        values[i] = 200.0 + static_cast<double>(i) * 0.125;
    }

    // Serialize to little-endian bytes regardless of host endianness.
    std::vector<std::uint8_t> raw_bytes(N * sizeof(double));
    for (std::size_t i = 0; i < N; ++i) {
        std::uint64_t bits;
        std::memcpy(&bits, &values[i], sizeof(double));
        for (std::size_t j = 0; j < 8; ++j) {
            raw_bytes[i * 8 + j] = static_cast<std::uint8_t>(bits >> (j * 8));
        }
    }
    std::size_t raw_len = raw_bytes.size();

    // Build descriptor JSON for encoding="none" (raw pass-through).
    std::string json =
        R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[1024],)"
        R"("strides":[8],"dtype":"float64","byte_order":"little",)"
        R"("encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {raw_bytes.data(), raw_len}
    };

    auto wire = tensogram::encode_pre_encoded(json, objects);

    // Decode and extract payload.
    auto msg = tensogram::decode(wire.data(), wire.size());
    if (msg.num_objects() != 1) {
        std::fprintf(stderr, "Expected 1 object, got %zu\n", msg.num_objects());
        return 1;
    }

    auto obj = msg.object(0);
    auto sha = sha256_hex(obj.data(), obj.data_size());

    // Print ONLY the hex digest to stdout (no newline, no extra text).
    std::printf("%s", sha.c_str());
    return 0;
}
