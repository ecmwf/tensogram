/// @file 11_encode_pre_encoded.cpp
/// @brief Example 11 — Pre-encoded payloads using the C++ wrapper.
///
/// Demonstrates tensogram::encode_pre_encoded(), which lets callers hand
/// already-encoded payload bytes to the library without re-running the
/// encoding pipeline.
///
/// This is useful for GPU pipelines, HPC frameworks, or any system that
/// produces encoded data outside the library — the caller packs the data
/// themselves and Tensogram wraps it into the wire format.
///
/// Build:
///   cmake -B build && cmake --build build
///   Then compile manually (or add to your CMakeLists.txt):
///   g++ -std=c++17 -I include -I crates/tensogram-ffi \
///       examples/cpp/11_encode_pre_encoded.cpp \
///       -L target/release -ltensogram_ffi \
///       -framework CoreFoundation -framework Security \
///       -framework SystemConfiguration -lc++ -lm \
///       -o build/example_11

#include <tensogram.hpp>

extern "C" {
#include "tensogram.h"
}

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// -----------------------------------------------------------------------
// Manual bit-packing helper (MSB-first, big-endian bit order)
// -----------------------------------------------------------------------

static std::vector<std::uint8_t> bit_pack(const std::vector<std::uint64_t>& values,
                                          int bits_per_value) {
    std::size_t total_bits = values.size() * static_cast<std::size_t>(bits_per_value);
    std::size_t total_bytes = (total_bits + 7) / 8;
    std::vector<std::uint8_t> buf(total_bytes, 0);

    std::size_t bit_pos = 0;
    for (auto val : values) {
        int remaining = bits_per_value;
        while (remaining > 0) {
            std::size_t byte_idx = bit_pos / 8;
            int bit_offset = static_cast<int>(bit_pos % 8);
            int space = 8 - bit_offset;
            int write_bits = std::min(remaining, space);
            int shift = remaining - write_bits;
            auto bits = static_cast<std::uint8_t>(
                (val >> shift) & ((1ULL << write_bits) - 1));
            buf[byte_idx] |= static_cast<std::uint8_t>(bits << (space - write_bits));
            bit_pos += static_cast<std::size_t>(write_bits);
            remaining -= write_bits;
        }
    }
    return buf;
}

int main() {
    // ── 1. Synthesise pre-encoded payload (simple_packing) ─────────────

    constexpr int N = 1000;
    std::vector<double> temps(N);
    for (int i = 0; i < N; ++i)
        temps[i] = 249.15 + i * 0.1;
    std::printf("Source: %d float64 values  raw=%zu bytes\n",
                N, temps.size() * sizeof(double));

    // Compute packing parameters via the C FFI helper
    double reference_value = 0.0;
    std::int32_t binary_scale_factor = 0;
    constexpr std::uint32_t bits_per_value = 16;
    constexpr std::int32_t decimal_scale_factor = 0;

    tgm_error err = tgm_simple_packing_compute_params(
        temps.data(), temps.size(),
        bits_per_value, decimal_scale_factor,
        &reference_value, &binary_scale_factor);
    assert(err == TGM_ERROR_OK);

    std::printf("Packing params: ref=%.4f  bsf=%d  dsf=%d  bpv=%u\n",
                reference_value, binary_scale_factor,
                decimal_scale_factor, bits_per_value);

    // Manual quantisation: packed_int = round((value - ref) * 10^dsf / 2^bsf)
    double scale = std::pow(10.0, decimal_scale_factor) /
                   std::pow(2.0, binary_scale_factor);
    std::vector<std::uint64_t> packed_ints(N);
    for (int i = 0; i < N; ++i) {
        packed_ints[i] = static_cast<std::uint64_t>(
            std::round((temps[i] - reference_value) * scale));
    }

    auto packed_bytes = bit_pack(packed_ints, static_cast<int>(bits_per_value));
    std::printf("Packed payload: %zu bytes  (%u bits x %d values)\n",
                packed_bytes.size(), bits_per_value, N);

    // ── 2. Build descriptor and encode_pre_encoded ─────────────────────

    char ref_buf[64];
    std::snprintf(ref_buf, sizeof(ref_buf), "%.17g", reference_value);

    std::string json =
        R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[)" +
        std::to_string(N) +
        R"(],"strides":[8],"dtype":"float64","byte_order":"little",)"
        R"("encoding":"simple_packing","filter":"none","compression":"none",)"
        R"("bits_per_value":)" + std::to_string(bits_per_value) +
        R"(,"reference_value":)" + std::string(ref_buf) +
        R"(,"binary_scale_factor":)" + std::to_string(binary_scale_factor) +
        R"(,"decimal_scale_factor":)" + std::to_string(decimal_scale_factor) +
        R"(}]})";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {packed_bytes.data(), packed_bytes.size()}
    };

    auto message = tensogram::encode_pre_encoded(json, objects);
    std::printf("Wire message: %zu bytes\n", message.size());

    // ── 3. Decode and verify ───────────────────────────────────────────

    auto msg = tensogram::decode(message.data(), message.size());
    assert(msg.num_objects() == 1);

    auto obj = msg.object(0);
    assert(obj.dtype_string() == "float64");

    const double* decoded = obj.data_as<double>();
    const std::size_t count = obj.element_count<double>();
    assert(count == static_cast<std::size_t>(N));

    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double e = std::abs(temps[i] - decoded[i]);
        if (e > max_err) max_err = e;
    }
    std::printf("Max quantisation error: %.6f\n", max_err);
    assert(max_err < 0.01);

    // ── 4. encoding=none variant ───────────────────────────────────────

    std::vector<float> raw_data(50);
    for (int i = 0; i < 50; ++i) raw_data[i] = static_cast<float>(i);

    std::string raw_json =
        R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[50],)"
        R"("strides":[4],"dtype":"float32","byte_order":"little",)"
        R"("encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> raw_objects = {
        {reinterpret_cast<const std::uint8_t*>(raw_data.data()),
         raw_data.size() * sizeof(float)}
    };

    auto raw_msg = tensogram::encode_pre_encoded(raw_json, raw_objects);
    auto raw_decoded = tensogram::decode(raw_msg.data(), raw_msg.size());
    auto raw_obj = raw_decoded.object(0);
    const float* rp = raw_obj.data_as<float>();
    for (int i = 0; i < 50; ++i) {
        assert(rp[i] == static_cast<float>(i));
    }
    std::printf("Raw encoding=none round-trip: OK\n");

    // ── 5. StreamingEncoder variant ────────────────────────────────────

    // Write to a temp file, then decode
    const char* tmp_path = "/tmp/tensogram_example_11.tgm";
    {
        tensogram::streaming_encoder enc(tmp_path, R"({"version":2})");

        // Pre-encoded simple_packing object
        std::string desc_sp =
            R"({"type":"ndarray","ndim":1,"shape":[)" + std::to_string(N) +
R"(],"strides":[8],"dtype":"float64","byte_order":"little",)"
            R"("encoding":"simple_packing","filter":"none","compression":"none",)"
            R"("bits_per_value":)" + std::to_string(bits_per_value) +
            R"(,"reference_value":)" + std::string(ref_buf) +
            R"(,"binary_scale_factor":)" + std::to_string(binary_scale_factor) +
            R"(,"decimal_scale_factor":)" + std::to_string(decimal_scale_factor) +
            R"(})";
        enc.write_object_pre_encoded(desc_sp,
                                     packed_bytes.data(), packed_bytes.size());

        // Pre-encoded encoding=none object
        std::string desc_raw =
            R"({"type":"ndarray","ndim":1,"shape":[50],"strides":[4],)"
            R"("dtype":"float32","byte_order":"little",)"
            R"("encoding":"none","filter":"none","compression":"none"})";
        enc.write_object_pre_encoded(desc_raw,
                                     reinterpret_cast<const std::uint8_t*>(raw_data.data()),
                                     raw_data.size() * sizeof(float));

        enc.finish();
    }

    auto f = tensogram::file::open(tmp_path);
    assert(f.message_count() == 1);
    auto stream_msg = f.decode_message(0);
    assert(stream_msg.num_objects() == 2);

    // Verify simple_packing object
    auto s0 = stream_msg.object(0);
    const double* sp0 = s0.data_as<double>();
    double s_max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double e = std::abs(temps[i] - sp0[i]);
        if (e > s_max_err) s_max_err = e;
    }
    assert(s_max_err < 0.01);

    // Verify encoding=none object
    auto s1 = stream_msg.object(1);
    const float* sp1 = s1.data_as<float>();
    for (int i = 0; i < 50; ++i) {
        assert(sp1[i] == static_cast<float>(i));
    }
    std::printf("Streaming pre-encoded: OK\n");

    // Clean up temp file
    std::remove(tmp_path);

    std::printf("\nOK: all pre-encoded round-trips succeeded\n");
    return 0;
}
