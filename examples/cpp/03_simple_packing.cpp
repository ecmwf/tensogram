/// Example 03 — Simple packing (C++)
///
/// Demonstrates lossy compression via simple_packing.
/// The packing parameters (bits_per_value etc.) are computed by the library
/// and embedded in the CBOR metadata, so the decoder is self-contained.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>

#include "tensogram.h"

static void check(tgm_error_t err, const char *ctx) {
    if (err != TGM_OK) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s: %s", ctx, tgm_error_string(err));
        throw std::runtime_error(buf);
    }
}

int main() {
    constexpr int N = 1000;

    // Source data: 1000 temperature values as f64
    std::vector<double> temps(N);
    for (int i = 0; i < N; ++i)
        temps[i] = 249.15 + i * 0.1;

    // ── Metadata requests simple_packing at 16 bits per value ─────────────────
    //
    // The library computes reference_value, binary_scale_factor, etc. from
    // the actual data and writes them into the CBOR metadata automatically.
    // The caller does not need to compute these parameters.
    const char *metadata_json = R"({
        "version": 1,
        "objects": [{
            "type": "ntensor",
            "ndim": 1,
            "shape": [1000],
            "strides": [1],
            "dtype": "float64"
        }],
        "payload": [{
            "byte_order": "big",
            "encoding": "simple_packing",
            "bits_per_value": 16,
            "decimal_scale_factor": 0,
            "filter": "none",
            "compression": "none"
        }]
    })";

    // NOTE: The C API accepts encoding parameters in the JSON and computes
    // the remaining parameters internally (reference_value, binary_scale_factor)
    // from the actual data. This differs from the Rust API where you call
    // compute_params() yourself.

    const uint8_t *data_ptr = reinterpret_cast<const uint8_t*>(temps.data());
    size_t         data_len = N * sizeof(double);
    const uint8_t *ptrs[]   = { data_ptr };
    size_t         lens[]   = { data_len };

    uint8_t *msg_buf = nullptr;
    size_t   msg_len = 0;
    check(tgm_encode(metadata_json, ptrs, lens, 1, &msg_buf, &msg_len), "encode");

    std::size_t expected_packed = (N * 16 + 7) / 8;
    std::printf("Raw:    %zu bytes\n", data_len);
    std::printf("Packed: ~%zu bytes (estimate)\n", expected_packed);
    std::printf("Total message: %zu bytes\n", msg_len);
    std::printf("Compression ratio: ~%.1fx\n",
                static_cast<double>(data_len) / expected_packed);

    // ── Decode ────────────────────────────────────────────────────────────────
    tgm_message_t *raw_msg = nullptr;
    check(tgm_decode(msg_buf, msg_len, &raw_msg), "decode");

    size_t decoded_len = 0;
    const uint8_t *decoded = tgm_object_data(raw_msg, 0, &decoded_len);

    // Decoded values are always f64 regardless of original dtype
    const double *decoded_temps = reinterpret_cast<const double*>(decoded);
    size_t n_decoded = decoded_len / sizeof(double);
    assert(n_decoded == static_cast<size_t>(N));

    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(temps[i] - decoded_temps[i]);
        if (err > max_err) max_err = err;
    }
    std::printf("Max error: %.6f K\n", max_err);
    assert(max_err < 0.01);
    std::printf("Precision OK (< 0.01 K)\n");

    tgm_message_free(raw_msg);
    tgm_free(msg_buf);
    return 0;
}
