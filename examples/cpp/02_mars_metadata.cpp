/// Example 02 — MARS-namespaced metadata (C++)
///
/// Shows how to read MARS keys from a decoded message and how to build
/// metadata with MARS namespace keys using the JSON metadata path.

#include <cassert>
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
    // ── Encode with MARS metadata ─────────────────────────────────────────────
    const char *metadata_json = R"({
        "version": 1,
        "objects": [{
            "type": "ntensor",
            "ndim": 2,
            "shape": [721, 1440],
            "strides": [1440, 1],
            "dtype": "float32",
            "mars": {"param": "2t", "levtype": "sfc"}
        }],
        "payload": [{
            "byte_order": "big",
            "encoding": "none",
            "filter": "none",
            "compression": "none"
        }],
        "mars": {
            "class": "od",
            "date": "20260401",
            "step": 6,
            "time": "0000",
            "type": "fc"
        }
    })";

    std::vector<float> data(721 * 1440, 273.15f);
    const uint8_t *data_ptr = reinterpret_cast<const uint8_t*>(data.data());
    size_t         data_len = data.size() * sizeof(float);
    const uint8_t *ptrs[] = { data_ptr };
    size_t         lens[] = { data_len };

    uint8_t *msg_buf = nullptr;
    size_t   msg_len = 0;
    check(tgm_encode(metadata_json, ptrs, lens, 1, &msg_buf, &msg_len), "encode");

    // ── Decode metadata only (no payload) ─────────────────────────────────────
    //
    // tgm_decode_metadata() reads only the CBOR section.
    // No object payload bytes are allocated or read.
    tgm_metadata_t *meta = nullptr;
    check(tgm_decode_metadata(msg_buf, msg_len, &meta), "decode_metadata");

    // ── Read keys using dot-notation ──────────────────────────────────────────
    //
    // tgm_metadata_get_string() accepts "mars.param" notation.
    // Returns NULL if the key is missing.
    const char *class_   = tgm_metadata_get_string(meta, "mars.class");
    const char *date      = tgm_metadata_get_string(meta, "mars.date");
    const char *type      = tgm_metadata_get_string(meta, "mars.type");
    int64_t     step      = tgm_metadata_get_int(meta, "mars.step", -1);
    const char *param     = tgm_metadata_get_string(meta, "objects[0].mars.param");
    const char *levtype   = tgm_metadata_get_string(meta, "objects[0].mars.levtype");

    std::printf("Message-level:\n");
    std::printf("  class   = %s\n", class_ ? class_ : "(null)");
    std::printf("  date    = %s\n", date   ? date   : "(null)");
    std::printf("  type    = %s\n", type   ? type   : "(null)");
    std::printf("  step    = %lld\n", (long long)step);
    std::printf("Object 0:\n");
    std::printf("  param   = %s\n", param   ? param   : "(null)");
    std::printf("  levtype = %s\n", levtype ? levtype : "(null)");
    std::printf("  version = %llu\n", (unsigned long long)tgm_metadata_version(meta));
    std::printf("  objects = %zu\n",  tgm_metadata_num_objects(meta));

    assert(class_  && std::strcmp(class_, "od") == 0);
    assert(date    && std::strcmp(date,   "20260401") == 0);
    assert(step    == 6);
    assert(param   && std::strcmp(param, "2t") == 0);
    std::printf("All assertions passed.\n");

    tgm_metadata_free(meta);
    tgm_free(msg_buf);
    return 0;
}
