/// Example 05 — Iterator APIs (C++)
///
/// Shows three iterator patterns:
///   tgm_buffer_iter_*  — iterate over messages in a byte buffer
///   tgm_file_iter_*    — iterate over messages in a file (seek-based)
///   tgm_object_iter_*  — iterate over objects within a message
///
/// All iterators follow the create → next → free pattern.
/// tgm_*_iter_next returns TGM_OK while items remain, TGM_END_OF_ITER when done.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "tensogram.h"

static void check(tgm_error_t err, const char *ctx) {
    if (err != TGM_OK) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s: %s", ctx, tgm_error_string(err));
        throw std::runtime_error(buf);
    }
}

// Encode a small 4×4 float32 message for demonstration.
static tgm_bytes_t encode_demo_message(const char *param) {
    char json[512];
    std::snprintf(json, sizeof(json), R"({
        "version": 1,
        "objects": [{
            "type": "ntensor", "ndim": 2,
            "shape": [4, 4], "strides": [4, 1],
            "dtype": "float32"
        }],
        "payload": [{
            "byte_order": "little",
            "encoding": "none", "filter": "none", "compression": "none"
        }],
        "mars": {"param": "%s"}
    })", param);

    float data[16] = {};
    const uint8_t *ptrs[] = { reinterpret_cast<const uint8_t*>(data) };
    size_t lens[] = { sizeof(data) };

    tgm_bytes_t out;
    check(tgm_encode(json, ptrs, lens, 1, nullptr, &out), "encode");
    return out;
}

int main() {
    // ── 1. Buffer iteration ─────────────────────────────────────────────────
    //
    // Encode 3 messages, concatenate, iterate with tgm_buffer_iter.
    printf("=== Buffer iterator ===\n");

    std::vector<tgm_bytes_t> encoded;
    const char *params[] = { "2t", "10u", "msl" };
    std::vector<uint8_t> buf;

    for (auto p : params) {
        auto msg = encode_demo_message(p);
        buf.insert(buf.end(), msg.data, msg.data + msg.len);
        encoded.push_back(msg);
    }

    tgm_buffer_iter_t *iter = nullptr;
    check(tgm_buffer_iter_create(buf.data(), buf.size(), &iter), "buffer_iter_create");
    printf("  %zu messages found\n", tgm_buffer_iter_count(iter));

    const uint8_t *msg_ptr;
    size_t msg_len;
    int idx = 0;
    while (tgm_buffer_iter_next(iter, &msg_ptr, &msg_len) == TGM_OK) {
        printf("  [%d] %zu bytes\n", idx++, msg_len);
    }
    tgm_buffer_iter_free(iter);

    // ── 2. Object iteration ─────────────────────────────────────────────────
    //
    // Iterate over objects in the first message.
    printf("\n=== Object iterator ===\n");

    tgm_object_iter_t *obj_iter = nullptr;
    check(tgm_object_iter_create(encoded[0].data, encoded[0].len, 0, &obj_iter),
          "object_iter_create");

    tgm_message_t *obj_msg = nullptr;
    idx = 0;
    while (tgm_object_iter_next(obj_iter, &obj_msg) == TGM_OK) {
        uint64_t ndim = tgm_object_ndim(obj_msg, 0);
        const uint64_t *shape = tgm_object_shape(obj_msg, 0);
        printf("  object[%d] ndim=%llu shape=[%llu, %llu]\n",
               idx++, (unsigned long long)ndim,
               (unsigned long long)shape[0],
               (unsigned long long)shape[1]);
        tgm_message_free(obj_msg);
    }
    tgm_object_iter_free(obj_iter);

    // Clean up encoded messages
    for (auto &m : encoded) {
        tgm_bytes_free(m);
    }

    printf("\nIterator example complete.\n");
    return 0;
}
