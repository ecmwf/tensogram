/// Example 04 — File API (C++)
///
/// Shows create, append, open, message_count, decode_message.
/// All functions that need the message list trigger a lazy scan on first call.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "tensogram.h"

static void check(tgm_error_t err, const char *ctx) {
    if (err != TGM_OK) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s: %s", ctx, tgm_error_string(err));
        throw std::runtime_error(buf);
    }
}

static void write_field(tgm_file_t *file, const char *param, int step) {
    char json[512];
    std::snprintf(json, sizeof(json), R"({
        "version": 1,
        "objects": [{
            "type": "ntensor", "ndim": 2,
            "shape": [10, 20], "strides": [20, 1],
            "dtype": "float32",
            "mars": {"param": "%s"}
        }],
        "payload": [{"byte_order":"big","encoding":"none","filter":"none","compression":"none"}],
        "mars": {"date":"20260401","step":%d,"type":"fc"}
    })", param, step);

    float data[10 * 20] = {};  // zeros stand in for real data
    const uint8_t *ptrs[] = { reinterpret_cast<const uint8_t*>(data) };
    size_t         lens[] = { sizeof(data) };

    // Encode the message
    uint8_t *msg_buf = nullptr;
    size_t   msg_len = 0;
    check(tgm_encode(json, ptrs, lens, 1, &msg_buf, &msg_len), "encode");

    // Append to file (library accepts raw bytes)
    check(tgm_file_append_raw(file, msg_buf, msg_len), "append");
    tgm_free(msg_buf);
}

int main() {
    const char *path = "/tmp/tensogram_example.tgm";

    // ── 1. Create and write ───────────────────────────────────────────────────
    {
        tgm_file_t *file = nullptr;
        check(tgm_file_create(path, &file), "create");

        const char *params[] = {"2t", "10u", "10v", "msl"};
        const int   steps[]  = {0, 6, 12};

        for (int step : steps)
            for (const char *param : params)
                write_field(file, param, step);

        size_t count = 0;
        check(tgm_file_message_count(file, &count), "count");
        std::printf("Written %zu messages\n", count);

        tgm_file_close(file);
    }

    // ── 2. Open and read ──────────────────────────────────────────────────────
    {
        tgm_file_t *file = nullptr;
        check(tgm_file_open(path, &file), "open");

        // Lazy scan triggers on first call that needs the index
        size_t count = 0;
        check(tgm_file_message_count(file, &count), "count");
        std::printf("Opened: %zu messages\n", count);
        assert(count == 12);

        // Random access by index
        for (size_t i : {0, 5, 11}) {
            tgm_message_t *msg = nullptr;
            check(tgm_file_decode_message(file, i, &msg), "decode_message");

            tgm_metadata_t *meta = tgm_message_metadata(msg);
            const char *param = tgm_metadata_get_string(meta, "objects[0].mars.param");
            int64_t     step  = tgm_metadata_get_int(meta, "mars.step", -1);

            size_t data_len = 0;
            const uint8_t *data = tgm_object_data(msg, 0, &data_len);
            std::printf("  [%zu] param=%-5s step=%lld  data=%zu bytes\n",
                        i, param ? param : "?", (long long)step, data_len);

            tgm_message_free(msg);
        }

        tgm_file_close(file);
    }

    return 0;
}
