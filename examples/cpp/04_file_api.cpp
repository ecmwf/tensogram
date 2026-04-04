/// @file 04_file_api.cpp
/// @brief Example 04 — File API using the C++ wrapper.
///
/// Shows create, append, open, message_count, decode_message.

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

/// Build JSON for a simple 10x20 float32 message with MARS metadata.
static std::string make_json(const char* param, int step) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        R"({"version":2,"descriptors":[{"type":"ndarray","ndim":2,"shape":[10,20],"strides":[80,4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}],"mars":{"date":"20260401","step":"%d","type":"fc","param":"%s"}})",
        step, param);
    return buf;
}

int main() {
    const char* path = "/tmp/tensogram_example_cpp.tgm";

    // -- 1. Create and write --
    {
        auto f = tensogram::file::create(path);

        const char* params[] = {"2t", "10u", "10v", "msl"};
        const int steps[] = {0, 6, 12};

        for (int step : steps) {
            for (const char* param : params) {
                std::vector<float> data(10 * 20, 0.0f);
                auto json = make_json(param, step);
                std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
                    {reinterpret_cast<const std::uint8_t*>(data.data()),
                     data.size() * sizeof(float)}
                };
                f.append(json, objects);
            }
        }

        std::printf("Written %zu messages\n", f.message_count());
    }

    // -- 2. Open and read --
    {
        auto f = tensogram::file::open(path);
        const std::size_t count = f.message_count();
        std::printf("Opened: %zu messages\n", count);
        assert(count == 12);

        // Random access by index
        for (std::size_t i : {std::size_t{0}, std::size_t{5}, std::size_t{11}}) {
            auto msg = f.decode_message(i);
            auto meta = msg.get_metadata();
            auto obj = msg.object(0);

            std::printf("  [%zu] mars.param=%s  data=%zu bytes\n",
                        i,
                        meta.get_string("mars.param").c_str(),
                        obj.data_size());
        }
    }

    std::remove(path);
    return 0;
}
