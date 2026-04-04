/// @file 05_iterators.cpp
/// @brief Example 05 — Iterator APIs using the C++ wrapper.
///
/// Shows three iterator patterns:
///   buffer_iterator  — iterate over messages in a byte buffer
///   file_iterator    — iterate over messages in a file
///   object_iterator  — iterate over objects within a message

#include <tensogram.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    // -- 1. Buffer iteration --
    std::printf("=== Buffer iterator ===\n");

    // Encode 3 messages with different element counts
    auto e1 = tensogram::encode(
        R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})",
        {{reinterpret_cast<const std::uint8_t*>("\0\0\x80\x3f\0\0\0\x40"), 8}});
    auto e2 = tensogram::encode(
        R"({"version":2,"descriptors":[{"type":"ndarray","ndim":1,"shape":[3],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})",
        {{reinterpret_cast<const std::uint8_t*>("\0\0\x80\x3f\0\0\0\x40\0\0\x40\x40"), 12}});

    // Concatenate into a single buffer
    std::vector<std::uint8_t> buf;
    buf.insert(buf.end(), e1.begin(), e1.end());
    buf.insert(buf.end(), e2.begin(), e2.end());

    tensogram::buffer_iterator iter(buf.data(), buf.size());
    std::printf("  %zu messages found\n", iter.count());

    const std::uint8_t* msg_ptr = nullptr;
    std::size_t msg_len = 0;
    int idx = 0;
    while (iter.next(msg_ptr, msg_len)) {
        auto msg = tensogram::decode(msg_ptr, msg_len);
        std::printf("  [%d] %zu bytes, %zu objects\n",
                    idx++, msg_len, msg.num_objects());
    }

    // -- 2. File iteration --
    std::printf("\n=== File iterator ===\n");

    const char* path = "/tmp/tensogram_iter_example.tgm";
    {
        auto f = tensogram::file::create(path);
        f.append_raw(e1);
        f.append_raw(e2);
    }

    {
        auto f = tensogram::file::open(path);
        tensogram::file_iterator fiter(f);
        std::vector<std::uint8_t> raw;
        idx = 0;
        while (fiter.next(raw)) {
            auto msg = tensogram::decode(raw.data(), raw.size());
            std::printf("  [%d] %zu bytes, %zu objects\n",
                        idx++, raw.size(), msg.num_objects());
        }
    }
    std::remove(path);

    // -- 3. Object iteration --
    std::printf("\n=== Object iterator ===\n");

    // Build a 2-object message
    std::vector<float> fvals = {1.0f, 2.0f};
    std::vector<double> dvals = {10.0, 20.0, 30.0};

    std::string multi_json = R"({"version":2,"descriptors":[)"
        R"({"type":"ndarray","ndim":1,"shape":[2],"strides":[4],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"},)"
        R"({"type":"ndarray","ndim":1,"shape":[3],"strides":[8],"dtype":"float64","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";

    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects = {
        {reinterpret_cast<const std::uint8_t*>(fvals.data()),
         fvals.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(dvals.data()),
         dvals.size() * sizeof(double)}
    };

    auto multi_encoded = tensogram::encode(multi_json, objects);

    // Need a message to receive each iteration result
    auto dummy = tensogram::decode(multi_encoded.data(), multi_encoded.size());
    tensogram::object_iterator oiter(multi_encoded.data(), multi_encoded.size());
    idx = 0;
    while (oiter.next(dummy)) {
        auto obj = dummy.object(0);
        std::printf("  object[%d] dtype=%s  data=%zu bytes\n",
                    idx++, obj.dtype_string().c_str(), obj.data_size());
    }

    // -- 4. Range-based for over message objects --
    std::printf("\n=== Range-based for ===\n");

    auto msg = tensogram::decode(multi_encoded.data(), multi_encoded.size());
    idx = 0;
    for (const auto& obj : msg) {
        std::printf("  object[%d] dtype=%s\n",
                    idx++, obj.dtype_string().c_str());
    }

    std::printf("\nIterator example complete.\n");
    return 0;
}
