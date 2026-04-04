# C++ API

Tensogram provides a header-only C++17 wrapper at `include/tensogram.hpp`. It delegates all work to the C FFI and adds RAII handle management, typed exceptions, and idiomatic C++ patterns.

## Requirements

- C++17 compiler (GCC 7+, Clang 5+, MSVC 19.14+)
- Rust static library built via `cargo build --release`
- CMake 3.16+ (recommended)

## Build

```bash
cargo build --release
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Quick Start

```cpp
#include <tensogram.hpp>

// Encode
std::string meta_json = R"({"version": 2, "descriptors": [...]})";
std::vector<float> data(100 * 200, 0.0f);
auto encoded = tensogram::encode(
    meta_json,
    {{reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(float)}});

// Decode
auto msg = tensogram::decode(encoded.data(), encoded.size());
auto obj = msg.object(0);
const float* values = obj.data_as<float>();
```

## RAII Classes

| Class | Wraps | Cleanup |
|-------|-------|---------|
| `message` | `tgm_message_t` | `tgm_message_free` |
| `metadata` | `tgm_metadata_t` | `tgm_metadata_free` |
| `file` | `tgm_file_t` | `tgm_file_close` |
| `buffer_iterator` | `tgm_buffer_iter_t` | `tgm_buffer_iter_free` |
| `file_iterator` | `tgm_file_iter_t` | `tgm_file_iter_free` |
| `object_iterator` | `tgm_object_iter_t` | `tgm_object_iter_free` |
| `streaming_encoder` | `tgm_streaming_encoder_t` | `tgm_streaming_encoder_free` |

All classes are move-only (copy deleted). Handles are released automatically when the object goes out of scope.

## Error Handling

C error codes are mapped to a typed exception hierarchy:

```cpp
try {
    auto msg = tensogram::decode(buf, len);
} catch (const tensogram::framing_error& e) {
    // Invalid message framing
} catch (const tensogram::hash_mismatch_error& e) {
    // Payload integrity check failed
} catch (const tensogram::error& e) {
    // Any Tensogram error (base class)
    std::cerr << e.what() << " (code=" << e.code() << ")\n";
}
```

## Iterators

See [Iterators](iterators.md#c-api) for buffer, file, and object iterator usage.

## Examples

See `examples/cpp/` for complete working examples covering encode/decode, metadata, file API, simple packing, and iterators.
