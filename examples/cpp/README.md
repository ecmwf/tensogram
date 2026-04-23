# C++ Examples

These examples demonstrate the Tensogram C++ wrapper API (`tensogram.hpp`),
which provides RAII handle management, typed exceptions, and an idiomatic
C++17 interface over the C FFI.

## Examples

| File | Description |
|------|-------------|
| `01_encode_decode.cpp` | Basic encode/decode round-trip with a 2D float32 grid |
| `02_mars_metadata.cpp` | Per-object metadata (ECMWF MARS vocabulary example): encode, decode, dot-notation key lookup |
| `02b_generic_metadata.cpp` | Per-object metadata with a generic application namespace (non-MARS) |
| `03_simple_packing.cpp` | Lossy quantization via simple_packing with error measurement |
| `04_file_api.cpp` | File API: create, append, open, random-access decode |
| `05_iterators.cpp` | All iterator patterns: buffer, file, object, and range-based for |
| `06_hash_and_errors.cpp` | Inline hash slots, `validate()` for corruption detection, typed-exception hierarchy |
| `07_scan_buffer.cpp` | Scanning a multi-message buffer with `scan()`, skipping corrupt regions |
| `08_decode_variants.cpp` | `decode` / `decode_metadata` / `decode_object` / `decode_range` |
| `09_streaming_consumer.cpp` | Consumer-side streaming: scan a growing buffer, decode messages as they arrive |
| `13_validate.cpp` | Structural, integrity, and fidelity validation at four levels |
| `16_multi_threaded_pipeline.cpp` | Caller-controlled `threads=N` encode/decode with determinism invariants |

## API Overview

```cpp
#include <tensogram.hpp>

// Encode
auto encoded = tensogram::encode(metadata_json, objects, opts);

// Decode
auto msg = tensogram::decode(buf, len);
auto meta = tensogram::decode_metadata(buf, len);
auto msg2 = tensogram::decode_object(buf, len, index);

// Access decoded data
msg.version();          // wire-format version
msg.num_objects();      // number of data objects
auto obj = msg.object(0);
obj.dtype_string();     // "float32", "int64", etc.
obj.shape();            // std::vector<uint64_t>
obj.data_as<float>();   // typed pointer to payload

// Range-based for over objects
for (const auto& obj : msg) { ... }

// Metadata key lookup (dot-notation)
meta.get_string("mars.class");
meta.get_int("custom_int", default_val);
meta.get_float("custom_float", default_val);

// File API
auto f = tensogram::file::create(path);
f.append(json, objects);
f.append_raw(encoded_bytes);
auto msg = f.decode_message(index);

// Iterators
tensogram::buffer_iterator iter(buf, len);
tensogram::file_iterator iter(file);
tensogram::object_iterator iter(buf, len);

// Streaming encoder
tensogram::streaming_encoder enc(path, metadata_json, opts);
enc.write_object(descriptor_json, data, len);
enc.finish();

// Utilities
auto entries = tensogram::scan(buf, len);
auto hash = tensogram::compute_hash(data, len, "xxh3");
```

## Error Handling

All Tensogram functions throw typed exceptions on failure:

```cpp
try {
    auto msg = tensogram::decode(buf, len);
} catch (const tensogram::framing_error& e) {
    // Invalid message framing
} catch (const tensogram::hash_mismatch_error& e) {
    // Payload hash mismatch
} catch (const tensogram::error& e) {
    // Any Tensogram error (base class)
    e.code();   // C-level tgm_error code
    e.what();   // Human-readable message
}
```

## Build

### Prerequisites

Build the Rust static library first (required by both methods):

```bash
cargo build --release
```

### CMake (recommended)

The `cpp/CMakeLists.txt` already builds the Rust library and sets up
include paths. Add example targets there, or build from the project root:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Manual g++ (fallback)

```bash
# Linux:
g++ -std=c++17 \
    -I cpp/include \
    examples/cpp/01_encode_decode.cpp \
    -L target/release -ltensogram_ffi \
    -ldl -lpthread -lm \
    -o example_01

# macOS:
g++ -std=c++17 \
    -I cpp/include \
    examples/cpp/01_encode_decode.cpp \
    -L target/release -ltensogram_ffi \
    -framework CoreFoundation -framework Security -framework SystemConfiguration \
    -lc++ -lm \
    -o example_01
```
