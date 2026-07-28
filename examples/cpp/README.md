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
| `10_doctor.cpp` | Environment diagnostics via `doctor()` (build / features / self-test JSON report) |
| `13_validate.cpp` | Structural, integrity, and fidelity validation at four levels |
| `16_multi_threaded_pipeline.cpp` | Caller-controlled `threads=N` encode/decode with determinism invariants |
| `17_masks_scan_validate.cpp` | Mask-aware `decode_with_masks()` (`has_masks()`/`mask()`), whole-buffer `validate_buffer()`, inline hashes, and `scan_file()` |
| `19_async_decode_remote.cpp` | Async remote decode over an object store / `file://` (coroutine frontend) |
| `20_async_producer.cpp` | Async streaming producer: stream forecast steps into a `.tgm` (coroutine frontend) |
| `21_async_consumer.cpp` | Async streaming consumer: walk messages with `async_for_each` (coroutine frontend) |
| `22_async_callback.cpp` | Async read via the callback frontend (C++17): open, count, decode with completion handlers |
| `23_async_stdfuture.cpp` | Async read via the `std::future` frontend (C++17) |
| `24_async_cancellation.cpp` | Cancellation tokens and timeouts (callback frontend) |
| `25_async_decode_object_range.cpp` | Async single-object + partial-range decode, `path()`, and the pull-model task handle (`ready()`/`cancel()`/`join()`) |
| `26_frame_walker.cpp` | Lazy `frames()` walk + `read_message_header()` envelope, borrowed frame payloads, `encode_options::aggregate_hash` / `codec_backend`, typed `dtype_enum()`, and `is_remote_url()` |

### Async examples (19–25)

These require the async surface (`TENSOGRAM_ASYNC=ON`, the default). The
callback / `std::future` examples (22–24) compile on C++17. The
coroutine examples (19–21) are built as separate C++20 targets — disable
with `-DTENSOGRAM_ASYNC_CORO_EXAMPLES=OFF` if your compiler lacks C++20
coroutines. Example 19 additionally needs the object-store backend:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DTENSOGRAM_ASYNC_REMOTE=ON
cmake --build build -j
./build/bin/19_async_decode_remote
```

See the [C++ Async API](../../docs/src/guide/cpp-async.md) and
[C++ Async Streaming](../../docs/src/guide/cpp-streaming-async.md) guides
for the API reference and the producer/consumer recipe.

### Synchronous remote sources

`tensogram::is_remote_url()` and `tensogram::file::open_remote()` link in
every build, but they need the C API's `remote` Cargo feature to actually
reach an object store. Without it `is_remote_url()` answers `false` for
every URL (this build genuinely cannot open one) and `open_remote()`
throws `tensogram::remote_error` explaining how to enable it:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DTENSOGRAM_REMOTE=ON
cmake --build build -j
```

`open_remote()` returns an ordinary `tensogram::file`, so `message_count()`,
`read_message()`, `decode_message()` and `file_iterator` work unchanged
against `s3://`, `gs://`, `az://`, `http(s)://` and `file://` sources. It is
independent of the async options — no async runtime is pulled in.

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
obj.dtype_enum();       // tensogram::dtype::float32 — switchable
obj.byte_order_enum();  // tensogram::byte_order::little
obj.shape();            // std::vector<uint64_t>
obj.data_as<float>();   // typed pointer to payload

// Message envelope + lazy frame walk (payloads borrow `buf`)
auto header = tensogram::read_message_header(buf, len);
header.total_length();  // 0 for a streaming message never back-filled
for (const auto& f : tensogram::frames(buf, len)) {
    f.type();           // tensogram::frame_type::ntensor, ...
    f.offset();         // where the frame starts in `buf`
    f.payload();        // std::string_view INTO `buf`
    f.has_hash();       // per-frame HASH_PRESENT flag
}

// Remote sources (needs the C API's `remote` feature)
tensogram::is_remote_url("s3://bucket/key.tgm");
auto remote = tensogram::file::open_remote("s3://bucket/key.tgm",
                                           {{"aws_region", "eu-west-1"}});

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

// Encode options (every field defaults to the library default)
tensogram::encode_options opts;
opts.hash_algo = "xxh3";
opts.aggregate_hash = tensogram::aggregate_hash_policy::both;   // buffered only
opts.codec_backend = tensogram::compression_backend::pure;      // szip / zstd

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
