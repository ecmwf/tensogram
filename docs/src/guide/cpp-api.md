# C++ API

Tensogram provides a header-only C++17 wrapper at `cpp/include/tensogram.hpp`. It delegates all work to the C FFI and adds RAII handle management, typed exceptions, and idiomatic C++ patterns.

> The C ABI underneath this wrapper is documented in [C API](c-api.md). The build flow on this page is the **in-tree** wrapper build: the bundled CMake reads the cbindgen-generated header from `rust/tensogram-ffi/` and links against `cargo build`'s `libtensogram_ffi.{a,so}`. The C API page also covers the **distribution** paths (pre-built tarballs, `cargo cinstall`) used when shipping the C library to consumers; the SONAME / versioning policy described there applies to those distributed binaries. Building the C++ wrapper against a cargo-c-installed `libtensogram` (rather than the in-tree static library) is not currently wired into `cpp/CMakeLists.txt`.

## Requirements

- C++17 compiler (GCC 7+, Clang 5+, MSVC 19.14+)
- Rust static library built via `cargo build --release`
- CMake 3.16+ (recommended)

## Build

```bash
cargo build --release
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Quick Start

```cpp
#include <tensogram.hpp>

// Encode
std::string meta_json = R"({"descriptors": [...]})";
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
| `frame_range` | `tgm_frame_iter_t` | `tgm_frame_iter_free` |
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

## Validation

Two free functions validate messages and files, returning JSON strings:

```cpp
// Validate a single message buffer (default level)
auto report = tensogram::validate(buf, len);

// Full validation with canonical CBOR check
auto full_report = tensogram::validate(buf, len, "full", /*check_canonical=*/true);

// Validate a .tgm file
auto file_report = tensogram::validate_file("data.tgm");
auto file_full   = tensogram::validate_file("data.tgm", "full");
```

Validation levels: `"quick"`, `"default"`, `"checksum"`, `"full"`.

The returned JSON contains `issues`, `object_count`, and `hash_verified` for single messages, or `file_issues` and `messages` for files. Parse with your preferred JSON library.

An invalid level string or a missing file throws `tensogram::invalid_arg_error` or `tensogram::io_error` respectively. Validation issues (corrupted data, hash mismatches) are reported in the JSON — they do not throw.

## Iterators

See [Iterators](iterators.md#c-api) for buffer, file, and object iterator usage.

## Frame walker and message header

`tensogram::frames()` returns a lazy, move-only `frame_range` over the frames of **one** message, usable in a range-for; `tensogram::read_message_header()` decodes that message's 24-byte envelope into a `message_header` value:

```cpp
const auto h = tensogram::read_message_header(msg.data(), msg.size());
if (h.has_header_index()) { /* random-access layout */ }

for (const auto& f : tensogram::frames(msg.data(), msg.size())) {
    if (f.is_data_object()) {                   // frame_type::ntensor
        std::string_view content = f.payload(); // borrows `msg`
        use(f.offset(), f.length(), f.has_hash(), content);
    }
}
```

`frame::payload()` / `payload_data()` view the caller's buffer — nothing to free, and the views stay valid after the range is destroyed, for as long as the buffer lives. A clean end simply terminates the loop; a truncated or inconsistent frame chain throws `tensogram::framing_error` from the increment that discovers it, after the intact frames have been yielded. The full cross-language contract — offsets, payload boundaries, and why a streaming message's header flags are advisory — is in [Frame Introspection](frame-introspection.md).

## Synchronous remote reads

`tensogram::file::open_remote()` opens an S3 / GCS / Azure / HTTP `.tgm` through the ordinary blocking API. The result is a plain `file`, so `message_count()`, `read_message()`, `decode_message()`, `file_iterator`, … all work unchanged:

```cpp
if (tensogram::is_remote_url(source)) {
    auto f = tensogram::file::open_remote(
        source,
        {{"aws_region", "eu-west-1"}},          // storage options
        tensogram::remote_scan_options{});      // bidirectional by default
    auto msg = f.decode_message(0);
} else {
    auto f = tensogram::file::open(source);
}
```

Requires the FFI built with `--features=remote` (`cmake -S cpp -B build -DTENSOGRAM_REMOTE=ON`; the default is **OFF**, and the option is independent of `TENSOGRAM_ASYNC` / `TENSOGRAM_ASYNC_REMOTE`). The symbols always link: in a build without the feature `is_remote_url` returns `false` for every input and `open_remote` throws `tensogram::remote_error` with a message naming the missing feature. See the [C API notes on the `remote` feature](c-api.md#synchronous-remote-access-opt-in-remote-feature) — in particular, it is **not** in the published C-API tarballs.

## Typed enums and encode options

`enum class dtype`, `byte_order`, `aggregate_hash_policy`, and `compression_backend` mirror the C enums, and each one's default is the library default. `decoded_object` gained the `switch`-able companions of the string getters (which are unchanged):

```cpp
auto obj = msg.object(0);
switch (obj.dtype_enum()) {                 // vs obj.dtype_string()
    case tensogram::dtype::float32: /* … */ break;
    default: break;
}
if (obj.byte_order_enum() == tensogram::byte_order::big) { /* … */ }
```

Both throw `tensogram::invalid_arg_error` for an out-of-range object index — the C accessors can only return their zero variant there, so the wrapper bounds-checks through the paired string getter first.

`encode_options` gained two fields:

```cpp
tensogram::encode_options opts;
opts.aggregate_hash = tensogram::aggregate_hash_policy::both;
opts.codec_backend  = tensogram::compression_backend::pure;
auto bytes = tensogram::encode(meta_json, objects, opts);
```

- `aggregate_hash` — where the aggregate hash frame goes. Ignored when hashing is off; `::header` and `::both` are buffered-mode only and make `streaming_encoder` throw `tensogram::encoding_error`.
- `codec_backend` — which szip / zstd implementation to prefer. Spelled `codec_backend` because a member named `compression_backend` would shadow the enum type inside the struct.

There are deliberately **no** `*_with_encode_options` overloads: they would be signature-identical to the existing ones. Instead the existing entry points — `encode()`, `file::append()`, `streaming_encoder` — pick the narrowest C function that can carry what you actually set, escalating to `*_with_encode_options` only when one of these two fields is non-default. A caller who sets neither keeps the exact previous call path (and the exact previous bytes). The one exception the C ABI imposes is `encode_pre_encoded()`, which has no full-option entry point and therefore **rejects** both fields rather than silently ignoring them.

## Examples

See `examples/cpp/` for complete working examples covering encode/decode, metadata, file API, simple packing, and iterators. `examples/cpp/26_frame_walker.cpp` covers the frame walker, the message header, the typed enums, and the new `encode_options` fields.
