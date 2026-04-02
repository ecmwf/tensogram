# C++ Examples

> **Status:** The C FFI (`tensogram-ffi`) is not yet implemented.
> These examples show the **intended API** once the FFI bindings are complete.
> They document the design contract so implementors have a clear target.

The C API uses opaque handles and typed getters — a pattern that lets C++
callers use it directly via `extern "C"` declarations without a wrapper layer.

## Planned Header

```c
// tensogram.h (planned)
#pragma once
#include <stdint.h>
#include <stddef.h>

// Opaque handles
typedef struct tgm_message_t   tgm_message_t;
typedef struct tgm_metadata_t  tgm_metadata_t;
typedef struct tgm_file_t      tgm_file_t;

// Error codes
typedef enum {
    TGM_OK              = 0,
    TGM_ERR_FRAMING     = 1,
    TGM_ERR_METADATA    = 2,
    TGM_ERR_ENCODING    = 3,
    TGM_ERR_HASH        = 4,
    TGM_ERR_OBJECT      = 5,
    TGM_ERR_IO          = 6,
} tgm_error_t;

// Encode
tgm_error_t tgm_encode(
    const char    *cbor_metadata_json,  // JSON → CBOR is done by the library
    const uint8_t *const *data_ptrs,
    const size_t  *data_lens,
    size_t         num_objects,
    uint8_t      **out_buf,             // caller must free with tgm_free()
    size_t        *out_len
);

// Decode
tgm_error_t tgm_decode(
    const uint8_t *buf,
    size_t         buf_len,
    tgm_message_t **out_message
);

tgm_error_t tgm_decode_metadata(
    const uint8_t  *buf,
    size_t          buf_len,
    tgm_metadata_t **out_metadata
);

// Metadata accessors
uint64_t    tgm_metadata_version(const tgm_metadata_t *m);
size_t      tgm_metadata_num_objects(const tgm_metadata_t *m);
const char *tgm_metadata_get_string(const tgm_metadata_t *m, const char *key);
int64_t     tgm_metadata_get_int(const tgm_metadata_t *m, const char *key, int64_t default_val);

// Object accessors (from decoded message)
size_t      tgm_message_num_objects(const tgm_message_t *msg);
size_t      tgm_object_ndim(const tgm_message_t *msg, size_t index);
const uint64_t *tgm_object_shape(const tgm_message_t *msg, size_t index);
const char *tgm_object_dtype(const tgm_message_t *msg, size_t index);
const uint8_t *tgm_object_data(const tgm_message_t *msg, size_t index, size_t *out_len);

// File API
tgm_error_t tgm_file_open(const char *path, tgm_file_t **out_file);
tgm_error_t tgm_file_create(const char *path, tgm_file_t **out_file);
tgm_error_t tgm_file_message_count(tgm_file_t *file, size_t *out_count);
tgm_error_t tgm_file_decode_message(
    tgm_file_t     *file,
    size_t          index,
    tgm_message_t **out_message
);
void tgm_file_close(tgm_file_t *file);

// Memory management
void tgm_message_free(tgm_message_t *msg);
void tgm_metadata_free(tgm_metadata_t *meta);
void tgm_free(void *ptr);

// Error string
const char *tgm_error_string(tgm_error_t err);
```

## Build (planned)

```cmake
find_package(tensogram REQUIRED)
target_link_libraries(my_app PRIVATE tensogram::tensogram)
```

Or manually with the `.a` / `.so` from `cargo build --release -p tensogram-ffi`.
