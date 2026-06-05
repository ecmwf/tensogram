// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// Clean-room C FFI shim for the SZ3 header-only C++ library.
//
// This file exposes a flat C ABI consumed by the Rust `tensogram-sz3-sys` crate.
// The SZ3 C++ Config class is mapped to a plain-old-data struct (SZ3_Config_C)
// that matches the Rust `#[repr(C)] SZ3_Config` layout exactly.

#include <SZ3/api/sz.hpp>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// C-compatible configuration struct
// ---------------------------------------------------------------------------

struct SZ3_Config_C {
    uint8_t   N;
    size_t*   dims;
    size_t    num;
    uint8_t   errorBoundMode;
    double    absErrorBound;
    double    relErrorBound;
    double    l2normErrorBound;
    double    psnrErrorBound;
    uint8_t   cmprAlgo;
    bool      lorenzo;
    bool      lorenzo2;
    bool      regression;
    bool      openmp;
    uint8_t   dataType;
    int32_t   blockSize;
    int32_t   quantbinCnt;
};

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

static SZ3::Config config_c_to_cpp(SZ3_Config_C c) {
    // Build dimension vector from the C pointer
    std::vector<size_t> dims_vec(c.dims, c.dims + c.N);

    SZ3::Config cfg;
    cfg.setDims(dims_vec.begin(), dims_vec.end());

    cfg.errorBoundMode  = c.errorBoundMode;
    cfg.absErrorBound   = c.absErrorBound;
    cfg.relErrorBound   = c.relErrorBound;
    cfg.l2normErrorBound = c.l2normErrorBound;
    cfg.psnrErrorBound  = c.psnrErrorBound;
    cfg.cmprAlgo        = c.cmprAlgo;
    cfg.lorenzo         = c.lorenzo;
    cfg.lorenzo2        = c.lorenzo2;
    cfg.regression      = c.regression;
    cfg.openmp          = c.openmp;
    cfg.dataType        = c.dataType;
    cfg.blockSize       = c.blockSize;
    cfg.quantbinCnt     = c.quantbinCnt;

    return cfg;
}

static SZ3_Config_C config_cpp_to_c(const SZ3::Config& cfg) {
    SZ3_Config_C c{};

    c.N = static_cast<uint8_t>(cfg.N);

    // Heap-allocate dims — the Rust caller must free via sz3_dealloc_size_t.
    c.dims = new size_t[cfg.dims.size()];
    std::memcpy(c.dims, cfg.dims.data(), cfg.dims.size() * sizeof(size_t));

    c.num             = cfg.num;
    c.errorBoundMode  = cfg.errorBoundMode;
    c.absErrorBound   = cfg.absErrorBound;
    c.relErrorBound   = cfg.relErrorBound;
    c.l2normErrorBound = cfg.l2normErrorBound;
    c.psnrErrorBound  = cfg.psnrErrorBound;
    c.cmprAlgo        = cfg.cmprAlgo;
    c.lorenzo         = cfg.lorenzo;
    c.lorenzo2        = cfg.lorenzo2;
    c.regression      = cfg.regression;
    c.openmp          = cfg.openmp;
    c.dataType        = cfg.dataType;
    c.blockSize       = cfg.blockSize;
    c.quantbinCnt     = cfg.quantbinCnt;

    return c;
}

// ---------------------------------------------------------------------------
// Macro to generate the 3 per-type FFI functions
// ---------------------------------------------------------------------------

#define SZ3_FFI_IMPL(T, suffix)                                               \
                                                                              \
extern "C" size_t sz3_compress_size_bound_##suffix(SZ3_Config_C config) {     \
    auto cfg = config_c_to_cpp(config);                                       \
    using namespace SZ3;                                                      \
    return SZ_compress_size_bound<T>(cfg);                                    \
}                                                                             \
                                                                              \
extern "C" size_t sz3_compress_##suffix(                                      \
        SZ3_Config_C config,                                                  \
        const T* data,                                                        \
        char* compressed_data,                                                \
        size_t compressed_capacity) {                                         \
    auto cfg = config_c_to_cpp(config);                                       \
    return SZ_compress<T>(cfg, data, compressed_data, compressed_capacity);   \
}                                                                             \
                                                                              \
extern "C" void sz3_decompress_##suffix(                                      \
        const char* compressed_data,                                          \
        size_t compressed_len,                                                \
        T* decompressed_data) {                                               \
    using namespace SZ3;                                                      \
    Config cfg;                                                               \
    SZ_decompress<T>(cfg, compressed_data, compressed_len, decompressed_data);\
}

// Instantiate for all 10 supported types
SZ3_FFI_IMPL(float,    f32)
SZ3_FFI_IMPL(double,   f64)
SZ3_FFI_IMPL(uint8_t,  u8)
SZ3_FFI_IMPL(int8_t,   i8)
SZ3_FFI_IMPL(uint16_t, u16)
SZ3_FFI_IMPL(int16_t,  i16)
SZ3_FFI_IMPL(uint32_t, u32)
SZ3_FFI_IMPL(int32_t,  i32)
SZ3_FFI_IMPL(uint64_t, u64)
SZ3_FFI_IMPL(int64_t,  i64)

// ---------------------------------------------------------------------------
// Decompress-config: parse the SZ3 header + trailing config without
// decompressing the payload.
// ---------------------------------------------------------------------------

// Sentinel "invalid" config: N == 0 and dims == nullptr.  The Rust
// caller (`ParsedConfig::from_compressed`) treats N == 0 as a malformed
// stream and returns an `Err` instead of proceeding.
static SZ3_Config_C invalid_config() {
    SZ3_Config_C c{};
    c.N    = 0;
    c.dims = nullptr;
    return c;
}

extern "C" SZ3_Config_C sz3_decompress_config(const char* data, size_t len) {
    using namespace SZ3;

    // SECURITY (SEC-010): this parser reads an SZ3 header and a config
    // trailer from an attacker-controlled buffer.  SZ3's `read()` and
    // `Config::load()` do NOT bounds-check against the buffer length, so
    // a truncated or hostile stream causes out-of-bounds reads (ASan
    // SEGV), reachable from a `.tgm` with `compression=sz3`.  Validate
    // every access against `len` here.

    // Fixed 16-byte header: magic(4) + version(4) + compressed_size(8).
    constexpr size_t kHeaderSize = 4 + 4 + 8;
    if (data == nullptr || len < kHeaderSize) {
        return invalid_config();
    }

    auto base = reinterpret_cast<const unsigned char*>(data);
    auto pos  = base;

    Config cfg;
    read(cfg.sz3MagicNumber, pos);
    read(cfg.sz3DataVer, pos);

    uint64_t cmpDataSize = 0;
    read(cmpDataSize, pos);
    // `pos` is now `base + 16`.  The compressed payload of `cmpDataSize`
    // bytes follows, then the config trailer.  Reject any `cmpDataSize`
    // that would push the config offset at or past the end of the
    // buffer (no room for even a minimal trailer), or that overflows.
    const size_t consumed = static_cast<size_t>(pos - base);  // == 16
    if (cmpDataSize > len || consumed > len - cmpDataSize) {
        return invalid_config();
    }
    const size_t conf_off = consumed + static_cast<size_t>(cmpDataSize);
    if (conf_off >= len) {
        // No bytes remain for the config trailer.
        return invalid_config();
    }

    // SZ3's `Config::load` walks the trailer WITHOUT an explicit end
    // pointer, so a trailer that claims more bytes than remain would
    // read past `data + len`.  Since `Config::load` is upstream SZ3
    // (treated as a black box), contain the over-read by copying the
    // trailer into a generously zero-padded heap buffer: any read past
    // the real trailer lands in our zero padding instead of out of
    // bounds.  The padding (64 KiB) comfortably exceeds any legitimate
    // serialised SZ3 config.
    const size_t trailer_len = len - conf_off;
    constexpr size_t kPad     = 64 * 1024;
    std::vector<unsigned char> trailer;
    try {
        trailer.assign(trailer_len + kPad, 0);
    } catch (...) {
        return invalid_config();
    }
    std::memcpy(trailer.data(), base + conf_off, trailer_len);

    const unsigned char* confPos = trailer.data();
    try {
        cfg.load(confPos);
    } catch (...) {
        // A malformed trailer can make SZ3 throw; surface as invalid
        // rather than letting the exception cross the C ABI boundary.
        return invalid_config();
    }

    return config_cpp_to_c(cfg);
}

// ---------------------------------------------------------------------------
// Deallocator for the heap-allocated dims array
// ---------------------------------------------------------------------------

extern "C" void sz3_dealloc_size_t(size_t* ptr) {
    delete[] ptr;
}
