// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF

/// @file tensogram.hpp
/// @brief Header-only C++17 wrapper for the Tensogram C API.
///
/// Each class holds an opaque C handle via std::unique_ptr with a custom
/// deleter, providing move-only RAII semantics.  C error codes are mapped
/// to a typed exception hierarchy rooted at tensogram::error.
///
/// @note Thread safety: individual handles are **not** thread-safe.
///       Concurrent access to the same handle from multiple threads
///       requires external synchronisation.  Different handles may be
///       used concurrently without synchronisation.
///
/// @note Lifetime: pointers returned by decoded_object accessors
///       (data(), data_as<T>(), shape(), strides(), etc.) borrow from the
///       parent message handle and are valid only until that handle is
///       destroyed or moved-from.

#ifndef TENSOGRAM_HPP
#define TENSOGRAM_HPP

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" {
#include "tensogram.h"
}

namespace tensogram {

// ============================================================
// Error hierarchy
// ============================================================

/// Base exception for all Tensogram errors.
///
/// Carries the underlying C error code so callers can inspect the error
/// category programmatically.  Subclasses provide finer-grained catch
/// clauses (e.g. tensogram::framing_error).
class error : public std::runtime_error {
public:
    error(tgm_error code, const std::string& msg)
        : std::runtime_error(msg), code_(code) {}
    /// Return the C-level error code for this exception.
    [[nodiscard]] tgm_error code() const noexcept { return code_; }
private:
    tgm_error code_;
};

/// Thrown when message framing is invalid or corrupted.
class framing_error : public error {
public:
    framing_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown when CBOR metadata parsing fails.
class metadata_error : public error {
public:
    metadata_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown when data encoding/decoding fails (e.g. simple_packing).
class encoding_error : public error {
public:
    encoding_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown when compression/decompression fails.
class compression_error : public error {
public:
    compression_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown for data-object-level errors.
class object_error : public error {
public:
    object_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown for file I/O errors.
class io_error : public error {
public:
    io_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown when a payload hash does not match the expected value.
class hash_mismatch_error : public error {
public:
    hash_mismatch_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown when an invalid argument is passed to a Tensogram function.
class invalid_arg_error : public error {
public:
    invalid_arg_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown when a remote object-store operation fails
/// (S3, GCS, Azure, or HTTP(S)).
class remote_error : public error {
public:
    remote_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

// ============================================================
// detail — error checking helper
// ============================================================

namespace detail {

/// Check a C error code and throw the appropriate typed exception.
inline void check(tgm_error err) {
    if (err == TGM_ERROR_OK) return;
    const char* msg = tgm_last_error();
    std::string message = msg ? msg : tgm_error_string(err);
    switch (err) {
        case TGM_ERROR_FRAMING:       throw framing_error(err, message);
        case TGM_ERROR_METADATA:      throw metadata_error(err, message);
        case TGM_ERROR_ENCODING:      throw encoding_error(err, message);
        case TGM_ERROR_COMPRESSION:   throw compression_error(err, message);
        case TGM_ERROR_OBJECT:        throw object_error(err, message);
        case TGM_ERROR_IO:            throw io_error(err, message);
        case TGM_ERROR_HASH_MISMATCH: throw hash_mismatch_error(err, message);
        case TGM_ERROR_INVALID_ARG:   throw invalid_arg_error(err, message);
        case TGM_ERROR_REMOTE:        throw remote_error(err, message);
        case TGM_ERROR_END_OF_ITER:   throw error(err, message);
        default:                      throw error(err, message);
    }
}

/// Adapter that splits a vector of (pointer, length) pairs into parallel
/// pointer and length arrays for the C API scatter-gather calling convention.
///
/// The C functions `tgm_encode()` and `tgm_file_append()` accept separate
/// `const uint8_t**` and `size_t*` arrays.  This struct builds those arrays
/// from a single `std::vector<std::pair<const uint8_t*, size_t>>` so callers
/// do not have to maintain two parallel vectors manually.
///
/// @note The resulting `ptrs` and `lens` vectors borrow from the original
///       pairs — the input vector must remain valid while the arrays are used.
struct scatter_gather {
    std::vector<const std::uint8_t*> ptrs;
    std::vector<std::size_t> lens;

    explicit scatter_gather(
        const std::vector<std::pair<const std::uint8_t*, std::size_t>>& objects) {
        ptrs.reserve(objects.size());
        lens.reserve(objects.size());
        for (const auto& [ptr, len] : objects) {
            ptrs.push_back(ptr);
            lens.push_back(len);
        }
    }
};

} // namespace detail

// ============================================================
// Value types
// ============================================================

/// A (offset, length) pair describing one message's location in a buffer.
struct scan_entry {
    std::size_t offset;  ///< Byte offset of the message start.
    std::size_t length;  ///< Byte length of the message.
};

/// Options controlling message encoding.
struct encode_options {
    /// Hash algorithm name (e.g. "xxh3").  Empty string disables hashing.
    std::string hash_algo = "xxh3";
    /// Thread budget for the multi-threaded coding pipeline.
    ///
    /// - `0` (default): sequential execution (pre-0.13.0 behaviour); may
    ///   still be overridden by the `TENSOGRAM_THREADS` environment variable.
    /// - `1`: single worker thread.
    /// - `N ≥ 2`: scoped pool of `N` workers.  For the transparent pipeline
    ///   stages the encoded payload is byte-identical across thread counts;
    ///   for opaque codecs (blosc2, zstd with workers) the compressed bytes
    ///   may differ but round-trip losslessly.
    std::uint32_t threads = 0;
    /// Reject NaN values in float payloads before the encoding pipeline
    /// runs.  Default `false` (backwards-compatible).
    ///
    /// When `true`, raw input for float dtypes
    /// (`float16`/`bfloat16`/`float32`/`float64`/`complex64`/`complex128`)
    /// is scanned element by element.  On the first NaN, encode fails
    /// with `encoding_error`.  Integer and bitmask dtypes are skipped
    /// (zero cost).  The check runs **before** the pipeline so the
    /// guarantee is pipeline-independent.
    bool reject_nan = false;
    /// Reject `+Inf` / `-Inf` values in float payloads before the
    /// encoding pipeline runs.  Default `false`.
    ///
    /// See `plans/RESEARCH_NAN_HANDLING.md` §3.1 for the user-visible
    /// corner this closes: `simple_packing` accepts Inf input and
    /// silently produces numerically-useless params that decode to NaN
    /// everywhere.  Turning this flag on catches the problem up front.
    bool reject_inf = false;
};

/// Options controlling message decoding.
struct decode_options {
    /// When true, payload hashes are verified during decode.
    bool verify_hash = false;
    /// When true (the default), decoded payloads are converted to the
    /// caller's native byte order.  Set to false to receive bytes in the
    /// message's declared wire byte order.
    bool native_byte_order = true;
    /// Thread budget for the multi-threaded decoding pipeline.
    /// See `encode_options::threads` for semantics.
    std::uint32_t threads = 0;
};

// ============================================================
// Forward declarations
// ============================================================

class message;
class metadata;
class file;
class buffer_iterator;
class file_iterator;
class object_iterator;
class streaming_encoder;

// Free functions (forward-declared for friend access)
[[nodiscard]] inline message decode(const std::uint8_t* buf, std::size_t len,
                      const decode_options& opts = {});
[[nodiscard]] inline message decode_object(const std::uint8_t* buf, std::size_t len,
                             std::size_t index, const decode_options& opts = {});
[[nodiscard]] inline metadata decode_metadata(const std::uint8_t* buf, std::size_t len);

// ============================================================
// decoded_object — non-owning view into a message's object
// ============================================================

/// Non-owning view of a single decoded data object within a message.
///
/// Instances are lightweight (two pointers) and are returned by value
/// from message::object() and the message iterator.
///
/// @warning All pointers returned by accessors (data(), data_as<T>(),
///          shape(), strides()) borrow from the parent message and
///          become dangling once the message is destroyed or moved-from.
class decoded_object {
public:
    /// Number of tensor dimensions.
    [[nodiscard]] std::uint64_t ndim() const { return tgm_object_ndim(msg_, index_); }

    /// Tensor shape as a vector of dimension sizes.
    [[nodiscard]] std::vector<std::uint64_t> shape() const {
        const auto n = ndim();
        const auto* p = tgm_object_shape(msg_, index_);
        return p ? std::vector<std::uint64_t>(p, p + n)
                 : std::vector<std::uint64_t>{};
    }

    /// Tensor strides (in bytes) as a vector.
    [[nodiscard]] std::vector<std::uint64_t> strides() const {
        const auto n = ndim();
        const auto* p = tgm_object_strides(msg_, index_);
        return p ? std::vector<std::uint64_t>(p, p + n)
                 : std::vector<std::uint64_t>{};
    }

    /// Dtype string (e.g. "float32", "int64").
    [[nodiscard]] std::string dtype_string() const {
        const char* s = tgm_object_dtype(msg_, index_);
        return s ? s : "";
    }

    /// Object type string (e.g. "ndarray").
    [[nodiscard]] std::string object_type() const {
        const char* s = tgm_object_type(msg_, index_);
        return s ? s : "";
    }

    /// Byte order string ("big" or "little").
    [[nodiscard]] std::string byte_order_string() const {
        const char* s = tgm_object_byte_order(msg_, index_);
        return s ? s : "";
    }

    /// Encoding pipeline string (e.g. "none", "simple_packing").
    [[nodiscard]] std::string encoding() const {
        const char* s = tgm_payload_encoding(msg_, index_);
        return s ? s : "";
    }

    /// Pre-processing filter string (e.g. "none", "shuffle").
    [[nodiscard]] std::string filter() const {
        const char* s = tgm_object_filter(msg_, index_);
        return s ? s : "";
    }

    /// Compression string (e.g. "none", "zstd", "szip").
    [[nodiscard]] std::string compression() const {
        const char* s = tgm_object_compression(msg_, index_);
        return s ? s : "";
    }

    /// True if this object carries a payload integrity hash.
    [[nodiscard]] bool has_hash() const { return tgm_payload_has_hash(msg_, index_) != 0; }

    /// Hash algorithm name (e.g. "xxh3"), or "" if no hash.
    [[nodiscard]] std::string hash_type() const {
        const char* s = tgm_object_hash_type(msg_, index_);
        return s ? s : "";
    }

    /// Hex-encoded hash value, or "" if no hash.
    [[nodiscard]] std::string hash_value() const {
        const char* s = tgm_object_hash_value(msg_, index_);
        return s ? s : "";
    }

    /// Raw pointer to decoded payload bytes.
    /// Valid until the parent message is destroyed or moved-from.
    [[nodiscard]] const std::uint8_t* data() const {
        std::size_t len = 0;
        return tgm_object_data(msg_, index_, &len);
    }

    /// Size of the decoded payload in bytes.
    [[nodiscard]] std::size_t data_size() const {
        std::size_t len = 0;
        tgm_object_data(msg_, index_, &len);
        return len;
    }

    /// Reinterpret the raw payload as a typed array.
    ///
    /// @tparam T  Element type (e.g. float, double, std::int32_t).
    /// @return Pointer to the first element, valid until the parent
    ///         message is destroyed or moved-from.
    template <typename T>
    [[nodiscard]] const T* data_as() const {
        return reinterpret_cast<const T*>(data());
    }

    /// Number of elements of type T that fit in the payload.
    template <typename T>
    [[nodiscard]] std::size_t element_count() const {
        return data_size() / sizeof(T);
    }

private:
    friend class message;
    decoded_object(const tgm_message_t* msg, std::size_t index)
        : msg_(msg), index_(index) {}
    const tgm_message_t* msg_;
    std::size_t index_;
};

// ============================================================
// message — RAII wrapper for tgm_message_t
// ============================================================

/// RAII wrapper for a decoded Tensogram message.
///
/// Owns the underlying C handle; move-only (copy is deleted).
/// Provides access to global metadata and per-object decoded payloads.
class message {
public:
    message(message&&) noexcept = default;
    message& operator=(message&&) noexcept = default;
    ~message() = default;

    message(const message&) = delete;
    message& operator=(const message&) = delete;

    /// Wire-format version of this message.
    [[nodiscard]] std::uint64_t version() const { return tgm_message_version(handle_.get()); }

    /// Number of data objects in this message.
    [[nodiscard]] std::size_t num_objects() const { return tgm_message_num_objects(handle_.get()); }

    /// Access a single decoded object by index.
    ///
    /// @param index  Zero-based object index (must be < num_objects()).
    /// @return A non-owning view; valid until this message is destroyed.
    [[nodiscard]] decoded_object object(std::size_t index) const {
        return decoded_object(handle_.get(), index);
    }

    /// Extract an independent metadata handle from this message.
    [[nodiscard]] metadata get_metadata() const; // defined after metadata class

    // -- range-based for support over decoded_objects ---------

    /// Input iterator over the decoded objects in a message.
    class iterator {
    public:
        using value_type        = decoded_object;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void;
        using reference         = decoded_object;
        using iterator_category = std::input_iterator_tag;

        [[nodiscard]] decoded_object operator*() const { return decoded_object(msg_, index_); }
        iterator& operator++() { ++index_; return *this; }
        iterator operator++(int) { auto tmp = *this; ++index_; return tmp; }
        [[nodiscard]] bool operator==(const iterator& other) const { return index_ == other.index_; }
        [[nodiscard]] bool operator!=(const iterator& other) const { return index_ != other.index_; }

    private:
        friend class message;
        iterator(const tgm_message_t* msg, std::size_t index)
            : msg_(msg), index_(index) {}
        const tgm_message_t* msg_;
        std::size_t index_;
    };

    [[nodiscard]] iterator begin() const { return iterator(handle_.get(), 0); }
    [[nodiscard]] iterator end()   const { return iterator(handle_.get(), num_objects()); }

private:
    // Only construction sites that receive a raw C handle may build a message.
    friend message tensogram::decode(const std::uint8_t*, std::size_t,
                                     const decode_options&);
    friend message tensogram::decode_object(const std::uint8_t*, std::size_t,
                                            std::size_t, const decode_options&);
    friend class file;
    friend class object_iterator;

    struct deleter {
        void operator()(tgm_message_t* p) const noexcept { tgm_message_free(p); }
    };

    explicit message(tgm_message_t* raw) : handle_(raw) {}
    std::unique_ptr<tgm_message_t, deleter> handle_;
};

// ============================================================
// metadata — RAII wrapper for tgm_metadata_t
// ============================================================

/// RAII wrapper for message-level (global) metadata.
///
/// Provides dot-notation key lookup for string, integer, and float values.
/// The lookup searches per-object `base` entries first (skipping internal
/// `_reserved_` keys), then the message-level `extra` section.
/// Move-only (copy is deleted).
class metadata {
public:
    metadata(metadata&&) noexcept = default;
    metadata& operator=(metadata&&) noexcept = default;
    ~metadata() = default;

    metadata(const metadata&) = delete;
    metadata& operator=(const metadata&) = delete;

    /// Wire-format version of the originating message.
    [[nodiscard]] std::uint64_t version() const { return tgm_metadata_version(handle_.get()); }

    /// Number of objects described in this metadata.
    ///
    /// Returns the length of the `base` array in the global metadata.
    /// If the encoding JSON did not include a `"base"` key, this returns 0.
    /// Use message::num_objects() for a count that always reflects the
    /// number of decoded data objects in the message.
    [[nodiscard]] std::size_t num_objects() const { return tgm_metadata_num_objects(handle_.get()); }

    /// Look up a string value by dot-notation key (e.g. "mars.class").
    ///
    /// Searches `base` entries first, then `extra`.
    /// @return The value as a string, or "" if not found.
    [[nodiscard]] std::string get_string(std::string_view key) const {
        const std::string k(key);
        const char* s = tgm_metadata_get_string(handle_.get(), k.c_str());
        return s ? s : "";
    }

    /// Look up an integer value by dot-notation key.
    ///
    /// Searches `base` entries first, then `extra`.
    /// @return The value, or @p default_val if not found.
    [[nodiscard]] std::int64_t get_int(std::string_view key, std::int64_t default_val = 0) const {
        const std::string k(key);
        return tgm_metadata_get_int(handle_.get(), k.c_str(), default_val);
    }

    /// Look up a floating-point value by dot-notation key.
    ///
    /// Searches `base` entries first, then `extra`.
    /// @return The value, or @p default_val if not found.
    [[nodiscard]] double get_float(std::string_view key, double default_val = 0.0) const {
        const std::string k(key);
        return tgm_metadata_get_float(handle_.get(), k.c_str(), default_val);
    }

private:
    friend metadata tensogram::decode_metadata(const std::uint8_t*, std::size_t);
    friend class message;

    struct deleter {
        void operator()(tgm_metadata_t* p) const noexcept { tgm_metadata_free(p); }
    };

    explicit metadata(tgm_metadata_t* raw) : handle_(raw) {}
    std::unique_ptr<tgm_metadata_t, deleter> handle_;
};

// -- deferred inline definition (needs metadata class complete) ----

inline metadata message::get_metadata() const {
    tgm_metadata_t* raw = nullptr;
    detail::check(tgm_message_metadata(handle_.get(), &raw));
    return metadata(raw);
}

// ============================================================
// file — RAII wrapper for tgm_file_t
// ============================================================

/// RAII wrapper for a Tensogram file (read or read-write).
///
/// Use the static factory methods open() and create() to construct.
/// Move-only (copy is deleted).
class file {
public:
    /// Open an existing Tensogram file for reading.
    ///
    /// @throws io_error if the file does not exist or cannot be opened.
    [[nodiscard]] static file open(const std::string& path) {
        tgm_file_t* raw = nullptr;
        detail::check(tgm_file_open(path.c_str(), &raw));
        return file(raw);
    }

    /// Create a new Tensogram file (truncates if it already exists).
    ///
    /// @throws io_error if the file cannot be created.
    [[nodiscard]] static file create(const std::string& path) {
        tgm_file_t* raw = nullptr;
        detail::check(tgm_file_create(path.c_str(), &raw));
        return file(raw);
    }

    file(file&&) noexcept = default;
    file& operator=(file&&) noexcept = default;
    ~file() = default;

    file(const file&) = delete;
    file& operator=(const file&) = delete;

    /// Number of messages in the file.
    ///
    /// @note This method is **non-const** because the first call triggers a
    ///       lazy scan of the file to build the internal message index.
    ///       Subsequent calls return the cached count without I/O.
    [[nodiscard]] std::size_t message_count() {
        std::size_t count = 0;
        detail::check(tgm_file_message_count(handle_.get(), &count));
        return count;
    }

    /// Decode the message at @p index into a message handle.
    [[nodiscard]] message decode_message(std::size_t index,
                           const decode_options& opts = {}) {
        tgm_message_t* raw = nullptr;
        detail::check(tgm_file_decode_message(
            handle_.get(), index, opts.verify_hash ? 1 : 0,
            opts.native_byte_order ? 1 : 0, opts.threads, &raw));
        return message(raw);
    }

    /// Read raw (undecoded) message bytes at @p index.
    [[nodiscard]] std::vector<std::uint8_t> read_message(std::size_t index) {
        tgm_bytes_t bytes{};
        detail::check(tgm_file_read_message(handle_.get(), index, &bytes));
        std::vector<std::uint8_t> result(bytes.data, bytes.data + bytes.len);
        tgm_bytes_free(bytes);
        return result;
    }

    /// Append pre-encoded raw message bytes to the file.
    void append_raw(const std::uint8_t* data, std::size_t len) {
        detail::check(tgm_file_append_raw(handle_.get(), data, len));
    }

    /// Append pre-encoded raw message bytes to the file.
    void append_raw(const std::vector<std::uint8_t>& data) {
        append_raw(data.data(), data.size());
    }

    /// Encode a message from JSON metadata and data slices, then append it.
    void append(const std::string& metadata_json,
                const std::vector<std::pair<const std::uint8_t*, std::size_t>>& objects,
                const encode_options& opts = {}) {
        detail::scatter_gather sg(objects);
        const char* hash = opts.hash_algo.empty() ? nullptr
                                                     : opts.hash_algo.c_str();
        detail::check(tgm_file_append(handle_.get(), metadata_json.c_str(),
                                       sg.ptrs.data(), sg.lens.data(),
                                       objects.size(), hash, opts.threads,
                                       opts.reject_nan, opts.reject_inf));
    }

    /// File path as a string.
    [[nodiscard]] std::string path() const {
        const char* p = tgm_file_path(handle_.get());
        return p ? p : "";
    }

    /// Expose the raw C handle for use by file_iterator.
    ///
    /// @warning The returned pointer is **non-owning**.  Do not call
    ///          `tgm_file_close()` on it or store it beyond the lifetime of
    ///          this `file` object.
    [[nodiscard]] tgm_file_t* raw() { return handle_.get(); }

private:
    struct deleter {
        void operator()(tgm_file_t* p) const noexcept { tgm_file_close(p); }
    };

    explicit file(tgm_file_t* raw) : handle_(raw) {}
    std::unique_ptr<tgm_file_t, deleter> handle_;
};

// ============================================================
// buffer_iterator — RAII wrapper for tgm_buffer_iter_t
// ============================================================

/// Iterate over concatenated Tensogram messages in a byte buffer.
///
/// Scans the buffer once at construction to locate message boundaries.
/// The caller's buffer must remain valid and unmodified for the lifetime
/// of this iterator.
class buffer_iterator {
public:
    /// Construct from a byte buffer.  Scans immediately.
    buffer_iterator(const std::uint8_t* buf, std::size_t len) {
        tgm_buffer_iter_t* raw = nullptr;
        detail::check(tgm_buffer_iter_create(buf, len, &raw));
        handle_.reset(raw);
    }

    buffer_iterator(buffer_iterator&&) noexcept = default;
    buffer_iterator& operator=(buffer_iterator&&) noexcept = default;
    ~buffer_iterator() = default;

    buffer_iterator(const buffer_iterator&) = delete;
    buffer_iterator& operator=(const buffer_iterator&) = delete;

    /// Total number of messages found during the initial scan.
    [[nodiscard]] std::size_t count() const { return tgm_buffer_iter_count(handle_.get()); }

    /// Advance to the next message slice.
    ///
    /// @param[out] out_buf  Pointer into the original buffer.
    /// @param[out] out_len  Length of the message slice.
    /// @return true if a message was returned, false when exhausted.
    bool next(const std::uint8_t*& out_buf, std::size_t& out_len) {
        const std::uint8_t* buf = nullptr;
        std::size_t len = 0;
        const tgm_error err = tgm_buffer_iter_next(handle_.get(), &buf, &len);
        if (err == TGM_ERROR_END_OF_ITER) return false;
        detail::check(err);
        out_buf = buf;
        out_len = len;
        return true;
    }

private:
    struct deleter {
        void operator()(tgm_buffer_iter_t* p) const noexcept { tgm_buffer_iter_free(p); }
    };
    std::unique_ptr<tgm_buffer_iter_t, deleter> handle_;
};

// ============================================================
// file_iterator — RAII wrapper for tgm_file_iter_t
// ============================================================

/// Iterate over messages in a Tensogram file.
///
/// Scans the file at construction to locate message boundaries,
/// then yields raw message bytes on each call to next().
class file_iterator {
public:
    /// Construct from an open file handle.
    explicit file_iterator(file& f) {
        tgm_file_iter_t* raw = nullptr;
        detail::check(tgm_file_iter_create(f.raw(), &raw));
        handle_.reset(raw);
    }

    file_iterator(file_iterator&&) noexcept = default;
    file_iterator& operator=(file_iterator&&) noexcept = default;
    ~file_iterator() = default;

    file_iterator(const file_iterator&) = delete;
    file_iterator& operator=(const file_iterator&) = delete;

    /// Advance to the next message.
    ///
    /// @param[out] out  Receives the raw message bytes.
    /// @return true if a message was returned, false when exhausted.
    bool next(std::vector<std::uint8_t>& out) {
        tgm_bytes_t bytes{};
        const tgm_error err = tgm_file_iter_next(handle_.get(), &bytes);
        if (err == TGM_ERROR_END_OF_ITER) return false;
        detail::check(err);
        out.assign(bytes.data, bytes.data + bytes.len);
        tgm_bytes_free(bytes);
        return true;
    }

private:
    struct deleter {
        void operator()(tgm_file_iter_t* p) const noexcept { tgm_file_iter_free(p); }
    };
    std::unique_ptr<tgm_file_iter_t, deleter> handle_;
};

// ============================================================
// object_iterator — RAII wrapper for tgm_object_iter_t
// ============================================================

/// Iterate over individual data objects within a single Tensogram message.
///
/// Parses metadata once at construction, then lazily decodes each object
/// on successive calls to next().
class object_iterator {
public:
    /// Construct from raw message bytes.
    object_iterator(const std::uint8_t* buf, std::size_t len,
                    const decode_options& opts = {}) {
        tgm_object_iter_t* raw = nullptr;
        detail::check(tgm_object_iter_create(
            buf, len, opts.verify_hash ? 1 : 0,
            opts.native_byte_order ? 1 : 0, &raw));
        handle_.reset(raw);
    }

    object_iterator(object_iterator&&) noexcept = default;
    object_iterator& operator=(object_iterator&&) noexcept = default;
    ~object_iterator() = default;

    object_iterator(const object_iterator&) = delete;
    object_iterator& operator=(const object_iterator&) = delete;

    /// Decode the next data object.
    ///
    /// @param[out] out  Receives a message containing exactly one object.
    /// @return true if an object was returned, false when exhausted.
    bool next(message& out) {
        tgm_message_t* raw = nullptr;
        const tgm_error err = tgm_object_iter_next(handle_.get(), &raw);
        if (err == TGM_ERROR_END_OF_ITER) return false;
        detail::check(err);
        out = message(raw);
        return true;
    }

private:
    struct deleter {
        void operator()(tgm_object_iter_t* p) const noexcept { tgm_object_iter_free(p); }
    };
    std::unique_ptr<tgm_object_iter_t, deleter> handle_;
};

// ============================================================
// streaming_encoder — RAII wrapper for tgm_streaming_encoder_t
// ============================================================

/// Progressive encoder that writes data objects one at a time to a file.
///
/// After writing all objects, call finish() to write the footer and close
/// the file.  If the encoder is destroyed without calling finish(), the
/// partial file is abandoned (the destructor will free the handle but
/// will not produce a valid Tensogram message).
class streaming_encoder {
public:
    /// Create a streaming encoder writing to @p path.
    ///
    /// @param path          Output file path (created/truncated).
    /// @param metadata_json JSON with "version" and optional extra keys.
    /// @param opts          Encoding options (hash algorithm, etc.).
    streaming_encoder(const std::string& path,
                      const std::string& metadata_json,
                      const encode_options& opts = {}) {
        tgm_streaming_encoder_t* raw = nullptr;
        const char* hash = opts.hash_algo.empty() ? nullptr
                                                    : opts.hash_algo.c_str();
        detail::check(tgm_streaming_encoder_create(
            path.c_str(), metadata_json.c_str(), hash, opts.threads,
            opts.reject_nan, opts.reject_inf, &raw));
        handle_.reset(raw);
    }

    streaming_encoder(streaming_encoder&&) noexcept = default;
    streaming_encoder& operator=(streaming_encoder&&) noexcept = default;
    ~streaming_encoder() = default;

    streaming_encoder(const streaming_encoder&) = delete;
    streaming_encoder& operator=(const streaming_encoder&) = delete;

    /// Write a PrecederMetadata frame for the next data object.
    ///
    /// @p metadata_json is a JSON object with per-object metadata keys
    /// (e.g. `{"mars":{"param":"2t"}, "units":"K"}`).
    ///
    /// Must be followed by exactly one write_object() or
    /// write_object_pre_encoded() call before another write_preceder()
    /// or finish().
    void write_preceder(const std::string& metadata_json) {
        detail::check(tgm_streaming_encoder_write_preceder(
            handle_.get(), metadata_json.c_str()));
    }

    /// Write a single data object.
    void write_object(const std::string& descriptor_json,
                      const std::uint8_t* data, std::size_t len) {
        detail::check(tgm_streaming_encoder_write(
            handle_.get(), descriptor_json.c_str(), data, len));
    }

    /// Write a single pre-encoded data object.
    ///
    /// Like write_object(), but @p data must already be encoded according
    /// to the descriptor's pipeline (`encoding` / `filter` / `compression`).
    /// The library does not run the encoding pipeline — it validates the
    /// descriptor's pipeline configuration and writes the bytes as-is.
    /// The hash (if configured on the encoder) is recomputed over the
    /// caller's bytes.
    ///
    /// For `szip` compression, the caller SHOULD include
    /// `szip_block_offsets` (bit offsets) in the descriptor's params so
    /// that `decode_range()` can locate compressed block boundaries.
    /// Other pipeline params (e.g. `simple_packing` reference value,
    /// scale factors) must also be present in the descriptor.
    void write_object_pre_encoded(const std::string& descriptor_json,
                                   const std::uint8_t* data, std::size_t len) {
        detail::check(tgm_streaming_encoder_write_pre_encoded(
            handle_.get(), descriptor_json.c_str(), data, len));
    }

    /// Number of objects written so far.
    [[nodiscard]] std::size_t object_count() const {
        return tgm_streaming_encoder_count(handle_.get());
    }

    /// Finalize the message (writes footer, closes file).
    ///
    /// After calling this, the handle is still valid but empty — the
    /// destructor will free the shell.  Do not write further objects.
    void finish() {
        detail::check(tgm_streaming_encoder_finish(handle_.get()));
        // Handle is now empty but still valid — destructor will free the shell.
    }

private:
    struct deleter {
        void operator()(tgm_streaming_encoder_t* p) const noexcept {
            tgm_streaming_encoder_free(p);
        }
    };
    std::unique_ptr<tgm_streaming_encoder_t, deleter> handle_;
};

// ============================================================
// Free functions
// ============================================================

/// Encode a Tensogram message from JSON metadata and raw data slices.
///
/// @param metadata_json  JSON with "version", "descriptors", optional "base"
///                       (list of per-object metadata dicts), and optional
///                       extra keys (e.g. "mars") which become message-level
///                       annotations in the `_extra_` CBOR section.
/// @param objects        Vector of (pointer, length) pairs — one per
///                       descriptor entry.
/// @param opts           Encoding options (hash algorithm, etc.).
/// @return The encoded message as a byte vector.
[[nodiscard]] inline std::vector<std::uint8_t> encode(
    const std::string& metadata_json,
    const std::vector<std::pair<const std::uint8_t*, std::size_t>>& objects,
    const encode_options& opts = {})
{
    detail::scatter_gather sg(objects);
    const char* hash = opts.hash_algo.empty() ? nullptr : opts.hash_algo.c_str();
    tgm_bytes_t bytes{};
    detail::check(tgm_encode(metadata_json.c_str(),
                              sg.ptrs.data(), sg.lens.data(), objects.size(),
                              hash, opts.threads,
                              opts.reject_nan, opts.reject_inf,
                              &bytes));
    std::vector<std::uint8_t> result(bytes.data, bytes.data + bytes.len);
    tgm_bytes_free(bytes);
    return result;
}

/// Encode a Tensogram message from JSON metadata and pre-encoded payload bytes.
///
/// Like encode(), but each entry in @p objects must already be encoded
/// according to the matching descriptor's `encoding` / `filter` /
/// `compression` pipeline. The library does not run the encoding pipeline
/// again — it writes the caller-provided bytes directly into the wire-format
/// payload after validating that the descriptor's pipeline configuration
/// is well-formed.
///
/// The library always recomputes the hash over the caller's bytes; any
/// `hash` field embedded in the descriptor JSON is ignored and overwritten.
///
/// For `szip` compression, callers SHOULD include `szip_block_offsets`
/// (bit offsets into the compressed payload) inside the matching
/// descriptor's params so that `decode_range()` can locate szip block
/// boundaries without rescanning the compressed stream. Other pipeline
/// params (e.g. `simple_packing` reference value, scale factors) must
/// also be present in the descriptor — they are not inferred from the
/// bytes.
///
/// The resulting wire format is identical to what encode() produces — a
/// decoder cannot distinguish the two sources.
///
/// @param metadata_json  Same JSON schema as encode() (`version`,
///                       `descriptors`, optional `base`, plus arbitrary
///                       extra top-level keys).
/// @param objects        Vector of (pointer, length) pairs pointing at
///                       already-encoded payload bytes — one per descriptor
///                       entry.
/// @param opts           Encoding options (hash algorithm, etc.).
/// @return The encoded message as a byte vector.
[[nodiscard]] inline std::vector<std::uint8_t> encode_pre_encoded(
    const std::string& metadata_json,
    const std::vector<std::pair<const std::uint8_t*, std::size_t>>& objects,
    const encode_options& opts = {})
{
    // Strict-finite flags are raw-input-only — pre-encoded bytes are
    // opaque to the library and cannot be meaningfully scanned for
    // NaN/Inf.  The underlying C FFI does not accept these flags, but
    // we catch the case here to give the C++ caller a clear error
    // rather than silently discarding their intent.
    if (opts.reject_nan || opts.reject_inf) {
        throw encoding_error(
            TGM_ERROR_ENCODING,
            "reject_nan / reject_inf do not apply to encode_pre_encoded: "
            "pre-encoded bytes are opaque to the library. Clear these "
            "fields before calling encode_pre_encoded, or use encode() "
            "on raw data.");
    }
    detail::scatter_gather sg(objects);
    const char* hash = opts.hash_algo.empty() ? nullptr : opts.hash_algo.c_str();
    tgm_bytes_t bytes{};
    detail::check(tgm_encode_pre_encoded(metadata_json.c_str(),
                                          sg.ptrs.data(), sg.lens.data(),
                                          objects.size(), hash,
                                          opts.threads, &bytes));
    std::vector<std::uint8_t> result(bytes.data, bytes.data + bytes.len);
    tgm_bytes_free(bytes);
    return result;
}

/// Decode a complete message (global metadata + all object payloads).
[[nodiscard]] inline message decode(const std::uint8_t* buf, std::size_t len,
                      const decode_options& opts)
{
    tgm_message_t* raw = nullptr;
    detail::check(tgm_decode(buf, len, opts.verify_hash ? 1 : 0,
                              opts.native_byte_order ? 1 : 0,
                              opts.threads, &raw));
    return message(raw);
}

/// Decode only the global metadata (no payload bytes are read).
/// The returned metadata contains `base` entries and `extra` keys.
[[nodiscard]] inline metadata decode_metadata(const std::uint8_t* buf, std::size_t len)
{
    tgm_metadata_t* raw = nullptr;
    detail::check(tgm_decode_metadata(buf, len, &raw));
    return metadata(raw);
}

/// Decode a single object by index.
[[nodiscard]] inline message decode_object(const std::uint8_t* buf, std::size_t len,
                             std::size_t index, const decode_options& opts)
{
    tgm_message_t* raw = nullptr;
    detail::check(tgm_decode_object(buf, len, index,
                                     opts.verify_hash ? 1 : 0,
                                     opts.native_byte_order ? 1 : 0,
                                     opts.threads, &raw));
    return message(raw);
}

/// Decode partial ranges from a data object (split mode — one vector per range).
[[nodiscard]] inline std::vector<std::vector<std::uint8_t>> decode_range(
    const std::uint8_t* buf, std::size_t len,
    std::size_t object_index,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
    const decode_options& opts = {})
{
    std::vector<std::uint64_t> offsets, counts;
    offsets.reserve(ranges.size());
    counts.reserve(ranges.size());
    for (const auto& r : ranges) {
        offsets.push_back(r.first);
        counts.push_back(r.second);
    }
    std::vector<tgm_bytes_t> bufs(ranges.size());
    std::size_t out_count = 0;
    detail::check(tgm_decode_range(buf, len, object_index,
                                    offsets.data(), counts.data(), ranges.size(),
                                    opts.verify_hash ? 1 : 0,
                                    opts.native_byte_order ? 1 : 0,
                                    opts.threads, 0,
                                    bufs.data(), &out_count));
    if (out_count > ranges.size()) {
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            tgm_bytes_free(bufs[i]);
        }
        throw std::runtime_error("tgm_decode_range returned out_count > ranges.size()");
    }
    std::vector<std::vector<std::uint8_t>> result;
    result.reserve(out_count);
    for (std::size_t i = 0; i < out_count; ++i) {
        result.emplace_back(bufs[i].data, bufs[i].data + bufs[i].len);
        tgm_bytes_free(bufs[i]);
    }
    return result;
}

/// Decode partial ranges from a data object (joined — single concatenated vector).
[[nodiscard]] inline std::vector<std::uint8_t> decode_range_joined(
    const std::uint8_t* buf, std::size_t len,
    std::size_t object_index,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
    const decode_options& opts = {})
{
    std::vector<std::uint64_t> offsets, counts;
    offsets.reserve(ranges.size());
    counts.reserve(ranges.size());
    for (const auto& r : ranges) {
        offsets.push_back(r.first);
        counts.push_back(r.second);
    }
    tgm_bytes_t bytes{};
    std::size_t out_count = 0;
    detail::check(tgm_decode_range(buf, len, object_index,
                                    offsets.data(), counts.data(), ranges.size(),
                                    opts.verify_hash ? 1 : 0,
                                    opts.native_byte_order ? 1 : 0,
                                    opts.threads, 1,
                                    &bytes, &out_count));
    if (out_count != 1) {
        tgm_bytes_free(bytes);
        throw std::runtime_error("tgm_decode_range returned unexpected out_count in joined mode");
    }
    std::vector<std::uint8_t> result(bytes.data, bytes.data + bytes.len);
    tgm_bytes_free(bytes);
    return result;
}

/// Scan a buffer for message boundaries.
[[nodiscard]] inline std::vector<scan_entry> scan(const std::uint8_t* buf, std::size_t len) {
    tgm_scan_result_t* raw = nullptr;
    detail::check(tgm_scan(buf, len, &raw));
    const std::size_t n = tgm_scan_count(raw);
    std::vector<scan_entry> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto e = tgm_scan_entry(raw, i);
        result.push_back({e.offset, e.length});
    }
    tgm_scan_free(raw);
    return result;
}

/// Compute a hash of the given data. Returns hex-encoded string.
[[nodiscard]] inline std::string compute_hash(const std::uint8_t* data, std::size_t len,
                                const std::string& algo = "xxh3") {
    tgm_bytes_t bytes{};
    detail::check(tgm_compute_hash(data, len, algo.c_str(), &bytes));
    std::string result(reinterpret_cast<const char*>(bytes.data), bytes.len);
    tgm_bytes_free(bytes);
    return result;
}

/// Validate a single Tensogram message buffer.
///
/// Returns a JSON string describing the validation report.
/// The report contains `issues`, `object_count`, and `hash_verified`.
/// Even structurally broken messages normally produce a report (with
/// errors in `issues`). Exceptions are reserved for non-OK API results
/// such as invalid arguments or failures while producing the JSON report.
///
/// @param buf               Wire-format message bytes (nullptr with len=0
///                          is valid for empty-buffer validation).
/// @param len               Length of @p buf.
/// @param level             "quick", "default", "checksum", or "full"
///                          (default: "default").
/// @param check_canonical   Check RFC 8949 CBOR key ordering.
/// @return JSON string with validation report.
/// @throws tensogram::invalid_arg_error If @p level is unrecognized.
/// @throws tensogram::encoding_error If JSON serialization fails.
[[nodiscard]] inline std::string validate(const std::uint8_t* buf, std::size_t len,
                                          const char* level = "default",
                                          bool check_canonical = false) {
    tgm_bytes_t bytes{};
    detail::check(tgm_validate(buf, len, level, check_canonical ? 1 : 0, &bytes));
    std::string result(reinterpret_cast<const char*>(bytes.data), bytes.len);
    tgm_bytes_free(bytes);
    return result;
}

/// Validate all messages in a `.tgm` file.
///
/// Returns a JSON string with `file_issues` and `messages` arrays.
/// Validation findings are reported in the JSON; only API/operational
/// failures throw.
///
/// @param path              Path to the `.tgm` file.
/// @param level             "quick", "default", "checksum", or "full"
///                          (default: "default").
/// @param check_canonical   Check CBOR key ordering.
/// @return JSON string with file validation report.
/// @throws tensogram::invalid_arg_error If @p path or @p level is invalid.
/// @throws tensogram::io_error If the file cannot be opened or read.
/// @throws tensogram::encoding_error If JSON serialization fails.
[[nodiscard]] inline std::string validate_file(const char* path,
                                               const char* level = "default",
                                               bool check_canonical = false) {
    tgm_bytes_t bytes{};
    detail::check(tgm_validate_file(path, level, check_canonical ? 1 : 0, &bytes));
    std::string result(reinterpret_cast<const char*>(bytes.data), bytes.len);
    tgm_bytes_free(bytes);
    return result;
}

} // namespace tensogram

#endif // TENSOGRAM_HPP
