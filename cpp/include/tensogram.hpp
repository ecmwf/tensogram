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
///
/// @note Lifetime: the frame walker borrows differently — frames() views the
///       caller's message buffer directly, so frame::payload() stays valid
///       for as long as *that buffer* lives, outliving the frame_range that
///       produced it.  See frame_range for the full contract.

#ifndef TENSOGRAM_HPP
#define TENSOGRAM_HPP

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
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
// Constants
// ============================================================

/// Wire-format version emitted and required by this build of the
/// library.
///
/// Mirrors the C macro `TGM_WIRE_VERSION` and the Rust constant
/// `tensogram::WIRE_VERSION`.  The value lives in the tensogram
/// preamble (see `plans/WIRE_FORMAT.md` §3) — never in the CBOR
/// metadata frame.  Decoders reject any other preamble version.
inline constexpr std::uint16_t wire_version = TGM_WIRE_VERSION;

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

/// Base for decode-time integrity verification failures.
///
/// Subclasses surface specific failure modes when the user has
/// opted in to verification via `decode_options::verify_hash =
/// true`:
///   * `hash_mismatch_error` — frame's `HASH_PRESENT` flag is set
///     and the inline hash slot disagrees with the recomputed
///     digest.
///   * `missing_hash_error` — frame's `HASH_PRESENT` flag is
///     clear (no digest was recorded for this frame).
///
/// Catch this base when you want to handle either kind
/// uniformly; catch the subclasses to discriminate.  See
/// `plans/DESIGN.md` §"Integrity Hashing".
///
/// Carries the 0-based index of the offending data object (the frame
/// whose inline hash was missing or mismatched) via object_index(),
/// mirroring the C ABI's `tgm_last_error_object_index()`.
class integrity_error : public error {
public:
    integrity_error(tgm_error code, const std::string& msg,
                    std::int64_t object_index = -1)
        : error(code, msg), object_index_(object_index) {}

    /// The 0-based index of the data object this integrity failure
    /// refers to, or `-1` when the error carries no object index.
    ///
    /// Sourced from the C ABI's `tgm_last_error_object_index()` at the
    /// throw site.  For `hash_mismatch_error` / `missing_hash_error`
    /// this pinpoints the frame that failed decode-time verification,
    /// so callers can react per-object without parsing `what()`.
    [[nodiscard]] std::int64_t object_index() const noexcept { return object_index_; }

private:
    std::int64_t object_index_;
};

/// Thrown when decode-time hash verification was requested,
/// the frame's `HASH_PRESENT` flag is set, but the inline-hash
/// slot disagrees with the recomputed digest of the frame body.
/// Distinct from `missing_hash_error` — the digest WAS recorded
/// but doesn't match the body, indicating transport / storage
/// corruption or a deliberately tampered frame.
///
/// The expected (stored) and actual (recomputed) 16-character
/// xxh3-64 hex digests are embedded in the exception message
/// (`what()`); a future binding extension may surface them as
/// structured fields.  See `plans/WIRE_FORMAT.md` §2.5.
class hash_mismatch_error : public integrity_error {
public:
    hash_mismatch_error(tgm_error code, const std::string& msg,
                        std::int64_t object_index = -1)
        : integrity_error(code, msg, object_index) {}
};

/// Thrown when decode-time hash verification was requested but
/// the frame's `HASH_PRESENT` flag is clear (no inline hash
/// recorded for that frame).  Distinct from `hash_mismatch_error`
/// — the digest didn't disagree; there was no digest to compare
/// against.  See `plans/WIRE_FORMAT.md` §2.5.
class missing_hash_error : public integrity_error {
public:
    missing_hash_error(tgm_error code, const std::string& msg,
                       std::int64_t object_index = -1)
        : integrity_error(code, msg, object_index) {}
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

/// Thrown when an async task exceeds its deadline.
class timeout_error : public error {
public:
    timeout_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

/// Thrown when an async task is cancelled via its
/// `cancellation_token`.
class cancelled_error : public error {
public:
    cancelled_error(tgm_error code, const std::string& msg) : error(code, msg) {}
};

// ============================================================
// detail — error checking helper
// ============================================================

namespace detail {

/// Throw the typed C++ exception that matches a C error code, using
/// `message` as the exception text.
///
/// This is `[[noreturn]]` and must be called with a *non-OK* code:
/// `TGM_ERROR_OK` has no corresponding exception and falls through to
/// the generic [`error`], so callers screen for OK first.  [`check()`]
/// only calls this on a non-OK code, and the async frontends only on a
/// failed `result`.  Used both by [`check()`] (which captures the FFI's
/// thread-local last-error string) and by the async frontends (which
/// capture the message inside completion handlers, before any other FFI
/// call on the thread can clobber the last-error slot).
///
/// `object_index` is the 0-based data-object index the error refers to
/// (from `tgm_last_error_object_index()`), or `-1` when none.  It is
/// carried only by the integrity subclasses; every other exception
/// ignores it.
[[noreturn]] inline void throw_for_code(tgm_error err, const std::string& message,
                                        std::int64_t object_index = -1) {
    switch (err) {
        case TGM_ERROR_FRAMING:       throw framing_error(err, message);
        case TGM_ERROR_METADATA:      throw metadata_error(err, message);
        case TGM_ERROR_ENCODING:      throw encoding_error(err, message);
        case TGM_ERROR_COMPRESSION:   throw compression_error(err, message);
        case TGM_ERROR_OBJECT:        throw object_error(err, message);
        case TGM_ERROR_IO:            throw io_error(err, message);
        case TGM_ERROR_HASH_MISMATCH: throw hash_mismatch_error(err, message, object_index);
        case TGM_ERROR_MISSING_HASH:  throw missing_hash_error(err, message, object_index);
        case TGM_ERROR_INVALID_ARG:   throw invalid_arg_error(err, message);
        case TGM_ERROR_REMOTE:        throw remote_error(err, message);
        case TGM_ERROR_TIMEOUT:       throw timeout_error(err, message);
        case TGM_ERROR_CANCELLED:     throw cancelled_error(err, message);
        case TGM_ERROR_END_OF_ITER:   throw error(err, message);
        default:                      throw error(err, message);
    }
}

/// Check a C error code and throw the appropriate typed exception.
inline void check(tgm_error err) {
    if (err == TGM_ERROR_OK) return;
    const char* msg = tgm_last_error();
    // Snapshot the object index immediately after the message pointer,
    // before any other FFI call on this thread can clobber the shared
    // thread-local error slot.  `tgm_error_string` (below) is a pure
    // lookup and does not touch the slot, so this ordering is safe.
    const std::int64_t object_index = tgm_last_error_object_index();
    std::string message = msg ? msg : tgm_error_string(err);
    throw_for_code(err, message, object_index);
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

/// Which non-finite bitmask kind to fetch from a mask-aware decode.
///
/// Mirrors the C `tgm_mask_kind` enum and the three slots of the Rust
/// `DecodedMaskSet` (`nan` / `pos_inf` / `neg_inf`).  Passed to
/// decoded_object::mask() on an object produced by decode_with_masks().
enum class mask_kind {
    nan,      ///< Positions that held `NaN` at encode time.
    pos_inf,  ///< Positions that held `+∞` at encode time.
    neg_inf,  ///< Positions that held `−∞` at encode time.
};

namespace detail {

/// Map the C++ mask_kind to the C `tgm_mask_kind` discriminant.
[[nodiscard]] inline tgm_mask_kind mask_kind_to_c(mask_kind k) noexcept {
    switch (k) {
        case mask_kind::nan:     return TGM_MASK_KIND_NAN;
        case mask_kind::pos_inf: return TGM_MASK_KIND_POS_INF;
        case mask_kind::neg_inf: return TGM_MASK_KIND_NEG_INF;
    }
    // Defensive: an unknown enumerator decays to the NaN slot.
    return TGM_MASK_KIND_NAN;
}

}  // namespace detail

/// An element type, mirroring the C `tgm_dtype` enum and
/// `tensogram::Dtype` one-for-one.
///
/// The wire format stores the dtype as a **string** (see
/// `plans/WIRE_FORMAT.md` §6.1), so these codes are a binding convenience
/// rather than a wire value — but they are part of the frozen C ABI.
/// decoded_object::dtype_string() returns the same information as text;
/// this enum is what you want in a `switch`.
enum class dtype {
    float16    = TGM_DTYPE_FLOAT16,
    bfloat16   = TGM_DTYPE_BFLOAT16,
    float32    = TGM_DTYPE_FLOAT32,
    float64    = TGM_DTYPE_FLOAT64,
    complex64  = TGM_DTYPE_COMPLEX64,
    complex128 = TGM_DTYPE_COMPLEX128,
    int8       = TGM_DTYPE_INT8,
    int16      = TGM_DTYPE_INT16,
    int32      = TGM_DTYPE_INT32,
    int64      = TGM_DTYPE_INT64,
    uint8      = TGM_DTYPE_UINT8,
    uint16     = TGM_DTYPE_UINT16,
    uint32     = TGM_DTYPE_UINT32,
    uint64     = TGM_DTYPE_UINT64,
    /// Sub-byte packed bitmask; the element width is 1 bit, so the byte
    /// width of this dtype is reported as 0.
    bitmask    = TGM_DTYPE_BITMASK,
};

/// A payload's byte order, mirroring the C `tgm_byte_order` enum and
/// `tensogram::ByteOrder`.
///
/// Like dtype these codes are a binding convenience — the wire stores
/// `"little"` / `"big"` as text — and are frozen as part of the C ABI.
/// decoded_object::byte_order_string() returns the same information as text.
enum class byte_order {
    little = TGM_BYTE_ORDER_LITTLE,
    big    = TGM_BYTE_ORDER_BIG,
};

/// Where to place the aggregate hash frame, mirroring the C
/// `tgm_aggregate_hash_policy` enum and `tensogram::AggregateHashPolicy`.
///
/// The default is aggregate_hash_policy::automatic, so a
/// default-constructed encode_options asks the encoder to choose — header
/// when buffering, footer when streaming — exactly as before this knob
/// existed.
///
/// `header` and `both` are **buffered-mode only**: a streaming encoder
/// writes its header before any data object exists, so the per-object
/// hashes are not yet known.  Passing either to streaming_encoder throws
/// tensogram::encoding_error.
enum class aggregate_hash_policy {
    /// Encoder picks: buffered → header, streaming → footer.
    automatic = TGM_AGGREGATE_HASH_POLICY_AUTO,
    /// Emit no aggregate hash frame.  Per-frame inline hash slots are
    /// unaffected — see frame::has_hash().
    none      = TGM_AGGREGATE_HASH_POLICY_NONE,
    /// Emit a frame_type::header_hash frame.  Buffered mode only.
    header    = TGM_AGGREGATE_HASH_POLICY_HEADER,
    /// Emit a frame_type::footer_hash frame.  Valid in both modes.
    footer    = TGM_AGGREGATE_HASH_POLICY_FOOTER,
    /// Emit both a header and a footer hash frame carrying identical hash
    /// lists.  Buffered mode only.
    both      = TGM_AGGREGATE_HASH_POLICY_BOTH,
};

/// Which codec implementation to use where both are compiled in, mirroring
/// the C `tgm_compression_backend` enum and `tensogram::CompressionBackend`.
///
/// `automatic` (the default) consults the `TENSOGRAM_COMPRESSION_BACKEND`
/// environment variable, then the platform default.  `ffi` and `pure` are
/// explicit overrides that always win over the environment.  The choice
/// affects szip and zstd, the two codecs with both a C and a pure-Rust
/// implementation; the decoded payload is identical either way, only the
/// compressed bytes may differ.
enum class compression_backend {
    /// Consult the environment, then the platform default.
    automatic = TGM_COMPRESSION_BACKEND_AUTO,
    /// Use the C FFI codecs (libaec for szip, libzstd for zstd).
    ffi       = TGM_COMPRESSION_BACKEND_FFI,
    /// Use the pure-Rust codecs (tensogram-szip, ruzstd).
    pure      = TGM_COMPRESSION_BACKEND_PURE,
};

namespace detail {

/// Map a C `tgm_dtype` discriminant to the C++ dtype enum.
[[nodiscard]] inline dtype dtype_from_c(tgm_dtype t) noexcept {
    switch (t) {
        case TGM_DTYPE_FLOAT16:    return dtype::float16;
        case TGM_DTYPE_BFLOAT16:   return dtype::bfloat16;
        case TGM_DTYPE_FLOAT32:    return dtype::float32;
        case TGM_DTYPE_FLOAT64:    return dtype::float64;
        case TGM_DTYPE_COMPLEX64:  return dtype::complex64;
        case TGM_DTYPE_COMPLEX128: return dtype::complex128;
        case TGM_DTYPE_INT8:       return dtype::int8;
        case TGM_DTYPE_INT16:      return dtype::int16;
        case TGM_DTYPE_INT32:      return dtype::int32;
        case TGM_DTYPE_INT64:      return dtype::int64;
        case TGM_DTYPE_UINT8:      return dtype::uint8;
        case TGM_DTYPE_UINT16:     return dtype::uint16;
        case TGM_DTYPE_UINT32:     return dtype::uint32;
        case TGM_DTYPE_UINT64:     return dtype::uint64;
        case TGM_DTYPE_BITMASK:    return dtype::bitmask;
    }
    // Defensive: an unknown discriminant from a future ABI decays to the
    // zero variant, which is also what the C accessor reports on failure.
    return dtype::float16;
}

/// Map a C `tgm_byte_order` discriminant to the C++ byte_order enum.
[[nodiscard]] inline byte_order byte_order_from_c(tgm_byte_order o) noexcept {
    switch (o) {
        case TGM_BYTE_ORDER_LITTLE: return byte_order::little;
        case TGM_BYTE_ORDER_BIG:    return byte_order::big;
    }
    // Defensive: an unknown discriminant from a future ABI decays to the
    // zero variant.
    return byte_order::little;
}

/// Map the C++ aggregate_hash_policy to the C discriminant.
[[nodiscard]] inline tgm_aggregate_hash_policy aggregate_hash_policy_to_c(
    aggregate_hash_policy p) noexcept {
    switch (p) {
        case aggregate_hash_policy::automatic: return TGM_AGGREGATE_HASH_POLICY_AUTO;
        case aggregate_hash_policy::none:      return TGM_AGGREGATE_HASH_POLICY_NONE;
        case aggregate_hash_policy::header:    return TGM_AGGREGATE_HASH_POLICY_HEADER;
        case aggregate_hash_policy::footer:    return TGM_AGGREGATE_HASH_POLICY_FOOTER;
        case aggregate_hash_policy::both:      return TGM_AGGREGATE_HASH_POLICY_BOTH;
    }
    // Defensive: an unknown enumerator decays to the library default.
    return TGM_AGGREGATE_HASH_POLICY_AUTO;
}

/// Map the C++ compression_backend to the C discriminant.
[[nodiscard]] inline tgm_compression_backend compression_backend_to_c(
    compression_backend b) noexcept {
    switch (b) {
        case compression_backend::automatic: return TGM_COMPRESSION_BACKEND_AUTO;
        case compression_backend::ffi:       return TGM_COMPRESSION_BACKEND_FFI;
        case compression_backend::pure:      return TGM_COMPRESSION_BACKEND_PURE;
    }
    // Defensive: an unknown enumerator decays to the library default.
    return TGM_COMPRESSION_BACKEND_AUTO;
}

/// Bounds-check an object index on behalf of the typed enum accessors.
///
/// The C enum accessors have no spare code for failure: on a NULL handle or
/// an out-of-range index they record the reason in the thread-local error
/// slot and return their zero variant, which is indistinguishable from a
/// genuine `float16` / `little` answer.  The paired **string** getter returns
/// NULL for exactly those inputs, so it is the unambiguous check — pass its
/// result as @p probe and the wrapper raises instead of handing back a
/// plausible-looking value.
inline void require_object_index(const char* probe, std::size_t index, const char* what) {
    if (probe == nullptr) {
        throw invalid_arg_error(TGM_ERROR_INVALID_ARG,
                                std::string(what) + ": object index " +
                                    std::to_string(index) + " is out of range");
    }
}

}  // namespace detail

/// Options controlling scan_with_options() and scan_file_with_options().
///
/// Mirrors the C `TgmScanOptions` POD and `tensogram::ScanOptions`.  A
/// default-constructed scan_options requests the library defaults
/// (bidirectional meet-in-the-middle walk, 4 GiB message-size cap).
struct scan_options {
    /// Enable the meet-in-the-middle (bidirectional) walk.  `false`
    /// forces the pure forward scan (still correct, no hop halving).
    bool bidirectional = true;
    /// Upper bound on the `total_length` a postamble may advertise;
    /// anything larger is treated as corruption and the backward walker
    /// yields to the forward walker.  `0` selects the library default
    /// (4 GiB).
    std::uint64_t max_message_size = 0;
};

/// simple_packing (GRIB-style lossy quantisation) parameters computed
/// from a set of sample values by compute_simple_packing_params().
///
/// These are the three scale parameters a `simple_packing` descriptor
/// needs (`sp_reference_value`, `sp_binary_scale_factor`,
/// `sp_decimal_scale_factor`); pass them straight into the descriptor
/// JSON instead of hand-rolling the C out-params.
struct simple_packing_params {
    /// Reference (minimum) value subtracted from every sample before
    /// packing — the descriptor's `sp_reference_value`.
    double reference_value = 0.0;
    /// Base-2 scale exponent chosen to fit the value range into
    /// `bits_per_value` bits — the descriptor's `sp_binary_scale_factor`.
    std::int32_t binary_scale_factor = 0;
    /// Base-10 scale exponent, echoed back from the request so callers
    /// have all three params in one place — the descriptor's
    /// `sp_decimal_scale_factor`.
    std::int32_t decimal_scale_factor = 0;
};

/// Options controlling message encoding.
///
/// Maps onto the C `TgmEncodeOptions` POD (plus the `threads` argument the
/// C entry points take separately).  Every field's default selects the
/// library default, so a default-constructed encode_options behaves exactly
/// like passing no options at all.
///
/// The encode entry points — encode(), file::append() and
/// streaming_encoder — pick the narrowest C function that can carry the
/// options actually set: the plain one, the mask-aware one, or the full
/// `*_with_encode_options` one.  No field is ever silently dropped, and a
/// caller who sets nothing new keeps the exact call path (and bytes) they
/// had before these knobs existed.  encode_pre_encoded() is the exception
/// the C ABI imposes: it has no full-option entry point, so it rejects
/// `aggregate_hash` / `codec_backend` rather than ignoring them.
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
    /// When true, NaN values in float / complex payloads are substituted
    /// with `0.0` and recorded in a bitmask companion section of the
    /// data-object frame.  When false (the default), any NaN input is a
    /// hard encode error.  See `docs/src/guide/nan-inf-handling.md`
    /// and `plans/WIRE_FORMAT.md` §6.5.
    bool allow_nan = false;
    /// When true, `+Inf` AND `-Inf` values are substituted with `0.0` and
    /// recorded in per-sign bitmasks.  Default false (reject).
    bool allow_inf = false;
    /// Mask compression method for the NaN mask.  Empty string ("") uses
    /// the library default ("roaring").  Accepts "none" | "rle" |
    /// "roaring" | "lz4" | "zstd" | "blosc2".
    std::string nan_mask_method = "";
    /// Mask compression method for the +Inf mask.  Default "roaring".
    std::string pos_inf_mask_method = "";
    /// Mask compression method for the -Inf mask.  Default "roaring".
    std::string neg_inf_mask_method = "";
    /// Uncompressed byte-count threshold below which mask blobs are
    /// written as "none" regardless of the requested method.  Default
    /// 128.  Set to 0 to disable the fallback.  Negative values use the
    /// library default.
    std::ptrdiff_t small_mask_threshold_bytes = -1;
    /// Where to write the aggregate hash frame.  Ignored when hashing is
    /// off (`hash_algo` empty or "none") — there is nothing to aggregate.
    ///
    /// aggregate_hash_policy::header and ::both are buffered-mode only;
    /// streaming_encoder rejects them with tensogram::encoding_error.
    aggregate_hash_policy aggregate_hash = aggregate_hash_policy::automatic;
    /// Which codec implementation to prefer for szip / zstd.
    ///
    /// Maps to `TgmEncodeOptions::compression_backend` and to
    /// `compression_backend=` / `compressionBackend` in the other bindings.
    /// Spelled `codec_backend` here because a member named
    /// `compression_backend` would shadow the enum type of the same name
    /// inside this struct.
    compression_backend codec_backend = compression_backend::automatic;
};

/// Options controlling message decoding.
struct decode_options {
    /// When true (the default), decoded payloads are converted to the
    /// caller's native byte order.  Set to false to receive bytes in the
    /// message's declared wire byte order.
    bool native_byte_order = true;
    /// Thread budget for the multi-threaded decoding pipeline.
    /// See `encode_options::threads` for semantics.
    std::uint32_t threads = 0;
    /// When true (the default) AND the message carries a mask companion,
    /// decode writes canonical NaN / ±Inf at the masked positions.  Set
    /// to false to receive the 0.0-substituted bytes as on disk.
    bool restore_non_finite = true;
    /// When true, every data-object frame the decoder touches is
    /// verified against its inline xxh3-64 hash.  See
    /// `plans/DESIGN.md` §"Integrity Hashing" for the contract:
    ///   * frame's `HASH_PRESENT` flag clear → throws
    ///     `tensogram::missing_hash_error`.
    ///   * flag set + slot disagrees → throws
    ///     `tensogram::hash_mismatch_error`.
    /// Both subclass `tensogram::integrity_error`.  Default
    /// `false` — opt in when the consumer wants end-to-end
    /// integrity beyond what the transport layer provides.
    ///
    /// **Not respected by `decode_range` / `file::decode_range`:**
    /// the inline hash covers the whole post-encoding payload of
    /// the source frame, so verifying it would require reading
    /// every byte that range decode is designed to avoid.  Use
    /// `decode_object` with `verify_hash=true` when integrity
    /// matters on a partial-load workflow.
    bool verify_hash = false;
};

namespace detail {
/// Build a TgmEncodeMaskOptions from encode_options or return nullptr
/// when no mask policy was requested.  Lifetime of the returned
/// struct is tied to the `holder` argument; the struct's string
/// pointers alias strings owned by `opts`.
struct mask_opts_holder {
    TgmEncodeMaskOptions value;
    bool active;
};

inline mask_opts_holder build_mask_opts(const encode_options& opts) {
    mask_opts_holder h{};
    // If none of the mask-specific fields are set, skip the options
    // struct entirely — the FFI will use library defaults.
    const bool any_mask_field = opts.allow_nan || opts.allow_inf
        || !opts.nan_mask_method.empty()
        || !opts.pos_inf_mask_method.empty()
        || !opts.neg_inf_mask_method.empty()
        || opts.small_mask_threshold_bytes >= 0;
    if (!any_mask_field) {
        h.active = false;
        return h;
    }
    h.active = true;
    h.value.allow_nan = opts.allow_nan;
    h.value.allow_inf = opts.allow_inf;
    h.value.nan_mask_method =
        opts.nan_mask_method.empty() ? nullptr : opts.nan_mask_method.c_str();
    h.value.pos_inf_mask_method =
        opts.pos_inf_mask_method.empty() ? nullptr : opts.pos_inf_mask_method.c_str();
    h.value.neg_inf_mask_method =
        opts.neg_inf_mask_method.empty() ? nullptr : opts.neg_inf_mask_method.c_str();
    h.value.small_mask_threshold_bytes = opts.small_mask_threshold_bytes;
    return h;
}

/// True when encode_options carries something only the full
/// `TgmEncodeOptions` entry points can express — the aggregate-hash
/// placement or the codec backend.
///
/// The encode paths use this to escalate: with neither knob set they keep
/// calling the narrower C functions (`tgm_encode` / `tgm_encode_with_options`
/// and friends), so existing callers keep their exact behaviour; with either
/// set they switch to `*_with_encode_options`, so nothing is dropped.
[[nodiscard]] inline bool needs_encode_options(const encode_options& opts) noexcept {
    return opts.aggregate_hash != aggregate_hash_policy::automatic
        || opts.codec_backend != compression_backend::automatic;
}

/// Build the full `TgmEncodeOptions` POD from encode_options.
///
/// @warning The struct's `const char*` fields alias strings owned by @p opts,
///          so the returned value must not outlive it.
[[nodiscard]] inline TgmEncodeOptions build_encode_opts(const encode_options& opts) noexcept {
    TgmEncodeOptions c_opts{};
    // The FFI convention is "name it to get it": a NULL hash means no hashing.
    c_opts.hash = opts.hash_algo.empty() ? nullptr : opts.hash_algo.c_str();
    c_opts.aggregate_hash = aggregate_hash_policy_to_c(opts.aggregate_hash);
    c_opts.compression_backend = compression_backend_to_c(opts.codec_backend);
    c_opts.allow_nan = opts.allow_nan;
    c_opts.allow_inf = opts.allow_inf;
    c_opts.nan_mask_method =
        opts.nan_mask_method.empty() ? nullptr : opts.nan_mask_method.c_str();
    c_opts.pos_inf_mask_method =
        opts.pos_inf_mask_method.empty() ? nullptr : opts.pos_inf_mask_method.c_str();
    c_opts.neg_inf_mask_method =
        opts.neg_inf_mask_method.empty() ? nullptr : opts.neg_inf_mask_method.c_str();
    c_opts.small_mask_threshold_bytes = opts.small_mask_threshold_bytes;
    return c_opts;
}
}  // namespace detail

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

// Async-frontend bridge helpers (forward-declared for friend access).
// Defined inline in `tensogram/async/callback.hpp` once the message
// and metadata types are complete.
namespace detail {
    inline message message_from_raw(tgm_message_t* raw);
    inline metadata metadata_from_raw(tgm_metadata_t* raw);
}

// Free functions (forward-declared for friend access)
[[nodiscard]] inline message decode(const std::uint8_t* buf, std::size_t len,
                      const decode_options& opts = {});
[[nodiscard]] inline message decode_with_masks(const std::uint8_t* buf, std::size_t len,
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

    /// Dtype as a typed enumerator (e.g. dtype::float32) — the `switch`-able
    /// companion to dtype_string().
    ///
    /// Wraps `tgm_object_dtype_enum`.  That accessor has no spare code to
    /// signal failure, so this wrapper bounds-checks through the paired
    /// string getter first and raises rather than returning the zero variant.
    ///
    /// @throws tensogram::invalid_arg_error If the object index is out of
    ///         range (dtype_string() returns "" for exactly the same input).
    [[nodiscard]] tensogram::dtype dtype_enum() const {
        detail::require_object_index(tgm_object_dtype(msg_, index_), index_, "dtype_enum");
        return detail::dtype_from_c(tgm_object_dtype_enum(msg_, index_));
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

    /// Byte order as a typed enumerator — the `switch`-able companion to
    /// byte_order_string().
    ///
    /// Wraps `tgm_object_byte_order_enum`; see dtype_enum() for why the
    /// bounds check goes through the string getter.
    ///
    /// @throws tensogram::invalid_arg_error If the object index is out of
    ///         range (byte_order_string() returns "" for the same input).
    [[nodiscard]] tensogram::byte_order byte_order_enum() const {
        detail::require_object_index(tgm_object_byte_order(msg_, index_), index_,
                                     "byte_order_enum");
        return detail::byte_order_from_c(tgm_object_byte_order_enum(msg_, index_));
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

    /// True if this object retained at least one non-finite bitmask
    /// (NaN, `+∞`, or `−∞`).
    ///
    /// Only ever true for an object belonging to a message produced by
    /// decode_with_masks(); every other decode path applies restoration
    /// in place and keeps no masks, so this returns false for them.
    [[nodiscard]] bool has_masks() const {
        return tgm_object_has_masks(msg_, index_);
    }

    /// Fetch one non-finite bitmask kind as a boolean byte-mask.
    ///
    /// Returns one byte per tensor element in row-major (descriptor-shape)
    /// order: `0x01` where the element held the requested non-finite kind
    /// at encode time, `0x00` otherwise.  The vector length equals the
    /// element count (the shape product).  For complex dtypes a single
    /// byte covers BOTH the real and imaginary components.
    ///
    /// Returns an **empty** vector when the object carried no mask of that
    /// kind (or the parent message was not produced by
    /// decode_with_masks()) — pair with has_masks() to distinguish "no
    /// masks at all" from "this kind absent".
    ///
    /// @warning Unlike the borrowing data() accessors this returns an
    ///          owned copy, so it may safely outlive the parent message.
    [[nodiscard]] std::vector<std::uint8_t> mask(mask_kind kind) const {
        tgm_bytes_t bytes = tgm_object_mask(msg_, index_, detail::mask_kind_to_c(kind));
        std::vector<std::uint8_t> result(bytes.data, bytes.data + bytes.len);
        tgm_bytes_free(bytes);
        return result;
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
    friend message tensogram::decode_with_masks(const std::uint8_t*, std::size_t,
                                                const decode_options&);
    friend message tensogram::decode_object(const std::uint8_t*, std::size_t,
                                            std::size_t, const decode_options&);
    friend class file;
    friend class object_iterator;
    // Async frontends (callback / coro / std_future) build a message
    // from an FFI handle returned by tgm_async_task_join_message.
    friend message tensogram::detail::message_from_raw(tgm_message_t*);

    struct deleter {
        void operator()(tgm_message_t* p) const noexcept { tgm_message_free(p); }
    };

    explicit message(tgm_message_t* raw) : handle_(raw) {}
    std::unique_ptr<tgm_message_t, deleter> handle_;
};

// ============================================================
// value_type + meta_value — borrowed metadata value cursor
// ============================================================

/// The kind of a metadata value, mirroring the C `tgm_value_type` enum.
///
/// CBOR integers are i128-backed and surface as a single
/// value_type::integer; use meta_value::as_int() / as_uint() to extract
/// with range checking.  `floating`/`integer` are spelled out (rather
/// than `float`/`int`) to avoid clashing with the C++ keywords.
enum class value_type {
    null,       ///< CBOR null (distinct from an absent key).
    boolean,    ///< CBOR bool.
    integer,    ///< CBOR integer (extract via as_int() / as_uint()).
    floating,   ///< CBOR float.
    string,     ///< CBOR text string (UTF-8).
    bytes,      ///< CBOR byte string.
    array,      ///< CBOR array (index via at() / operator[] / range-for).
    map,        ///< CBOR map or a section view (enumerate via key_at()/value_at()).
};

namespace detail {

/// Map a C `tgm_value_type` discriminant to the C++ value_type enum.
[[nodiscard]] inline value_type value_type_from_c(tgm_value_type t) noexcept {
    switch (t) {
        case TGM_VALUE_TYPE_NULL:   return value_type::null;
        case TGM_VALUE_TYPE_BOOL:   return value_type::boolean;
        case TGM_VALUE_TYPE_INT:    return value_type::integer;
        case TGM_VALUE_TYPE_FLOAT:  return value_type::floating;
        case TGM_VALUE_TYPE_STRING: return value_type::string;
        case TGM_VALUE_TYPE_BYTES:  return value_type::bytes;
        case TGM_VALUE_TYPE_ARRAY:  return value_type::array;
        case TGM_VALUE_TYPE_MAP:    return value_type::map;
    }
    // Defensive: an unknown discriminant from a future ABI decays to null.
    return value_type::null;
}

} // namespace detail

/// Non-owning cursor over a single metadata value (wraps a borrowed
/// `const tgm_value_t*`).
///
/// A meta_value is a lightweight handle (one pointer).  It borrows from the
/// parent tensogram::metadata and is only valid until that metadata handle is
/// destroyed or moved-from.
///
/// @warning A meta_value — and any std::string_view / meta_value it yields —
///          **must not outlive the parent metadata**.  Copying a meta_value
///          copies the borrow, not the data.
///
/// The extraction accessors are **precise** (no cross-type coercion): as_int()
/// only succeeds on integers, as_string() only on text, etc.  as_double() is
/// the sole widening accessor (an integer widens to double).  Every accessor
/// on an invalid cursor (see valid()) yields the empty result — std::nullopt,
/// a null type, or size 0 — so navigation never dereferences a bad pointer.
class meta_value {
public:
    /// The kind of this value (value_type::null for an invalid cursor).
    [[nodiscard]] value_type type() const {
        return detail::value_type_from_c(tgm_value_get_type(v_));
    }

    /// True if this cursor references a value (false only for the invalid
    /// cursor returned by e.g. metadata::object() on an out-of-range index).
    /// A genuine CBOR null is valid() — use is_null() to detect that.
    [[nodiscard]] bool valid() const noexcept { return v_ != nullptr; }

    [[nodiscard]] bool is_null()   const { return type() == value_type::null; }
    [[nodiscard]] bool is_bool()   const { return type() == value_type::boolean; }
    [[nodiscard]] bool is_int()    const { return type() == value_type::integer; }
    [[nodiscard]] bool is_float()  const { return type() == value_type::floating; }
    [[nodiscard]] bool is_string() const { return type() == value_type::string; }
    [[nodiscard]] bool is_bytes()  const { return type() == value_type::bytes; }
    [[nodiscard]] bool is_array()  const { return type() == value_type::array; }
    [[nodiscard]] bool is_map()    const { return type() == value_type::map; }

    /// Extract a signed 64-bit integer (integers in i64 range only).
    /// @return The value, or std::nullopt for a non-integer / out-of-range.
    [[nodiscard]] std::optional<std::int64_t> as_int() const {
        std::int64_t out = 0;
        if (tgm_value_as_i64(v_, &out)) return out;
        return std::nullopt;
    }

    /// Extract an unsigned 64-bit integer (integers in u64 range only).
    /// @return The value, or std::nullopt for a non-integer / out-of-range.
    [[nodiscard]] std::optional<std::uint64_t> as_uint() const {
        std::uint64_t out = 0;
        if (tgm_value_as_u64(v_, &out)) return out;
        return std::nullopt;
    }

    /// Extract a double.  This is the one widening accessor: a float extracts
    /// as-is and an integer is widened to double (may lose precision).
    /// @return The value, or std::nullopt for a non-numeric value.
    [[nodiscard]] std::optional<double> as_double() const {
        double out = 0.0;
        if (tgm_value_as_f64(v_, &out)) return out;
        return std::nullopt;
    }

    /// Extract a bool (no coercion).
    /// @return The value, or std::nullopt for a non-boolean value.
    [[nodiscard]] std::optional<bool> as_bool() const {
        bool out = false;
        if (tgm_value_as_bool(v_, &out)) return out;
        return std::nullopt;
    }

    /// Borrow a text string as a std::string_view (text values only).
    ///
    /// The view uses the C API's `ptr + len` (never `strlen`), so interior NUL
    /// bytes are preserved and the buffer need not be NUL-terminated.
    /// @return A view borrowing from the parent metadata, or std::nullopt for
    ///         a non-string value.
    /// @warning The view must not outlive the parent metadata.
    [[nodiscard]] std::optional<std::string_view> as_string() const {
        std::size_t len = 0;
        const char* p = tgm_value_as_string(v_, &len);
        if (p == nullptr) return std::nullopt;
        return std::string_view(p, len);
    }

    /// Copy a byte string into an owned vector (byte-string values only).
    ///
    /// Returns an owned copy (not a borrow) so the bytes may safely outlive
    /// the parent metadata.  The project targets C++17, so this copies rather
    /// than returning a std::span.
    /// @return The bytes, or std::nullopt for a non-bytes value.
    [[nodiscard]] std::optional<std::vector<std::uint8_t>> as_bytes() const {
        std::size_t len = 0;
        const std::uint8_t* p = tgm_value_as_bytes(v_, &len);
        if (p == nullptr) return std::nullopt;
        return std::vector<std::uint8_t>(p, p + len);
    }

    /// Number of elements in an array, or entries in a map; 0 otherwise.
    [[nodiscard]] std::size_t size() const {
        switch (type()) {
            case value_type::array: return tgm_value_array_len(v_);
            case value_type::map:   return tgm_value_map_len(v_);
            default:                return 0;
        }
    }

    // -- array access -----------------------------------------

    /// Index into an array value.
    /// @return The element, or std::nullopt for a non-array or out-of-range
    ///         index (distinguishing "no element" from a present null).
    [[nodiscard]] std::optional<meta_value> at(std::size_t index) const {
        const tgm_value_t* child = tgm_value_array_get(v_, index);
        if (child == nullptr) return std::nullopt;
        return meta_value(child);
    }

    /// Unchecked array index.
    /// @return The element, or an invalid meta_value (valid() == false) for a
    ///         non-array or out-of-range index.  Prefer at() when the index
    ///         may be out of range.
    [[nodiscard]] meta_value operator[](std::size_t index) const {
        return meta_value(tgm_value_array_get(v_, index));
    }

    /// Input iterator over the elements of an array value.
    ///
    /// Iterating a non-array yields nothing (begin() == end()).
    class iterator {
    public:
        using value_type        = meta_value;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void;
        using reference         = meta_value;
        using iterator_category = std::input_iterator_tag;

        [[nodiscard]] meta_value operator*() const {
            return meta_value(tgm_value_array_get(v_, index_));
        }
        iterator& operator++() { ++index_; return *this; }
        iterator operator++(int) { auto tmp = *this; ++index_; return tmp; }
        // Compare the source handle as well as the index, so iterators from two
        // different arrays never compare equal (only same-array iterators are
        // meaningfully comparable).
        [[nodiscard]] bool operator==(const iterator& other) const {
            return v_ == other.v_ && index_ == other.index_;
        }
        [[nodiscard]] bool operator!=(const iterator& other) const { return !(*this == other); }

    private:
        friend class meta_value;
        iterator(const tgm_value_t* v, std::size_t index) : v_(v), index_(index) {}
        const tgm_value_t* v_;
        std::size_t index_;
    };

    [[nodiscard]] iterator begin() const { return iterator(v_, 0); }
    [[nodiscard]] iterator end()   const { return iterator(v_, size()); }

    // -- map access -------------------------------------------

    /// True if a map value has an entry for @p key (single, non-dotted key).
    [[nodiscard]] bool contains(std::string_view key) const {
        const std::string k(key);
        return tgm_value_map_get(v_, k.c_str()) != nullptr;
    }

    /// Look up a single (non-dotted) @p key in a map value.
    ///
    /// Does not split on `.` — use metadata::get() for dotted message-level
    /// paths.  On a section view (from metadata::object()) this can reach
    /// `_reserved_`.
    /// @return The value, or std::nullopt for a non-map or absent key.
    [[nodiscard]] std::optional<meta_value> get(std::string_view key) const {
        const std::string k(key);
        const tgm_value_t* child = tgm_value_map_get(v_, k.c_str());
        if (child == nullptr) return std::nullopt;
        return meta_value(child);
    }

    /// Borrow the @p index-th key of a map value (iteration order).
    /// @return The key view, or std::nullopt for a non-map or out-of-range
    ///         index.
    /// @warning The view must not outlive the parent metadata.
    [[nodiscard]] std::optional<std::string_view> key_at(std::size_t index) const {
        std::size_t len = 0;
        const char* p = tgm_value_map_key_at(v_, index, &len);
        if (p == nullptr) return std::nullopt;
        return std::string_view(p, len);
    }

    /// Borrow the @p index-th value of a map value (same order as key_at()).
    /// @return The value, or std::nullopt for a non-map or out-of-range index.
    [[nodiscard]] std::optional<meta_value> value_at(std::size_t index) const {
        const tgm_value_t* child = tgm_value_map_value_at(v_, index);
        if (child == nullptr) return std::nullopt;
        return meta_value(child);
    }

private:
    friend class metadata;

    explicit meta_value(const tgm_value_t* v) noexcept : v_(v) {}
    const tgm_value_t* v_;
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
///
/// Two access surfaces coexist:
///   * The **precise** cursor API — contains() / get() / try_get_*() and the
///     section views object() / extra() / reserved() returning meta_value —
///     distinguishes absent from wrong-type from a real value equal to a
///     default, and navigates nested maps and arrays without JSON.
///   * The **legacy** default-based getters — get_string() / get_int() /
///     get_float() and the `_at` variants — coerce across types and fold
///     absent into a caller-supplied default.  They are retained unchanged;
///     prefer the precise API for new code.
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
    ///
    /// @note Legacy default-based getter: this coerces int/float/bool leaves
    ///       to text and cannot distinguish an absent key from a stored `""`.
    ///       For new code prefer the precise API — contains(), get() and
    ///       try_get_string() — which reports absence distinctly.
    [[nodiscard]] std::string get_string(std::string_view key) const {
        const std::string k(key);
        const char* s = tgm_metadata_get_string(handle_.get(), k.c_str());
        return s ? s : "";
    }

    /// Look up an integer value by dot-notation key.
    ///
    /// Searches `base` entries first, then `extra`.
    /// @return The value, or @p default_val if not found.
    ///
    /// @note Legacy default-based getter: a returned @p default_val cannot be
    ///       told apart from a stored value equal to it.  For new code prefer
    ///       the precise API — contains() and try_get_int() — which reports
    ///       absence and wrong-type as std::nullopt.
    [[nodiscard]] std::int64_t get_int(std::string_view key, std::int64_t default_val = 0) const {
        const std::string k(key);
        return tgm_metadata_get_int(handle_.get(), k.c_str(), default_val);
    }

    /// Look up a floating-point value by dot-notation key.
    ///
    /// Searches `base` entries first, then `extra`.
    /// @return The value, or @p default_val if not found.
    ///
    /// @note Legacy default-based getter: a returned @p default_val cannot be
    ///       told apart from a stored value equal to it.  For new code prefer
    ///       the precise API — contains() and try_get_double() — which reports
    ///       absence and wrong-type as std::nullopt.
    [[nodiscard]] double get_float(std::string_view key, double default_val = 0.0) const {
        const std::string k(key);
        return tgm_metadata_get_float(handle_.get(), k.c_str(), default_val);
    }

    /// Look up a string value in object @p obj_index by dot-notation key.
    ///
    /// Scoped to `base[obj_index]` only (no cross-object first-match), so a
    /// caller can walk every object of a multi-object message. Dot-paths
    /// navigate nested maps (e.g. "geometry.gridType"); `_reserved_` is skipped
    /// at the first path segment (matching the C FFI and Rust docs).
    /// @return The value as a string, or "" if not found / out of range.
    ///
    /// @note Legacy default-based getter (see get_string()).  For new code
    ///       prefer the precise per-object API — contains_at(), get_at() and
    ///       try_get_*() — plus object() to enumerate a whole `base[i]`.
    [[nodiscard]] std::string get_string_at(std::size_t obj_index, std::string_view key) const {
        const std::string k(key);
        const char* s = tgm_metadata_get_string_at(handle_.get(), obj_index, k.c_str());
        return s ? s : "";
    }

    /// Look up an integer value in object @p obj_index by dot-notation key.
    /// @return The value, or @p default_val if not found / out of range.
    [[nodiscard]] std::int64_t get_int_at(std::size_t obj_index, std::string_view key,
                                          std::int64_t default_val = 0) const {
        const std::string k(key);
        return tgm_metadata_get_int_at(handle_.get(), obj_index, k.c_str(), default_val);
    }

    /// Look up a floating-point value in object @p obj_index by dot-notation key.
    /// @return The value, or @p default_val if not found / out of range.
    [[nodiscard]] double get_float_at(std::size_t obj_index, std::string_view key,
                                      double default_val = 0.0) const {
        const std::string k(key);
        return tgm_metadata_get_float_at(handle_.get(), obj_index, k.c_str(), default_val);
    }

    /// Serialise object @p obj_index's full metadata (`base[obj_index]`) to a
    /// JSON object string — the enumerate-everything companion to the typed
    /// getters, for callers that do not know the key set ahead of time.
    /// @return The JSON text, or "" if @p obj_index is out of range.
    ///
    /// @note This is a serialization/debug helper, **not** an access path.
    ///       To enumerate keys programmatically without parsing JSON, use
    ///       object() and the meta_value map accessors (key_at()/value_at()).
    [[nodiscard]] std::string object_to_json(std::size_t obj_index) const {
        const char* s = tgm_metadata_object_to_json(handle_.get(), obj_index);
        return s ? s : "";
    }

    /// Serialise the whole global metadata to a JSON object string, shaped
    /// `{ "base": [...], "_reserved_": {...}, "_extra_": {...} }`.
    ///
    /// @note Serialization/debug helper, not an access path — use get(),
    ///       contains() and the section views for programmatic access.
    [[nodiscard]] std::string to_json() const {
        const char* s = tgm_metadata_to_json(handle_.get());
        return s ? s : "";
    }

    /// Split this metadata's per-object `base` entries into the keys
    /// common to every object and each object's remaining keys.
    ///
    /// Wraps `tgm_compute_common`, mirroring `tensogram::compute_common`.
    /// Returns UTF-8 JSON of the shape:
    ///
    /// ```json
    /// { "common": { …keys identical across ALL objects… },
    ///   "remaining": [ { …object 0's varying keys… }, … ] }
    /// ```
    ///
    /// `common` holds every key present with an identical value in all
    /// objects; `remaining[i]` holds object `i`'s keys that are not
    /// common.  The library-managed `_reserved_` key is excluded from
    /// both (it varies per object).  With zero objects both are empty.
    ///
    /// @note Serialization helper — the result is intended for a JSON
    ///       reader (`nlohmann::json::parse` or equivalent).  For
    ///       programmatic access to individual keys prefer get() and the
    ///       section views.
    /// @return JSON string with the `{common, remaining}` split.
    /// @throws tensogram::encoding_error If JSON serialization fails.
    [[nodiscard]] std::string compute_common() const {
        tgm_bytes_t bytes{};
        detail::check(tgm_compute_common(handle_.get(), &bytes));
        std::string result(reinterpret_cast<const char*>(bytes.data), bytes.len);
        tgm_bytes_free(bytes);
        return result;
    }

    // -- precise cursor API (message-level) -------------------

    /// True if a message-level dot-notation @p path exists (any type,
    /// including a CBOR null).
    ///
    /// Same scoping as get(): first match across `base` (skipping `_reserved_`
    /// at the first segment), then `_extra_`.  Distinguishes a present `0` /
    /// `""` / null from an absent key.
    [[nodiscard]] bool contains(std::string_view path) const {
        const std::string p(path);
        return tgm_metadata_has(handle_.get(), p.c_str());
    }

    /// True if dot-notation @p path exists within object @p obj (base[obj]).
    ///
    /// Per-object scoping: no cross-object first-match, no `_extra_` fallback,
    /// `_reserved_` hidden at the first segment.
    [[nodiscard]] bool contains_at(std::size_t obj, std::string_view path) const {
        const std::string p(path);
        return tgm_metadata_has_at(handle_.get(), obj, p.c_str());
    }

    /// Look up a message-level dot-notation @p path.
    ///
    /// First match across `base` (skipping `_reserved_` at the first segment),
    /// then `_extra_`.  Presence ⇔ a value, so a stored `0` / `""` / null is
    /// distinguishable from an absent key.
    /// @return A borrowed cursor, or std::nullopt if the path is absent.
    /// @warning The cursor must not outlive this metadata.
    [[nodiscard]] std::optional<meta_value> get(std::string_view path) const {
        const std::string p(path);
        const tgm_value_t* v = tgm_metadata_get(handle_.get(), p.c_str());
        if (v == nullptr) return std::nullopt;
        return meta_value(v);
    }

    /// Look up a dot-notation @p path scoped to object @p obj (base[obj]).
    ///
    /// Per-object scoping (see contains_at()).
    /// @return A borrowed cursor, or std::nullopt if absent / out of range.
    /// @warning The cursor must not outlive this metadata.
    [[nodiscard]] std::optional<meta_value> get_at(std::size_t obj, std::string_view path) const {
        const std::string p(path);
        const tgm_value_t* v = tgm_metadata_get_at(handle_.get(), obj, p.c_str());
        if (v == nullptr) return std::nullopt;
        return meta_value(v);
    }

    /// Convenience: message-level lookup extracted as a signed integer.
    /// @return The value, or std::nullopt if absent or not an integer.
    [[nodiscard]] std::optional<std::int64_t> try_get_int(std::string_view path) const {
        if (const auto v = get(path)) return v->as_int();
        return std::nullopt;
    }

    /// Convenience: message-level lookup extracted as an unsigned integer.
    /// @return The value, or std::nullopt if absent or not an integer.
    [[nodiscard]] std::optional<std::uint64_t> try_get_uint(std::string_view path) const {
        if (const auto v = get(path)) return v->as_uint();
        return std::nullopt;
    }

    /// Convenience: message-level lookup extracted as a double (integers
    /// widen).
    /// @return The value, or std::nullopt if absent or not numeric.
    [[nodiscard]] std::optional<double> try_get_double(std::string_view path) const {
        if (const auto v = get(path)) return v->as_double();
        return std::nullopt;
    }

    /// Convenience: message-level lookup extracted as a string view.
    /// @return A borrowed view, or std::nullopt if absent or not a string.
    /// @warning The view must not outlive this metadata.
    [[nodiscard]] std::optional<std::string_view> try_get_string(std::string_view path) const {
        if (const auto v = get(path)) return v->as_string();
        return std::nullopt;
    }

    /// Convenience: message-level lookup extracted as a bool.
    /// @return The value, or std::nullopt if absent or not a bool.
    [[nodiscard]] std::optional<bool> try_get_bool(std::string_view path) const {
        if (const auto v = get(path)) return v->as_bool();
        return std::nullopt;
    }

    // -- section views (for enumeration) ----------------------

    /// View object @p obj's full metadata map (`base[obj]`) for enumeration.
    ///
    /// **Includes** `_reserved_` (parity with the JSON export), unlike the
    /// path getters which hide it.  Enumerate via meta_value::size() /
    /// key_at() / value_at(), or look up a key with meta_value::get().
    /// @return A map cursor, or an invalid cursor (valid() == false) if @p obj
    ///         is out of range.
    [[nodiscard]] meta_value object(std::size_t obj) const {
        return meta_value(tgm_metadata_object(handle_.get(), obj));
    }

    /// View the message-level `_extra_` section as a map for enumeration.
    /// @return A map cursor; an empty section is an empty map (size() == 0),
    ///         still valid().
    [[nodiscard]] meta_value extra() const {
        return meta_value(tgm_metadata_extra(handle_.get()));
    }

    /// View the message-level `_reserved_` section (library-managed keys such
    /// as tensor descriptors) as a map for enumeration.
    /// @return A map cursor; an empty section is an empty map (size() == 0),
    ///         still valid().
    [[nodiscard]] meta_value reserved() const {
        return meta_value(tgm_metadata_reserved(handle_.get()));
    }

private:
    friend metadata tensogram::decode_metadata(const std::uint8_t*, std::size_t);
    friend class message;
    // Async frontends construct a metadata wrapper from an FFI handle
    // returned by tgm_async_task_join_metadata.
    friend metadata tensogram::detail::metadata_from_raw(tgm_metadata_t*);

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
// Remote sources — is_remote_url / remote_scan_options
// ============================================================

/// Reader-side scan-walker options for file::open_remote().
///
/// Mirrors the C `TgmRemoteScanOptions` POD and
/// `tensogram::RemoteScanOptions`.  A default-constructed
/// remote_scan_options requests the library defaults.
struct remote_scan_options {
    /// Enable the meet-in-the-middle (bidirectional) remote walk, which
    /// pairs forward preamble fetches with backward postamble fetches and
    /// roughly halves wall-clock layout discovery on real networks.
    /// `false` forces a forward-only walk.
    bool bidirectional = true;
};

namespace detail {

/// Reject a string that cannot survive the trip through a C string.
///
/// The C ABI takes NUL-terminated `const char*`, so an interior NUL would
/// silently truncate the value — opening a *different* object than the
/// caller named.  std::string / std::string_view can carry one, so the
/// wrapper checks rather than letting the truncation happen.
inline void require_nul_free(std::string_view value, const char* what) {
    if (value.find('\0') != std::string_view::npos) {
        throw invalid_arg_error(TGM_ERROR_INVALID_ARG,
                                std::string(what) + " must not contain an embedded NUL");
    }
}

}  // namespace detail

/// True when @p source is a URL this build can open remotely.
///
/// Wraps `tgm_is_remote_url`, binding `tensogram::is_remote_url`.  The
/// recognised schemes are `s3`, `s3a`, `gs`, `az`, `azure`, `http` and
/// `https`, compared case-insensitively.  Plain paths and `file://` URLs are
/// **not** remote — they belong to the local backend (file::open()).
///
/// Returns false for **every** input when the C API was built without the
/// `remote` Cargo feature: such a build genuinely cannot open any remote
/// URL, so "not remote for me" is the honest answer.
///
/// @return true if @p source names a remote object this build can open.
///         Never throws — a source with an embedded NUL is simply not a
///         remote URL.
[[nodiscard]] inline bool is_remote_url(std::string_view source) {
    if (source.find('\0') != std::string_view::npos) return false;
    const std::string src(source);
    return tgm_is_remote_url(src.c_str());
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

    /// Open a remote `.tgm` (S3 / GCS / Azure / HTTP) for **synchronous**
    /// reading.
    ///
    /// Wraps `tgm_file_open_remote`.  The result is an ordinary file, so the
    /// whole existing API — message_count(), read_message(),
    /// decode_message(), file_iterator, … — works unchanged against the
    /// remote source, and the handle closes the same way.
    ///
    /// @param source          Remote URL (see is_remote_url() for the
    ///                        recognised schemes).
    /// @param storage_options Backend options (credentials, region,
    ///                        endpoint, …) forwarded verbatim to the
    ///                        object-store backend.
    /// @param opts            Scan-walker options; the default requests the
    ///                        bidirectional walk.
    /// @throws tensogram::invalid_arg_error If @p source or any storage
    ///         option contains an embedded NUL (it could not be passed to
    ///         the C ABI without silently truncating).
    /// @throws tensogram::remote_error If the URL is unparseable, the object
    ///         is missing, a storage option is rejected, the transport
    ///         fails — or the C API was built without the `remote` Cargo
    ///         feature, in which case the message says how to enable it.
    [[nodiscard]] static file open_remote(
        std::string_view source,
        const std::map<std::string, std::string>& storage_options = {},
        remote_scan_options opts = {})
    {
        detail::require_nul_free(source, "remote source");
        const std::string source_str(source);

        // The C call takes parallel key / value arrays of `const char*`.
        // The map owns the strings, so the pointers stay valid for the call.
        std::vector<const char*> keys;
        std::vector<const char*> values;
        keys.reserve(storage_options.size());
        values.reserve(storage_options.size());
        for (const auto& [key, value] : storage_options) {
            detail::require_nul_free(key, "storage option key");
            detail::require_nul_free(value, "storage option value");
            keys.push_back(key.c_str());
            values.push_back(value.c_str());
        }

        TgmRemoteScanOptions c_opts{};
        c_opts.bidirectional = opts.bidirectional;

        tgm_file_t* raw = nullptr;
        detail::check(tgm_file_open_remote(
            source_str.c_str(),
            keys.empty() ? nullptr : keys.data(),
            values.empty() ? nullptr : values.data(),
            keys.size(), &c_opts, &raw));
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
            handle_.get(), index,
            opts.native_byte_order ? 1 : 0, opts.threads,
            opts.verify_hash ? 1 : 0,
            &raw));
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
        auto mask_holder = detail::build_mask_opts(opts);
        if (detail::needs_encode_options(opts)) {
            const TgmEncodeOptions c_opts = detail::build_encode_opts(opts);
            detail::check(tgm_file_append_with_encode_options(
                handle_.get(), metadata_json.c_str(),
                sg.ptrs.data(), sg.lens.data(),
                objects.size(), opts.threads, &c_opts));
        } else if (mask_holder.active) {
            detail::check(tgm_file_append_with_options(
                handle_.get(), metadata_json.c_str(),
                sg.ptrs.data(), sg.lens.data(),
                objects.size(), hash, opts.threads,
                &mask_holder.value));
        } else {
            detail::check(tgm_file_append(handle_.get(), metadata_json.c_str(),
                                           sg.ptrs.data(), sg.lens.data(),
                                           objects.size(), hash, opts.threads));
        }
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
            buf, len,
            opts.native_byte_order ? 1 : 0,
            opts.verify_hash ? 1 : 0,
            &raw));
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
// Frame walker — frame_type / frame / frame_range
// ============================================================

/// The kind of a wire frame, mirroring the C `tgm_frame_type` enum.
///
/// The enumerator values **are** the wire's frame-type numbers (see
/// `plans/WIRE_FORMAT.md` §4), not a binding invention.  Type 4 is
/// reserved — it held the obsolete v2 data-object layout — and therefore
/// has no enumerator, which is why the sequence skips from 3 to 5.
enum class frame_type : std::uint16_t {
    /// CBOR global metadata, written in the header (random-access mode).
    header_metadata   = TGM_FRAME_TYPE_HEADER_METADATA,
    /// Object index, written in the header (random-access mode).
    header_index      = TGM_FRAME_TYPE_HEADER_INDEX,
    /// Aggregate hash frame, written in the header.
    header_hash       = TGM_FRAME_TYPE_HEADER_HASH,
    /// Aggregate hash frame, written in the footer.
    footer_hash       = TGM_FRAME_TYPE_FOOTER_HASH,
    /// Object index, written in the footer (streaming mode).
    footer_index      = TGM_FRAME_TYPE_FOOTER_INDEX,
    /// CBOR global metadata, written in the footer (streaming mode).
    footer_metadata   = TGM_FRAME_TYPE_FOOTER_METADATA,
    /// Per-object metadata frame immediately preceding a data-object frame.
    preceder_metadata = TGM_FRAME_TYPE_PRECEDER_METADATA,
    /// N-dimensional tensor data-object frame — the only data-object type
    /// in v3, and the only frame type with a 20-byte footer.
    ntensor           = TGM_FRAME_TYPE_NTENSOR,
};

namespace detail {

/// Map the C `tgm_frame_type` discriminant to the C++ frame_type enum.
///
/// The two enums carry identical values (both are the wire number), so this
/// is an exact, total conversion rather than a lookup table — an unknown
/// discriminant from a future ABI is preserved as its number instead of
/// being silently folded into a known variant.
[[nodiscard]] inline frame_type frame_type_from_c(tgm_frame_type t) noexcept {
    return static_cast<frame_type>(static_cast<std::uint16_t>(t));
}

}  // namespace detail

/// One frame's structural description plus a **borrowed** view of its
/// content — the C++ face of the C `TgmFrame` POD.
///
/// A frame is a small value type (copyable, cheap) yielded by frame_range.
///
/// @warning Lifetime: payload() / payload_data() point **into the message
///          buffer** passed to frames() — a view, never a copy, with nothing
///          to free.  They stay valid for exactly as long as that buffer
///          does, independently of the range that produced them: destroying
///          the frame_range does **not** invalidate them.  Copy the bytes
///          out if you need them to outlive the message buffer.
class frame {
public:
    /// Which kind of frame this is.
    [[nodiscard]] frame_type type() const noexcept {
        return detail::frame_type_from_c(raw_.frame_type);
    }

    /// Frame-type-specific version field from the frame header.
    [[nodiscard]] std::uint16_t version() const noexcept { return raw_.version; }

    /// Raw 16-bit frame flags; bit 1 is `HASH_PRESENT` (see has_hash()).
    [[nodiscard]] std::uint16_t flags() const noexcept { return raw_.flags; }

    /// Byte offset of the frame header, relative to the start of the message.
    [[nodiscard]] std::size_t offset() const noexcept { return raw_.offset; }

    /// Whole-frame span in bytes: frame header through `ENDF`, excluding any
    /// alignment padding that follows.
    [[nodiscard]] std::size_t length() const noexcept { return raw_.length; }

    /// Borrowed pointer to the content bytes — everything between the
    /// 16-byte frame header and the type-specific footer (20 bytes for
    /// frame_type::ntensor, 12 for every other type).
    /// @warning Borrows the caller's message buffer; see the class warning.
    [[nodiscard]] const std::uint8_t* payload_data() const noexcept { return raw_.payload; }

    /// Length of the borrowed content bytes.
    [[nodiscard]] std::size_t payload_size() const noexcept { return raw_.payload_len; }

    /// The content bytes as a std::string_view (a byte view — the payload is
    /// binary CBOR or tensor data, not text).
    ///
    /// The project targets C++17, so this is the borrowing byte-range type
    /// available; it is built from `ptr + len` and never `strlen`, so interior
    /// NUL bytes are preserved.
    /// @warning Borrows the caller's message buffer; see the class warning.
    [[nodiscard]] std::string_view payload() const noexcept {
        if (raw_.payload == nullptr) return {};
        return std::string_view(reinterpret_cast<const char*>(raw_.payload),
                                raw_.payload_len);
    }

    /// True if this frame's `HASH_PRESENT` flag is set, i.e. its inline hash
    /// slot holds a meaningful digest (`plans/WIRE_FORMAT.md` §2.5).
    ///
    /// This is the authoritative answer for a single frame —
    /// message_header::has_hashes_present() is only an advisory
    /// message-wide summary.
    [[nodiscard]] bool has_hash() const noexcept { return tgm_frame_has_hash(&raw_); }

    /// True if this frame carries a data object (frame_type::ntensor is the
    /// only data-object type in v3).
    [[nodiscard]] bool is_data_object() const noexcept {
        return type() == frame_type::ntensor;
    }

private:
    friend class frame_range;

    frame() = default;
    explicit frame(const TgmFrame& raw) noexcept : raw_(raw) {}
    TgmFrame raw_{};
};

/// A lazy, single-pass range over the frames of one message.
///
/// Returned by frames().  Owns the underlying C cursor (RAII, move-only) and
/// pulls exactly one frame per `++` — nothing is materialised up front, so a
/// caller that only needs the first frames never pays for the rest.
///
/// Supports range-for:
/// @code
/// for (const auto& f : tensogram::frames(msg.data(), msg.size())) {
///     if (f.is_data_object()) { ... }
/// }
/// @endcode
///
/// @warning **Lifetime**: the range borrows the message buffer — the buffer
///          must outlive the range and must not be moved, reallocated, or
///          mutated while the range lives.  Frames yielded by the walk keep
///          pointing into that buffer and stay valid after the range is
///          destroyed (see the frame class documentation).
///
/// @note **End vs malformed**: the underlying cursor stops for two different
///       reasons.  A clean end simply terminates the loop; a truncated or
///       inconsistent frame chain throws tensogram::framing_error from the
///       increment that discovers it, after the intact frames have been
///       yielded.  A buffer whose preamble does not parse at all throws from
///       frames() itself.
class frame_range {
public:
    /// End-of-walk marker.  C++17 permits a range-for `end()` of a different
    /// type from `begin()`, which lets the cursor answer "am I done?" from
    /// its own state instead of synthesising a comparable end iterator.
    struct sentinel {};

    /// Input iterator over the frames of a message.
    ///
    /// Single-pass by construction: every copy shares the one C cursor owned
    /// by the frame_range, exactly like std::istream_iterator.
    class iterator {
    public:
        using value_type        = frame;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const frame*;
        using reference         = const frame&;
        using iterator_category = std::input_iterator_tag;

        /// Construct an exhausted iterator (compares equal to any sentinel).
        iterator() = default;

        [[nodiscard]] const frame& operator*() const noexcept { return current_; }
        [[nodiscard]] const frame* operator->() const noexcept { return &current_; }

        /// Pull the next frame.
        /// @throws tensogram::framing_error If the frame chain is malformed.
        iterator& operator++() { advance(); return *this; }

        /// Post-increment returns void, as C++20's `std::input_iterator`
        /// permits: every copy shares the one C cursor, so a returned copy
        /// could not be advanced independently of this one.  Use the
        /// pre-increment form.
        /// @throws tensogram::framing_error If the frame chain is malformed.
        void operator++(int) { advance(); }

        [[nodiscard]] bool operator==(sentinel) const noexcept { return done_; }
        [[nodiscard]] bool operator!=(sentinel) const noexcept { return !done_; }
        [[nodiscard]] friend bool operator==(sentinel s, const iterator& it) noexcept {
            return it == s;
        }
        [[nodiscard]] friend bool operator!=(sentinel s, const iterator& it) noexcept {
            return it != s;
        }
        /// Two iterators are equal when they share a cursor and its state.
        [[nodiscard]] bool operator==(const iterator& other) const noexcept {
            return it_ == other.it_ && done_ == other.done_;
        }
        [[nodiscard]] bool operator!=(const iterator& other) const noexcept {
            return !(*this == other);
        }

    private:
        friend class frame_range;

        explicit iterator(tgm_frame_iter_t* it) : it_(it), done_(false) { advance(); }

        /// Pull one frame, translating the C cursor's single `false` return
        /// into the two outcomes it actually stands for.
        ///
        /// `tgm_frame_iter_next` clears the thread-local last error on a
        /// clean exhaustion and sets it on a malformed frame, so the error
        /// slot — not the return value — is what tells the two apart.
        void advance() {
            if (done_ || it_ == nullptr) {
                done_ = true;
                return;
            }
            TgmFrame raw{};
            if (tgm_frame_iter_next(it_, &raw)) {
                current_ = frame(raw);
                return;
            }
            done_ = true;
            current_ = frame();
            if (const char* msg = tgm_last_error()) {
                throw framing_error(TGM_ERROR_FRAMING, msg);
            }
        }

        /// Borrowed from the owning frame_range — the cursor's address is
        /// stable across a move of the range, so iterators survive it.
        tgm_frame_iter_t* it_ = nullptr;
        frame current_{};
        bool done_ = true;
    };

    /// Start a walk over the frames of the message in `[msg, msg + len)`.
    ///
    /// @throws tensogram::invalid_arg_error If @p msg is null.
    /// @throws tensogram::framing_error If the preamble does not parse
    ///         (truncated buffer, wrong magic, unsupported version).
    frame_range(const std::uint8_t* msg, std::size_t len) {
        if (msg == nullptr) {
            throw invalid_arg_error(TGM_ERROR_INVALID_ARG,
                                    "frames: message buffer is null");
        }
        tgm_frame_iter_t* raw = tgm_frame_iter_create(msg, len);
        if (raw == nullptr) {
            const char* err = tgm_last_error();
            throw framing_error(TGM_ERROR_FRAMING,
                                err ? err : "message preamble does not parse");
        }
        handle_.reset(raw);
    }

    frame_range(frame_range&&) noexcept = default;
    frame_range& operator=(frame_range&&) noexcept = default;
    ~frame_range() = default;

    frame_range(const frame_range&) = delete;
    frame_range& operator=(const frame_range&) = delete;

    /// Begin the walk, pulling the first frame.
    ///
    /// Single-pass: calling begin() a second time resumes where the previous
    /// iterator stopped rather than restarting the walk.
    /// @throws tensogram::framing_error If the first frame is malformed.
    [[nodiscard]] iterator begin() { return iterator(handle_.get()); }

    /// The end-of-walk sentinel.
    [[nodiscard]] sentinel end() const noexcept { return sentinel{}; }

private:
    struct deleter {
        void operator()(tgm_frame_iter_t* p) const noexcept { tgm_frame_iter_free(p); }
    };
    std::unique_ptr<tgm_frame_iter_t, deleter> handle_;
};

/// Walk the frames of a single message, lazily.
///
/// Wraps `tgm_frame_iter_create`, binding `tensogram::frames`.  @p msg must
/// point at the start of a message (the `TENSOGRM` preamble magic) —
/// typically a slice located with scan().  Only the type 1–9 `FR` frames are
/// yielded; the preamble and postamble are not frames, use
/// read_message_header() for the envelope.
///
/// @param msg  Wire-format bytes of a single message.
/// @param len  Length of @p msg.
/// @return A lazy range; see frame_range for the lifetime and
///         end-vs-malformed contracts.
/// @throws tensogram::invalid_arg_error If @p msg is null.
/// @throws tensogram::framing_error If the preamble does not parse.
[[nodiscard]] inline frame_range frames(const std::uint8_t* msg, std::size_t len) {
    return frame_range(msg, len);
}

// ============================================================
// message_header — the message envelope, without walking frames
// ============================================================

class message_header;
[[nodiscard]] inline message_header read_message_header(const std::uint8_t* msg,
                                                        std::size_t len);

/// A message's envelope (the 24-byte preamble) as a value type.
///
/// Produced by read_message_header().  The eight `has_*` accessors are the
/// preamble's structural flags decoded into named predicates, so callers
/// never touch a raw bitset.  Together they say whether a message is
/// *random-access* (metadata / index / hashes in the **header**) or
/// *streaming* (in the **footer**) without reading a single frame.
class message_header {
public:
    /// Wire-format version (tensogram::wire_version for messages this build
    /// writes).
    [[nodiscard]] std::uint16_t version() const noexcept { return raw_.version; }

    /// Total message length in bytes, preamble through postamble, or `0` if
    /// a streaming writer never back-filled it.  Zero is not an error — it
    /// means "unknown at write time".
    [[nodiscard]] std::uint64_t total_length() const noexcept { return raw_.total_length; }

    /// A frame_type::header_metadata frame is present (random-access mode).
    [[nodiscard]] bool has_header_metadata() const noexcept {
        return raw_.has_header_metadata;
    }
    /// A frame_type::footer_metadata frame is present (streaming mode).
    [[nodiscard]] bool has_footer_metadata() const noexcept {
        return raw_.has_footer_metadata;
    }
    /// A frame_type::header_index frame is present.
    [[nodiscard]] bool has_header_index() const noexcept { return raw_.has_header_index; }
    /// A frame_type::footer_index frame is present.
    [[nodiscard]] bool has_footer_index() const noexcept { return raw_.has_footer_index; }
    /// A frame_type::header_hash frame is present.
    [[nodiscard]] bool has_header_hashes() const noexcept { return raw_.has_header_hashes; }
    /// A frame_type::footer_hash frame is present.
    [[nodiscard]] bool has_footer_hashes() const noexcept { return raw_.has_footer_hashes; }
    /// At least one frame_type::preceder_metadata frame appears in the body.
    ///
    /// **Advisory in streaming mode**: the encoder writes the preamble before
    /// it knows whether any preceder will follow, so `true` does not
    /// guarantee a frame.  Only "frame present ⇒ flag set" holds there; in
    /// buffered mode the flag is exact.
    [[nodiscard]] bool has_preceder_metadata() const noexcept {
        return raw_.has_preceder_metadata;
    }
    /// Advisory: every frame in this message has its per-frame
    /// `HASH_PRESENT` bit set.  For any single frame, frame::has_hash()
    /// stays authoritative.
    [[nodiscard]] bool has_hashes_present() const noexcept {
        return raw_.has_hashes_present;
    }

private:
    friend message_header tensogram::read_message_header(const std::uint8_t*, std::size_t);

    message_header() = default;
    TgmMessageHeader raw_{};
};

/// Read a message's envelope without walking its frames.
///
/// Wraps `tgm_message_header`, binding `tensogram::message_header`.  @p msg
/// must point at the start of a message (the `TENSOGRM` preamble magic) —
/// typically a slice located with scan().
///
/// @param msg  Wire-format bytes of a single message.
/// @param len  Length of @p msg.
/// @return The decoded envelope.
/// @throws tensogram::invalid_arg_error If @p msg is null.
/// @throws tensogram::framing_error If the preamble does not parse
///         (truncated message, wrong magic, unsupported version).
[[nodiscard]] inline message_header read_message_header(const std::uint8_t* msg,
                                                        std::size_t len) {
    message_header header;
    detail::check(tgm_message_header(msg, len, &header.raw_));
    return header;
}

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
    /// @throws tensogram::encoding_error If `opts.aggregate_hash` asks for a
    ///         header-side placement (aggregate_hash_policy::header or
    ///         ::both): a streaming writer emits its header before any data
    ///         object exists, so the per-object hashes are not yet known.
    ///         Use aggregate_hash_policy::automatic (which resolves to the
    ///         footer when streaming) or ::footer.
    streaming_encoder(const std::string& path,
                      const std::string& metadata_json,
                      const encode_options& opts = {}) {
        tgm_streaming_encoder_t* raw = nullptr;
        const char* hash = opts.hash_algo.empty() ? nullptr
                                                    : opts.hash_algo.c_str();
        auto mask_holder = detail::build_mask_opts(opts);
        if (detail::needs_encode_options(opts)) {
            const TgmEncodeOptions c_opts = detail::build_encode_opts(opts);
            detail::check(tgm_streaming_encoder_create_with_encode_options(
                path.c_str(), metadata_json.c_str(), opts.threads, &c_opts, &raw));
        } else if (mask_holder.active) {
            detail::check(tgm_streaming_encoder_create_with_options(
                path.c_str(), metadata_json.c_str(), hash, opts.threads,
                &mask_holder.value, &raw));
        } else {
            detail::check(tgm_streaming_encoder_create(
                path.c_str(), metadata_json.c_str(), hash, opts.threads, &raw));
        }
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
    auto mask_holder = detail::build_mask_opts(opts);
    const TgmEncodeMaskOptions* mask_ptr = mask_holder.active ? &mask_holder.value : nullptr;
    if (detail::needs_encode_options(opts)) {
        // Aggregate-hash placement / codec backend live only in the full
        // option struct, which also carries the hash algorithm and the mask
        // policy — so there is no separate `hash_algo` argument here.
        const TgmEncodeOptions c_opts = detail::build_encode_opts(opts);
        detail::check(tgm_encode_with_encode_options(metadata_json.c_str(),
                                                     sg.ptrs.data(), sg.lens.data(),
                                                     objects.size(), opts.threads,
                                                     &c_opts, &bytes));
    } else if (mask_ptr != nullptr) {
        detail::check(tgm_encode_with_options(metadata_json.c_str(),
                                              sg.ptrs.data(), sg.lens.data(), objects.size(),
                                              hash, opts.threads,
                                              mask_ptr,
                                              &bytes));
    } else {
        detail::check(tgm_encode(metadata_json.c_str(),
                                 sg.ptrs.data(), sg.lens.data(), objects.size(),
                                 hash, opts.threads,
                                 &bytes));
    }
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
/// @param opts           Encoding options (hash algorithm, threads).  The
///                       mask fields do not apply to opaque pre-encoded
///                       bytes and are ignored.
/// @return The encoded message as a byte vector.
/// @throws tensogram::invalid_arg_error If `opts.aggregate_hash` or
///         `opts.codec_backend` is set to a non-default value: the C ABI
///         has no pre-encoded entry point that carries them, so honouring
///         them here is impossible and ignoring them would emit a message
///         that contradicts the request.  Use encode() for those.
[[nodiscard]] inline std::vector<std::uint8_t> encode_pre_encoded(
    const std::string& metadata_json,
    const std::vector<std::pair<const std::uint8_t*, std::size_t>>& objects,
    const encode_options& opts = {})
{
    // The mask fields are raw-input-only — pre-encoded bytes are opaque to
    // the library and cannot be meaningfully scanned for NaN / Inf — and
    // `tgm_encode_pre_encoded` accordingly takes no mask options, so they
    // are inapplicable here.
    //
    // The aggregate-hash placement and the codec backend are different:
    // they describe the *message* the encoder writes, not the payload
    // bytes, yet the C ABI has no pre-encoded entry point that carries
    // them.  Accepting them would emit a message that contradicts what the
    // caller asked for, so the mismatch is reported instead of dropped.
    if (detail::needs_encode_options(opts)) {
        throw invalid_arg_error(
            TGM_ERROR_INVALID_ARG,
            "encode_pre_encoded: encode_options::aggregate_hash and "
            "::codec_backend are not supported on the pre-encoded path "
            "(the C ABI has no pre-encoded entry point that carries them); "
            "use encode() for those, or leave both at their default");
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
    // Pass the restore_non_finite flag via the TgmDecodeMaskOptions
    // variant only when the caller asked for a non-default value.
    // Default-constructed decode_options leaves restore_non_finite =
    // true, which matches the library default; in that case we use
    // the legacy entry point.
    if (opts.restore_non_finite) {
        detail::check(tgm_decode(buf, len,
                                  opts.native_byte_order ? 1 : 0,
                                  opts.threads,
                                  opts.verify_hash ? 1 : 0,
                                  &raw));
    } else {
        TgmDecodeMaskOptions mask_opts{};
        mask_opts.restore_non_finite = false;
        detail::check(tgm_decode_with_options(
            buf, len,
            opts.native_byte_order ? 1 : 0, opts.threads,
            opts.verify_hash ? 1 : 0,
            &mask_opts, &raw));
    }
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

/// Decode a complete message AND retain the raw per-object NaN / ±Inf
/// bitmasks (advanced entry point).
///
/// Wraps `tgm_decode_with_masks`, binding `tensogram::decode_with_masks`.
/// Behaves like decode() for every existing accessor with one deliberate
/// difference: restoration is **not** applied, so each object's payload
/// holds the canonical `0.0` at every position that was non-finite at
/// encode time.  The original NaN / `+∞` / `−∞` positions are recoverable
/// through decoded_object::has_masks() and decoded_object::mask().
///
/// Use this when you want to map missing values to your own sentinel,
/// aggregate missing-count statistics, or apply the masks lazily.
/// Callers that just want the canonical NaN / Inf bytes restored in place
/// should use decode() (or decode() with `decode_options`).
///
/// @note `decode_options::restore_non_finite` is ignored here — this
///       entry point always retains masks without restoring in place.
///       `native_byte_order`, `threads`, and `verify_hash` are honoured.
[[nodiscard]] inline message decode_with_masks(const std::uint8_t* buf, std::size_t len,
                             const decode_options& opts)
{
    tgm_message_t* raw = nullptr;
    detail::check(tgm_decode_with_masks(buf, len,
                                        opts.native_byte_order ? 1 : 0,
                                        opts.threads,
                                        opts.verify_hash ? 1 : 0,
                                        &raw));
    return message(raw);
}

/// Decode a single object by index.
[[nodiscard]] inline message decode_object(const std::uint8_t* buf, std::size_t len,
                             std::size_t index, const decode_options& opts)
{
    tgm_message_t* raw = nullptr;
    detail::check(tgm_decode_object(buf, len, index,
                                     opts.native_byte_order ? 1 : 0,
                                     opts.threads,
                                     opts.verify_hash ? 1 : 0,
                                     &raw));
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

namespace detail {

/// Drain a `tgm_scan_result_t` into a vector of scan_entry and free the
/// handle.  Shared by the scan_file / *_with_options scanners so each
/// entry point only has to build its options and pick the C call.
[[nodiscard]] inline std::vector<scan_entry> collect_scan_result(tgm_scan_result_t* raw) {
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

/// Convert a C++ scan_options into the C `TgmScanOptions` POD.
[[nodiscard]] inline TgmScanOptions to_c_scan_options(const scan_options& opts) noexcept {
    TgmScanOptions c_opts{};
    c_opts.bidirectional = opts.bidirectional;
    c_opts.max_message_size = opts.max_message_size;
    return c_opts;
}

}  // namespace detail

/// Scan a buffer for message boundaries.
[[nodiscard]] inline std::vector<scan_entry> scan(const std::uint8_t* buf, std::size_t len) {
    tgm_scan_result_t* raw = nullptr;
    detail::check(tgm_scan(buf, len, &raw));
    return detail::collect_scan_result(raw);
}

/// Scan a buffer for message boundaries with explicit options.
///
/// Wraps `tgm_scan_with_options`.  Like scan() but takes a scan_options
/// controlling the bidirectional walk and the message-size cap.
[[nodiscard]] inline std::vector<scan_entry> scan_with_options(
    const std::uint8_t* buf, std::size_t len, const scan_options& opts)
{
    const TgmScanOptions c_opts = detail::to_c_scan_options(opts);
    tgm_scan_result_t* raw = nullptr;
    detail::check(tgm_scan_with_options(buf, len, &c_opts, &raw));
    return detail::collect_scan_result(raw);
}

/// Scan a `.tgm` file for message boundaries using streaming seek-and-read
/// I/O — the whole file is never loaded into memory.
///
/// Wraps `tgm_scan_file`.  The file-based counterpart to scan().
///
/// @param path  Path to the `.tgm` file.
/// @return The `(offset, length)` of every message found.
/// @throws tensogram::io_error If the file cannot be opened, seeked, or read.
/// @throws tensogram::invalid_arg_error For a non-UTF-8 path.
[[nodiscard]] inline std::vector<scan_entry> scan_file(std::string_view path) {
    const std::string path_str(path);
    tgm_scan_result_t* raw = nullptr;
    detail::check(tgm_scan_file(path_str.c_str(), &raw));
    return detail::collect_scan_result(raw);
}

/// Scan a `.tgm` file for message boundaries with explicit options.
///
/// Wraps `tgm_scan_file_with_options`.  Like scan_file() but takes a
/// scan_options.
///
/// @param path  Path to the `.tgm` file.
/// @param opts  Scan options (bidirectional walk, message-size cap).
/// @throws tensogram::io_error If the file cannot be opened, seeked, or read.
/// @throws tensogram::invalid_arg_error For a non-UTF-8 path.
[[nodiscard]] inline std::vector<scan_entry> scan_file_with_options(
    std::string_view path, const scan_options& opts)
{
    const std::string path_str(path);
    const TgmScanOptions c_opts = detail::to_c_scan_options(opts);
    tgm_scan_result_t* raw = nullptr;
    detail::check(tgm_scan_file_with_options(path_str.c_str(), &c_opts, &raw));
    return detail::collect_scan_result(raw);
}

namespace detail {

/// Parse the constrained JSON array emitted by
/// `tgm_data_object_inline_hashes` into one std::optional per element.
///
/// The array is flat and its elements are either a JSON string (a
/// 16-char lowercase-hex xxh3-64 digest) or JSON `null` (a zero / unhashed
/// slot).  The digests never contain escapes or structural characters, so
/// a minimal scanner is sufficient and deliberately avoids pulling in a
/// full JSON dependency — the binding stays header-only and dependency
/// free.  On any malformed input the scan stops early and returns what it
/// parsed so far (the C producer only ever emits the well-formed shape).
[[nodiscard]] inline std::vector<std::optional<std::string>>
parse_inline_hashes_json(std::string_view json) {
    std::vector<std::optional<std::string>> out;
    std::size_t i = 0;
    const std::size_t n = json.size();
    const auto skip_ws = [&] {
        while (i < n && (json[i] == ' ' || json[i] == '\t' ||
                         json[i] == '\n' || json[i] == '\r')) {
            ++i;
        }
    };
    skip_ws();
    if (i >= n || json[i] != '[') return out;  // not an array
    ++i;                                        // consume '['
    skip_ws();
    if (i < n && json[i] == ']') return out;    // empty array
    while (i < n) {
        skip_ws();
        if (i >= n) break;
        if (json[i] == '"') {
            ++i;  // opening quote
            std::string digest;
            while (i < n && json[i] != '"') {
                digest.push_back(json[i]);
                ++i;
            }
            if (i < n) ++i;  // closing quote
            out.emplace_back(std::move(digest));
        } else if (json.compare(i, 4, "null") == 0) {
            i += 4;
            out.emplace_back(std::nullopt);
        } else {
            break;  // malformed — stop defensively
        }
        skip_ws();
        if (i < n && json[i] == ',') { ++i; continue; }
        break;  // ']' or end of input
    }
    return out;
}

}  // namespace detail

/// Extract every data-object frame's inline hash slot from a single wire
/// message, without a full CBOR decode.
///
/// Wraps `tgm_data_object_inline_hashes`, binding
/// `tensogram::data_object_inline_hashes`.  `buf` must hold a **single**
/// message; for a multi-message buffer, locate boundaries with scan()
/// first and pass each message slice.
///
/// Returns one entry per data object in frame-emission order: the
/// populated slot's 16-char lowercase-hex xxh3-64 digest, or std::nullopt
/// when the slot is zero (message-wide `HASHES_PRESENT = 0`, or the frame
/// was not hashed).  This is the parsed form of the JSON array the C ABI
/// returns (e.g. `["1122334455667788",null]`), chosen over the raw JSON
/// so callers can consume the digests directly without a JSON reader —
/// mirroring Python's `object_inline_hashes` and the TS
/// `dataObjectInlineHashes`.
///
/// @param buf  Wire-format bytes of a single message.
/// @param len  Length of @p buf.
/// @throws tensogram::framing_error If @p buf is not a parseable message.
/// @throws tensogram::invalid_arg_error For null pointers.
/// @throws tensogram::encoding_error If the C JSON serialization fails.
[[nodiscard]] inline std::vector<std::optional<std::string>> data_object_inline_hashes(
    const std::uint8_t* buf, std::size_t len)
{
    tgm_bytes_t bytes{};
    detail::check(tgm_data_object_inline_hashes(buf, len, &bytes));
    const std::string_view json(reinterpret_cast<const char*>(bytes.data), bytes.len);
    auto result = detail::parse_inline_hashes_json(json);
    tgm_bytes_free(bytes);
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

/// Compute simple_packing (lossy quantisation) parameters for a set of
/// f64 sample values.
///
/// Wraps `tgm_simple_packing_compute_params`: given the samples, the
/// target `bits_per_value`, and a `decimal_scale_factor`, it returns the
/// `reference_value` + `binary_scale_factor` the packer will use (plus
/// the decimal scale echoed back).  Feed the returned
/// simple_packing_params into a `simple_packing` descriptor so callers
/// do not have to touch the C out-params.
///
/// @param values               Pointer to the f64 samples.
/// @param num_values           Number of samples.
/// @param bits_per_value       Target bit width per packed value.
/// @param decimal_scale_factor Base-10 pre-scale (usually 0).
/// @return The computed simple_packing_params.
/// @throws tensogram::encoding_error If the samples contain NaN.
[[nodiscard]] inline simple_packing_params compute_simple_packing_params(
    const double* values, std::size_t num_values,
    int bits_per_value, int decimal_scale_factor = 0) {
    simple_packing_params params{};
    params.decimal_scale_factor = static_cast<std::int32_t>(decimal_scale_factor);
    detail::check(tgm_simple_packing_compute_params(
        values, num_values,
        static_cast<std::uint32_t>(bits_per_value),
        static_cast<std::int32_t>(decimal_scale_factor),
        &params.reference_value, &params.binary_scale_factor));
    return params;
}

/// Compute simple_packing parameters for a vector of f64 sample values.
///
/// Convenience overload of compute_simple_packing_params() taking a
/// std::vector; see that overload for semantics and throwing behaviour.
[[nodiscard]] inline simple_packing_params compute_simple_packing_params(
    const std::vector<double>& values,
    int bits_per_value, int decimal_scale_factor = 0) {
    return compute_simple_packing_params(values.data(), values.size(),
                                         bits_per_value, decimal_scale_factor);
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

/// Validate every message in a multi-message wire buffer.
///
/// The in-memory companion to validate_file(): where validate() checks a
/// *single* message, this wraps `tgm_validate_buffer` to scan @p buf for
/// message boundaries and validate each message, additionally reporting
/// file-level anomalies — gaps between messages, trailing junk, or a
/// buffer with no parseable messages.
///
/// Returns a JSON `FileValidationReport` string shaped
/// `{"file_issues":[…],"messages":[…]}`.  Validation findings — including
/// file-level ones — are carried in the JSON; only API/operational
/// failures throw.
///
/// @param buf               Wire-format bytes holding zero or more
///                          concatenated messages (nullptr with len=0 is
///                          valid for empty-buffer validation).
/// @param len               Length of @p buf.
/// @param level             "quick", "default", "checksum", or "full"
///                          (default: "default").
/// @param check_canonical   Check RFC 8949 CBOR key ordering.
/// @return JSON string with the file validation report.
/// @throws tensogram::invalid_arg_error If @p level is unrecognized or
///         @p buf is null with a non-zero @p len.
/// @throws tensogram::encoding_error If JSON serialization fails.
[[nodiscard]] inline std::string validate_buffer(const std::uint8_t* buf, std::size_t len,
                                                 std::string_view level = "default",
                                                 bool check_canonical = false) {
    const std::string level_str(level);
    tgm_bytes_t bytes{};
    detail::check(tgm_validate_buffer(buf, len, level_str.c_str(),
                                      check_canonical ? 1 : 0, &bytes));
    std::string result(reinterpret_cast<const char*>(bytes.data), bytes.len);
    tgm_bytes_free(bytes);
    return result;
}

/// Verify that a CBOR byte string is in RFC 8949 §4.2.1 canonical form.
///
/// Wraps `tgm_verify_canonical_cbor`, binding
/// `tensogram::verify_canonical_cbor`.  Canonical form requires every map
/// key sorted length-first, then lexicographically by their encoded bytes.
///
/// @param cbor  Pointer to the CBOR bytes (nullptr is only valid with
///              len=0, which is not valid CBOR and returns false).
/// @param len   Length of @p cbor.
/// @return true when the bytes parse AND are canonical; false when they
///         are non-canonical, fail to parse as CBOR, or are empty.  This
///         predicate never throws — a false result reports "not canonical"
///         rather than an error condition.
[[nodiscard]] inline bool verify_canonical_cbor(const std::uint8_t* cbor, std::size_t len) {
    return tgm_verify_canonical_cbor(cbor, len);
}

/// Run environment diagnostics and return the report as a JSON string.
///
/// Wraps `tgm_doctor_to_json`, mirroring `tensogram::doctor::run_diagnostics()`
/// and the `tensogram doctor` CLI subcommand.  The same JSON shape is
/// produced by Python's `tensogram.doctor()` and the WASM `doctor()`
/// export (see `docs/src/cli/doctor.md` for the schema).  The report
/// covers the library version, build/runtime environment, the codecs
/// compiled into this build, and a `self_test` array for the core
/// encode/decode pipeline; the C FFI build does not run the CLI-only
/// GRIB / NetCDF converter self-tests.
///
/// The returned `tgm_bytes_t` is managed with the same RAII /
/// `tgm_bytes_free` pattern as validate() and metadata::to_json().
///
/// @return JSON diagnostics report (UTF-8) ready for
///         `nlohmann::json::parse` or any JSON reader.
/// @throws tensogram::encoding_error If serialising the report fails.
[[nodiscard]] inline std::string doctor() {
    tgm_bytes_t bytes{};
    detail::check(tgm_doctor_to_json(&bytes));
    std::string result(reinterpret_cast<const char*>(bytes.data), bytes.len);
    tgm_bytes_free(bytes);
    return result;
}

} // namespace tensogram

#endif // TENSOGRAM_HPP
