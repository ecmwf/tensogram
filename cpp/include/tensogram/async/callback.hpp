// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF

/// @file tensogram/async/callback.hpp
/// @brief Callback-based asynchronous frontend (PR 4 of the cpp-async plan).
///
/// Header-only, C++17, always available wherever the FFI ships
/// `--features async`.  The minimum-viable async surface that all
/// other frontends (`coro.hpp`, `std_future.hpp`) layer on top of.

#ifndef TENSOGRAM_ASYNC_CALLBACK_HPP
#define TENSOGRAM_ASYNC_CALLBACK_HPP

// The async frontends call `tgm_async_*` symbols that exist only when the
// FFI is built with the `async` Cargo feature.  CMake defines
// `TENSOGRAM_ASYNC=1` on the `tensogram` target exactly when that build is
// selected (`-DTENSOGRAM_ASYNC=ON`, the default).  Including an async
// header in an async-disabled build would otherwise fail at link time with
// undefined `tgm_async_*` references; fail fast with a clear message
// instead.  (coro.hpp and std_future.hpp include this header, so the guard
// covers the whole `async/` family.)
#if !defined(TENSOGRAM_ASYNC)
#  error "tensogram/async/*.hpp requires the async build: configure CMake with -DTENSOGRAM_ASYNC=ON (the default), or define TENSOGRAM_ASYNC if you build the FFI with --features=async by hand."
#endif

#include "../../tensogram.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensogram::async_callback {

// ============================================================
// result<T> — never-throw discriminated union
// ============================================================

/// Lightweight success/error union for callback results.  Doesn't
/// throw on construction or access; consumers check `.ok()` before
/// reading `.value()`.
template <typename T>
class result {
public:
    static result<T> ok_value(T v) {
        result<T> r;
        r.ok_ = true;
        r.value_ = std::make_unique<T>(std::move(v));
        return r;
    }
    static result<T> err(tgm_error code, std::string msg) {
        result<T> r;
        r.ok_ = false;
        r.code_ = code;
        r.message_ = std::move(msg);
        return r;
    }

    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] tgm_error code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

    /// Access the success value.  Caller must check `.ok()` first;
    /// calling `.value()` on a failed result is undefined behaviour.
    T& value() noexcept { return *value_; }
    const T& value() const noexcept { return *value_; }

    /// Move-out the success value.
    T take() noexcept { return std::move(*value_); }

private:
    bool ok_ = false;
    tgm_error code_ = TGM_ERROR_OK;
    std::string message_;
    std::unique_ptr<T> value_;
};

template <>
class result<void> {
public:
    static result<void> ok_value() {
        result<void> r;
        r.ok_ = true;
        return r;
    }
    static result<void> err(tgm_error code, std::string msg) {
        result<void> r;
        r.ok_ = false;
        r.code_ = code;
        r.message_ = std::move(msg);
        return r;
    }
    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] tgm_error code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }
private:
    bool ok_ = false;
    tgm_error code_ = TGM_ERROR_OK;
    std::string message_;
};

// ============================================================
// cancellation_token
// ============================================================

class cancellation_token {
public:
    cancellation_token()
        : handle_(tgm_cancellation_token_create(),
                  &tgm_cancellation_token_free) {}

    void cancel() noexcept {
        tgm_cancellation_token_cancel(handle_.get());
    }
    [[nodiscard]] bool cancelled() const noexcept {
        return tgm_cancellation_token_is_cancelled(handle_.get());
    }

    cancellation_token(const cancellation_token&) = delete;
    cancellation_token& operator=(const cancellation_token&) = delete;
    cancellation_token(cancellation_token&&) noexcept = default;
    cancellation_token& operator=(cancellation_token&&) noexcept = default;

    /// Internal: raw handle for FFI plumbing.
    [[nodiscard]] tgm_cancellation_token_t* raw() noexcept { return handle_.get(); }

private:
    using deleter_t = void (*)(tgm_cancellation_token_t*);
    std::unique_ptr<tgm_cancellation_token_t, deleter_t> handle_;
};

// ============================================================
// detail — task adapters
// ============================================================

namespace detail {

/// Snapshot the FFI's thread-local last-error string into a `std::string`.
///
/// The dispatcher worker calls this immediately after the join
/// returns an error; the FFI's thread-local error slot is shared
/// with any tgm_* call on the same OS thread, so the snapshot must
/// happen before any other FFI call on that thread.
inline std::string error_message_from_last() {
    const char* m = tgm_last_error();
    return m ? std::string(m) : std::string();
}

/// State held alive across the FFI completion callback.  When
/// `tgm_async_task_set_completion` fires, the dispatcher worker
/// invokes `trampoline(state)` which joins the task, packages the
/// typed result, and hands it to the user's `cb`.  The state then
/// frees the task handle and itself.
template <typename T>
struct completion_state {
    std::function<void(result<T>)> cb;
    tgm_async_task_t* task;
};

/// Trampoline: invoked by the FFI dispatcher pool when the task
/// completes.  Templated on the result type and the typed-join
/// function so we share the framing across every async entry point.
template <typename T,
          tgm_error (*JoinFn)(tgm_async_task_t*, T*)>
inline void trampoline_join_value(void* userdata) noexcept {
    auto* state = static_cast<completion_state<T>*>(userdata);
    T raw{};
    tgm_error e = JoinFn(state->task, &raw);
    if (e == TGM_ERROR_OK) {
        state->cb(result<T>::ok_value(std::move(raw)));
    } else {
        state->cb(result<T>::err(e, error_message_from_last()));
    }
    tgm_async_task_free(state->task);
    delete state;
}

/// Trampoline for the void-result case (writes / finishes).
inline void trampoline_join_void(void* userdata) noexcept {
    auto* state = static_cast<completion_state<void>*>(userdata);
    tgm_error e = tgm_async_task_join_void(state->task);
    if (e == TGM_ERROR_OK) {
        state->cb(result<void>::ok_value());
    } else {
        state->cb(result<void>::err(e, error_message_from_last()));
    }
    tgm_async_task_free(state->task);
    delete state;
}

/// Register a completion callback on `task` that joins via `JoinFn`,
/// converts the raw result via `Convert`, and invokes the user `cb`.
/// Used for results that need post-join transformation
/// (`tgm_async_file_t*` → `async_file`, `tgm_bytes_t` → vector).
template <typename UserT, typename RawT,
          tgm_error (*JoinFn)(tgm_async_task_t*, RawT*),
          UserT (*Convert)(RawT&&)>
inline void trampoline_join_convert(void* userdata) noexcept {
    auto* state = static_cast<completion_state<UserT>*>(userdata);
    RawT raw{};
    tgm_error e = JoinFn(state->task, &raw);
    if (e == TGM_ERROR_OK) {
        state->cb(result<UserT>::ok_value(Convert(std::move(raw))));
    } else {
        state->cb(result<UserT>::err(e, error_message_from_last()));
    }
    tgm_async_task_free(state->task);
    delete state;
}

/// Owned array of byte buffers returned by an async range decode.
using range_buffers = std::vector<std::vector<std::uint8_t>>;

/// Transfer an FFI `tgm_bytes_t[count]` array into owned vectors and
/// release the array via `tgm_multi_bytes_free`.  Shared by the
/// callback trampoline and the pull-model blocking joiner.
inline range_buffers multi_bytes_to_vectors(tgm_bytes_t* array, std::size_t count) {
    range_buffers out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.emplace_back(array[i].data, array[i].data + array[i].len);
    }
    tgm_multi_bytes_free(array, count);
    return out;
}

/// Trampoline for the multi-byte-buffer join (`decode_range`).  The FFI
/// hands back an owned array of `tgm_bytes_t` plus a count; we copy each
/// into an owned vector and release the array with `tgm_multi_bytes_free`.
/// This join has two out-params, so it cannot reuse the single-out
/// `trampoline_join_convert`.
inline void trampoline_join_multi_bytes(void* userdata) noexcept {
    auto* state = static_cast<completion_state<range_buffers>*>(userdata);
    tgm_bytes_t* array = nullptr;
    std::size_t count = 0;
    tgm_error e = tgm_async_task_join_multi_bytes(state->task, &array, &count);
    if (e == TGM_ERROR_OK) {
        state->cb(result<range_buffers>::ok_value(multi_bytes_to_vectors(array, count)));
    } else {
        state->cb(result<range_buffers>::err(e, error_message_from_last()));
    }
    tgm_async_task_free(state->task);
    delete state;
}

}  // namespace detail

// ============================================================
// async_file
// ============================================================

namespace detail {

/// Trivial conversion helpers used by `trampoline_join_convert` so each
/// async entry point can route raw FFI types into the user-facing C++
/// types without bespoke trampolines.
inline tensogram::message message_from_raw_handle(tgm_message_t*&& raw) {
    return tensogram::detail::message_from_raw(raw);
}

inline tensogram::metadata metadata_from_raw_handle(tgm_metadata_t*&& raw) {
    return tensogram::detail::metadata_from_raw(raw);
}

inline std::vector<std::uint8_t> bytes_to_vector(tgm_bytes_t&& b) {
    std::vector<std::uint8_t> v(b.data, b.data + b.len);
    tgm_bytes_free(b);
    return v;
}

inline std::size_t size_to_size_t(std::uint64_t&& n) {
    return static_cast<std::size_t>(n);
}

// -- pull-model blocking joiners --------------------------------------
//
// The completion (push) trampolines above are paired with blocking
// joiners here so the same typed-join + convert logic backs both the
// callback frontend and the pull-model `task<T>` handle.  Each returns a
// `result<T>` (never throws), snapshotting the FFI last-error message on
// the joining thread before it can be clobbered.

inline result<std::size_t> join_size_result(tgm_async_task_t* task) {
    std::uint64_t n = 0;
    tgm_error e = tgm_async_task_join_size(task, &n);
    if (e == TGM_ERROR_OK) return result<std::size_t>::ok_value(static_cast<std::size_t>(n));
    return result<std::size_t>::err(e, error_message_from_last());
}

inline result<tensogram::message> join_message_result(tgm_async_task_t* task) {
    tgm_message_t* raw = nullptr;
    tgm_error e = tgm_async_task_join_message(task, &raw);
    if (e == TGM_ERROR_OK) {
        return result<tensogram::message>::ok_value(tensogram::detail::message_from_raw(raw));
    }
    return result<tensogram::message>::err(e, error_message_from_last());
}

inline result<range_buffers> join_multi_bytes_result(tgm_async_task_t* task) {
    tgm_bytes_t* array = nullptr;
    std::size_t count = 0;
    tgm_error e = tgm_async_task_join_multi_bytes(task, &array, &count);
    if (e == TGM_ERROR_OK) return result<range_buffers>::ok_value(multi_bytes_to_vectors(array, count));
    return result<range_buffers>::err(e, error_message_from_last());
}

}  // namespace detail

// ============================================================
// task<T> — pull-model (poll / cancel / join) task handle
// ============================================================

/// Move-only RAII owner of a single in-flight FFI task in the *pull*
/// model.  Where the completion-based `async_file` methods hand the task
/// to a dispatcher trampoline, `task<T>` keeps ownership so the caller
/// can poll `ready()`, optionally `cancel()`, and then `join()` exactly
/// once to consume the typed result.
///
/// Obtain one from the `*_task` launchers on async_file (e.g.
/// `decode_object_task`).  This is the C++ surface for the FFI's
/// `tgm_async_task_is_ready` / `tgm_async_task_cancel`; the completion
/// frontends (coro / std_future) instead consume readiness through their
/// awaiter / future and cancellation through `cancellation_token`.
///
/// @note Not thread-safe: poll / cancel / join from one thread at a time.
template <typename T>
class task {
public:
    task(task&&) noexcept = default;
    task& operator=(task&&) noexcept = default;
    task(const task&) = delete;
    task& operator=(const task&) = delete;

    /// Non-blocking readiness poll (`tgm_async_task_is_ready`).  Returns
    /// true once the task has resolved (success, error, timeout, or
    /// cancellation), so a subsequent join() will not block.
    [[nodiscard]] bool ready() const noexcept {
        return tgm_async_task_is_ready(handle_.get());
    }

    /// Request cooperative cancellation (`tgm_async_task_cancel`).  The
    /// task transitions to `TGM_ERROR_CANCELLED` at its next yield point.
    /// Idempotent; safe after the task has already resolved.
    void cancel() noexcept { tgm_async_task_cancel(handle_.get()); }

    /// Block until ready, then consume the typed result exactly once.
    /// A second join() returns an `invalid_arg` result rather than
    /// double-joining the underlying task.
    result<T> join() {
        if (joined_) {
            return result<T>::err(TGM_ERROR_INVALID_ARG, "task result already consumed");
        }
        joined_ = true;
        return joiner_(handle_.get());
    }

private:
    friend class async_file;
    using deleter_t = void (*)(tgm_async_task_t*);
    using joiner_t = result<T> (*)(tgm_async_task_t*);

    task(tgm_async_task_t* raw, joiner_t joiner)
        : handle_(raw, &tgm_async_task_free), joiner_(joiner) {}

    std::unique_ptr<tgm_async_task_t, deleter_t> handle_;
    joiner_t joiner_;
    bool joined_ = false;
};

class async_file {
public:
    async_file(async_file&&) noexcept = default;
    async_file& operator=(async_file&&) noexcept = default;
    async_file(const async_file&) = delete;
    async_file& operator=(const async_file&) = delete;

    /// Open a local file asynchronously.  `cb` is invoked once on the
    /// FFI dispatcher pool when the open completes.
    ///
    /// `timeout` is in milliseconds; pass `0` for no timeout.
    static void open(const std::string& path,
                     std::function<void(result<async_file>)> cb,
                     cancellation_token* token = nullptr,
                     std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_file_open(
            path.c_str(),
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<async_file>::err(err, detail::error_message_from_last()));
            return;
        }
        register_async_file_completion(task, std::move(cb));
    }

    /// Open a remote `.tgm` over S3 / GCS / Azure / HTTP asynchronously.
    ///
    /// `url` is an object-store URL (`s3://bucket/key.tgm`, `gs://…`,
    /// `az://…`, `https://…`) or a `file://…` URL routed through the
    /// same object-store backend.  `storage_options` carries backend
    /// credentials / configuration as key→value pairs (e.g.
    /// `{{"aws_region", "eu-west-1"}}`); pass an empty vector to rely
    /// on ambient credentials (environment, instance role).
    /// `bidirectional` selects the pipelined two-ended remote scan.
    ///
    /// Requires the FFI to be built with the `async-remote` Cargo
    /// feature.  Without it the callback fires with `TGM_ERROR_REMOTE`
    /// and a diagnostic naming the missing feature — the symbol is
    /// always linkable, so this never becomes an undefined reference.
    /// `timeout` is in milliseconds; pass `0` for no timeout.
    static void open_remote(
        const std::string& url,
        const std::vector<std::pair<std::string, std::string>>& storage_options,
        bool bidirectional,
        std::function<void(result<async_file>)> cb,
        cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        // The FFI copies every key/value string into an owned map
        // before returning, so these `c_str()` views only need to
        // stay valid across the synchronous launch call below.
        std::vector<const char*> keys;
        std::vector<const char*> values;
        keys.reserve(storage_options.size());
        values.reserve(storage_options.size());
        for (const auto& [k, v] : storage_options) {
            keys.push_back(k.c_str());
            values.push_back(v.c_str());
        }
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_file_open_remote(
            url.c_str(),
            keys.empty() ? nullptr : keys.data(),
            values.empty() ? nullptr : values.data(),
            storage_options.size(),
            bidirectional,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<async_file>::err(err, detail::error_message_from_last()));
            return;
        }
        register_async_file_completion(task, std::move(cb));
    }

    /// Async message-count.
    void message_count(std::function<void(result<std::size_t>)> cb,
                       cancellation_token* token = nullptr,
                       std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_file_message_count(
            handle_.get(),
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<std::size_t>::err(err, detail::error_message_from_last()));
            return;
        }
        auto* state = new detail::completion_state<std::size_t>{std::move(cb), task};
        tgm_async_task_set_completion(
            task,
            &detail::trampoline_join_convert<std::size_t, std::uint64_t,
                &tgm_async_task_join_size, &detail::size_to_size_t>,
            state);
    }

    /// Async decode of message at `index`, returning a `tensogram::message`.
    void decode_message(std::size_t index,
                        std::function<void(result<tensogram::message>)> cb,
                        cancellation_token* token = nullptr,
                        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_file_decode_message(
            handle_.get(), index,
            true,   // native_byte_order
            0,      // threads
            true,   // restore_non_finite
            false,  // verify_hash
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<tensogram::message>::err(err, detail::error_message_from_last()));
            return;
        }
        auto* state = new detail::completion_state<tensogram::message>{std::move(cb), task};
        tgm_async_task_set_completion(
            task,
            &detail::trampoline_join_convert<tensogram::message, tgm_message_t*,
                &tgm_async_task_join_message, &detail::message_from_raw_handle>,
            state);
    }

    /// Async decode of metadata only.
    void decode_metadata(std::size_t index,
                         std::function<void(result<tensogram::metadata>)> cb,
                         cancellation_token* token = nullptr,
                         std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_file_decode_metadata(
            handle_.get(), index,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<tensogram::metadata>::err(err, detail::error_message_from_last()));
            return;
        }
        auto* state = new detail::completion_state<tensogram::metadata>{std::move(cb), task};
        tgm_async_task_set_completion(
            task,
            &detail::trampoline_join_convert<tensogram::metadata, tgm_metadata_t*,
                &tgm_async_task_join_metadata, &detail::metadata_from_raw_handle>,
            state);
    }

    /// Async raw read of message bytes.
    void read_message(std::size_t index,
                      std::function<void(result<std::vector<std::uint8_t>>)> cb,
                      cancellation_token* token = nullptr,
                      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_file_read_message(
            handle_.get(), index,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<std::vector<std::uint8_t>>::err(err, detail::error_message_from_last()));
            return;
        }
        auto* state = new detail::completion_state<std::vector<std::uint8_t>>{
            std::move(cb), task};
        tgm_async_task_set_completion(
            task,
            &detail::trampoline_join_convert<std::vector<std::uint8_t>, tgm_bytes_t,
                &tgm_async_task_join_bytes, &detail::bytes_to_vector>,
            state);
    }

    /// Async decode of a single object `obj_index` from message
    /// `msg_index`, returning a single-object `tensogram::message` so the
    /// usual object accessors apply.  Flags mirror decode_message().
    void decode_object(std::size_t msg_index, std::size_t obj_index,
                       std::function<void(result<tensogram::message>)> cb,
                       cancellation_token* token = nullptr,
                       std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_file_decode_object(
            handle_.get(), msg_index, obj_index,
            true,   // native_byte_order
            0,      // threads
            true,   // restore_non_finite
            false,  // verify_hash
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<tensogram::message>::err(err, detail::error_message_from_last()));
            return;
        }
        auto* state = new detail::completion_state<tensogram::message>{std::move(cb), task};
        tgm_async_task_set_completion(
            task,
            &detail::trampoline_join_convert<tensogram::message, tgm_message_t*,
                &tgm_async_task_join_message, &detail::message_from_raw_handle>,
            state);
    }

    /// Async decode of partial ranges from object `obj_index` in message
    /// `msg_index`.  Each `(element_offset, element_count)` pair resolves
    /// to one byte buffer (split mode), returned as a vector of vectors.
    void decode_range(std::size_t msg_index, std::size_t obj_index,
                      const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
                      std::function<void(result<std::vector<std::vector<std::uint8_t>>>)> cb,
                      cancellation_token* token = nullptr,
                      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        // The FFI copies the offset/count arrays before returning, so
        // these locals only need to outlive the synchronous launch call.
        std::vector<std::uint64_t> offsets;
        std::vector<std::uint64_t> counts;
        offsets.reserve(ranges.size());
        counts.reserve(ranges.size());
        for (const auto& r : ranges) {
            offsets.push_back(r.first);
            counts.push_back(r.second);
        }
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_file_decode_range(
            handle_.get(), msg_index, obj_index,
            offsets.empty() ? nullptr : offsets.data(),
            counts.empty() ? nullptr : counts.data(),
            ranges.size(),
            true,   // native_byte_order
            0,      // threads
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<detail::range_buffers>::err(err, detail::error_message_from_last()));
            return;
        }
        auto* state = new detail::completion_state<detail::range_buffers>{std::move(cb), task};
        tgm_async_task_set_completion(task, &detail::trampoline_join_multi_bytes, state);
    }

    /// Borrowed file path (`tgm_async_file_path`).  Valid until this
    /// handle is closed; empty for a null handle.
    [[nodiscard]] std::string path() const {
        const char* p = tgm_async_file_path(handle_.get());
        return p ? p : "";
    }

    // -- pull-model launchers (poll / cancel / join) --------------------
    //
    // Return a `task<T>` the caller owns and consumes with ready() /
    // cancel() / join(), instead of a dispatcher completion.  They throw
    // the typed `tensogram::error` hierarchy if the launch itself fails
    // (unlike the completion methods, which report launch errors through
    // the callback).

    /// Pull-model message count.  Join yields the count.
    [[nodiscard]] task<std::size_t> message_count_task(
        cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        tgm_async_task_t* raw = nullptr;
        tensogram::detail::check(tgm_async_file_message_count(
            handle_.get(), token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()), &raw));
        return task<std::size_t>(raw, &detail::join_size_result);
    }

    /// Pull-model decode of message `index`.  Join yields a message.
    [[nodiscard]] task<tensogram::message> decode_message_task(
        std::size_t index, cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        tgm_async_task_t* raw = nullptr;
        tensogram::detail::check(tgm_async_file_decode_message(
            handle_.get(), index, true, 0, true, false,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()), &raw));
        return task<tensogram::message>(raw, &detail::join_message_result);
    }

    /// Pull-model decode of object `obj_index` from message `msg_index`.
    /// Join yields a single-object message.
    [[nodiscard]] task<tensogram::message> decode_object_task(
        std::size_t msg_index, std::size_t obj_index,
        cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        tgm_async_task_t* raw = nullptr;
        tensogram::detail::check(tgm_async_file_decode_object(
            handle_.get(), msg_index, obj_index, true, 0, true, false,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()), &raw));
        return task<tensogram::message>(raw, &detail::join_message_result);
    }

    /// Pull-model partial-range decode.  Join yields one byte buffer per
    /// `(element_offset, element_count)` range (split mode).
    [[nodiscard]] task<std::vector<std::vector<std::uint8_t>>> decode_range_task(
        std::size_t msg_index, std::size_t obj_index,
        const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
        cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        std::vector<std::uint64_t> offsets;
        std::vector<std::uint64_t> counts;
        offsets.reserve(ranges.size());
        counts.reserve(ranges.size());
        for (const auto& r : ranges) {
            offsets.push_back(r.first);
            counts.push_back(r.second);
        }
        tgm_async_task_t* raw = nullptr;
        tensogram::detail::check(tgm_async_file_decode_range(
            handle_.get(), msg_index, obj_index,
            offsets.empty() ? nullptr : offsets.data(),
            counts.empty() ? nullptr : counts.data(),
            ranges.size(), true, 0,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()), &raw));
        return task<detail::range_buffers>(raw, &detail::join_multi_bytes_result);
    }

    /// Internal helper: raw FFI handle.  Reserved for the
    /// other frontends.
    [[nodiscard]] tgm_async_file_t* raw() const noexcept { return handle_.get(); }

private:
    using deleter_t = void (*)(tgm_async_file_t*);
    std::unique_ptr<tgm_async_file_t, deleter_t> handle_;

    explicit async_file(tgm_async_file_t* raw)
        : handle_(raw, &tgm_async_file_close) {}

    /// Inline-defined here so the trampoline can construct an
    /// `async_file` from a raw FFI handle without exposing the
    /// constructor publicly.
    static void register_async_file_completion(
        tgm_async_task_t* task,
        std::function<void(result<async_file>)> cb) {
        struct State {
            std::function<void(result<async_file>)> cb;
            tgm_async_task_t* task;
        };
        auto* state = new State{std::move(cb), task};
        tgm_async_task_set_completion(
            task,
            [](void* userdata) noexcept {
                auto* s = static_cast<State*>(userdata);
                tgm_async_file_t* raw = nullptr;
                tgm_error e = tgm_async_task_join_async_file(s->task, &raw);
                if (e == TGM_ERROR_OK) {
                    s->cb(result<async_file>::ok_value(async_file(raw)));
                } else {
                    s->cb(result<async_file>::err(e, detail::error_message_from_last()));
                }
                tgm_async_task_free(s->task);
                delete s;
            },
            state);
    }
};

// ============================================================
// async_streaming_encoder
// ============================================================

class async_streaming_encoder {
public:
    async_streaming_encoder(async_streaming_encoder&&) noexcept = default;
    async_streaming_encoder& operator=(async_streaming_encoder&&) noexcept = default;
    async_streaming_encoder(const async_streaming_encoder&) = delete;
    async_streaming_encoder& operator=(const async_streaming_encoder&) = delete;

    /// Create an async streaming encoder writing to a local file.
    /// `cb` fires once on the FFI dispatcher pool with the
    /// constructed handle (or an error).
    static void create(const std::string& path,
                       const std::string& metadata_json,
                       std::function<void(result<async_streaming_encoder>)> cb,
                       const std::string& hash_algo = "xxh3",
                       std::uint32_t threads = 0,
                       cancellation_token* token = nullptr,
                       std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_streaming_encoder_create(
            path.c_str(),
            metadata_json.c_str(),
            hash_algo.empty() ? nullptr : hash_algo.c_str(),
            threads,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        if (err != TGM_ERROR_OK) {
            cb(result<async_streaming_encoder>::err(err, detail::error_message_from_last()));
            return;
        }
        struct State {
            std::function<void(result<async_streaming_encoder>)> cb;
            tgm_async_task_t* task;
        };
        auto* state = new State{std::move(cb), task};
        tgm_async_task_set_completion(
            task,
            [](void* userdata) noexcept {
                auto* s = static_cast<State*>(userdata);
                tgm_async_streaming_encoder_t* raw = nullptr;
                tgm_error e = tgm_async_task_join_async_streaming_encoder(s->task, &raw);
                if (e == TGM_ERROR_OK) {
                    s->cb(result<async_streaming_encoder>::ok_value(
                        async_streaming_encoder(raw)));
                } else {
                    s->cb(result<async_streaming_encoder>::err(
                        e, detail::error_message_from_last()));
                }
                tgm_async_task_free(s->task);
                delete s;
            },
            state);
    }

    /// Async write of an encoded data object.  `cb` fires with void
    /// success or an error.
    void write_object(const std::string& descriptor_json,
                      const std::uint8_t* data, std::size_t len,
                      std::function<void(result<void>)> cb,
                      cancellation_token* token = nullptr,
                      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_streaming_encoder_write_object(
            handle_.get(),
            descriptor_json.c_str(),
            data, len,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        launch_void_completion(task, err, std::move(cb));
    }

    /// Async write of a pre-encoded data object (descriptor must
    /// describe the payload's encoding/compression accurately).
    void write_pre_encoded(const std::string& descriptor_json,
                           const std::uint8_t* data, std::size_t len,
                           std::function<void(result<void>)> cb,
                           cancellation_token* token = nullptr,
                           std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_streaming_encoder_write_pre_encoded(
            handle_.get(),
            descriptor_json.c_str(),
            data, len,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        launch_void_completion(task, err, std::move(cb));
    }

    /// Async write of a `PrecederMetadata` frame.
    void write_preceder(const std::string& metadata_json,
                        std::function<void(result<void>)> cb,
                        cancellation_token* token = nullptr,
                        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_streaming_encoder_write_preceder(
            handle_.get(),
            metadata_json.c_str(),
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        launch_void_completion(task, err, std::move(cb));
    }

    /// Async finish — writes footer frames + postamble.  When
    /// `backfill = true` (default) and the sink supports seek (local
    /// files do), the preamble + postamble `total_length` slots are
    /// patched.
    void finish(std::function<void(result<void>)> cb,
                bool backfill = true,
                cancellation_token* token = nullptr,
                std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        tgm_async_task_t* task = nullptr;
        tgm_error err = tgm_async_streaming_encoder_finish(
            handle_.get(),
            backfill,
            token ? token->raw() : nullptr,
            static_cast<uint64_t>(timeout.count()),
            &task);
        launch_void_completion(task, err, std::move(cb));
    }

    /// Outcome of `try_object_count`.  Distinguishes the four cases
    /// the convenience accessor (`object_count`) collapses to a
    /// single `static_cast<std::size_t>(-1)` sentinel.
    enum class object_count_status {
        ok = 0,           ///< `value` is the current object count.
        null_handle = 1,  ///< Handle is null/invalid.
        busy = 2,         ///< Encoder locked by another in-flight call.
        finished = 3,     ///< `finish` already consumed the encoder.
    };

    /// Discriminated `try` accessor: returns both the status and (if
    /// `status == ok`) the current count.  Use this in preference to
    /// `object_count()` when you need to distinguish "0 written" from
    /// "busy" or "finished".
    struct object_count_result {
        object_count_status status;
        std::size_t value;
    };

    [[nodiscard]] object_count_result try_object_count() const noexcept {
        std::size_t count = 0;
        TgmObjectCountStatus s =
            tgm_async_streaming_encoder_try_object_count(handle_.get(), &count);
        return object_count_result{
            static_cast<object_count_status>(static_cast<int>(s)),
            count,
        };
    }

    /// Best-effort object-count snapshot.  Returns
    /// `static_cast<std::size_t>(-1)` if the handle is null, the
    /// encoder is currently busy with another in-flight call, or
    /// `finish` has already been called.  See `try_object_count`
    /// for a typed version that distinguishes those cases.
    [[nodiscard]] std::size_t object_count() const noexcept {
        return tgm_async_streaming_encoder_object_count(handle_.get());
    }

    /// Borrowed encoder output path (`tgm_async_streaming_encoder_path`).
    /// Valid for the lifetime of the handle; empty for a null handle.
    [[nodiscard]] std::string path() const {
        const char* p = tgm_async_streaming_encoder_path(handle_.get());
        return p ? p : "";
    }

private:
    using deleter_t = void (*)(tgm_async_streaming_encoder_t*);
    std::unique_ptr<tgm_async_streaming_encoder_t, deleter_t> handle_;

    explicit async_streaming_encoder(tgm_async_streaming_encoder_t* raw)
        : handle_(raw, &tgm_async_streaming_encoder_free) {}

    static void launch_void_completion(tgm_async_task_t* task, tgm_error err,
                                       std::function<void(result<void>)> cb) {
        if (err != TGM_ERROR_OK) {
            cb(result<void>::err(err, detail::error_message_from_last()));
            return;
        }
        auto* state = new detail::completion_state<void>{std::move(cb), task};
        tgm_async_task_set_completion(task, &detail::trampoline_join_void, state);
    }
};

// ============================================================
// runtime — one-shot configuration
// ============================================================

/// Configure the FFI tokio runtime.  Must be called before any other
/// async call; subsequent calls throw `invalid_arg_error`.
///
/// Pass `0` for any field to use the default.
inline void runtime_configure(std::uint32_t workers = 0,
                              std::uint32_t dispatcher_workers = 0,
                              std::uint64_t multipart_part_size_bytes = 0) {
    tgm_error e = tgm_runtime_configure(workers, dispatcher_workers,
                                        multipart_part_size_bytes);
    tensogram::detail::check(e);
}

/// Drain in-flight tasks and return the count that did not finish
/// within `timeout`.
inline std::uint64_t runtime_shutdown_blocking(std::chrono::milliseconds timeout) {
    return tgm_runtime_shutdown_blocking(static_cast<uint64_t>(timeout.count()));
}

}  // namespace tensogram::async_callback

// ============================================================
// Helpers exposed back to ::tensogram for friend access
// ============================================================

namespace tensogram::detail {

/// Construct a `tensogram::message` from a raw FFI handle.  Used by
/// the async frontend to convert join results.  Uses the `message`'s
/// private constructor through friend access.
inline tensogram::message message_from_raw(tgm_message_t* raw) {
    return tensogram::message(raw);
}

inline tensogram::metadata metadata_from_raw(tgm_metadata_t* raw) {
    return tensogram::metadata(raw);
}

inline std::string error_string(tgm_error err) {
    const char* s = tgm_error_string(err);
    return s ? std::string(s) : std::string();
}

}  // namespace tensogram::detail

#endif  // TENSOGRAM_ASYNC_CALLBACK_HPP
