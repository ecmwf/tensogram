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
///
/// Plan: `plans/PLAN_CPP_ASYNC.md` §4.1.

#ifndef TENSOGRAM_ASYNC_CALLBACK_HPP
#define TENSOGRAM_ASYNC_CALLBACK_HPP

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

}  // namespace detail

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

    /// Best-effort object-count snapshot.  Returns `static_cast<std::size_t>(-1)`
    /// if the encoder is currently busy with another in-flight call.
    [[nodiscard]] std::size_t object_count() const noexcept {
        return tgm_async_streaming_encoder_object_count(handle_.get());
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
