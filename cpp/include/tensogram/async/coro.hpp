// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF

/// @file tensogram/async/coro.hpp
/// @brief C++20 coroutine frontend (PR 5 of the cpp-async plan).
///
/// Two types are provided:
///
///   * `task<T>` — a proper coroutine return type.  Users write
///     `task<int> my_func() { co_return 42; }` and other functions
///     can `co_await my_func()`.  Lazy: nothing happens until awaited.
///
///   * `awaiter<T>` — what the async I/O methods return.  Itself
///     awaitable; suspends until the underlying FFI task resolves.
///     Users typically don't construct these directly.
///
/// Plan: `plans/PLAN_CPP_ASYNC.md` §4.2.

#ifndef TENSOGRAM_ASYNC_CORO_HPP
#define TENSOGRAM_ASYNC_CORO_HPP

#if __cplusplus < 202002L
#  error "tensogram/async/coro.hpp requires C++20 (-std=c++20). Use tensogram/async/callback.hpp on C++17."
#endif

#include "callback.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensogram::coro {

namespace tac = tensogram::async_callback;

// `throw_for_error` lives in `tensogram::detail` so all async frontends
// share a single typed-exception dispatch (single source of truth).
// See `tensogram.hpp` → `detail::throw_for_code`.

// ============================================================
// awaiter<T> — wraps a callback-style async operation
// ============================================================

template <typename T>
class awaiter {
public:
    struct shared_state {
        std::mutex mtx;
        std::optional<T> value;
        tgm_error code = TGM_ERROR_OK;
        std::string error_message;
        bool ready = false;
        std::coroutine_handle<> awaiter_handle;
    };

    explicit awaiter(std::shared_ptr<shared_state> s) : state_(std::move(s)) {}

    bool await_ready() const noexcept {
        std::lock_guard lock(state_->mtx);
        return state_->ready;
    }
    void await_suspend(std::coroutine_handle<> h) noexcept {
        std::lock_guard lock(state_->mtx);
        if (state_->ready) {
            h.resume();
        } else {
            state_->awaiter_handle = h;
        }
    }
    T await_resume() {
        std::lock_guard lock(state_->mtx);
        if (state_->code != TGM_ERROR_OK) {
            tensogram::detail::throw_for_code(state_->code, state_->error_message);
        }
        return std::move(*state_->value);
    }

    /// Synchronously block the calling thread until ready.  Used by
    /// the block_on adapter and when_all when the value is needed
    /// outside a coroutine.
    T sync_get() {
        std::unique_lock lock(state_->mtx);
        // Wait via spin-yield since shared_state has no condvar; fine
        // for top-level entry but we add a condvar for efficiency
        // when polling many awaiters in when_all.
        // We ship a separate condvar by promoting the awaiter to the
        // block_on variant where needed.
        while (!state_->ready) {
            lock.unlock();
            std::this_thread::yield();
            lock.lock();
        }
        if (state_->code != TGM_ERROR_OK) {
            tensogram::detail::throw_for_code(state_->code, state_->error_message);
        }
        return std::move(*state_->value);
    }

private:
    std::shared_ptr<shared_state> state_;
};

template <>
class awaiter<void> {
public:
    struct shared_state {
        std::mutex mtx;
        tgm_error code = TGM_ERROR_OK;
        std::string error_message;
        bool ready = false;
        std::coroutine_handle<> awaiter_handle;
    };

    explicit awaiter(std::shared_ptr<shared_state> s) : state_(std::move(s)) {}

    bool await_ready() const noexcept {
        std::lock_guard lock(state_->mtx);
        return state_->ready;
    }
    void await_suspend(std::coroutine_handle<> h) noexcept {
        std::lock_guard lock(state_->mtx);
        if (state_->ready) {
            h.resume();
        } else {
            state_->awaiter_handle = h;
        }
    }
    void await_resume() {
        std::lock_guard lock(state_->mtx);
        if (state_->code != TGM_ERROR_OK) {
            tensogram::detail::throw_for_code(state_->code, state_->error_message);
        }
    }
    void sync_get() {
        std::unique_lock lock(state_->mtx);
        while (!state_->ready) {
            lock.unlock();
            std::this_thread::yield();
            lock.lock();
        }
        if (state_->code != TGM_ERROR_OK) {
            tensogram::detail::throw_for_code(state_->code, state_->error_message);
        }
    }

private:
    std::shared_ptr<shared_state> state_;
};

namespace detail {

template <typename T, typename Launcher>
inline awaiter<T> launch_awaiter(Launcher&& launch) {
    auto state = std::make_shared<typename awaiter<T>::shared_state>();
    auto state_capture = state;
    launch([state_capture](tac::result<T> r) mutable {
        std::coroutine_handle<> h_to_resume;
        {
            std::lock_guard lock(state_capture->mtx);
            if (r.ok()) {
                state_capture->value = r.take();
            } else {
                state_capture->code = r.code();
                state_capture->error_message = r.message();
            }
            state_capture->ready = true;
            h_to_resume = state_capture->awaiter_handle;
            state_capture->awaiter_handle = nullptr;
        }
        if (h_to_resume) h_to_resume.resume();
    });
    return awaiter<T>(state);
}

template <typename Launcher>
inline awaiter<void> launch_void_awaiter(Launcher&& launch) {
    auto state = std::make_shared<awaiter<void>::shared_state>();
    auto state_capture = state;
    launch([state_capture](tac::result<void> r) mutable {
        std::coroutine_handle<> h_to_resume;
        {
            std::lock_guard lock(state_capture->mtx);
            if (!r.ok()) {
                state_capture->code = r.code();
                state_capture->error_message = r.message();
            }
            state_capture->ready = true;
            h_to_resume = state_capture->awaiter_handle;
            state_capture->awaiter_handle = nullptr;
        }
        if (h_to_resume) h_to_resume.resume();
    });
    return awaiter<void>(state);
}

}  // namespace detail

// ============================================================
// task<T> — coroutine return type
// ============================================================

template <typename T>
struct task;

namespace detail {

template <typename T>
struct task_promise_base {
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    std::exception_ptr ex;
    std::coroutine_handle<> continuation;

    struct final_awaiter {
        bool await_ready() noexcept { return false; }
        template <typename Promise>
        std::coroutine_handle<> await_suspend(std::coroutine_handle<Promise> h) noexcept {
            auto& promise = h.promise();
            std::coroutine_handle<> cont;
            {
                std::lock_guard lock(promise.mtx);
                promise.ready = true;
                cont = promise.continuation;
            }
            promise.cv.notify_all();
            if (cont) return cont;
            return std::noop_coroutine();
        }
        void await_resume() noexcept {}
    };

    std::suspend_always initial_suspend() noexcept { return {}; }
    final_awaiter final_suspend() noexcept { return {}; }
    void unhandled_exception() {
        std::lock_guard lock(mtx);
        ex = std::current_exception();
    }
};

}  // namespace detail

template <typename T>
struct task {
    struct promise_type : detail::task_promise_base<T> {
        std::optional<T> value;
        task get_return_object() {
            return task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        void return_value(T v) {
            std::lock_guard lock(this->mtx);
            value = std::move(v);
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    explicit task(handle_type h) : handle_(h) {}
    task(const task&) = delete;
    task& operator=(const task&) = delete;
    task(task&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    task& operator=(task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    ~task() { if (handle_) handle_.destroy(); }

    bool await_ready() const noexcept { return false; }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiter_h) {
        auto& promise = handle_.promise();
        std::lock_guard lock(promise.mtx);
        if (promise.ready) return awaiter_h;
        promise.continuation = awaiter_h;
        return handle_;
    }

    T await_resume() {
        auto& promise = handle_.promise();
        if (promise.ex) std::rethrow_exception(promise.ex);
        return std::move(*promise.value);
    }

    /// Synchronously run the task to completion on the calling thread.
    T sync_run() {
        // Run the coroutine to completion by resuming until ready.
        // initial_suspend is suspend_always so we kick it off here.
        handle_.resume();
        auto& promise = handle_.promise();
        std::unique_lock lock(promise.mtx);
        promise.cv.wait(lock, [&] { return promise.ready; });
        if (promise.ex) std::rethrow_exception(promise.ex);
        return std::move(*promise.value);
    }

private:
    handle_type handle_;
};

// task<void> specialization
template <>
struct task<void> {
    struct promise_type : detail::task_promise_base<void> {
        task get_return_object() {
            return task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        void return_void() {
            std::lock_guard lock(this->mtx);
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    explicit task(handle_type h) : handle_(h) {}
    task(const task&) = delete;
    task& operator=(const task&) = delete;
    task(task&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    task& operator=(task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    ~task() { if (handle_) handle_.destroy(); }

    bool await_ready() const noexcept { return false; }
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiter_h) {
        auto& promise = handle_.promise();
        std::lock_guard lock(promise.mtx);
        if (promise.ready) return awaiter_h;
        promise.continuation = awaiter_h;
        return handle_;
    }
    void await_resume() {
        auto& promise = handle_.promise();
        if (promise.ex) std::rethrow_exception(promise.ex);
    }

    void sync_run() {
        handle_.resume();
        auto& promise = handle_.promise();
        std::unique_lock lock(promise.mtx);
        promise.cv.wait(lock, [&] { return promise.ready; });
        if (promise.ex) std::rethrow_exception(promise.ex);
    }

private:
    handle_type handle_;
};

/// Synchronously run a `task<T>` to completion on the calling thread.
template <typename T>
inline T block_on(task<T> t) {
    return t.sync_run();
}

inline void block_on(task<void> t) {
    t.sync_run();
}

// ============================================================
// async_file — coroutine-friendly file handle
// ============================================================

class async_file {
public:
    [[nodiscard]] static awaiter<async_file> open(
        const std::string& path,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_awaiter<async_file>([path, token, timeout](auto cb) {
            tac::async_file::open(path,
                [cb = std::move(cb)](tac::result<tac::async_file> r) mutable {
                    if (r.ok()) {
                        cb(tac::result<async_file>::ok_value(async_file(r.take())));
                    } else {
                        cb(tac::result<async_file>::err(r.code(), r.message()));
                    }
                }, token, timeout);
        });
    }

    [[nodiscard]] awaiter<std::size_t> message_count(
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        auto inner = inner_;
        return detail::launch_awaiter<std::size_t>([inner, token, timeout](auto cb) {
            inner->message_count(std::move(cb), token, timeout);
        });
    }

    [[nodiscard]] awaiter<tensogram::message> decode_message(
        std::size_t index,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        auto inner = inner_;
        return detail::launch_awaiter<tensogram::message>([inner, index, token, timeout](auto cb) {
            inner->decode_message(index, std::move(cb), token, timeout);
        });
    }

    [[nodiscard]] awaiter<tensogram::metadata> decode_metadata(
        std::size_t index,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        auto inner = inner_;
        return detail::launch_awaiter<tensogram::metadata>(
            [inner, index, token, timeout](auto cb) {
                inner->decode_metadata(index, std::move(cb), token, timeout);
            });
    }

    [[nodiscard]] awaiter<std::vector<std::uint8_t>> read_message(
        std::size_t index,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        auto inner = inner_;
        return detail::launch_awaiter<std::vector<std::uint8_t>>(
            [inner, index, token, timeout](auto cb) {
                inner->read_message(index, std::move(cb), token, timeout);
            });
    }

    async_file(async_file&&) noexcept = default;
    async_file& operator=(async_file&&) noexcept = default;
    async_file(const async_file&) = delete;
    async_file& operator=(const async_file&) = delete;

private:
    explicit async_file(tac::async_file inner)
        : inner_(std::make_shared<tac::async_file>(std::move(inner))) {}
    std::shared_ptr<tac::async_file> inner_;
};

// ============================================================
// async_streaming_encoder
// ============================================================

class async_streaming_encoder {
public:
    [[nodiscard]] static awaiter<async_streaming_encoder> create(
        const std::string& path,
        const std::string& metadata_json,
        const std::string& hash_algo = "xxh3",
        std::uint32_t threads = 0,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_awaiter<async_streaming_encoder>(
            [path, metadata_json, hash_algo, threads, token, timeout](auto cb) {
                tac::async_streaming_encoder::create(path, metadata_json,
                    [cb = std::move(cb)](tac::result<tac::async_streaming_encoder> r) mutable {
                        if (r.ok()) {
                            cb(tac::result<async_streaming_encoder>::ok_value(
                                async_streaming_encoder(r.take())));
                        } else {
                            cb(tac::result<async_streaming_encoder>::err(r.code(), r.message()));
                        }
                    }, hash_algo, threads, token, timeout);
            });
    }

    [[nodiscard]] awaiter<void> write_object(
        const std::string& descriptor_json,
        const std::uint8_t* data, std::size_t len,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        auto inner = inner_;
        return detail::launch_void_awaiter(
            [inner, descriptor_json, data, len, token, timeout](auto cb) {
                inner->write_object(descriptor_json, data, len, std::move(cb), token, timeout);
            });
    }

    [[nodiscard]] awaiter<void> write_pre_encoded(
        const std::string& descriptor_json,
        const std::uint8_t* data, std::size_t len,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        auto inner = inner_;
        return detail::launch_void_awaiter(
            [inner, descriptor_json, data, len, token, timeout](auto cb) {
                inner->write_pre_encoded(descriptor_json, data, len, std::move(cb),
                                         token, timeout);
            });
    }

    [[nodiscard]] awaiter<void> write_preceder(
        const std::string& metadata_json,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        auto inner = inner_;
        return detail::launch_void_awaiter(
            [inner, metadata_json, token, timeout](auto cb) {
                inner->write_preceder(metadata_json, std::move(cb), token, timeout);
            });
    }

    [[nodiscard]] awaiter<void> finish(
        bool backfill = true,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        auto inner = inner_;
        return detail::launch_void_awaiter([inner, backfill, token, timeout](auto cb) {
            inner->finish(std::move(cb), backfill, token, timeout);
        });
    }

    [[nodiscard]] std::size_t object_count() const noexcept {
        return inner_->object_count();
    }

    async_streaming_encoder(async_streaming_encoder&&) noexcept = default;
    async_streaming_encoder& operator=(async_streaming_encoder&&) noexcept = default;
    async_streaming_encoder(const async_streaming_encoder&) = delete;
    async_streaming_encoder& operator=(const async_streaming_encoder&) = delete;

private:
    explicit async_streaming_encoder(tac::async_streaming_encoder inner)
        : inner_(std::make_shared<tac::async_streaming_encoder>(std::move(inner))) {}
    std::shared_ptr<tac::async_streaming_encoder> inner_;
};

// ============================================================
// async_for_each — drive a coroutine over every message in a file
// ============================================================

/// Apply `fn` to every decoded message in `file`.  `fn` is called with
/// each `tensogram::message` in sequence; if `fn` returns `bool`,
/// iteration stops on `false`.  See plan Q9.
template <typename Fn>
[[nodiscard]] inline task<void> async_for_each(async_file& file, Fn fn) {
    std::size_t n = co_await file.message_count();
    for (std::size_t i = 0; i < n; ++i) {
        auto msg = co_await file.decode_message(i);
        if constexpr (std::is_same_v<std::invoke_result_t<Fn, tensogram::message>, bool>) {
            if (!fn(std::move(msg))) co_return;
        } else {
            fn(std::move(msg));
        }
    }
}

}  // namespace tensogram::coro

#endif  // TENSOGRAM_ASYNC_CORO_HPP
