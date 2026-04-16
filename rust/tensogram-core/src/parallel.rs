// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Thread-budget resolution and scoped-pool execution helpers for the
//! multi-threaded encoding/decoding pipeline.
//!
//! The design goals are:
//! - Byte-identical output regardless of `threads` setting (determinism).
//! - Zero overhead when `threads == 0` and no env override is active.
//! - Graceful degradation when the `threads` cargo feature is disabled —
//!   callers pay for an `Option<u32>` field and a single branch.
//!
//! ## Thread budget resolution
//!
//! The resolution order is:
//!
//! 1. If the caller set `threads > 0`, use that value verbatim (option
//!    beats environment).
//! 2. If `threads == 0` and the `TENSOGRAM_THREADS` env var parses as a
//!    non-zero `u32`, use that value.
//! 3. Otherwise, `threads` is `0` (sequential).
//!
//! The env var is cached in a `OnceLock` so repeated calls do not pay
//! the `std::env::var` cost.  This matches the
//! `TENSOGRAM_COMPRESSION_BACKEND` pattern used elsewhere.
//!
//! ## Small-message threshold
//!
//! Building a rayon `ThreadPool` costs tens of microseconds and is
//! pointless for tiny payloads.  [`should_parallelise`] skips the pool
//! when the total work bytes are below a configurable threshold
//! (default 64 KiB).  Callers with a known-small workload can either
//! request `threads = 0` or pass `parallel_threshold_bytes = Some(usize::MAX)`
//! to force the sequential path.

use std::sync::OnceLock;

/// Default threshold below which the library runs sequentially even when
/// `threads > 0`.  Chosen to be well above the per-call rayon pool
/// construction cost (~10 µs) but small enough not to starve encode
/// paths that want parallelism.
///
/// Callers can override via `EncodeOptions.parallel_threshold_bytes`
/// or `DecodeOptions.parallel_threshold_bytes`.
pub const DEFAULT_PARALLEL_THRESHOLD_BYTES: usize = 64 * 1024;

/// Env var consulted when `threads == 0`.  Values accepted: any
/// non-negative integer.  Zero, empty, or unparseable values fall back
/// to sequential execution.
pub const ENV_THREADS: &str = "TENSOGRAM_THREADS";

fn env_threads() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var(ENV_THREADS)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(0)
    })
}

/// Resolve the effective thread budget given a caller-provided value.
///
/// - If `requested > 0`, returns `requested` unchanged (option beats env).
/// - If `requested == 0`, returns `TENSOGRAM_THREADS` (0 when unset).
///
/// Values are not clamped to CPU count — rayon handles that naturally,
/// and pathological N values simply see diminishing returns.
#[inline]
pub fn resolve_budget(requested: u32) -> u32 {
    if requested > 0 {
        requested
    } else {
        env_threads()
    }
}

/// Returns `true` iff the caller should run the parallel path given the
/// effective budget and the total byte workload.
///
/// `threshold_bytes == None` uses [`DEFAULT_PARALLEL_THRESHOLD_BYTES`].
#[inline]
pub fn should_parallelise(budget: u32, work_bytes: usize, threshold_bytes: Option<usize>) -> bool {
    if budget == 0 {
        return false;
    }
    let threshold = threshold_bytes.unwrap_or(DEFAULT_PARALLEL_THRESHOLD_BYTES);
    work_bytes >= threshold
}

/// Returns `true` iff a (encoding, filter, compression) triple is
/// axis-B-friendly — i.e. at least one stage can usefully spend an
/// `intra_codec_threads > 0` budget.
///
/// Keep this in sync with the codec dispatch in `tensogram-encodings`:
/// whenever a codec grows internal parallelism, add its identifier here.
#[inline]
pub fn is_axis_b_friendly(encoding: &str, filter: &str, compression: &str) -> bool {
    matches!(compression, "blosc2" | "zstd")
        || matches!(encoding, "simple_packing")
        || matches!(filter, "shuffle")
}

/// Policy decision: given a multi-object workload with a total thread
/// budget, should we spread work across objects (axis A) or across a
/// single object's codec (axis B)?
///
/// Returns `true` for axis A (`par_iter` over objects), `false` for
/// axis B (sequential over objects; each object spends the full
/// budget inside its codec).
///
/// The rule is:
/// - `budget <= 1` or `n_objects <= 1` → axis A is impossible/useless,
///   return `false`.
/// - `any_object_axis_b_friendly == true` → axis B wins.  This matches
///   the "Tensogram messages tend to carry a small number of very
///   large objects" heuristic and avoids N\u{00b2} thread over-subscription
///   when blosc2 or zstd spawn their own internal worker pool per call.
/// - otherwise → axis A (`par_iter` across objects).
#[inline]
pub fn use_axis_a(n_objects: usize, budget: u32, any_object_axis_b_friendly: bool) -> bool {
    if budget <= 1 || n_objects <= 1 {
        return false;
    }
    !any_object_axis_b_friendly
}

/// Execute `f` with a rayon thread pool of size `budget`.
///
/// - `budget == 0` or `budget == 1`: runs `f` on the calling thread
///   (no pool built).  With `budget == 1` callers get deterministic
///   single-threaded execution but still go through the same code
///   paths — useful for testing.
/// - `budget >= 2` and the `threads` feature is enabled: builds a
///   scoped pool of `budget` workers and runs `f` via
///   `ThreadPool::install`.  The pool is dropped when `f` returns.
/// - `budget >= 2` and the `threads` feature is disabled: logs a
///   `tracing::warn!` on first use and falls back to sequential.
///
/// The pool build is intentionally scoped (not global) so that
/// different call sites can pick different thread counts without
/// interfering with each other.
#[inline]
pub fn with_pool<F, R>(budget: u32, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if budget <= 1 {
        return f();
    }

    #[cfg(feature = "threads")]
    {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(budget as usize)
            .thread_name(|i| format!("tensogram-worker-{i}"))
            .build()
        {
            Ok(pool) => pool.install(f),
            Err(e) => {
                warn_pool_build_failure(&e.to_string());
                f()
            }
        }
    }

    #[cfg(not(feature = "threads"))]
    {
        warn_threads_feature_disabled();
        f()
    }
}

#[cfg(feature = "threads")]
fn warn_pool_build_failure(msg: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        tracing::warn!(
            error = msg,
            "failed to build rayon thread pool; falling back to sequential execution"
        );
    });
}

#[cfg(not(feature = "threads"))]
fn warn_threads_feature_disabled() {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        tracing::warn!(
            "threads > 1 requested but the 'threads' cargo feature is disabled; \
             falling back to sequential execution"
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_zero_with_no_env_returns_zero() {
        // Env may or may not be set in the test environment; if set to 0
        // or unset the result must be 0.
        if std::env::var(ENV_THREADS).is_err() {
            assert_eq!(resolve_budget(0), 0);
        }
    }

    #[test]
    fn resolve_nonzero_ignores_env() {
        // Even if TENSOGRAM_THREADS is set, a non-zero caller request wins.
        assert_eq!(resolve_budget(4), 4);
    }

    #[test]
    fn should_parallelise_below_threshold() {
        assert!(!should_parallelise(8, 1024, None));
        assert!(!should_parallelise(0, usize::MAX, None));
    }

    #[test]
    fn should_parallelise_above_threshold() {
        assert!(should_parallelise(2, 1024 * 1024, None));
    }

    #[test]
    fn should_parallelise_custom_threshold() {
        assert!(should_parallelise(2, 1, Some(0)));
        assert!(!should_parallelise(2, 1024 * 1024, Some(usize::MAX)));
    }

    #[test]
    fn with_pool_budget_zero_runs_inline() {
        let result = with_pool(0, || 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn with_pool_budget_one_runs_inline() {
        let result = with_pool(1, || 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn is_axis_b_friendly_reports_known_codecs() {
        assert!(is_axis_b_friendly("none", "none", "blosc2"));
        assert!(is_axis_b_friendly("none", "none", "zstd"));
        assert!(is_axis_b_friendly("simple_packing", "none", "none"));
        assert!(is_axis_b_friendly("none", "shuffle", "none"));
        assert!(!is_axis_b_friendly("none", "none", "none"));
        assert!(!is_axis_b_friendly("none", "none", "lz4"));
        assert!(!is_axis_b_friendly("none", "none", "szip"));
        assert!(!is_axis_b_friendly("none", "none", "zfp"));
    }

    #[test]
    fn use_axis_a_single_object_is_always_false() {
        assert!(!use_axis_a(1, 8, false));
        assert!(!use_axis_a(1, 8, true));
        assert!(!use_axis_a(0, 8, false));
    }

    #[test]
    fn use_axis_a_low_budget_is_always_false() {
        assert!(!use_axis_a(10, 0, false));
        assert!(!use_axis_a(10, 1, false));
    }

    #[test]
    fn use_axis_a_multi_object_b_friendly_prefers_b() {
        assert!(!use_axis_a(10, 4, true));
    }

    #[test]
    fn use_axis_a_multi_object_non_b_friendly_uses_a() {
        assert!(use_axis_a(10, 4, false));
    }

    #[cfg(feature = "threads")]
    #[test]
    fn with_pool_budget_four_uses_pool() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = AtomicUsize::new(0);
        let sum: usize = with_pool(4, || {
            use rayon::prelude::*;
            (0..1000u64)
                .into_par_iter()
                .map(|i| {
                    counter.fetch_add(1, Ordering::Relaxed);
                    i as usize
                })
                .sum()
        });
        assert_eq!(sum, (0..1000).sum::<u64>() as usize);
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }
}
