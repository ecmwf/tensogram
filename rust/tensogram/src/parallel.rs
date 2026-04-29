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
//!   callers pay for an extra `u32` field, an `Option<usize>` field, and
//!   a single branch per encode/decode call.
//!
//! ## Thread budget resolution
//!
//! The resolution order in [`resolve_budget`] is:
//!
//! 1. If the caller set `threads > 0`, use that value verbatim (option
//!    beats environment).
//! 2. If `threads == 0`, fall back to `TENSOGRAM_THREADS`.  Any value
//!    that does not parse as a `u32` (or is missing entirely) is treated
//!    as `0`, i.e. sequential.
//!
//! The env var is parsed once per process and cached in a [`OnceLock`]
//! so repeated calls pay no `std::env::var` cost.  This matches the
//! `TENSOGRAM_COMPRESSION_BACKEND` pattern used elsewhere.
//!
//! ## Small-message threshold
//!
//! Building a rayon [`rayon::ThreadPool`] costs tens of microseconds
//! and is pointless for tiny payloads.  [`should_parallelise`] skips
//! the pool when the total work bytes are below a configurable
//! threshold (default 64 KiB).  Callers with a known-small workload
//! can either request `threads = 0` or pass
//! `parallel_threshold_bytes = Some(usize::MAX)` to force the
//! sequential path.

use std::sync::OnceLock;

use crate::error::{Result, TensogramError};

/// Default threshold below which the library runs sequentially even when
/// `threads > 0`.  Chosen to be well above the per-call rayon pool
/// construction cost (~10 µs) but small enough not to starve encode
/// paths that want parallelism.
///
/// Callers can override via `EncodeOptions.parallel_threshold_bytes`
/// or `DecodeOptions.parallel_threshold_bytes`.
pub const DEFAULT_PARALLEL_THRESHOLD_BYTES: usize = 64 * 1024;

/// Env var consulted when the caller-provided `threads` is `0`.
///
/// **Strict-input contract** (Wave 1.1): the value must parse as a
/// `u32` (decimal).  An empty / unset variable falls through to `0`
/// (sequential execution).  Any other unparseable value (e.g. `"four"`,
/// `"-2"`) surfaces as a [`TensogramError::Encoding`] from the next
/// encode / decode call rather than being silently swallowed.
pub const ENV_THREADS: &str = "TENSOGRAM_THREADS";

/// Pure parser for the `TENSOGRAM_THREADS` env variable shape.
/// Extracted so unit tests can exercise every input without
/// fighting the [`env_threads`] OnceLock cache.
///
/// Contract:
/// - `None` (env unset) → `Ok(0)` (sequential).
/// - `Some(empty / whitespace)` → `Ok(0)` (treat blank as unset).
/// - `Some("N")` where N parses as `u32` → `Ok(N)`.
/// - Any other value → `Err(_)` with a message naming the env var
///   and the offending input.
fn parse_env_threads(raw: Option<&str>) -> std::result::Result<u32, String> {
    match raw {
        None => Ok(0),
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Ok(0);
            }
            trimmed.parse::<u32>().map_err(|e| {
                format!(
                    "invalid {ENV_THREADS} value {s:?}: {e}; \
                     expected a non-negative integer (e.g. 0, 1, 4)"
                )
            })
        }
    }
}

fn env_threads() -> Result<u32> {
    // Cache the parse result (Ok or typed Err) so repeated callers
    // do not pay a syscall per call.  An invalid value is sticky:
    // every encode in the process surfaces the same diagnostic until
    // the env is fixed, matching `resolve_compression_backend`.
    static CACHED: OnceLock<std::result::Result<u32, String>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let var = std::env::var(ENV_THREADS).ok();
            parse_env_threads(var.as_deref())
        })
        .clone()
        .map_err(TensogramError::Encoding)
}

/// Resolve the effective thread budget given a caller-provided value.
///
/// - If `requested > 0`, returns `requested` unchanged (option beats env).
/// - If `requested == 0`, returns `TENSOGRAM_THREADS` (0 when unset).
/// - If `TENSOGRAM_THREADS` is set but unparseable, returns
///   [`TensogramError::Encoding`] (Wave 1.1: strict-input contract,
///   replaces the previous silent fall-back to 0).
///
/// Values are not clamped to CPU count — rayon handles that naturally,
/// and pathological N values simply see diminishing returns.
#[inline]
pub(crate) fn resolve_budget(requested: u32) -> Result<u32> {
    if requested > 0 {
        Ok(requested)
    } else {
        env_threads()
    }
}

/// Returns `true` iff the caller should run the parallel path given the
/// effective budget and the total byte workload.
///
/// `threshold_bytes == None` uses [`DEFAULT_PARALLEL_THRESHOLD_BYTES`].
#[inline]
pub(crate) fn should_parallelise(
    budget: u32,
    work_bytes: usize,
    threshold_bytes: Option<usize>,
) -> bool {
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
pub(crate) fn is_axis_b_friendly(encoding: &str, filter: &str, compression: &str) -> bool {
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
///   large objects" heuristic and avoids N² thread over-subscription
///   when blosc2 or zstd spawn their own internal worker pool per call.
/// - otherwise → axis A (`par_iter` across objects).
#[inline]
pub(crate) fn use_axis_a(n_objects: usize, budget: u32, any_object_axis_b_friendly: bool) -> bool {
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
///   [`rayon::ThreadPool::install`].  The pool is dropped when `f`
///   returns.
/// - `budget >= 2` and the `threads` feature is disabled: logs a
///   [`tracing::warn!`] on first use and falls back to sequential.
///
/// The pool build is intentionally scoped (not global) so that
/// different call sites can pick different thread counts without
/// interfering with each other.
#[inline]
pub(crate) fn with_pool<F, R>(budget: u32, f: F) -> R
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

/// Run a sequential object-loop either inside a scoped thread pool (so
/// any nested `par_iter` inside a codec picks it up via
/// [`rayon::current_num_threads`]) or inline.
///
/// This is the common "axis-B or purely sequential" path shared by
/// `encode_inner`, `decode`, `decode_object`, and
/// `decode_range_from_payload`.  Build a pool iff both `parallel` is
/// true *and* the intra-codec budget is large enough to benefit from
/// having a pool installed — otherwise run `f` on the caller thread
/// with no allocation.
#[inline]
pub(crate) fn run_maybe_pooled<F, R>(
    budget: u32,
    parallel: bool,
    intra_codec_threads: u32,
    f: F,
) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if parallel && intra_codec_threads > 1 {
        with_pool(budget, f)
    } else {
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
            assert_eq!(resolve_budget(0).unwrap(), 0);
        }
    }

    #[test]
    fn resolve_nonzero_ignores_env() {
        // Even if TENSOGRAM_THREADS is set, a non-zero caller request wins.
        assert_eq!(resolve_budget(4).unwrap(), 4);
    }

    // ── Pure parser for the env-var shape (Wave 1.1) ─────────────────

    #[test]
    fn parse_env_threads_unset_is_zero() {
        assert_eq!(parse_env_threads(None).unwrap(), 0);
    }

    #[test]
    fn parse_env_threads_empty_is_zero() {
        assert_eq!(parse_env_threads(Some("")).unwrap(), 0);
    }

    #[test]
    fn parse_env_threads_whitespace_is_zero() {
        assert_eq!(parse_env_threads(Some("   ")).unwrap(), 0);
    }

    #[test]
    fn parse_env_threads_valid_integer() {
        assert_eq!(parse_env_threads(Some("4")).unwrap(), 4);
        assert_eq!(parse_env_threads(Some("  4 ")).unwrap(), 4);
    }

    #[test]
    fn parse_env_threads_rejects_unparseable() {
        // Strict-input contract: anything that is not a valid u32 is
        // rejected with a typed error mentioning both the env var and
        // the offending value.  Earlier versions silently swallowed
        // such typos and ran sequentially.
        let err = parse_env_threads(Some("four")).unwrap_err();
        assert!(err.contains("TENSOGRAM_THREADS"), "msg: {err}");
        assert!(err.contains("four"), "msg: {err}");
        assert!(err.contains("non-negative integer"), "msg: {err}");
    }

    #[test]
    fn parse_env_threads_rejects_negative() {
        let err = parse_env_threads(Some("-2")).unwrap_err();
        assert!(err.contains("TENSOGRAM_THREADS"));
        assert!(err.contains("-2"));
    }

    #[test]
    fn parse_env_threads_rejects_float() {
        let err = parse_env_threads(Some("4.5")).unwrap_err();
        assert!(err.contains("TENSOGRAM_THREADS"));
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
    fn should_parallelise_boundary_values() {
        // exactly at threshold -> inclusive, returns true.
        assert!(should_parallelise(
            2,
            DEFAULT_PARALLEL_THRESHOLD_BYTES,
            None
        ));
        // one byte below -> false.
        assert!(!should_parallelise(
            2,
            DEFAULT_PARALLEL_THRESHOLD_BYTES - 1,
            None
        ));
        // zero budget always false regardless of bytes.
        assert!(!should_parallelise(0, usize::MAX, Some(0)));
        // explicit threshold 0 means "always parallel if budget > 0".
        assert!(should_parallelise(1, 0, Some(0)));
        // explicit threshold usize::MAX means "never parallel".
        assert!(!should_parallelise(
            u32::MAX,
            usize::MAX - 1,
            Some(usize::MAX)
        ));
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

    #[test]
    fn run_maybe_pooled_no_budget_runs_inline() {
        // budget=0 or parallel=false or intra<=1: never build a pool.
        assert_eq!(run_maybe_pooled(0, false, 0, || 7), 7);
        assert_eq!(run_maybe_pooled(4, false, 0, || 7), 7);
        assert_eq!(run_maybe_pooled(4, true, 0, || 7), 7);
        assert_eq!(run_maybe_pooled(4, true, 1, || 7), 7);
    }

    #[cfg(feature = "threads")]
    #[test]
    fn run_maybe_pooled_with_budget_installs_pool() {
        // intra_codec_threads >= 2 AND parallel=true: pool is installed,
        // so rayon::current_num_threads() reflects our budget.
        let observed = run_maybe_pooled(4, true, 4, rayon::current_num_threads);
        assert_eq!(observed, 4);
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
