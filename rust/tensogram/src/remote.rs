// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Remote object store backend for `TensogramFile`.
//!
//! Enables reading `.tgm` files from S3, GCS, Azure, and HTTP(S) via
//! the `object_store` crate. Feature-gated behind `remote`.

use std::collections::BTreeMap;
use std::ops::Range;
#[cfg(feature = "async")]
use std::sync::MutexGuard;
use std::sync::{Arc, Mutex, OnceLock};

use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt};
use url::Url;

use crate::decode::DecodeOptions;
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::metadata;
use crate::types::{DataObjectDescriptor, GlobalMetadata, IndexFrame};
use crate::wire::{
    DATA_OBJECT_FOOTER_SIZE, DataObjectFlags, FRAME_COMMON_FOOTER_SIZE, FRAME_END,
    FRAME_HEADER_SIZE, FrameHeader, FrameType, MAGIC, MessageFlags, POSTAMBLE_SIZE, PREAMBLE_SIZE,
    Postamble, Preamble,
};

// ── URL scheme detection ─────────────────────────────────────────────────────

const REMOTE_SCHEMES: &[&str] = &["s3", "s3a", "gs", "az", "azure", "http", "https"];

// ── Shared tokio runtime ─────────────────────────────────────────────────────

/// Process-wide shared tokio runtime for remote I/O.
///
/// Wrapped in `Result` so that `get_or_init` (which runs the closure
/// exactly once, no races) can cache a build failure without panicking.
static SHARED_RUNTIME: OnceLock<std::result::Result<tokio::runtime::Runtime, String>> =
    OnceLock::new();

fn shared_runtime() -> Result<&'static tokio::runtime::Runtime> {
    SHARED_RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name("tensogram-remote-io")
                .build()
                .map_err(|e| format!("tokio runtime: {e}"))
        })
        .as_ref()
        .map_err(|e| TensogramError::Remote(e.clone()))
}

/// Run an async operation synchronously using the shared runtime.
///
/// Three strategies based on the calling context:
///
/// - **Not in async context** (Python, CLI): direct `handle.block_on()`,
///   no extra thread creation.
/// - **Inside multi-thread tokio runtime** (`#[tokio::test]`, server
///   handler): `block_in_place` + `handle.block_on()`, which tells
///   tokio to spawn a replacement worker so the current one can block
///   without causing runtime starvation.
/// - **Inside current-thread tokio runtime**: scoped thread fallback,
///   since `block_in_place` is not supported on single-threaded
///   runtimes.
///
/// In all cases the shared runtime's event loop and connection pool
/// are reused, eliminating the per-call runtime creation overhead of
/// the old `block_on_thread` pattern.
fn block_on_shared<T, Fut>(future: Fut) -> Result<T>
where
    T: Send,
    Fut: std::future::Future<Output = std::result::Result<T, object_store::Error>> + Send,
{
    let rt = shared_runtime()?;
    let handle = rt.handle().clone();

    match tokio::runtime::Handle::try_current() {
        Err(_) => {
            // Not in an async context — drive the future directly.
            handle
                .block_on(future)
                .map_err(|e| TensogramError::Remote(e.to_string()))
        }
        Ok(current) if current.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            // Multi-thread runtime: block_in_place lets tokio spawn a
            // replacement worker so this one can block safely.
            tokio::task::block_in_place(|| {
                handle
                    .block_on(future)
                    .map_err(|e| TensogramError::Remote(e.to_string()))
            })
        }
        Ok(_) => {
            // Current-thread runtime: block_in_place is unsupported,
            // fall back to a scoped thread.
            std::thread::scope(|s| {
                match s
                    .spawn(|| {
                        handle
                            .block_on(future)
                            .map_err(|e| TensogramError::Remote(e.to_string()))
                    })
                    .join()
                {
                    Ok(result) => result,
                    Err(_) => Err(TensogramError::Remote(
                        "remote I/O thread panicked".to_string(),
                    )),
                }
            })
        }
    }
}

pub fn is_remote_url(source: &str) -> bool {
    match source.find("://") {
        Some(pos) => {
            let scheme = &source[..pos];
            REMOTE_SCHEMES
                .iter()
                .any(|s| s.eq_ignore_ascii_case(scheme))
        }
        None => false,
    }
}

// ── Scan options ─────────────────────────────────────────────────────────────

/// Internal knobs for the remote scan walker.
///
/// Crate-private until a follow-on sub-task promotes it to the public
/// API surface.  The default is `bidirectional = false` — the existing
/// forward-only behaviour — so every existing call site lands here
/// unchanged through [`RemoteBackend::open`] / [`RemoteBackend::open_async`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct RemoteScanOptions {
    /// Enable the bidirectional walker.  When `true`, the backend
    /// alternates forward and backward hops across the file using
    /// the v3 postamble's mirrored `total_length` field.
    ///
    /// `false` keeps the pre-bidirectional forward-only walker —
    /// today's default.
    pub(crate) bidirectional: bool,
    /// Upper bound on the `total_length` value advertised by a
    /// **postamble** during a backward hop.  Anything larger is
    /// treated as corruption and the backward walker yields to the
    /// forward walker.  Forward scanning is **not** affected by
    /// this cap (it uses `file_size` as its only bound).
    ///
    /// Default: 4 GiB.
    pub(crate) max_message_size: u64,
}

impl Default for RemoteScanOptions {
    fn default() -> Self {
        Self {
            bidirectional: false,
            max_message_size: 4 * 1024 * 1024 * 1024,
        }
    }
}

// ── Cached per-message layout ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MessageLayout {
    offset: u64,
    length: u64,
    preamble: Preamble,
    index: Option<IndexFrame>,
    global_metadata: Option<GlobalMetadata>,
}

// ── Backward hop primitives ──────────────────────────────────────────────────

/// Outcome of parsing a backward-fetch round before backward-preamble
/// validation (Round 2).  Pure — does not touch state.
#[derive(Debug)]
enum BackwardOutcome {
    /// Postamble parsed cleanly.  `msg_start` is the candidate
    /// message-start offset; the caller still has to validate the
    /// preamble at that offset (Round 2).
    NeedPreambleValidation { msg_start: u64, length: u64 },
    /// Format error — the backward walker yields and any provisional
    /// suffix layouts are discarded.  The string identifies which
    /// taxonomy row triggered.
    Format(&'static str),
    /// Streaming non-seekable producer (`postamble.total_length == 0`).
    /// Backward yields; forward continues.
    Streaming,
}

/// Parse a backward postamble fetch and decide what (if anything) to
/// validate next.  Pure function — independent of `RemoteState` and
/// `RemoteBackend`.
///
/// `pa_bytes` is expected to be the 24-byte postamble at
/// `[snap.prev - POSTAMBLE_SIZE, snap.prev)`.  `max_message_size`
/// caps **backward** postamble jumps only; forward scanning is bounded
/// by `file_size` (or the bidirectional dispatcher's `prev_scan_offset`
/// hand-off) and never consults this cap.
fn parse_backward_postamble(
    pa_bytes: &[u8],
    snap: &ScanSnapshot,
    max_message_size: u64,
) -> BackwardOutcome {
    let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;

    if pa_bytes.len() < POSTAMBLE_SIZE {
        return BackwardOutcome::Format("short-fetch-bwd");
    }
    if &pa_bytes[POSTAMBLE_SIZE - crate::wire::END_MAGIC.len()..] != crate::wire::END_MAGIC {
        return BackwardOutcome::Format("bad-end-magic-bwd");
    }
    let postamble = match Postamble::read_from(pa_bytes) {
        Ok(p) => p,
        Err(_) => return BackwardOutcome::Format("postamble-parse-error"),
    };
    let total = postamble.total_length;
    if total == 0 {
        return BackwardOutcome::Streaming;
    }
    if total < min_message_size {
        return BackwardOutcome::Format("length-below-minimum-bwd");
    }
    if total > max_message_size {
        return BackwardOutcome::Format("length-exceeds-max");
    }
    let msg_start = match snap.prev.checked_sub(total) {
        Some(s) => s,
        None => return BackwardOutcome::Format("backward-arith-underflow"),
    };
    if msg_start < snap.next {
        return BackwardOutcome::Format("backward-overlaps-forward");
    }
    BackwardOutcome::NeedPreambleValidation {
        msg_start,
        length: total,
    }
}

/// Outcome of validating Round 2 of a bidirectional round (the
/// backward preamble fetched at the candidate `msg_start`).
#[derive(Debug)]
enum BackwardCommit {
    Format(&'static str),
    Layout(MessageLayout),
}

/// Validate the backward preamble (Round 2) and produce a
/// commit-or-yield decision.  `msg_start` and `length` are the
/// candidate values from the matching [`BackwardOutcome`].
fn validate_backward_preamble(
    preamble_bytes: &[u8],
    msg_start: u64,
    length: u64,
) -> BackwardCommit {
    if preamble_bytes.len() < PREAMBLE_SIZE {
        return BackwardCommit::Format("short-fetch-bwd");
    }
    if &preamble_bytes[..MAGIC.len()] != MAGIC {
        return BackwardCommit::Format("bad-magic-bwd");
    }
    let preamble = match Preamble::read_from(preamble_bytes) {
        Ok(p) => p,
        Err(_) => return BackwardCommit::Format("preamble-parse-error-bwd"),
    };
    if preamble.total_length == 0 {
        return BackwardCommit::Format("streaming-preamble-non-tail");
    }
    if preamble.total_length != length {
        return BackwardCommit::Format("preamble-postamble-length-mismatch");
    }
    BackwardCommit::Layout(MessageLayout {
        offset: msg_start,
        length,
        preamble,
        index: None,
        global_metadata: None,
    })
}

/// Next step the eager-discovery accessors should take.  Computed
/// while holding the lock, then dispatched without it.
#[cfg(feature = "async")]
#[derive(Debug)]
enum EagerAction {
    /// Forward-only mode: combined chunk fetch (preamble + frames in
    /// one round trip) for the next message.
    ScanForwardEager,
    /// Bidirectional mode: paired preamble fetch.  Per-message
    /// frame discovery happens later via [`RemoteBackend::ensure_layout_async`].
    ScanBidir,
    /// Layout exists but metadata/index frames not yet fetched.
    Discover,
}

/// Outcome of parsing a forward preamble fetch.  Pure — does not
/// touch state.  Mirrors the format-error taxonomy of
/// [`scan_fwd_step_locked`] / [`scan_fwd_step_async`] so the
/// bidirectional round can apply the commit decision before
/// reacquiring the lock.
#[derive(Debug)]
enum ForwardOutcome {
    Hit {
        offset: u64,
        length: u64,
        preamble: Preamble,
        msg_end: u64,
    },
    /// Streaming preamble (`total_length == 0`).  Caller decides
    /// whether to record a streaming-tail layout (forward-only) or
    /// disable backward (bidirectional, see commit decision table).
    Streaming(u64),
    Terminate(&'static str),
}

/// `true` iff a successful forward Hit and a backward-discovered
/// layout describe exactly the same message.  Triggers on the
/// 1-message file (forward and backward both identify the only
/// message) and on odd-count crossovers where the meet-in-the-middle
/// lands on a single shared middle message.  In both cases the
/// commit decision table commits the forward layout once and skips
/// the backward record without clearing `suffix_rev` from earlier
/// rounds.
fn same_message_as_forward(fwd: &ForwardOutcome, layout: &MessageLayout) -> bool {
    matches!(
        fwd,
        ForwardOutcome::Hit { offset, length, .. }
            if *offset == layout.offset && *length == layout.length
    )
}

fn parse_forward_preamble(
    preamble_bytes: &[u8],
    pos: u64,
    file_size: u64,
    bound: u64,
) -> ForwardOutcome {
    let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;

    if preamble_bytes.len() < PREAMBLE_SIZE {
        return ForwardOutcome::Terminate("short-fetch-fwd");
    }
    if &preamble_bytes[..MAGIC.len()] != MAGIC {
        return ForwardOutcome::Terminate("bad-magic-fwd");
    }
    let preamble = match Preamble::read_from(preamble_bytes) {
        Ok(p) => p,
        Err(_) => return ForwardOutcome::Terminate("preamble-parse-error-fwd"),
    };
    let msg_len = preamble.total_length;
    if msg_len == 0 {
        let remaining = file_size - pos;
        return ForwardOutcome::Streaming(remaining);
    }
    match pos.checked_add(msg_len) {
        Some(end) if msg_len >= min_message_size && end <= bound => ForwardOutcome::Hit {
            offset: pos,
            length: msg_len,
            preamble,
            msg_end: end,
        },
        _ => ForwardOutcome::Terminate("length-out-of-range-fwd"),
    }
}

// ── Remote backend ───────────────────────────────────────────────────────────

pub(crate) struct RemoteBackend {
    source_url: String,
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    file_size: u64,
    state: Mutex<RemoteState>,
    scan_opts: RemoteScanOptions,
}

/// Two-cursor scan state for the bidirectional walker.
///
/// The forward cursor (`next_scan_offset`) and the backward cursor
/// (`prev_scan_offset`) bound the unknown gap.  In forward-only mode
/// (`bwd_active = false`), `suffix_rev` stays empty and
/// `prev_scan_offset` stays at `file_size`, collapsing the state
/// machine to today's behaviour — `scan_complete()` reduces to the
/// pre-existing `fwd_terminated` semantics.
///
/// Invariants pinned by the helpers below and the unit tests:
///
/// 1. `next_scan_offset <= prev_scan_offset <= file_size` while
///    scanning is in progress.
/// 2. `layouts` is a contiguous forward prefix; `next_scan_offset`
///    is the end of that prefix.
/// 3. `suffix_rev` is a contiguous EOF-first suffix; `prev_scan_offset`
///    is the start of its first entry while `bwd_active`.
/// 4. `bwd_active => !gap_closed && !fwd_terminated`.
/// 5. `gap_closed => suffix_rev.is_empty() && !bwd_active && scan_complete()`.
/// 6. `fwd_terminated => suffix_rev.is_empty() && !bwd_active`
///    (bidirectional is never recovery — `terminate_forward` cascades
///    to `disable_backward`).
#[derive(Debug, Default)]
struct RemoteState {
    /// Forward-discovered layouts.  `layouts[i]` always has stable
    /// absolute index `i` (forward indices grow contiguously from 0).
    layouts: Vec<MessageLayout>,
    /// Start of the unknown gap.  Forward walker advances this.
    next_scan_offset: u64,

    /// EOF-first backward-discovered layouts.  Always empty when
    /// `bwd_active = false`.  Reversed and merged into `layouts` on
    /// `gap_closed`.
    suffix_rev: Vec<MessageLayout>,
    /// End of the unknown gap.  Initialised to `file_size` in
    /// [`RemoteBackend::open`] / [`RemoteBackend::open_async`].
    prev_scan_offset: u64,
    /// `true` iff bidirectional mode is requested AND the backward
    /// walker has not yielded.  Once `false`, stays `false`.
    bwd_active: bool,

    /// Forward walker reached EOF or hit a format error and cannot
    /// advance further.
    fwd_terminated: bool,
    /// Forward and backward walkers met (`next == prev`); `suffix_rev`
    /// has been merged into `layouts`.
    gap_closed: bool,

    /// Monotone counter incremented on every state-machine transition.
    /// The lock-around-await protocol snapshots this together with
    /// `(next, prev)` and validates equality on reacquire so a stale
    /// paired fetch cannot commit after a fallback or termination has
    /// already mutated state.
    scan_epoch: u64,
}

/// Snapshot of the scan cursors taken before a network round-trip.
/// Validated for equality on lock reacquire — any mismatch means
/// another caller mutated state in the meantime, and the in-flight
/// work must be discarded.
#[derive(Debug, Clone, Copy)]
struct ScanSnapshot {
    next: u64,
    prev: u64,
    epoch: u64,
}

/// Emit a one-shot tracing event identifying which scan mode the
/// freshly-opened backend is using.  Filterable via
/// `RUST_LOG=tensogram::remote_scan=debug`.
fn emit_scan_mode(scan_opts: &RemoteScanOptions) {
    let mode = if scan_opts.bidirectional {
        "bidirectional"
    } else {
        "forward-only"
    };
    tracing::debug!(
        target: "tensogram::remote_scan",
        mode = mode,
        max_message_size = scan_opts.max_message_size,
        "remote scan mode",
    );
}

impl RemoteState {
    /// Replaces the old `scan_complete: bool` field.  When forward-only
    /// is active (`bwd_active = false` and `suffix_rev.is_empty()`)
    /// this collapses to `fwd_terminated`, byte-identical to the
    /// previous semantics.
    fn scan_complete(&self) -> bool {
        self.gap_closed || (self.fwd_terminated && self.suffix_rev.is_empty())
    }

    /// `true` once forward and backward cursors have met or crossed.
    fn walkers_overlap(&self) -> bool {
        self.next_scan_offset >= self.prev_scan_offset
    }

    /// Append a forward-discovered layout, advance the cursor, bump
    /// the epoch.
    fn record_forward_hop(&mut self, layout: MessageLayout) {
        debug_assert!(
            layout.offset.checked_add(layout.length).is_some(),
            "forward end must fit in u64; validated by scan_next before recording",
        );
        let end = layout.offset + layout.length;
        debug_assert!(end <= self.prev_scan_offset);
        let offset = layout.offset;
        let length = layout.length;
        self.layouts.push(layout);
        self.next_scan_offset = end;
        self.scan_epoch = self.scan_epoch.wrapping_add(1);
        tracing::debug!(
            target: "tensogram::remote_scan",
            direction = "fwd",
            offset = offset,
            length = length,
            "scan hop",
        );
    }

    /// Append a backward-discovered layout, retreat the cursor, bump
    /// the epoch.  Pre: `bwd_active`, layout extends from
    /// `prev_scan_offset` exactly back to `layout.offset`.
    fn record_backward_hop(&mut self, layout: MessageLayout) {
        debug_assert!(self.bwd_active);
        debug_assert!(layout.offset >= self.next_scan_offset);
        debug_assert_eq!(layout.offset + layout.length, self.prev_scan_offset);
        let offset = layout.offset;
        let length = layout.length;
        self.prev_scan_offset = layout.offset;
        self.suffix_rev.push(layout);
        self.scan_epoch = self.scan_epoch.wrapping_add(1);
        tracing::debug!(
            target: "tensogram::remote_scan",
            direction = "bwd",
            offset = offset,
            length = length,
            "scan hop",
        );
    }

    /// Snapshot for the lock-around-await protocol.
    fn snapshot(&self) -> ScanSnapshot {
        ScanSnapshot {
            next: self.next_scan_offset,
            prev: self.prev_scan_offset,
            epoch: self.scan_epoch,
        }
    }

    /// `true` iff the state still matches the snapshot taken before
    /// a network round-trip.  Used at commit time to detect any
    /// concurrent mutation.
    fn matches(&self, snap: &ScanSnapshot) -> bool {
        self.next_scan_offset == snap.next
            && self.prev_scan_offset == snap.prev
            && self.scan_epoch == snap.epoch
    }

    /// Disable the backward walker and discard provisional layouts.
    /// Bumps the epoch so any in-flight paired fetch fails its
    /// snapshot validation.  Idempotent — no-op when the walker is
    /// already inactive and `suffix_rev` is empty.
    fn disable_backward(&mut self, reason: &'static str) {
        if !self.bwd_active && self.suffix_rev.is_empty() {
            return;
        }
        self.bwd_active = false;
        self.suffix_rev.clear();
        self.scan_epoch = self.scan_epoch.wrapping_add(1);
        tracing::debug!(
            target: "tensogram::remote_scan",
            reason = reason,
            "backward walker disabled",
        );
    }

    /// Terminate the forward walker.  Per the state-machine
    /// invariant `fwd_terminated => suffix_rev.is_empty() &&
    /// !bwd_active`, this also disables the backward walker and
    /// discards any provisional suffix layouts — bidirectional is
    /// an optimisation, never a recovery mode.
    fn terminate_forward(&mut self, reason: &'static str) {
        if !self.fwd_terminated {
            self.fwd_terminated = true;
            self.scan_epoch = self.scan_epoch.wrapping_add(1);
            tracing::debug!(
                target: "tensogram::remote_scan",
                reason = reason,
                "forward walker terminated",
            );
        }
        self.disable_backward(reason);
    }

    /// Merge `suffix_rev` into `layouts` (reversed) and mark the scan
    /// complete.  Pre: walkers met cleanly (`next_scan_offset ==
    /// prev_scan_offset`); `bwd_active` was true before the call.
    fn close_gap(&mut self) {
        debug_assert!(self.bwd_active);
        debug_assert_eq!(self.next_scan_offset, self.prev_scan_offset);
        let mut tail = std::mem::take(&mut self.suffix_rev);
        tail.reverse();
        self.layouts.extend(tail);
        self.gap_closed = true;
        self.bwd_active = false;
        self.scan_epoch = self.scan_epoch.wrapping_add(1);
        tracing::debug!(
            target: "tensogram::remote_scan",
            messages = self.layouts.len(),
            "gap closed",
        );
    }
}

impl std::fmt::Debug for RemoteBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteBackend")
            .field("source", &self.source_url)
            .field("file_size", &self.file_size)
            .field("scan_opts", &self.scan_opts)
            .field(
                "messages",
                &self
                    .state
                    .lock()
                    .map(|state| state.layouts.len())
                    .unwrap_or(0),
            )
            .finish()
    }
}

impl RemoteBackend {
    pub(crate) fn source_url(&self) -> &str {
        &self.source_url
    }

    #[cfg(feature = "async")]
    fn lock_state(&self) -> Result<MutexGuard<'_, RemoteState>> {
        self.state
            .lock()
            .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))
    }

    pub(crate) fn open(source: &str, storage_options: &BTreeMap<String, String>) -> Result<Self> {
        Self::open_with_scan_opts(source, storage_options, RemoteScanOptions::default())
    }

    fn validate_scan_opts(opts: &RemoteScanOptions) -> Result<()> {
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        if opts.bidirectional && opts.max_message_size < min_message_size {
            return Err(TensogramError::Remote(format!(
                "RemoteScanOptions.max_message_size ({}) is below the minimum \
                 message size ({min_message_size}); the backward walker would \
                 yield on every postamble",
                opts.max_message_size,
            )));
        }
        Ok(())
    }

    pub(crate) fn open_with_scan_opts(
        source: &str,
        storage_options: &BTreeMap<String, String>,
        scan_opts: RemoteScanOptions,
    ) -> Result<Self> {
        Self::validate_scan_opts(&scan_opts)?;
        emit_scan_mode(&scan_opts);
        let url = Url::parse(source)
            .map_err(|e| TensogramError::Remote(format!("invalid URL '{source}': {e}")))?;

        let mut opts: Vec<(String, String)> = storage_options
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if url.scheme() == "http" && !opts.iter().any(|(k, _)| k == "allow_http") {
            opts.push(("allow_http".to_string(), "true".to_string()));
        }
        let (store, path) = object_store::parse_url_opts(&url, opts)
            .map_err(|e| TensogramError::Remote(format!("cannot open '{source}': {e}")))?;

        let store: Arc<dyn ObjectStore> = Arc::from(store);

        let head_store = store.clone();
        let head_path = path.clone();
        let meta = block_on_shared(async move { head_store.head(&head_path).await })?;

        let file_size = meta.size;
        if file_size < (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64 {
            return Err(TensogramError::Remote(format!(
                "remote file too small ({file_size} bytes)"
            )));
        }

        let backend = RemoteBackend {
            source_url: source.to_string(),
            store,
            path,
            file_size,
            state: Mutex::new(RemoteState {
                prev_scan_offset: file_size,
                bwd_active: scan_opts.bidirectional,
                ..RemoteState::default()
            }),
            scan_opts,
        };
        {
            let mut state = backend
                .state
                .lock()
                .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
            backend.scan_next_locked(&mut state)?;
            if state.layouts.is_empty() {
                return Err(TensogramError::Remote(
                    "no valid messages found in remote file".to_string(),
                ));
            }
        }
        Ok(backend)
    }

    // ── Range reads ──────────────────────────────────────────────────────

    fn get_range(&self, range: Range<u64>) -> Result<Bytes> {
        let store = self.store.clone();
        let path = self.path.clone();
        block_on_shared(async move { store.get_range(&path, range).await })
    }

    // ── Message scanning ─────────────────────────────────────────────────

    fn scan_next_locked(&self, state: &mut RemoteState) -> Result<()> {
        self.scan_fwd_step_locked(state, self.file_size)
    }

    /// One forward hop bounded by `bound` (exclusive upper offset).
    ///
    /// In forward-only mode `bound == self.file_size` (today's
    /// behaviour).  In bidirectional mode the bidirectional dispatcher
    /// passes `state.prev_scan_offset` so the forward walker cannot
    /// overrun into the region already claimed by the backward
    /// walker.  Streaming preambles (`total_length == 0`) are
    /// tail-only by spec; the streaming END_MAGIC is therefore read
    /// at `self.file_size - 8` regardless of `bound`.
    fn scan_fwd_step_locked(&self, state: &mut RemoteState, bound: u64) -> Result<()> {
        if state.scan_complete() {
            return Ok(());
        }
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let pos = state.next_scan_offset;

        if pos + min_message_size > bound {
            state.terminate_forward("eof");
            return Ok(());
        }

        let preamble_bytes = self.get_range(pos..pos + PREAMBLE_SIZE as u64)?;
        if &preamble_bytes[..MAGIC.len()] != MAGIC {
            state.terminate_forward("bad-magic-fwd");
            return Ok(());
        }

        let preamble = match Preamble::read_from(&preamble_bytes) {
            Ok(p) => p,
            Err(_) => {
                state.terminate_forward("preamble-parse-error-fwd");
                return Ok(());
            }
        };

        let msg_len = preamble.total_length;

        if msg_len == 0 {
            let remaining = self.file_size - pos;
            if remaining < min_message_size {
                state.terminate_forward("streaming-tail-too-small");
                return Ok(());
            }
            let end_magic_pos = self.file_size - crate::wire::END_MAGIC.len() as u64;
            let end_bytes =
                self.get_range(end_magic_pos..end_magic_pos + crate::wire::END_MAGIC.len() as u64)?;
            if &end_bytes[..] != crate::wire::END_MAGIC {
                state.terminate_forward("streaming-end-magic-mismatch");
                return Ok(());
            }
            state.record_forward_hop(MessageLayout {
                offset: pos,
                length: remaining,
                preamble,
                index: None,
                global_metadata: None,
            });
            state.terminate_forward("streaming-tail");
            return Ok(());
        }

        match pos.checked_add(msg_len) {
            Some(end) if msg_len >= min_message_size && end <= bound => {}
            _ => {
                state.terminate_forward("length-out-of-range-fwd");
                return Ok(());
            }
        }

        state.record_forward_hop(MessageLayout {
            offset: pos,
            length: msg_len,
            preamble,
            index: None,
            global_metadata: None,
        });
        Ok(())
    }

    /// One step of progress in whichever scan mode is active.  When
    /// `bwd_active`, dispatches to the bidirectional paired round;
    /// otherwise falls back to bounded forward-only.  This is the
    /// single funnel through which every read accessor scans, so
    /// adding new state-machine steps doesn't require touching the
    /// accessors a second time.
    fn scan_step_locked(&self, state: &mut RemoteState) -> Result<()> {
        if state.scan_complete() {
            return Ok(());
        }
        if state.bwd_active && !state.fwd_terminated {
            self.scan_bidir_round_locked(state)
        } else {
            let bound = self.forward_bound(state);
            self.scan_fwd_step_locked(state, bound)
        }
    }

    /// Upper bound for the next forward hop.  In bidirectional mode,
    /// caps at `prev_scan_offset` so the forward walker cannot
    /// overrun into the suffix already claimed by backward.  Once
    /// backward has yielded and `suffix_rev` is empty, the bound
    /// reverts to `file_size`.
    fn forward_bound(&self, state: &RemoteState) -> u64 {
        if state.suffix_rev.is_empty() {
            self.file_size
        } else {
            state.prev_scan_offset
        }
    }

    /// One bidirectional scan round: paired forward preamble + backward
    /// postamble fetch, followed by a backward-preamble validation
    /// fetch when the postamble parses cleanly.  Commits the parsed
    /// outcomes atomically through the [`RemoteState`] decision
    /// table.
    ///
    /// On any *transport* error (timeout, 503, abort), state is left
    /// unchanged and the error propagates so the caller can retry.
    /// *Format* errors are absorbed locally — backward yields,
    /// forward keeps going.
    fn scan_bidir_round_locked(&self, state: &mut RemoteState) -> Result<()> {
        let snap = state.snapshot();
        // In bidirectional mode `prev_scan_offset` IS the forward
        // bound: it equals `file_size` while `suffix_rev` is empty
        // and the start of the leftmost suffix layout otherwise.  In
        // forward-only mode this function is never reached (the
        // dispatcher routes to `scan_fwd_step_locked` directly).
        let bound = snap.prev;
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;

        // Walkers about to overlap?  One bounded forward step
        // closes the gap; backward never gets called at this point.
        if snap.prev < snap.next + 2 * min_message_size {
            return self.scan_fwd_step_locked(state, bound);
        }

        let fwd_r = snap.next..snap.next + PREAMBLE_SIZE as u64;
        let bwd_r = snap.prev - POSTAMBLE_SIZE as u64..snap.prev;

        let bytes = block_on_shared({
            let store = self.store.clone();
            let path = self.path.clone();
            let ranges = vec![fwd_r.clone(), bwd_r.clone()];
            async move { store.get_ranges(&path, &ranges).await }
        })?;
        if bytes.len() != 2 {
            return Err(TensogramError::Remote(format!(
                "get_ranges returned {} buffers, expected 2",
                bytes.len()
            )));
        }

        let bwd_outcome =
            parse_backward_postamble(&bytes[1], &snap, self.scan_opts.max_message_size);

        let bwd_round2 = match &bwd_outcome {
            BackwardOutcome::NeedPreambleValidation { msg_start, .. } => {
                Some(self.get_range(*msg_start..*msg_start + PREAMBLE_SIZE as u64)?)
            }
            _ => None,
        };

        if !state.matches(&snap) {
            return Ok(());
        }

        let fwd_kind = parse_forward_preamble(&bytes[0], snap.next, self.file_size, bound);
        self.apply_round_outcomes(state, fwd_kind, bwd_outcome, bwd_round2.as_deref());
        Ok(())
    }

    /// Apply the (forward, backward, optional Round-2) outcomes of a
    /// single bidirectional round to the locked state.  Backward is
    /// processed first so a format/streaming yield can fire before
    /// the forward commit; this keeps the "forward terminate clears
    /// suffix" path simple.
    fn apply_round_outcomes(
        &self,
        state: &mut RemoteState,
        fwd: ForwardOutcome,
        bwd: BackwardOutcome,
        bwd_round2: Option<&[u8]>,
    ) {
        match bwd {
            BackwardOutcome::Format(reason) => state.disable_backward(reason),
            BackwardOutcome::Streaming => state.disable_backward("streaming-zero-bwd"),
            BackwardOutcome::NeedPreambleValidation { msg_start, length } => {
                let validation = bwd_round2
                    .map(|bytes| validate_backward_preamble(bytes, msg_start, length))
                    .unwrap_or(BackwardCommit::Format("missing-round2-buffer"));
                Self::commit_or_yield_backward(state, &fwd, validation);
            }
        }

        match fwd {
            ForwardOutcome::Hit {
                offset,
                length,
                preamble,
                msg_end,
            } => {
                debug_assert_eq!(offset, state.next_scan_offset);
                debug_assert_eq!(msg_end, offset + length);
                state.record_forward_hop(MessageLayout {
                    offset,
                    length,
                    preamble,
                    index: None,
                    global_metadata: None,
                });
            }
            ForwardOutcome::Streaming(_remaining) => {
                // Streaming preamble in bidirectional mode is a
                // contradiction: backward has discovered tail
                // content, so the streaming "tail" can't actually
                // extend to file end.  Yield backward and let
                // forward-only logic re-handle this position on the
                // next step (which will set fwd_terminated on the
                // streaming branch).
                state.disable_backward("streaming-fwd-non-tail");
            }
            ForwardOutcome::Terminate(reason) => state.terminate_forward(reason),
        }

        if state.bwd_active && state.walkers_overlap() {
            state.close_gap();
        }
    }

    /// Decide what to do with a Round-2-validated backward layout
    /// before the forward commit fires.  Three branches:
    ///
    /// - `bwd_active == false`: a concurrent caller disabled the
    ///   walker between Round 1 and Round 2; silently drop the
    ///   layout.
    /// - `same_message_as_forward`: the 1-message file and odd-count
    ///   middle-meet case — forward will commit this exact layout
    ///   itself.  Yield silently (do NOT call `disable_backward` —
    ///   that would clear `suffix_rev` from earlier rounds).
    /// - true overlap (`fwd.msg_end > layout.offset`): only happens
    ///   on corruption; disable backward.
    /// - otherwise: commit the backward hop.
    fn commit_or_yield_backward(
        state: &mut RemoteState,
        fwd: &ForwardOutcome,
        validation: BackwardCommit,
    ) {
        let layout = match validation {
            BackwardCommit::Format(reason) => {
                state.disable_backward(reason);
                return;
            }
            BackwardCommit::Layout(layout) => layout,
        };
        if !state.bwd_active {
            return;
        }
        if same_message_as_forward(fwd, &layout) {
            return;
        }
        if let ForwardOutcome::Hit { msg_end, .. } = fwd {
            if *msg_end > layout.offset {
                state.disable_backward("backward-overlaps-forward");
                return;
            }
        }
        state.record_backward_hop(layout);
    }

    fn ensure_message_locked(&self, state: &mut RemoteState, msg_idx: usize) -> Result<()> {
        while msg_idx >= state.layouts.len() && !state.scan_complete() {
            self.scan_step_locked(state)?;
        }
        if msg_idx >= state.layouts.len() {
            return Err(TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                msg_idx,
                state.layouts.len()
            )));
        }
        Ok(())
    }

    fn scan_all_locked(&self, state: &mut RemoteState) -> Result<()> {
        while !state.scan_complete() {
            self.scan_step_locked(state)?;
        }
        Ok(())
    }

    fn scan_and_discover_next_locked(&self, state: &mut RemoteState) -> Result<()> {
        if state.scan_complete() {
            return Ok(());
        }
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let pos = state.next_scan_offset;

        if pos + min_message_size > self.file_size {
            state.terminate_forward("eof");
            return Ok(());
        }

        let chunk_size = (self.file_size - pos).min(256 * 1024);
        let chunk = self.get_range(pos..pos + chunk_size)?;

        if chunk.len() < PREAMBLE_SIZE || &chunk[..MAGIC.len()] != MAGIC {
            state.terminate_forward("bad-magic-fwd");
            return Ok(());
        }

        let preamble = match Preamble::read_from(&chunk[..PREAMBLE_SIZE]) {
            Ok(p) => p,
            Err(_) => {
                state.terminate_forward("preamble-parse-error-fwd");
                return Ok(());
            }
        };

        let msg_len = preamble.total_length;

        if msg_len == 0 {
            let remaining = self.file_size - pos;
            if remaining < min_message_size {
                state.terminate_forward("streaming-tail-too-small");
                return Ok(());
            }
            let end_magic_pos = self.file_size - crate::wire::END_MAGIC.len() as u64;
            let end_bytes =
                self.get_range(end_magic_pos..end_magic_pos + crate::wire::END_MAGIC.len() as u64)?;
            if &end_bytes[..] != crate::wire::END_MAGIC {
                state.terminate_forward("streaming-end-magic-mismatch");
                return Ok(());
            }
            state.record_forward_hop(MessageLayout {
                offset: pos,
                length: remaining,
                preamble,
                index: None,
                global_metadata: None,
            });
            state.terminate_forward("streaming-tail");
            return Ok(());
        }

        match pos.checked_add(msg_len) {
            Some(end) if msg_len >= min_message_size && end <= self.file_size => {}
            _ => {
                state.terminate_forward("length-out-of-range-fwd");
                return Ok(());
            }
        }

        let flags = preamble.flags;
        let msg_idx = state.layouts.len();

        state.record_forward_hop(MessageLayout {
            offset: pos,
            length: msg_len,
            preamble,
            index: None,
            global_metadata: None,
        });

        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            let chunk_end = (msg_len as usize).min(chunk.len());
            Self::parse_header_frames(state, msg_idx, &chunk[..chunk_end])?;
        } else if flags.has(MessageFlags::FOOTER_METADATA) && flags.has(MessageFlags::FOOTER_INDEX)
        {
            self.discover_footer_layout_from_suffix_locked(state, msg_idx)?;
        }

        Ok(())
    }

    /// Discover footer layout by reading a single suffix chunk from the
    /// message end.  The suffix covers both the postamble and the footer
    /// region (metadata + index frames) in one GET, halving the round
    /// trips compared to separate postamble + footer reads.
    ///
    /// The suffix size is capped at 256 KB — footer regions are typically
    /// a few KB.  If `first_footer_offset` points outside the suffix,
    /// falls back to a separate read for the footer region.
    fn discover_footer_layout_from_suffix_locked(
        &self,
        state: &mut RemoteState,
        msg_idx: usize,
    ) -> Result<()> {
        let msg_offset = state.layouts[msg_idx].offset;
        let msg_len = state.layouts[msg_idx].length;

        let suffix_size = msg_len.min(256 * 1024);
        let msg_end = msg_offset
            .checked_add(msg_len)
            .ok_or_else(|| TensogramError::Remote("message end overflow".to_string()))?;
        let suffix_start = msg_end - suffix_size;
        let suffix = self.get_range(suffix_start..msg_end)?;

        if suffix.len() < POSTAMBLE_SIZE {
            return Err(TensogramError::Remote(
                "suffix too short for postamble".to_string(),
            ));
        }

        let pa_bytes = &suffix[suffix.len() - POSTAMBLE_SIZE..];
        let postamble = Postamble::read_from(pa_bytes)?;

        if postamble.first_footer_offset < PREAMBLE_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "first_footer_offset ({}) is before preamble end ({PREAMBLE_SIZE})",
                postamble.first_footer_offset
            )));
        }

        let footer_abs_start = msg_offset
            .checked_add(postamble.first_footer_offset)
            .ok_or_else(|| TensogramError::Remote("footer offset overflow".to_string()))?;
        let footer_abs_end = msg_end - POSTAMBLE_SIZE as u64;

        if footer_abs_start >= footer_abs_end {
            return Err(TensogramError::Remote(
                "first_footer_offset points at or past postamble".to_string(),
            ));
        }

        if footer_abs_start >= suffix_start {
            let local_start = (footer_abs_start - suffix_start) as usize;
            let local_end = suffix.len() - POSTAMBLE_SIZE;
            Self::parse_footer_frames(state, msg_idx, &suffix[local_start..local_end])
        } else {
            let footer_bytes = self.get_range(footer_abs_start..footer_abs_end)?;
            Self::parse_footer_frames(state, msg_idx, &footer_bytes)
        }
    }

    fn ensure_layout_eager_locked(&self, state: &mut RemoteState, msg_idx: usize) -> Result<()> {
        while msg_idx >= state.layouts.len() && !state.scan_complete() {
            if state.bwd_active && !state.fwd_terminated {
                self.scan_bidir_round_locked(state)?;
            } else {
                self.scan_and_discover_next_locked(state)?;
            }
        }
        if msg_idx >= state.layouts.len() {
            return Err(TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                msg_idx,
                state.layouts.len()
            )));
        }
        if state.layouts[msg_idx].global_metadata.is_some()
            && state.layouts[msg_idx].index.is_some()
        {
            return Ok(());
        }
        self.ensure_layout_locked(state, msg_idx)
    }

    // ── Layout discovery (metadata + index for a single message) ─────────

    fn ensure_layout_locked(&self, state: &mut RemoteState, msg_idx: usize) -> Result<()> {
        self.ensure_message_locked(state, msg_idx)?;
        if state.layouts[msg_idx].global_metadata.is_some()
            && state.layouts[msg_idx].index.is_some()
        {
            return Ok(());
        }

        let flags = state.layouts[msg_idx].preamble.flags;

        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            self.discover_header_layout_locked(state, msg_idx)?;
        } else if flags.has(MessageFlags::FOOTER_METADATA) && flags.has(MessageFlags::FOOTER_INDEX)
        {
            self.discover_footer_layout_locked(state, msg_idx)?;
        } else {
            return Err(TensogramError::Remote(
                "remote access requires header-indexed or footer-indexed messages".to_string(),
            ));
        }

        Ok(())
    }

    fn discover_footer_layout_locked(&self, state: &mut RemoteState, msg_idx: usize) -> Result<()> {
        let msg_offset = state.layouts[msg_idx].offset;
        let msg_len = state.layouts[msg_idx].length;

        let pa_offset = msg_offset
            .checked_add(msg_len)
            .and_then(|end| end.checked_sub(POSTAMBLE_SIZE as u64))
            .ok_or_else(|| TensogramError::Remote("postamble offset overflow".to_string()))?;
        let pa_bytes = self.get_range(pa_offset..pa_offset + POSTAMBLE_SIZE as u64)?;
        let postamble = Postamble::read_from(&pa_bytes)?;

        if postamble.first_footer_offset < PREAMBLE_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "first_footer_offset ({}) is before preamble end ({})",
                postamble.first_footer_offset, PREAMBLE_SIZE
            )));
        }
        let footer_start = msg_offset
            .checked_add(postamble.first_footer_offset)
            .ok_or_else(|| TensogramError::Remote("footer offset overflow".to_string()))?;
        let footer_end = pa_offset;
        if footer_start >= footer_end {
            return Err(TensogramError::Remote(
                "first_footer_offset points at or past postamble".to_string(),
            ));
        }
        let footer_bytes = self.get_range(footer_start..footer_end)?;

        Self::parse_footer_frames(state, msg_idx, &footer_bytes)
    }

    fn discover_header_layout_locked(&self, state: &mut RemoteState, msg_idx: usize) -> Result<()> {
        let layout = &state.layouts[msg_idx];
        let msg_offset = layout.offset;
        let msg_len = layout.length;

        // Read a generous initial chunk: up to 256KB or the message size.
        // Header metadata + index are typically a few KB.
        let chunk_size = msg_len.min(256 * 1024);
        let header_bytes = self.get_range(msg_offset..msg_offset + chunk_size)?;

        Self::parse_header_frames(state, msg_idx, &header_bytes)
    }

    fn parse_header_frames(state: &mut RemoteState, msg_idx: usize, buf: &[u8]) -> Result<()> {
        let min_frame_size = FRAME_HEADER_SIZE + FRAME_END.len();
        let mut pos = PREAMBLE_SIZE;

        while pos + FRAME_HEADER_SIZE <= buf.len() {
            if &buf[pos..pos + 2] != b"FR" {
                pos += 1;
                continue;
            }
            let fh = FrameHeader::read_from(&buf[pos..])?;
            let frame_total = usize::try_from(fh.total_length).map_err(|_| {
                TensogramError::Remote("frame total_length does not fit in usize".to_string())
            })?;

            if frame_total < min_frame_size {
                return Err(TensogramError::Remote(format!(
                    "frame total_length ({frame_total}) smaller than minimum ({min_frame_size})"
                )));
            }
            let frame_end = match pos.checked_add(frame_total) {
                Some(end) if end <= buf.len() => end,
                _ => break,
            };

            if &buf[frame_end - FRAME_END.len()..frame_end] != FRAME_END {
                return Err(TensogramError::Remote(
                    "frame missing ENDF trailer".to_string(),
                ));
            }

            let payload = &buf[pos + FRAME_HEADER_SIZE..frame_end - FRAME_COMMON_FOOTER_SIZE];

            match fh.frame_type {
                FrameType::HeaderMetadata => {
                    let meta = metadata::cbor_to_global_metadata(payload)?;
                    state.layouts[msg_idx].global_metadata = Some(meta);
                }
                FrameType::HeaderIndex => {
                    let idx = metadata::cbor_to_index(payload)?;
                    state.layouts[msg_idx].index = Some(idx);
                }
                FrameType::NTensorFrame | FrameType::PrecederMetadata => {
                    break;
                }
                _ => {}
            }

            let aligned = (frame_end.saturating_add(7)) & !7;
            pos = aligned.min(buf.len());
        }

        if state.layouts[msg_idx].global_metadata.is_none() {
            return Err(TensogramError::Remote(
                "header region did not contain a metadata frame".to_string(),
            ));
        }
        if state.layouts[msg_idx].index.is_none() {
            return Err(TensogramError::Remote(
                "header region did not contain an index frame (header chunk may be too small)"
                    .to_string(),
            ));
        }

        Ok(())
    }

    fn parse_footer_frames(state: &mut RemoteState, msg_idx: usize, buf: &[u8]) -> Result<()> {
        let min_frame_size = FRAME_HEADER_SIZE + FRAME_END.len();
        let mut pos = 0;

        while pos + FRAME_HEADER_SIZE <= buf.len() {
            if &buf[pos..pos + 2] != b"FR" {
                pos += 1;
                continue;
            }
            let fh = FrameHeader::read_from(&buf[pos..])?;
            let frame_total = usize::try_from(fh.total_length).map_err(|_| {
                TensogramError::Remote(
                    "footer frame total_length does not fit in usize".to_string(),
                )
            })?;

            if frame_total < min_frame_size {
                return Err(TensogramError::Remote(format!(
                    "footer frame total_length ({frame_total}) smaller than minimum ({min_frame_size})"
                )));
            }
            let frame_end = match pos.checked_add(frame_total) {
                Some(end) if end <= buf.len() => end,
                _ => break,
            };

            if &buf[frame_end - FRAME_END.len()..frame_end] != FRAME_END {
                return Err(TensogramError::Remote(
                    "footer frame missing ENDF trailer".to_string(),
                ));
            }

            let payload = &buf[pos + FRAME_HEADER_SIZE..frame_end - FRAME_COMMON_FOOTER_SIZE];

            match fh.frame_type {
                FrameType::FooterMetadata => {
                    let meta = metadata::cbor_to_global_metadata(payload)?;
                    state.layouts[msg_idx].global_metadata = Some(meta);
                }
                FrameType::FooterIndex => {
                    let idx = metadata::cbor_to_index(payload)?;
                    state.layouts[msg_idx].index = Some(idx);
                }
                _ => {}
            }

            let aligned = (frame_end.saturating_add(7)) & !7;
            pos = aligned.min(buf.len());
        }

        if state.layouts[msg_idx].global_metadata.is_none() {
            return Err(TensogramError::Remote(
                "footer region did not contain a metadata frame".to_string(),
            ));
        }
        if state.layouts[msg_idx].index.is_none() {
            return Err(TensogramError::Remote(
                "footer region did not contain an index frame".to_string(),
            ));
        }

        Ok(())
    }

    // ── Public API used by TensogramFile ─────────────────────────────────

    pub(crate) fn message_count(&self) -> Result<usize> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
        self.scan_all_locked(&mut state)?;
        Ok(state.layouts.len())
    }

    pub(crate) fn read_message(&self, msg_idx: usize) -> Result<Vec<u8>> {
        let (offset, length) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
            self.ensure_message_locked(&mut state, msg_idx)?;
            let layout = &state.layouts[msg_idx];
            (layout.offset, layout.length)
        };
        let bytes = self.get_range(offset..offset + length)?;
        Ok(bytes.to_vec())
    }

    pub(crate) fn read_metadata(&self, msg_idx: usize) -> Result<GlobalMetadata> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
        self.ensure_layout_eager_locked(&mut state, msg_idx)?;
        state.layouts[msg_idx]
            .global_metadata
            .clone()
            .ok_or_else(|| TensogramError::Remote("metadata not found".to_string()))
    }

    pub(crate) fn read_descriptors(
        &self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        let layout = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
            self.ensure_layout_eager_locked(&mut state, msg_idx)?;
            state.layouts[msg_idx].clone()
        };
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            if index.offsets.len() != index.lengths.len() {
                return Err(TensogramError::Remote(format!(
                    "corrupt index: offsets.len()={} != lengths.len()={}",
                    index.offsets.len(),
                    index.lengths.len()
                )));
            }

            let meta = layout
                .global_metadata
                .clone()
                .ok_or_else(|| TensogramError::Remote("metadata not cached".to_string()))?;

            let msg_length = layout.length;
            let mut descriptors = Vec::with_capacity(index.offsets.len());
            for i in 0..index.offsets.len() {
                let desc = self.read_descriptor_only(
                    msg_offset,
                    msg_length,
                    index.offsets[i],
                    index.lengths[i],
                )?;
                descriptors.push(desc);
            }
            Ok((meta, descriptors))
        } else {
            let msg_bytes = self.read_message(msg_idx)?;
            crate::decode::decode_descriptors(&msg_bytes)
        }
    }

    /// Read only the CBOR descriptor from a data object frame, without
    /// downloading the full payload.
    ///
    /// For frames below `DESCRIPTOR_PREFIX_THRESHOLD` bytes, falls back
    /// to reading the entire frame (fewer round-trips).  For large frames,
    /// reads just the header, footer, and CBOR region — typically < 10 KB
    /// even when the payload is hundreds of megabytes.
    fn read_descriptor_only(
        &self,
        msg_offset: u64,
        msg_length: u64,
        frame_offset_in_msg: u64,
        frame_length: u64,
    ) -> Result<DataObjectDescriptor> {
        const DESCRIPTOR_PREFIX_THRESHOLD: u64 = 64 * 1024;

        let range =
            Self::checked_frame_range(msg_offset, msg_length, frame_offset_in_msg, frame_length)?;

        if frame_length <= DESCRIPTOR_PREFIX_THRESHOLD {
            let frame_bytes = self.get_range(range.clone())?;
            let (desc, _payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(&frame_bytes)?;
            return Ok(desc);
        }

        let frame_start = range.start;
        let frame_end = range.end;

        let header_bytes = self.get_range(frame_start..frame_start + FRAME_HEADER_SIZE as u64)?;
        let fh = FrameHeader::read_from(&header_bytes)?;

        if !fh.frame_type.is_data_object() {
            return Err(TensogramError::Remote(format!(
                "expected DataObject frame, got {:?}",
                fh.frame_type
            )));
        }

        let footer_start = frame_end - DATA_OBJECT_FOOTER_SIZE as u64;
        let footer_bytes = self.get_range(footer_start..frame_end)?;

        if footer_bytes.len() < DATA_OBJECT_FOOTER_SIZE {
            return Err(TensogramError::Remote("frame footer too short".to_string()));
        }
        // v3 data-object footer: [cbor_offset u64][hash u64][ENDF 4]
        // ENDF sits in the last 4 bytes.
        let endf_pos = footer_bytes.len() - FRAME_END.len();
        if &footer_bytes[endf_pos..] != FRAME_END {
            return Err(TensogramError::Remote(
                "frame missing ENDF trailer".to_string(),
            ));
        }

        let cbor_offset = u64::from_be_bytes(
            footer_bytes[..8]
                .try_into()
                .map_err(|_| TensogramError::Remote("footer cbor_offset truncated".to_string()))?,
        );

        if cbor_offset < FRAME_HEADER_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "cbor_offset ({cbor_offset}) below frame header size ({FRAME_HEADER_SIZE})"
            )));
        }

        let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;
        let cbor_start = frame_start
            .checked_add(cbor_offset)
            .ok_or_else(|| TensogramError::Remote("cbor_start overflow".to_string()))?;

        if cbor_after {
            if cbor_start >= footer_start {
                return Err(TensogramError::Remote(
                    "cbor_offset points at or past footer".to_string(),
                ));
            }
            let cbor_bytes = self.get_range(cbor_start..footer_start)?;
            metadata::cbor_to_object_descriptor(&cbor_bytes)
        } else {
            if cbor_start >= footer_start {
                return Err(TensogramError::Remote(
                    "cbor_offset beyond frame body".to_string(),
                ));
            }
            let max_cbor_len = footer_start - cbor_start;
            let mut prefix_size: u64 = 8192;
            loop {
                let read_end = (cbor_start + prefix_size).min(footer_start);
                let prefix_bytes = self.get_range(cbor_start..read_end)?;
                match metadata::cbor_to_object_descriptor(&prefix_bytes) {
                    Ok(desc) => return Ok(desc),
                    Err(_) if prefix_size < max_cbor_len => {
                        prefix_size = (prefix_size * 2).min(max_cbor_len);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }

    pub(crate) fn read_object(
        &self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        let layout = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
            self.ensure_layout_eager_locked(&mut state, msg_idx)?;
            state.layouts[msg_idx].clone()
        };
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            Self::validate_index_access(index, obj_idx)?;

            let meta = layout
                .global_metadata
                .clone()
                .ok_or_else(|| TensogramError::Remote("metadata not cached".to_string()))?;

            let range = Self::checked_frame_range(
                msg_offset,
                layout.length,
                index.offsets[obj_idx],
                index.lengths[obj_idx],
            )?;
            let frame_bytes = self.get_range(range)?;

            let (desc, payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(&frame_bytes)?;

            let decoded = crate::decode::decode_single_object(&desc, payload, options)?;

            Ok((meta, desc, decoded))
        } else {
            // No index — fall back to full message download
            let msg_bytes = self.read_message(msg_idx)?;
            crate::decode::decode_object(&msg_bytes, obj_idx, options)
        }
    }

    pub(crate) fn read_range_batch(
        &self,
        msg_indices: &[usize],
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<Vec<(DataObjectDescriptor, Vec<Vec<u8>>)>> {
        let mut byte_ranges = Vec::with_capacity(msg_indices.len());
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
            for &msg_idx in msg_indices {
                self.ensure_layout_eager_locked(&mut state, msg_idx)?;
            }
            for &msg_idx in msg_indices {
                let layout = &state.layouts[msg_idx];
                if let Some(ref index) = layout.index {
                    Self::validate_index_access(index, obj_idx)?;
                    byte_ranges.push(Self::checked_frame_range(
                        layout.offset,
                        layout.length,
                        index.offsets[obj_idx],
                        index.lengths[obj_idx],
                    )?);
                } else {
                    return Err(TensogramError::Remote(format!(
                        "message {} has no index frame; batch requires indexed messages",
                        msg_idx
                    )));
                }
            }
        }

        let store = self.store.clone();
        let path = self.path.clone();
        let all_bytes =
            block_on_shared(async move { store.get_ranges(&path, &byte_ranges).await })?;

        let mut results = Vec::with_capacity(msg_indices.len());
        for frame_bytes in &all_bytes {
            let (desc, payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(frame_bytes)?;
            let parts = crate::decode::decode_range_from_payload(&desc, payload, ranges, options)?;
            results.push((desc, parts));
        }
        Ok(results)
    }

    pub(crate) fn read_object_batch(
        &self,
        msg_indices: &[usize],
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<Vec<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)>> {
        let mut byte_ranges = Vec::with_capacity(msg_indices.len());
        let mut metas = Vec::with_capacity(msg_indices.len());
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
            for &msg_idx in msg_indices {
                self.ensure_layout_eager_locked(&mut state, msg_idx)?;
            }
            for &msg_idx in msg_indices {
                let layout = &state.layouts[msg_idx];
                if let Some(ref index) = layout.index {
                    Self::validate_index_access(index, obj_idx)?;
                    byte_ranges.push(Self::checked_frame_range(
                        layout.offset,
                        layout.length,
                        index.offsets[obj_idx],
                        index.lengths[obj_idx],
                    )?);
                    metas.push(layout.global_metadata.clone().ok_or_else(|| {
                        TensogramError::Remote("metadata not cached".to_string())
                    })?);
                } else {
                    return Err(TensogramError::Remote(format!(
                        "message {} has no index frame; batch decode requires indexed messages",
                        msg_idx
                    )));
                }
            }
        }

        let store = self.store.clone();
        let path = self.path.clone();
        let all_bytes =
            block_on_shared(async move { store.get_ranges(&path, &byte_ranges).await })?;

        let mut results = Vec::with_capacity(msg_indices.len());
        for (frame_bytes, meta) in all_bytes.iter().zip(metas) {
            let (desc, payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(frame_bytes)?;
            let decoded = crate::decode::decode_single_object(&desc, payload, options)?;
            results.push((meta, desc, decoded));
        }
        Ok(results)
    }

    pub(crate) fn read_range(
        &self,
        msg_idx: usize,
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<(DataObjectDescriptor, Vec<Vec<u8>>)> {
        let layout = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TensogramError::Remote("remote state lock poisoned".to_string()))?;
            self.ensure_layout_eager_locked(&mut state, msg_idx)?;
            state.layouts[msg_idx].clone()
        };
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            Self::validate_index_access(index, obj_idx)?;

            let range = Self::checked_frame_range(
                msg_offset,
                layout.length,
                index.offsets[obj_idx],
                index.lengths[obj_idx],
            )?;
            let frame_bytes = self.get_range(range)?;
            let (desc, payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(&frame_bytes)?;
            let parts = crate::decode::decode_range_from_payload(&desc, payload, ranges, options)?;
            Ok((desc, parts))
        } else {
            let msg_bytes = self.read_message(msg_idx)?;
            crate::decode::decode_range(&msg_bytes, obj_idx, ranges, options)
        }
    }

    fn validate_index_access(index: &IndexFrame, obj_idx: usize) -> Result<()> {
        if index.offsets.len() != index.lengths.len() {
            return Err(TensogramError::Remote(format!(
                "corrupt index: offsets.len()={} != lengths.len()={}",
                index.offsets.len(),
                index.lengths.len()
            )));
        }
        if obj_idx >= index.offsets.len() {
            return Err(TensogramError::Object(format!(
                "object index {} out of range (count={})",
                obj_idx,
                index.offsets.len()
            )));
        }
        Ok(())
    }

    fn checked_frame_range(
        msg_offset: u64,
        msg_length: u64,
        frame_offset_in_msg: u64,
        frame_length: u64,
    ) -> Result<Range<u64>> {
        let start = msg_offset
            .checked_add(frame_offset_in_msg)
            .ok_or_else(|| TensogramError::Remote("frame offset overflow".to_string()))?;
        let end = start
            .checked_add(frame_length)
            .ok_or_else(|| TensogramError::Remote("frame end overflow".to_string()))?;
        let msg_end = msg_offset
            .checked_add(msg_length)
            .ok_or_else(|| TensogramError::Remote("message end overflow".to_string()))?;
        if end > msg_end {
            return Err(TensogramError::Remote(format!(
                "indexed frame {start}..{end} exceeds message boundary {msg_end}"
            )));
        }
        Ok(start..end)
    }
}

// ── Native async path (remote + async) ───────────────────────────────────────

#[cfg(feature = "async")]
impl RemoteBackend {
    async fn get_range_async(&self, range: Range<u64>) -> Result<Bytes> {
        self.store
            .get_range(&self.path, range)
            .await
            .map_err(|e| TensogramError::Remote(e.to_string()))
    }

    pub(crate) async fn open_async(
        source: &str,
        storage_options: &BTreeMap<String, String>,
    ) -> Result<Self> {
        Self::open_async_with_scan_opts(source, storage_options, RemoteScanOptions::default()).await
    }

    pub(crate) async fn open_async_with_scan_opts(
        source: &str,
        storage_options: &BTreeMap<String, String>,
        scan_opts: RemoteScanOptions,
    ) -> Result<Self> {
        Self::validate_scan_opts(&scan_opts)?;
        emit_scan_mode(&scan_opts);
        let url = Url::parse(source)
            .map_err(|e| TensogramError::Remote(format!("invalid URL '{source}': {e}")))?;

        let mut opts: Vec<(String, String)> = storage_options
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if url.scheme() == "http" && !opts.iter().any(|(k, _)| k == "allow_http") {
            opts.push(("allow_http".to_string(), "true".to_string()));
        }
        let (store, path) = object_store::parse_url_opts(&url, opts)
            .map_err(|e| TensogramError::Remote(format!("cannot open '{source}': {e}")))?;

        let store: Arc<dyn ObjectStore> = Arc::from(store);
        let meta = store
            .head(&path)
            .await
            .map_err(|e| TensogramError::Remote(e.to_string()))?;

        let file_size = meta.size;
        if file_size < (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64 {
            return Err(TensogramError::Remote(format!(
                "remote file too small ({file_size} bytes)"
            )));
        }

        let backend = RemoteBackend {
            source_url: source.to_string(),
            store,
            path,
            file_size,
            state: Mutex::new(RemoteState {
                prev_scan_offset: file_size,
                bwd_active: scan_opts.bidirectional,
                ..RemoteState::default()
            }),
            scan_opts,
        };
        backend.scan_next_async().await?;
        {
            let state = backend.lock_state()?;
            if state.layouts.is_empty() {
                return Err(TensogramError::Remote(
                    "no valid messages found in remote file".to_string(),
                ));
            }
        }
        Ok(backend)
    }

    async fn scan_next_async(&self) -> Result<()> {
        self.scan_fwd_step_async(self.file_size).await
    }

    /// Async counterpart of [`Self::scan_fwd_step_locked`].  See that
    /// method's docstring for the `bound` parameter contract.
    async fn scan_fwd_step_async(&self, bound: u64) -> Result<()> {
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let pos = {
            let state = self.lock_state()?;
            if state.scan_complete() {
                return Ok(());
            }
            state.next_scan_offset
        };

        if pos + min_message_size > bound {
            let mut state = self.lock_state()?;
            if state.next_scan_offset == pos {
                state.terminate_forward("eof");
            }
            return Ok(());
        }

        let preamble_bytes = self
            .get_range_async(pos..pos + PREAMBLE_SIZE as u64)
            .await?;
        if &preamble_bytes[..MAGIC.len()] != MAGIC {
            let mut state = self.lock_state()?;
            if state.next_scan_offset == pos {
                state.terminate_forward("bad-magic-fwd");
            }
            return Ok(());
        }

        let preamble = match Preamble::read_from(&preamble_bytes) {
            Ok(preamble) => preamble,
            Err(_) => {
                let mut state = self.lock_state()?;
                if state.next_scan_offset == pos {
                    state.terminate_forward("preamble-parse-error-fwd");
                }
                return Ok(());
            }
        };

        let msg_len = preamble.total_length;

        if msg_len == 0 {
            let remaining = self.file_size - pos;
            if remaining < min_message_size {
                let mut state = self.lock_state()?;
                if state.next_scan_offset == pos {
                    state.terminate_forward("streaming-tail-too-small");
                }
                return Ok(());
            }

            let end_magic_pos = self.file_size - crate::wire::END_MAGIC.len() as u64;
            let end_bytes = self
                .get_range_async(end_magic_pos..end_magic_pos + crate::wire::END_MAGIC.len() as u64)
                .await?;
            if &end_bytes[..] != crate::wire::END_MAGIC {
                let mut state = self.lock_state()?;
                if state.next_scan_offset == pos {
                    state.terminate_forward("streaming-end-magic-mismatch");
                }
                return Ok(());
            }

            let mut state = self.lock_state()?;
            if state.scan_complete() || state.next_scan_offset != pos {
                return Ok(());
            }
            state.record_forward_hop(MessageLayout {
                offset: pos,
                length: remaining,
                preamble,
                index: None,
                global_metadata: None,
            });
            state.terminate_forward("streaming-tail");
            return Ok(());
        }

        match pos.checked_add(msg_len) {
            Some(end) if msg_len >= min_message_size && end <= bound => {}
            _ => {
                let mut state = self.lock_state()?;
                if state.next_scan_offset == pos {
                    state.terminate_forward("length-out-of-range-fwd");
                }
                return Ok(());
            }
        }

        let mut state = self.lock_state()?;
        if state.scan_complete() || state.next_scan_offset != pos {
            return Ok(());
        }
        state.record_forward_hop(MessageLayout {
            offset: pos,
            length: msg_len,
            preamble,
            index: None,
            global_metadata: None,
        });
        Ok(())
    }

    /// Async sibling of [`Self::scan_step_locked`]; see that method's
    /// docstring.  Returns immediately when the scan is already
    /// complete; otherwise dispatches to either the bidirectional
    /// paired round or a single bounded forward hop.  `None`
    /// signals bidirectional, `Some(bound)` signals forward-only.
    async fn scan_step_async(&self) -> Result<()> {
        let fwd_bound: Option<u64> = {
            let state = self.lock_state()?;
            if state.scan_complete() {
                return Ok(());
            }
            if state.bwd_active && !state.fwd_terminated {
                None
            } else {
                Some(self.forward_bound(&state))
            }
        };
        match fwd_bound {
            Some(bound) => self.scan_fwd_step_async(bound).await,
            None => self.scan_bidir_round_async().await,
        }
    }

    /// Async sibling of [`Self::scan_bidir_round_locked`]; see that
    /// method's docstring.  Lock-around-await: snapshot before the
    /// paired fetch, validate the same snapshot on reacquire, then
    /// commit through [`Self::apply_round_outcomes`].
    async fn scan_bidir_round_async(&self) -> Result<()> {
        let snap = {
            let state = self.lock_state()?;
            state.snapshot()
        };
        let bound = snap.prev;
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;

        if snap.prev < snap.next + 2 * min_message_size {
            return self.scan_fwd_step_async(bound).await;
        }

        let fwd_r = snap.next..snap.next + PREAMBLE_SIZE as u64;
        let bwd_r = snap.prev - POSTAMBLE_SIZE as u64..snap.prev;
        let bytes = self
            .store
            .get_ranges(&self.path, &[fwd_r.clone(), bwd_r.clone()])
            .await
            .map_err(|e| TensogramError::Remote(e.to_string()))?;
        if bytes.len() != 2 {
            return Err(TensogramError::Remote(format!(
                "get_ranges returned {} buffers, expected 2",
                bytes.len()
            )));
        }

        let bwd_outcome =
            parse_backward_postamble(&bytes[1], &snap, self.scan_opts.max_message_size);
        let bwd_round2 = match &bwd_outcome {
            BackwardOutcome::NeedPreambleValidation { msg_start, .. } => Some(
                self.get_range_async(*msg_start..*msg_start + PREAMBLE_SIZE as u64)
                    .await?,
            ),
            _ => None,
        };

        let mut state = self.lock_state()?;
        if !state.matches(&snap) {
            return Ok(());
        }
        let fwd_outcome = parse_forward_preamble(&bytes[0], snap.next, self.file_size, bound);
        self.apply_round_outcomes(&mut state, fwd_outcome, bwd_outcome, bwd_round2.as_deref());
        Ok(())
    }

    async fn ensure_message_async(&self, msg_idx: usize) -> Result<()> {
        loop {
            let ready = {
                let state = self.lock_state()?;
                if msg_idx < state.layouts.len() {
                    return Ok(());
                }
                if state.scan_complete() {
                    return Err(TensogramError::Framing(format!(
                        "message index {} out of range (count={})",
                        msg_idx,
                        state.layouts.len()
                    )));
                }
                false
            };

            if !ready {
                self.scan_step_async().await?;
            }
        }
    }

    async fn scan_and_discover_next_async(&self) -> Result<()> {
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let pos = {
            let state = self.lock_state()?;
            if state.scan_complete() {
                return Ok(());
            }
            state.next_scan_offset
        };

        if pos + min_message_size > self.file_size {
            let mut state = self.lock_state()?;
            if state.next_scan_offset == pos {
                state.terminate_forward("eof");
            }
            return Ok(());
        }

        let chunk_size = (self.file_size - pos).min(256 * 1024);
        let chunk = self.get_range_async(pos..pos + chunk_size).await?;

        if chunk.len() < PREAMBLE_SIZE || &chunk[..MAGIC.len()] != MAGIC {
            let mut state = self.lock_state()?;
            if state.next_scan_offset == pos {
                state.terminate_forward("bad-magic-fwd");
            }
            return Ok(());
        }

        let preamble = match Preamble::read_from(&chunk[..PREAMBLE_SIZE]) {
            Ok(preamble) => preamble,
            Err(_) => {
                let mut state = self.lock_state()?;
                if state.next_scan_offset == pos {
                    state.terminate_forward("preamble-parse-error-fwd");
                }
                return Ok(());
            }
        };

        let msg_len = preamble.total_length;

        if msg_len == 0 {
            let remaining = self.file_size - pos;
            if remaining < min_message_size {
                let mut state = self.lock_state()?;
                if state.next_scan_offset == pos {
                    state.terminate_forward("streaming-tail-too-small");
                }
                return Ok(());
            }

            let end_magic_pos = self.file_size - crate::wire::END_MAGIC.len() as u64;
            let end_bytes = self
                .get_range_async(end_magic_pos..end_magic_pos + crate::wire::END_MAGIC.len() as u64)
                .await?;
            if &end_bytes[..] != crate::wire::END_MAGIC {
                let mut state = self.lock_state()?;
                if state.next_scan_offset == pos {
                    state.terminate_forward("streaming-end-magic-mismatch");
                }
                return Ok(());
            }

            let mut state = self.lock_state()?;
            if state.scan_complete() || state.next_scan_offset != pos {
                return Ok(());
            }
            state.record_forward_hop(MessageLayout {
                offset: pos,
                length: remaining,
                preamble,
                index: None,
                global_metadata: None,
            });
            state.terminate_forward("streaming-tail");
            return Ok(());
        }

        match pos.checked_add(msg_len) {
            Some(end) if msg_len >= min_message_size && end <= self.file_size => {}
            _ => {
                let mut state = self.lock_state()?;
                if state.next_scan_offset == pos {
                    state.terminate_forward("length-out-of-range-fwd");
                }
                return Ok(());
            }
        }

        let flags = preamble.flags;
        let msg_idx = {
            let mut state = self.lock_state()?;
            if state.scan_complete() || state.next_scan_offset != pos {
                return Ok(());
            }
            let msg_idx = state.layouts.len();
            state.record_forward_hop(MessageLayout {
                offset: pos,
                length: msg_len,
                preamble,
                index: None,
                global_metadata: None,
            });
            msg_idx
        };

        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            let chunk_end = (msg_len as usize).min(chunk.len());
            let mut state = self.lock_state()?;
            if msg_idx < state.layouts.len()
                && state.layouts[msg_idx].offset == pos
                && state.layouts[msg_idx].global_metadata.is_none()
                && state.layouts[msg_idx].index.is_none()
            {
                Self::parse_header_frames(&mut state, msg_idx, &chunk[..chunk_end])?;
            }
        } else if flags.has(MessageFlags::FOOTER_METADATA) && flags.has(MessageFlags::FOOTER_INDEX)
        {
            self.discover_footer_layout_from_suffix_async(msg_idx)
                .await?;
        }

        Ok(())
    }

    async fn discover_footer_layout_from_suffix_async(&self, msg_idx: usize) -> Result<()> {
        let (msg_offset, msg_len) = {
            let state = self.lock_state()?;
            let layout = state.layouts.get(msg_idx).ok_or_else(|| {
                TensogramError::Framing(format!(
                    "message index {} out of range (count={})",
                    msg_idx,
                    state.layouts.len()
                ))
            })?;
            (layout.offset, layout.length)
        };

        let suffix_size = msg_len.min(256 * 1024);
        let msg_end = msg_offset
            .checked_add(msg_len)
            .ok_or_else(|| TensogramError::Remote("message end overflow".to_string()))?;
        let suffix_start = msg_end - suffix_size;
        let suffix = self.get_range_async(suffix_start..msg_end).await?;

        if suffix.len() < POSTAMBLE_SIZE {
            return Err(TensogramError::Remote(
                "suffix too short for postamble".to_string(),
            ));
        }

        let pa_bytes = &suffix[suffix.len() - POSTAMBLE_SIZE..];
        let postamble = Postamble::read_from(pa_bytes)?;

        if postamble.first_footer_offset < PREAMBLE_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "first_footer_offset ({}) is before preamble end ({PREAMBLE_SIZE})",
                postamble.first_footer_offset
            )));
        }

        let footer_abs_start = msg_offset
            .checked_add(postamble.first_footer_offset)
            .ok_or_else(|| TensogramError::Remote("footer offset overflow".to_string()))?;
        let footer_abs_end = msg_end - POSTAMBLE_SIZE as u64;

        if footer_abs_start >= footer_abs_end {
            return Err(TensogramError::Remote(
                "first_footer_offset points at or past postamble".to_string(),
            ));
        }

        if footer_abs_start >= suffix_start {
            let local_start = (footer_abs_start - suffix_start) as usize;
            let local_end = suffix.len() - POSTAMBLE_SIZE;
            let mut state = self.lock_state()?;
            if state.layouts[msg_idx].global_metadata.is_some()
                && state.layouts[msg_idx].index.is_some()
            {
                return Ok(());
            }
            Self::parse_footer_frames(&mut state, msg_idx, &suffix[local_start..local_end])
        } else {
            let footer_bytes = self
                .get_range_async(footer_abs_start..footer_abs_end)
                .await?;
            let mut state = self.lock_state()?;
            if state.layouts[msg_idx].global_metadata.is_some()
                && state.layouts[msg_idx].index.is_some()
            {
                return Ok(());
            }
            Self::parse_footer_frames(&mut state, msg_idx, &footer_bytes)
        }
    }

    async fn ensure_layout_eager_async(&self, msg_idx: usize) -> Result<()> {
        loop {
            let action = {
                let state = self.lock_state()?;
                if let Some(layout) = state.layouts.get(msg_idx) {
                    if layout.global_metadata.is_some() && layout.index.is_some() {
                        return Ok(());
                    }
                    EagerAction::Discover
                } else if state.scan_complete() {
                    return Err(TensogramError::Framing(format!(
                        "message index {} out of range (count={})",
                        msg_idx,
                        state.layouts.len()
                    )));
                } else if state.bwd_active && !state.fwd_terminated {
                    EagerAction::ScanBidir
                } else {
                    EagerAction::ScanForwardEager
                }
            };

            match action {
                EagerAction::ScanBidir => {
                    self.scan_bidir_round_async().await?;
                }
                EagerAction::ScanForwardEager => {
                    self.scan_and_discover_next_async().await?;
                }
                EagerAction::Discover => {
                    return self.ensure_layout_async(msg_idx).await;
                }
            }
        }
    }

    async fn ensure_layout_async(&self, msg_idx: usize) -> Result<()> {
        self.ensure_message_async(msg_idx).await?;

        let flags = {
            let state = self.lock_state()?;
            let layout = &state.layouts[msg_idx];
            if layout.global_metadata.is_some() && layout.index.is_some() {
                return Ok(());
            }
            layout.preamble.flags
        };

        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            self.discover_header_layout_async(msg_idx).await?;
        } else if flags.has(MessageFlags::FOOTER_METADATA) && flags.has(MessageFlags::FOOTER_INDEX)
        {
            self.discover_footer_layout_async(msg_idx).await?;
        } else {
            return Err(TensogramError::Remote(
                "remote access requires header-indexed or footer-indexed messages".to_string(),
            ));
        }

        Ok(())
    }

    async fn discover_header_layout_async(&self, msg_idx: usize) -> Result<()> {
        let (msg_offset, msg_len) = {
            let state = self.lock_state()?;
            let layout = state.layouts.get(msg_idx).ok_or_else(|| {
                TensogramError::Framing(format!(
                    "message index {} out of range (count={})",
                    msg_idx,
                    state.layouts.len()
                ))
            })?;
            (layout.offset, layout.length)
        };

        let chunk_size = msg_len.min(256 * 1024);
        let header_bytes = self
            .get_range_async(msg_offset..msg_offset + chunk_size)
            .await?;

        let mut state = self.lock_state()?;
        if state.layouts[msg_idx].global_metadata.is_some()
            && state.layouts[msg_idx].index.is_some()
        {
            return Ok(());
        }
        Self::parse_header_frames(&mut state, msg_idx, &header_bytes)
    }

    async fn discover_footer_layout_async(&self, msg_idx: usize) -> Result<()> {
        let (msg_offset, msg_len) = {
            let state = self.lock_state()?;
            let layout = state.layouts.get(msg_idx).ok_or_else(|| {
                TensogramError::Framing(format!(
                    "message index {} out of range (count={})",
                    msg_idx,
                    state.layouts.len()
                ))
            })?;
            (layout.offset, layout.length)
        };

        let pa_offset = msg_offset
            .checked_add(msg_len)
            .and_then(|end| end.checked_sub(POSTAMBLE_SIZE as u64))
            .ok_or_else(|| TensogramError::Remote("postamble offset overflow".to_string()))?;
        let pa_bytes = self
            .get_range_async(pa_offset..pa_offset + POSTAMBLE_SIZE as u64)
            .await?;
        let postamble = Postamble::read_from(&pa_bytes)?;

        if postamble.first_footer_offset < PREAMBLE_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "first_footer_offset ({}) is before preamble end ({})",
                postamble.first_footer_offset, PREAMBLE_SIZE
            )));
        }
        let footer_start = msg_offset
            .checked_add(postamble.first_footer_offset)
            .ok_or_else(|| TensogramError::Remote("footer offset overflow".to_string()))?;
        let footer_end = pa_offset;
        if footer_start >= footer_end {
            return Err(TensogramError::Remote(
                "first_footer_offset points at or past postamble".to_string(),
            ));
        }

        let footer_bytes = self.get_range_async(footer_start..footer_end).await?;
        let mut state = self.lock_state()?;
        if state.layouts[msg_idx].global_metadata.is_some()
            && state.layouts[msg_idx].index.is_some()
        {
            return Ok(());
        }
        Self::parse_footer_frames(&mut state, msg_idx, &footer_bytes)
    }

    pub(crate) async fn message_count_async(&self) -> Result<usize> {
        loop {
            let done = {
                let state = self.lock_state()?;
                state.scan_complete()
            };
            if done {
                break;
            }
            self.scan_step_async().await?;
        }
        let state = self.lock_state()?;
        Ok(state.layouts.len())
    }

    pub(crate) async fn read_message_async(&self, msg_idx: usize) -> Result<Vec<u8>> {
        self.ensure_message_async(msg_idx).await?;
        let (offset, length) = {
            let state = self.lock_state()?;
            let layout = &state.layouts[msg_idx];
            (layout.offset, layout.length)
        };
        let bytes = self.get_range_async(offset..offset + length).await?;
        Ok(bytes.to_vec())
    }

    pub(crate) async fn read_metadata_async(&self, msg_idx: usize) -> Result<GlobalMetadata> {
        self.ensure_layout_eager_async(msg_idx).await?;
        let state = self.lock_state()?;
        state.layouts[msg_idx]
            .global_metadata
            .clone()
            .ok_or_else(|| TensogramError::Remote("metadata not found".to_string()))
    }

    pub(crate) async fn read_descriptors_async(
        &self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        self.ensure_layout_eager_async(msg_idx).await?;
        let layout = {
            let state = self.lock_state()?;
            state.layouts[msg_idx].clone()
        };
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            if index.offsets.len() != index.lengths.len() {
                return Err(TensogramError::Remote(format!(
                    "corrupt index: offsets.len()={} != lengths.len()={}",
                    index.offsets.len(),
                    index.lengths.len()
                )));
            }

            let meta = layout
                .global_metadata
                .clone()
                .ok_or_else(|| TensogramError::Remote("metadata not cached".to_string()))?;

            let msg_length = layout.length;
            let mut descriptors = Vec::with_capacity(index.offsets.len());
            for i in 0..index.offsets.len() {
                let desc = self
                    .read_descriptor_only_async(
                        msg_offset,
                        msg_length,
                        index.offsets[i],
                        index.lengths[i],
                    )
                    .await?;
                descriptors.push(desc);
            }
            Ok((meta, descriptors))
        } else {
            let msg_bytes = self.read_message_async(msg_idx).await?;
            crate::decode::decode_descriptors(&msg_bytes)
        }
    }

    async fn read_descriptor_only_async(
        &self,
        msg_offset: u64,
        msg_length: u64,
        frame_offset_in_msg: u64,
        frame_length: u64,
    ) -> Result<DataObjectDescriptor> {
        const DESCRIPTOR_PREFIX_THRESHOLD: u64 = 64 * 1024;

        let range =
            Self::checked_frame_range(msg_offset, msg_length, frame_offset_in_msg, frame_length)?;

        if frame_length <= DESCRIPTOR_PREFIX_THRESHOLD {
            let frame_bytes = self.get_range_async(range.clone()).await?;
            let (desc, _payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(&frame_bytes)?;
            return Ok(desc);
        }

        let frame_start = range.start;
        let frame_end = range.end;

        let header_bytes = self
            .get_range_async(frame_start..frame_start + FRAME_HEADER_SIZE as u64)
            .await?;
        let fh = FrameHeader::read_from(&header_bytes)?;

        if !fh.frame_type.is_data_object() {
            return Err(TensogramError::Remote(format!(
                "expected DataObject frame, got {:?}",
                fh.frame_type
            )));
        }

        let footer_start = frame_end - DATA_OBJECT_FOOTER_SIZE as u64;
        let footer_bytes = self.get_range_async(footer_start..frame_end).await?;

        if footer_bytes.len() < DATA_OBJECT_FOOTER_SIZE {
            return Err(TensogramError::Remote("frame footer too short".to_string()));
        }
        // v3 data-object footer: [cbor_offset u64][hash u64][ENDF 4]
        let endf_pos = footer_bytes.len() - FRAME_END.len();
        if &footer_bytes[endf_pos..] != FRAME_END {
            return Err(TensogramError::Remote(
                "frame missing ENDF trailer".to_string(),
            ));
        }

        let cbor_offset = u64::from_be_bytes(
            footer_bytes[..8]
                .try_into()
                .map_err(|_| TensogramError::Remote("footer cbor_offset truncated".to_string()))?,
        );

        if cbor_offset < FRAME_HEADER_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "cbor_offset ({cbor_offset}) below frame header size ({FRAME_HEADER_SIZE})"
            )));
        }

        let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;
        let cbor_start = frame_start
            .checked_add(cbor_offset)
            .ok_or_else(|| TensogramError::Remote("cbor_start overflow".to_string()))?;

        if cbor_after {
            if cbor_start >= footer_start {
                return Err(TensogramError::Remote(
                    "cbor_offset points at or past footer".to_string(),
                ));
            }
            let cbor_bytes = self.get_range_async(cbor_start..footer_start).await?;
            metadata::cbor_to_object_descriptor(&cbor_bytes)
        } else {
            if cbor_start >= footer_start {
                return Err(TensogramError::Remote(
                    "cbor_offset beyond frame body".to_string(),
                ));
            }
            let max_cbor_len = footer_start - cbor_start;
            let mut prefix_size: u64 = 8192;
            loop {
                let read_end = (cbor_start + prefix_size).min(footer_start);
                let prefix_bytes = self.get_range_async(cbor_start..read_end).await?;
                match metadata::cbor_to_object_descriptor(&prefix_bytes) {
                    Ok(desc) => return Ok(desc),
                    Err(_) if prefix_size < max_cbor_len => {
                        prefix_size = (prefix_size * 2).min(max_cbor_len);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }

    pub(crate) async fn read_object_async(
        &self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        self.ensure_layout_eager_async(msg_idx).await?;
        let layout = {
            let state = self.lock_state()?;
            state.layouts[msg_idx].clone()
        };
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            Self::validate_index_access(index, obj_idx)?;

            let meta = layout
                .global_metadata
                .clone()
                .ok_or_else(|| TensogramError::Remote("metadata not cached".to_string()))?;

            let range = Self::checked_frame_range(
                msg_offset,
                layout.length,
                index.offsets[obj_idx],
                index.lengths[obj_idx],
            )?;
            let frame_bytes = self.get_range_async(range).await?;
            let (desc, payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(&frame_bytes)?;
            let decoded = crate::decode::decode_single_object(&desc, payload, options)?;
            Ok((meta, desc, decoded))
        } else {
            let msg_bytes = self.read_message_async(msg_idx).await?;
            crate::decode::decode_object(&msg_bytes, obj_idx, options)
        }
    }

    pub(crate) async fn ensure_all_layouts_batch_async(&self, msg_indices: &[usize]) -> Result<()> {
        if msg_indices.is_empty() {
            return Ok(());
        }
        let max_idx = msg_indices.iter().copied().max().unwrap_or(0);
        loop {
            let bidir = {
                let state = self.lock_state()?;
                if state.layouts.len() > max_idx || state.scan_complete() {
                    break;
                }
                state.bwd_active && !state.fwd_terminated
            };
            if bidir {
                self.scan_bidir_round_async().await?;
            } else {
                self.scan_and_discover_next_async().await?;
            }
        }

        {
            let state = self.lock_state()?;
            for &idx in msg_indices {
                if idx >= state.layouts.len() {
                    return Err(TensogramError::Framing(format!(
                        "message index {} out of range (count={})",
                        idx,
                        state.layouts.len()
                    )));
                }
            }
        }

        // Phase 2: find which messages still need layout discovery.
        let needs_layout: Vec<usize> = {
            let state = self.lock_state()?;
            msg_indices
                .iter()
                .copied()
                .filter(|&idx| {
                    state
                        .layouts
                        .get(idx)
                        .is_none_or(|l| l.global_metadata.is_none() || l.index.is_none())
                })
                .collect()
        };

        if needs_layout.is_empty() {
            return Ok(());
        }

        // Phase 3: compute header/footer byte ranges for all cold messages.
        let mut fetch_ranges: Vec<Range<u64>> = Vec::new();
        let mut fetch_map: Vec<(usize, bool)> = Vec::new(); // (msg_idx, is_header)
        {
            let state = self.lock_state()?;
            for &msg_idx in &needs_layout {
                let layout = &state.layouts[msg_idx];
                let flags = layout.preamble.flags;
                if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX)
                {
                    let chunk_size = layout.length.min(256 * 1024);
                    fetch_ranges.push(layout.offset..layout.offset + chunk_size);
                    fetch_map.push((msg_idx, true));
                } else if flags.has(MessageFlags::FOOTER_METADATA)
                    && flags.has(MessageFlags::FOOTER_INDEX)
                {
                    let pa_offset = layout
                        .offset
                        .checked_add(layout.length)
                        .and_then(|end| end.checked_sub(POSTAMBLE_SIZE as u64))
                        .ok_or_else(|| {
                            TensogramError::Remote("postamble offset overflow".to_string())
                        })?;
                    fetch_ranges.push(pa_offset..pa_offset + POSTAMBLE_SIZE as u64);
                    fetch_map.push((msg_idx, false));
                } else {
                    return Err(TensogramError::Remote(
                        "remote batch requires header-indexed or footer-indexed messages"
                            .to_string(),
                    ));
                }
            }
        }

        // Phase 4: batched HTTP fetch for all layout headers.
        let all_bytes = self
            .store
            .get_ranges(&self.path, &fetch_ranges)
            .await
            .map_err(|e| TensogramError::Remote(e.to_string()))?;

        // Phase 5: parse header layouts.
        let mut footer_fetches: Vec<(usize, Range<u64>)> = Vec::new();
        {
            let mut state = self.lock_state()?;
            for (bytes, &(msg_idx, is_header)) in all_bytes.iter().zip(fetch_map.iter()) {
                if state.layouts[msg_idx].global_metadata.is_some()
                    && state.layouts[msg_idx].index.is_some()
                {
                    continue;
                }
                if is_header {
                    Self::parse_header_frames(&mut state, msg_idx, bytes)?;
                } else {
                    let postamble = Postamble::read_from(bytes)?;
                    let layout = &state.layouts[msg_idx];
                    let footer_start = layout
                        .offset
                        .checked_add(postamble.first_footer_offset)
                        .ok_or_else(|| {
                            TensogramError::Remote("footer offset overflow".to_string())
                        })?;
                    let pa_offset = layout
                        .offset
                        .checked_add(layout.length)
                        .and_then(|end| end.checked_sub(POSTAMBLE_SIZE as u64))
                        .ok_or_else(|| {
                            TensogramError::Remote("postamble offset overflow".to_string())
                        })?;
                    footer_fetches.push((msg_idx, footer_start..pa_offset));
                }
            }
        }

        // Phase 6: batched footer fetch if any footer-indexed messages.
        if !footer_fetches.is_empty() {
            let footer_ranges: Vec<Range<u64>> =
                footer_fetches.iter().map(|(_, r)| r.clone()).collect();
            let footer_bytes = self
                .store
                .get_ranges(&self.path, &footer_ranges)
                .await
                .map_err(|e| TensogramError::Remote(e.to_string()))?;
            let mut state = self.lock_state()?;
            for (bytes, &(msg_idx, _)) in footer_bytes.iter().zip(footer_fetches.iter()) {
                if state.layouts[msg_idx].global_metadata.is_some()
                    && state.layouts[msg_idx].index.is_some()
                {
                    continue;
                }
                Self::parse_footer_frames(&mut state, msg_idx, bytes)?;
            }
        }

        Ok(())
    }

    pub(crate) async fn read_object_batch_async(
        &self,
        msg_indices: &[usize],
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<Vec<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)>> {
        self.ensure_all_layouts_batch_async(msg_indices).await?;

        let mut byte_ranges = Vec::with_capacity(msg_indices.len());
        let mut metas = Vec::with_capacity(msg_indices.len());
        {
            let state = self.lock_state()?;
            for &msg_idx in msg_indices {
                let layout = &state.layouts[msg_idx];
                if let Some(ref index) = layout.index {
                    Self::validate_index_access(index, obj_idx)?;
                    byte_ranges.push(Self::checked_frame_range(
                        layout.offset,
                        layout.length,
                        index.offsets[obj_idx],
                        index.lengths[obj_idx],
                    )?);
                    metas.push(layout.global_metadata.clone().ok_or_else(|| {
                        TensogramError::Remote("metadata not cached".to_string())
                    })?);
                } else {
                    return Err(TensogramError::Remote(format!(
                        "message {} has no index frame; batch decode requires indexed messages",
                        msg_idx
                    )));
                }
            }
        }

        let all_bytes = self
            .store
            .get_ranges(&self.path, &byte_ranges)
            .await
            .map_err(|e| TensogramError::Remote(e.to_string()))?;

        let mut results = Vec::with_capacity(msg_indices.len());
        for (frame_bytes, meta) in all_bytes.iter().zip(metas) {
            let (desc, payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(frame_bytes)?;
            let decoded = crate::decode::decode_single_object(&desc, payload, options)?;
            results.push((meta, desc, decoded));
        }
        Ok(results)
    }

    pub(crate) async fn read_range_batch_async(
        &self,
        msg_indices: &[usize],
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<Vec<(DataObjectDescriptor, Vec<Vec<u8>>)>> {
        // 1. Batch-discover all layouts (scan + batched header/footer fetch).
        self.ensure_all_layouts_batch_async(msg_indices).await?;

        // 2. Compute byte ranges for each message's data object frame.
        let mut byte_ranges = Vec::with_capacity(msg_indices.len());
        {
            let state = self.lock_state()?;
            for &msg_idx in msg_indices {
                let layout = &state.layouts[msg_idx];
                if let Some(ref index) = layout.index {
                    Self::validate_index_access(index, obj_idx)?;
                    let range = Self::checked_frame_range(
                        layout.offset,
                        layout.length,
                        index.offsets[obj_idx],
                        index.lengths[obj_idx],
                    )?;
                    byte_ranges.push(range);
                } else {
                    return Err(TensogramError::Remote(format!(
                        "message {} has no index frame; batch range decode requires indexed messages",
                        msg_idx
                    )));
                }
            }
        }

        // 3. Single batched HTTP fetch for all frames.
        let all_bytes = self
            .store
            .get_ranges(&self.path, &byte_ranges)
            .await
            .map_err(|e| TensogramError::Remote(e.to_string()))?;

        // 4. Decode each frame locally.
        let mut results = Vec::with_capacity(msg_indices.len());
        for frame_bytes in &all_bytes {
            let (desc, payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(frame_bytes)?;
            let parts = crate::decode::decode_range_from_payload(&desc, payload, ranges, options)?;
            results.push((desc, parts));
        }
        Ok(results)
    }

    pub(crate) async fn read_range_async(
        &self,
        msg_idx: usize,
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<(DataObjectDescriptor, Vec<Vec<u8>>)> {
        self.ensure_layout_eager_async(msg_idx).await?;
        let layout = {
            let state = self.lock_state()?;
            state.layouts[msg_idx].clone()
        };
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            Self::validate_index_access(index, obj_idx)?;

            let range = Self::checked_frame_range(
                msg_offset,
                layout.length,
                index.offsets[obj_idx],
                index.lengths[obj_idx],
            )?;
            let frame_bytes = self.get_range_async(range).await?;
            let (desc, payload, _mask_region, _consumed) =
                framing::decode_data_object_frame(&frame_bytes)?;
            let parts = crate::decode::decode_range_from_payload(&desc, payload, ranges, options)?;
            Ok((desc, parts))
        } else {
            let msg_bytes = self.read_message_async(msg_idx).await?;
            crate::decode::decode_range(&msg_bytes, obj_idx, ranges, options)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_layout(offset: u64, length: u64) -> MessageLayout {
        MessageLayout {
            offset,
            length,
            preamble: Preamble {
                version: 3,
                flags: MessageFlags::default(),
                reserved: 0,
                total_length: length,
            },
            index: None,
            global_metadata: None,
        }
    }

    #[test]
    fn scan_options_default_is_forward_only() {
        let opts = RemoteScanOptions::default();
        assert!(
            !opts.bidirectional,
            "default must keep current forward-only behaviour"
        );
        assert_eq!(
            opts.max_message_size,
            4 * 1024 * 1024 * 1024,
            "default backward-postamble cap is 4 GiB",
        );
    }

    #[test]
    fn scan_complete_truth_table_phase_1_equivalence() {
        for fwd_terminated in [false, true] {
            for gap_closed in [false, true] {
                let s = RemoteState {
                    fwd_terminated,
                    gap_closed,
                    ..RemoteState::default()
                };
                let expected = gap_closed || fwd_terminated;
                assert_eq!(
                    s.scan_complete(),
                    expected,
                    "fwd_terminated={fwd_terminated} gap_closed={gap_closed}",
                );
            }
        }
    }

    #[test]
    fn record_forward_hop_advances_cursor_and_bumps_epoch() {
        let mut s = RemoteState {
            prev_scan_offset: 1000,
            ..RemoteState::default()
        };
        let epoch_before = s.scan_epoch;
        s.record_forward_hop(dummy_layout(0, 100));
        assert_eq!(s.next_scan_offset, 100);
        assert_eq!(s.layouts.len(), 1);
        assert_ne!(s.scan_epoch, epoch_before, "epoch must bump on forward hop");
    }

    #[test]
    fn disable_backward_clears_suffix_and_disables() {
        let mut s = RemoteState {
            prev_scan_offset: 1000,
            bwd_active: true,
            ..RemoteState::default()
        };
        s.suffix_rev.push(dummy_layout(800, 200));
        s.disable_backward("test");
        assert!(!s.bwd_active);
        assert!(s.suffix_rev.is_empty());
    }

    #[test]
    fn disable_backward_is_idempotent_when_already_inactive() {
        let mut s = RemoteState::default();
        let epoch_before = s.scan_epoch;
        s.disable_backward("test");
        assert_eq!(
            s.scan_epoch, epoch_before,
            "no-op disable on already-inactive state must not bump epoch",
        );
    }

    #[test]
    fn terminate_forward_clears_provisional_backward_state() {
        let mut s = RemoteState {
            prev_scan_offset: 1000,
            bwd_active: true,
            ..RemoteState::default()
        };
        s.suffix_rev.push(dummy_layout(800, 200));
        s.terminate_forward("test");
        assert!(s.fwd_terminated);
        assert!(
            s.suffix_rev.is_empty(),
            "fwd_terminated => suffix_rev empty (bidirectional is never recovery)",
        );
        assert!(!s.bwd_active);
    }

    #[test]
    fn max_message_size_is_addressable_via_options() {
        let opts = RemoteScanOptions {
            bidirectional: true,
            max_message_size: 1024,
        };
        assert_eq!(opts.max_message_size, 1024);
        assert!(opts.bidirectional);
    }
}

#[cfg(all(test, feature = "async"))]
mod bidir_http_tests {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use http_body_util::Full;
    use hyper::body::Bytes as HyperBytes;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    use super::*;
    use crate::Dtype;
    use crate::encode::{self, EncodeOptions};
    use crate::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};

    /// Tiny hyper-based mock object store.  Serves a fixed byte
    /// buffer with single-Range support, counts every request, and
    /// gives the test a port URL to feed `RemoteBackend::open*`.
    struct MockObjectStore {
        request_count: Arc<AtomicUsize>,
        range_request_count: Arc<AtomicUsize>,
        addr: SocketAddr,
    }

    impl MockObjectStore {
        async fn start(data: Vec<u8>) -> std::io::Result<Self> {
            let data = Arc::new(data);
            let request_count = Arc::new(AtomicUsize::new(0));
            let range_request_count = Arc::new(AtomicUsize::new(0));
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;

            let data_clone = data.clone();
            let count_clone = request_count.clone();
            let range_count_clone = range_request_count.clone();
            tokio::spawn(async move {
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(v) => v,
                        Err(_) => break,
                    };
                    let io = TokioIo::new(stream);
                    let data = data_clone.clone();
                    let count = count_clone.clone();
                    let range_count = range_count_clone.clone();
                    tokio::spawn(async move {
                        let _ = http1::Builder::new()
                            .serve_connection(
                                io,
                                service_fn(move |req: Request<hyper::body::Incoming>| {
                                    let data = data.clone();
                                    let count = count.clone();
                                    let range_count = range_count.clone();
                                    async move { handle(req, data, count, range_count) }
                                }),
                            )
                            .await;
                    });
                }
            });

            Ok(MockObjectStore {
                request_count,
                range_request_count,
                addr,
            })
        }

        fn url(&self) -> String {
            format!("http://127.0.0.1:{}/test.tgm", self.addr.port())
        }

        fn request_count(&self) -> usize {
            self.request_count.load(Ordering::SeqCst)
        }

        fn range_request_count(&self) -> usize {
            self.range_request_count.load(Ordering::SeqCst)
        }
    }

    fn handle(
        req: Request<hyper::body::Incoming>,
        data: Arc<Vec<u8>>,
        request_count: Arc<AtomicUsize>,
        range_request_count: Arc<AtomicUsize>,
    ) -> std::io::Result<Response<Full<HyperBytes>>> {
        request_count.fetch_add(1, Ordering::SeqCst);

        if req.method() == hyper::Method::HEAD {
            return Response::builder()
                .status(StatusCode::OK)
                .header("Content-Length", data.len())
                .header("Accept-Ranges", "bytes")
                .body(Full::new(HyperBytes::new()))
                .map_err(std::io::Error::other);
        }

        if let Some(range_header) = req.headers().get("Range") {
            range_request_count.fetch_add(1, Ordering::SeqCst);
            let range_str = range_header.to_str().unwrap_or("");
            if let Some((start, end_exclusive)) = parse_range(range_str, data.len()) {
                let slice = &data[start..end_exclusive];
                return Response::builder()
                    .status(StatusCode::PARTIAL_CONTENT)
                    .header(
                        "Content-Range",
                        format!("bytes {}-{}/{}", start, end_exclusive - 1, data.len()),
                    )
                    .header("Content-Length", slice.len())
                    .body(Full::new(HyperBytes::copy_from_slice(slice)))
                    .map_err(std::io::Error::other);
            }
            return Response::builder()
                .status(StatusCode::RANGE_NOT_SATISFIABLE)
                .header("Content-Range", format!("bytes */{}", data.len()))
                .body(Full::new(HyperBytes::new()))
                .map_err(std::io::Error::other);
        }

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Length", data.len())
            .body(Full::new(HyperBytes::copy_from_slice(&data)))
            .map_err(std::io::Error::other)
    }

    fn parse_range(header: &str, total: usize) -> Option<(usize, usize)> {
        let header = header.strip_prefix("bytes=")?;
        if total == 0 {
            return None;
        }
        if let Some(suffix) = header.strip_prefix('-') {
            let n: usize = suffix.parse().ok()?;
            if n == 0 {
                return None;
            }
            Some((total.saturating_sub(n), total))
        } else if let Some((start_s, end_s)) = header.split_once('-') {
            let start: usize = start_s.parse().ok()?;
            if start >= total {
                return None;
            }
            if end_s.is_empty() {
                Some((start, total))
            } else {
                let end: usize = end_s.parse().ok()?;
                if end < start {
                    return None;
                }
                Some((start, end.min(total - 1) + 1))
            }
        } else {
            None
        }
    }

    fn make_message(shape: Vec<u64>, fill: u8) -> Vec<u8> {
        let strides = if shape.is_empty() {
            vec![]
        } else {
            let mut s = vec![1u64; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                s[i] = s[i + 1] * shape[i + 1];
            }
            s
        };
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: shape.len() as u64,
            shape: shape.clone(),
            strides,
            dtype: Dtype::Float32,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
        };
        let num_bytes = shape.iter().product::<u64>() as usize * 4;
        let data = vec![fill; num_bytes];
        let meta = GlobalMetadata {
            extra: BTreeMap::new(),
            ..Default::default()
        };
        encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())
            .expect("encode test message")
    }

    fn concat_messages(parts: Vec<Vec<u8>>) -> Vec<u8> {
        let total = parts.iter().map(|p| p.len()).sum();
        let mut out = Vec::with_capacity(total);
        for p in parts {
            out.extend_from_slice(&p);
        }
        out
    }

    fn empty_storage() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    fn forward_layouts(url: &str) -> Vec<(u64, u64)> {
        let backend =
            RemoteBackend::open_with_scan_opts(url, &empty_storage(), RemoteScanOptions::default())
                .expect("open forward-only");
        backend.message_count().expect("count");
        let state = backend.state.lock().expect("lock");
        state.layouts.iter().map(|l| (l.offset, l.length)).collect()
    }

    fn bidir_layouts(url: &str) -> Vec<(u64, u64)> {
        let backend = RemoteBackend::open_with_scan_opts(
            url,
            &empty_storage(),
            RemoteScanOptions {
                bidirectional: true,
                max_message_size: 4 * 1024 * 1024 * 1024,
            },
        )
        .expect("open bidirectional");
        backend.message_count().expect("count");
        let state = backend.state.lock().expect("lock");
        state.layouts.iter().map(|l| (l.offset, l.length)).collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bidir_one_message_no_duplicate() {
        let buf = make_message(vec![4], 42);
        let server = MockObjectStore::start(buf).await.expect("start");
        let layouts = bidir_layouts(&server.url());
        assert_eq!(
            layouts.len(),
            1,
            "1-message file must produce exactly one layout, got {layouts:?}",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bidir_two_messages_match_forward_only() {
        let buf = concat_messages(vec![make_message(vec![4], 10), make_message(vec![8], 20)]);
        let server = MockObjectStore::start(buf).await.expect("start");
        let fwd = forward_layouts(&server.url());
        let bidir = bidir_layouts(&server.url());
        assert_eq!(
            fwd, bidir,
            "2-message file: bidirectional must match forward-only"
        );
        assert_eq!(bidir.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bidir_three_messages_odd_count_meet() {
        let buf = concat_messages(vec![
            make_message(vec![4], 10),
            make_message(vec![8], 20),
            make_message(vec![16], 30),
        ]);
        let server = MockObjectStore::start(buf).await.expect("start");
        let fwd = forward_layouts(&server.url());
        let bidir = bidir_layouts(&server.url());
        assert_eq!(
            fwd, bidir,
            "3-message file: bidirectional must match forward-only"
        );
        assert_eq!(bidir.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bidir_ten_messages_match_forward_only() {
        let parts: Vec<Vec<u8>> = (0..10)
            .map(|i| make_message(vec![4 + i as u64], i as u8))
            .collect();
        let buf = concat_messages(parts);
        let server = MockObjectStore::start(buf).await.expect("start");
        let fwd = forward_layouts(&server.url());
        let bidir = bidir_layouts(&server.url());
        assert_eq!(
            fwd, bidir,
            "10-message file: bidirectional must match forward-only"
        );
        assert_eq!(bidir.len(), 10);
    }

    /// Streaming-mode (`postamble.total_length == 0`) encoder output:
    /// backward must yield with `streaming-zero-bwd`; forward
    /// completes the scan via the existing streaming-tail branch.
    /// Validates the format-vs-transport taxonomy `Streaming` row.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bidir_streaming_postamble_yields_backward() {
        let mut buf = make_message(vec![4], 99);
        let pa_start = buf.len() - POSTAMBLE_SIZE;
        let total_off = pa_start + 8;
        for byte in &mut buf[total_off..total_off + 8] {
            *byte = 0;
        }
        let preamble_total_off = 16;
        for byte in &mut buf[preamble_total_off..preamble_total_off + 8] {
            *byte = 0;
        }
        let server = MockObjectStore::start(buf).await.expect("start");
        let bidir = bidir_layouts(&server.url());
        assert_eq!(
            bidir.len(),
            1,
            "streaming-tail must still produce a single forward-discovered layout, got {bidir:?}",
        );
        assert!(server.request_count() > 0);
    }

    /// Corrupt `END_MAGIC` at file end: backward yields with
    /// `bad-end-magic-bwd`; forward parses both messages cleanly
    /// because non-streaming forward scanning never consults the
    /// file-end `END_MAGIC`.  Validates that bidirectional gracefully
    /// degrades to forward-only on backward-side corruption.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bidir_corrupt_end_magic_yields_backward() {
        let mut buf = concat_messages(vec![make_message(vec![4], 10), make_message(vec![8], 20)]);
        let len = buf.len();
        buf[len - 8..].copy_from_slice(b"BADMAGIC");
        let server = MockObjectStore::start(buf).await.expect("start");
        let backend = RemoteBackend::open_with_scan_opts(
            &server.url(),
            &empty_storage(),
            RemoteScanOptions {
                bidirectional: true,
                max_message_size: 4 * 1024 * 1024 * 1024,
            },
        )
        .expect("open");
        let count = backend.message_count().expect("count");
        assert_eq!(
            count, 2,
            "corrupt END_MAGIC: forward path still finds both messages",
        );
        assert!(server.range_request_count() > 0);
    }
}
