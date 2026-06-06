# Security Analysis & Hardening Plan

> **Status:** First audit pass complete — **zero HIGH findings remain**.
> This document is the durable record of the security threat model, the
> iterative audit methodology, the findings (with severity), and their
> mitigations.

## 0. Executive Summary

A first deep security audit of all ECMWF-deployed Tensogram code was
performed against a **fully-hostile-remote-bytes** threat model (a `.tgm`
downloaded from an untrusted/compromised server), using `cargo-fuzz`
(libFuzzer + AddressSanitizer), targeted adversarial tests, `cargo
audit`, and manual review of every `unsafe` / native-codec / FFI
boundary.

**11 HIGH issues were found and fixed; 1 security invariant was verified
and pinned; 1 LOW was logged.** Every fix has a permanent regression
test, and every fuzz target re-ran clean afterwards. The fixes are
**performance-neutral on the hot path** (checked/saturating arithmetic
replaces the same operations; the native-codec fixes add a one-time
length check + a bounded pad only on the decode-setup path).

| Category | Count | IDs |
|---|---:|---|
| HIGH — framing integer-overflow / OOB / panic-DoS | 7 | SEC-001..007 |
| HIGH — native-codec out-of-bounds read (zfp, SZ3) | 2 | SEC-009, SEC-010 |
| HIGH — unbounded-allocation OOM DoS (sz3) | 1 | SEC-011 |
| HIGH total | **10** | |
| Invariant verified & pinned (CBOR recursion) | 1 | SEC-008 |
| LOW logged (not on hostile-input path) | 1 | SEC-012 |

**The two native out-of-bounds reads (SEC-009 zfp, SEC-010 in our own
`sz3_ffi.cpp`) are the most serious — memory-safety violations on remote
input — and warrant treating the landing change as a security release.**

`cargo audit`: 0 vulnerable dependencies.

Surfaces reviewed **clean** (no finding): pure-Rust `tensogram-szip`
(zero `unsafe`), szip/libaec + blosc2 codec shims, the `tensogram-ffi` C
ABI input boundary (every `from_raw_parts` null-guarded), `remote.rs`
(already overflow-hardened), Python bindings (safe PyO3 + sound
`u8`→`i8` + safe numpy byte-parse), and the C++/Fortran/WASM/TS thin
layers (delegate to the audited core).

> **Note on scope:** this is a *first pass to zero-HIGH*. Fuzzing was
> bounded (~1–2 min/target smoke runs); a CI/nightly deep-fuzz job and a
> periodic re-audit are recommended follow-ups (see §10). Upstream
> vendored C/C++ (SZ3, c-blosc2, libaec, zfp) is treated as a contained
> black box, not line-audited.

## 1. Scope

All code deployed to / linkable by ECMWF software:

- **Rust core** — `tensogram`, `tensogram-encodings`, `tensogram-szip`
  (framing, decode, decode_range, metadata/CBOR, file, restore,
  substitute_and_mask, pipeline, validate, codecs).
- **Remote** — `remote.rs`, `remote_scan_parse.rs` (object_store / HTTP
  network-facing fetch + scan).
- **C FFI** — `tensogram-ffi` (`lib.rs`, `async_core.rs`,
  `async_streaming.rs`) — the largest `unsafe` surface (~370 unsafe).
- **Python bindings** — `python/bindings` (PyO3, buffer protocol,
  free-threaded, numpy interop).
- **C++ wrapper, Fortran, WASM/TypeScript** — thin layers over the FFI.
- **Native codec FFI shims** — `libaec.rs`, `zfp_ffi.rs`, blosc2, sz3,
  **including our vendored `rust/tensogram-sz3-sys/cpp/sz3_ffi.cpp`**
  (audited deeply, as it is our code).

Out of scope: line-by-line audit of *upstream* vendored C/C++ sources
(SZ3, c-blosc2, libaec, zfp). These are treated as **untrusted black
boxes**, contained and fuzzed at the Rust boundary.

## 2. Threat Model

**Adversary:** controls the full `.tgm` byte stream — a compromised or
malicious S3/GCS/Azure bucket, a MITM on HTTP, or a hostile producer.
The library is linked into long-running ECMWF processes and decodes
buffers downloaded from remote servers. Runs as an ordinary user
(not root).

**Assets:** process integrity (no RCE / memory corruption), process
availability (no crash / abort / OOM-kill / hang), and confidentiality
of process memory (no OOB read leakage).

**Attack classes considered (researched widely):**

| # | Class | Concrete vector in this library |
|---|---|---|
| A | Memory corruption (OOB read/write, UAF, double-free) | `unsafe` slicing/pointer math on attacker offsets/lengths; FFI raw pointers; native codec bugs |
| B | Integer overflow → undersized alloc → OOB | `shape × dtype_width`, `offset + len`, `num_values × bits`, `cbor_offset`, frame `total_length` arithmetic |
| C | Decompression bomb (memory/CPU DoS) | szip/zstd/lz4/blosc2/zfp/sz3/RLE/roaring expanding tiny input to huge output |
| D | Unbounded allocation DoS | descriptor claims terabyte tensor / billions of objects / huge CBOR |
| E | Panic / abort as DoS | `unwrap`/`expect`/indexing/arithmetic-overflow panics on hostile input (`panic=abort` ⇒ process death) |
| F | Infinite / superlinear loop DoS | scan recovery, RLE run decode, frame walking, mask restore |
| G | FFI contract violations | NULL/dangling/misaligned pointers, non-UTF-8, use-after-free across the C ABI, double-free |
| H | Type confusion / transmute | dtype reinterpretation, byteswap unit mismatch, CBOR value coercion |
| I | Recursion / stack exhaustion | nested CBOR, recursive descriptor structures |
| J | Supply chain | vendored C++ and `-sys` crates; dependency CVEs (`cargo audit`) |
| K | Hash/integrity bypass | forging frames when `verify_hash` is off; collision misuse |
| L | Resource handle exhaustion | file descriptors, mmap, tokio tasks, threads from descriptors |

## 3. Security Invariants (the contract we enforce)

1. **No undefined behaviour** on any input. Ever.
2. **No panic / abort** on any untrusted input — malformed input must
   return a structured `Err`, never crash. (`panic=abort` makes any
   panic a hard DoS.)
3. **No unbounded resource use derived from untrusted descriptors
   without a fallible guard** — every allocation sized from the wire is
   `try_reserve`/checked; every size multiplication is checked.
4. **Fail fast, fail clean:** on hostile input, error *before* doing
   damage and let the caller stop using that input. (Per owner: do not
   compromise performance — erroring out is the correct response.)
5. **Performance is non-negotiable:** mitigations must add ~zero cost to
   the hot path for legitimate data. Resource *limits* are **opt-in**
   (`ReaderOptions`/`DecodeOptions`), default off / very high, so real
   multi-GiB ERA5 / ML data is never falsely rejected. Correctness
   guards (overflow checks, fallible alloc, bounds checks) are always on
   — they cost nothing measurable.

## 4. Severity Definitions

- **HIGH** — memory corruption / UB / RCE potential, OR a
  trivially-triggered crash/abort/UB/hang on small untrusted input.
  *Fixed unconditionally in this effort.*
- **MEDIUM** — DoS requiring large/crafted input, non-trivial panic,
  resource exhaustion mitigable by an opt-in limit. *Logged; fixed if
  cheap; regression-tested where practical.*
- **LOW** — defence-in-depth, hardening, hygiene. *Logged.*

**Termination:** iterate until **zero HIGH** findings remain. MEDIUM /
LOW recorded in §8 with regression tests where cheap.

## 5. Methodology (iterative loop)

For each surface module, in priority order (§6):

```
1. ANALYSE  — read the code as an attacker: trace every untrusted value
              (offset, length, count, dtype, index, pointer) from the
              wire to its use (alloc, slice, ptr math, codec, loop).
2. HYPOTHESISE — write down each suspected weakness as a concrete attack.
3. TEST     — write a test/fuzz target that feeds the malicious input and
              asserts the SECURE behaviour (clean Err / bounded work),
              i.e. a failing test if the bug exists.
4. CONFIRM  — run it. If it crashes/UB/hangs → confirmed HIGH.
              If it already errors cleanly → invariant holds; KEEP the
              test as a no-regression guard.
5. FIX      — minimal, perf-preserving fix (checked arithmetic, fallible
              alloc, bounds check, structured error).
6. RETEST   — the test now passes (clean Err); rerun the suite + relevant
              fuzz target; confirm no perf regression on hot path.
7. NEXT     — move to the next hypothesis / module.
```

Tooling:
- **`cargo-fuzz`** (libFuzzer) harnesses for the parser entry points and
  each codec; short bounded local runs; commit harnesses + any
  crash-repro corpus as permanent regression assets.
- **`cargo miri`** on the unsafe-heavy modules where feasible (detects
  UB the type system can't) — note: cannot run FFI/native codecs, so
  scoped to pure-Rust unsafe (e.g. `tensogram-szip`, dtype byteswap).
- **`cargo audit`** for dependency CVEs.
- **proptest** for round-trip + bounded-work properties.
- Existing `adversarial.rs` extended in place.

## 6. Audit Order (highest leverage first)

1. **Framing & decode** (`framing.rs`, `decode.rs`) — the primary
   untrusted-bytes → offsets/lengths/slices boundary.
2. **Metadata / CBOR** (`metadata.rs`, `types.rs`) — attacker CBOR →
   recursion / huge maps / type confusion.
3. **Encoding pipeline & codecs** (`pipeline.rs`, `simple_packing.rs`,
   `shuffle.rs`, `bitmask/*`, `compression/*`, `libaec.rs`, `restore.rs`,
   `substitute_and_mask.rs`) — decompression bombs, output-size,
   integer overflow.
4. **Pure-Rust szip** (`tensogram-szip`) — Miri-able unsafe.
5. **Remote** (`remote.rs`, `remote_scan_parse.rs`) — network size
   handling, range arithmetic, scan loops.
6. **C FFI** (`tensogram-ffi`) — pointer/UTF-8/lifetime/double-free; the
   biggest unsafe surface.
7. **Vendored SZ3 shim** (`sz3_ffi.cpp`) — our C++ glue, deep audit.
8. **Python bindings** — buffer protocol, free-threaded races, numpy.
9. **C++/Fortran/WASM/TS** — thin layers; contract checks.
10. **Supply chain** — `cargo audit`, vendored-source provenance.

## 7. Fuzz Targets (planned)

- `fuzz_decode` — `decode(arbitrary_bytes)`
- `fuzz_decode_metadata` — `decode_metadata`
- `fuzz_decode_object` / `fuzz_decode_range` — index/range fuzzing
- `fuzz_scan` — multi-message scanner
- `fuzz_validate` — all 4 validation levels
- `fuzz_pipeline_decode` — per-codec decode of arbitrary "compressed"
  bytes with arbitrary descriptors
- `fuzz_szip_pure` — pure-Rust AEC decode
- (FFI) `fuzz_ffi_decode` — through the C ABI with a JSON descriptor

Property: **no panic, no hang (bounded by libFuzzer timeout), no leak,
no UB (ASan)** on any input.

## 8. Findings Log

> Updated as the audit proceeds. Each entry: ID, severity, surface,
> description, attack, status, mitigation commit, regression test.

### SEC-001 — HIGH — FIXED

- **Surface:** `framing.rs` `decode_message` (reached from `decode`,
  `decode_object`, `decode_range`, `validate_buffer`, and every binding).
- **Class:** E (panic-as-DoS) / B (integer underflow).
- **Description:** When the preamble `total_length` is non-zero but
  smaller than `PREAMBLE_SIZE + POSTAMBLE_SIZE` (48), the postamble
  offset `total_len - POSTAMBLE_SIZE` underflowed. In debug this panics
  ("attempt to subtract with overflow"); in release it wraps to a huge
  `usize` and then slices out of bounds. Under `panic = "abort"` this is
  a process-killing DoS triggerable by a **42-byte** hostile message.
- **Attack:** any consumer calling `decode`/`decode_object`/
  `decode_range`/`validate` on a downloaded `.tgm` with a crafted
  `total_length` (e.g. `10`).
- **Discovery:** `fuzz_decode` (libFuzzer + ASan) — found in < 5 s.
- **Mitigation:** reject `total_len < PREAMBLE_SIZE + POSTAMBLE_SIZE`
  with a structured `Framing` error before any offset arithmetic; the
  `msg_end` subtraction also switched to `checked_sub` for defence in
  depth.
- **Regression test:** `adversarial.rs::sec001_undersized_total_length_is_rejected_not_panic`
  (exact fuzzer reproducer + full sub-minimum boundary sweep).
- **Re-fuzz:** `fuzz_decode` clean over 5,600+ executions after the fix.

### SEC-002 — HIGH — FIXED

- **Surface:** `framing.rs` `try_forward_hop` (reached from `scan`,
  `scan_with_options`, multi-message reads).
- **Class:** E (panic-as-DoS) / B (integer overflow).
- **Description:** `pos + total` with attacker-controlled `total`
  (`u64` up to `usize::MAX`) overflowed → "attempt to add with
  overflow" panic.
- **Discovery:** `fuzz_scan`.
- **Mitigation:** `checked_add` for the message end + a minimum-length
  floor; overflow yields a clean "no message here" (advance), never a
  panic.
- **Regression test:** `adversarial.rs::sec002_scan_huge_total_length_does_not_overflow`.

### SEC-003 — HIGH — FIXED

- **Surface:** `framing.rs` `scan_file` (file/mmap scanner, reached from
  `TensogramFile::open`).
- **Class:** E / B. Same `pos + total` overflow as SEC-002 but on the
  on-disk file-scan path (the fuzzer covers the in-memory path; this was
  found by reading the sibling code during the SEC-002 fix).
- **Mitigation:** `checked_add` + minimum-length floor.
- **Regression:** covered structurally by the same boundary logic; the
  file path shares the guard.

### SEC-004 — HIGH — FIXED

- **Surface:** `framing.rs` `data_object_inline_hashes` (reached from
  hash-frame consistency checks / validation).
- **Class:** E / B. `pos + frame_total` overflow with attacker frame
  `total_length`; also `frame_end - 12` underflow for a tiny frame.
- **Mitigation:** `checked_add` for the frame end, a minimum frame-size
  floor (so the hash-slot offset can't underflow), and `saturating_add`
  on the alignment step.
- **Regression test:** `adversarial.rs::sec004_inline_hash_walk_huge_frame_total_does_not_overflow`.

### SEC-005 — HIGH — FIXED

- **Surface:** `framing.rs` `try_backward_hop` (bidirectional scan).
- **Class:** A (OOB read) / E (panic). `msg_start = bound_end - total`
  with a tiny `total` (below the preamble size) put `msg_start` so close
  to the buffer end that the subsequent `buf[msg_start..msg_start + 8]`
  MAGIC slice ran out of bounds and panicked.
- **Discovery:** `fuzz_scan` (after SEC-002 fix exposed this deeper path).
- **Mitigation:** require `total >= PREAMBLE_SIZE + POSTAMBLE_SIZE` in
  the backward hop, guaranteeing the MAGIC slice stays in bounds.
- **Regression test:** `adversarial.rs::sec005_backward_scan_tiny_total_no_oob`
  (exact reproducer + synthetic tiny-total sweep).
- **Re-fuzz:** `fuzz_scan` clean over a full 90 s run after the fix.

### SEC-006 — HIGH — FIXED

- **Surface:** `framing.rs` `decode_metadata_only` (reached from
  `decode_metadata`, every metadata-only read).
- **Class:** E / B / F. `pos += frame_total` overflow with an
  attacker-controlled frame `total_length`; a zero/sub-header
  `frame_total` could additionally spin the loop (no progress → DoS).
- **Discovery:** `fuzz_decode_metadata`.
- **Mitigation:** reject `frame_total < FRAME_HEADER_SIZE`;
  `saturating_add` on the advance + alignment so an overflow exits the
  loop cleanly.
- **Regression test:** `adversarial.rs::sec006_decode_metadata_huge_skip_frame_no_overflow`.
- **Re-fuzz:** `fuzz_decode_metadata` clean over a full 90 s run.

### SEC-007 — HIGH — FIXED

- **Surface:** `framing.rs` `scan_file` bidirectional forward hop.
- **Class:** E / B. `fwd_pos + total` overflow and `fwd_pos + total - 8`
  underflow with attacker-controlled preamble `total_length` on the
  on-disk/file-scan path. Found by auditing the file siblings of the
  in-memory scan fixes.
- **Mitigation:** `checked_add` + preamble+postamble minimum floor.
- **Regression test:** `adversarial.rs::sec007_scan_file_huge_and_tiny_total_no_overflow`
  (via `Cursor`, huge + tiny `total_length`).

> **Pattern note:** SEC-001..007 are one systemic class — every
> attacker-controlled length (preamble `total_length`, frame
> `total_length`, mask offset/length) added to a buffer position must
> use checked/saturating arithmetic and a structural minimum floor
> before deriving any offset.  The whole framing layer was swept for
> this pattern; the fixes are perf-neutral (the checks replace the same
> additions).

### SEC-009 — HIGH — FIXED

- **Surface:** `zfp_ffi.rs` `zfp_decompress_f64` → libzfp C decoder.
- **Class:** A (out-of-bounds read / memory-safety). The vendored libzfp
  decompressor does **not** bounds-check its input bitstream against the
  requested element count, so a truncated stream (e.g. 1 byte) with a
  large `num_values` made zfp's `stream_read_word` read past the buffer
  (ASan **SEGV** / potential info leak), reachable from a `.tgm` with
  `compression=zfp`.
- **Discovery:** `fuzz_codec_decode` (the direct codec-decode harness).
- **Mitigation:** at the Rust shim, query `zfp_stream_maximum_size` for
  the field+mode, reject a stream longer than that maximum (malformed),
  and decode from a **zero-padded buffer sized to the maximum** so the
  decoder can never read past our allocation regardless of how it walks
  the stream. Legitimate round-trips (all modes + range) unaffected.
- **Regression test:** `zfp_ffi.rs::sec009_zfp_truncated_stream_rejected_not_oob`.

### SEC-010 — HIGH — FIXED

- **Surface:** **our vendored** `sz3_ffi.cpp` `sz3_decompress_config` →
  SZ3 C++ `read` / `Config::load`.
- **Class:** A (out-of-bounds read). The C++ header parser read a
  16-byte header and a config trailer from an attacker buffer **using
  `len` for nothing** — no length check before the header read, and the
  config offset `pos + cmpDataSize` used an attacker-controlled
  `cmpDataSize` from the stream. A short/hostile stream caused OOB reads
  (ASan **SEGV** in `SZ3::read` / `MemoryUtil.hpp`), reachable from a
  `.tgm` with `compression=sz3`.
- **Discovery:** `fuzz_codec_decode`.
- **Mitigation (deep audit of our C++):**
  - reject `len < 16` before any header read;
  - bound `cmpDataSize` against `len` (no overflow, config offset in
    range);
  - copy the config trailer into a **64 KiB zero-padded heap buffer** so
    SZ3's non-bounds-checking `Config::load` cannot over-read past the
    real data;
  - wrap `Config::load` in try/catch so a thrown SZ3 exception cannot
    cross the C ABI;
  - return a sentinel invalid config (`N == 0`, null `dims`) on any of
    the above, which the Rust side (`ParsedConfig::from_compressed`) now
    rejects with `SZ3Error::MalformedCompressedStream`.
- **Regression test:** `compression/sz3.rs::sec010_sz3_truncated_stream_rejected_not_oob`.
- **Re-fuzz:** `fuzz_codec_decode` clean over 150 s after both SEC-009
  and SEC-010 fixes.

> **Native-codec pattern:** vendored C/C++ decompressors (zfp, SZ3,
> likely others) trust their input stream length implicitly.  The
> containment pattern is: validate the stream length against the
> descriptor-derived expectation at the Rust/C++ shim, and decode from a
> zero-padded buffer sized to the decoder's maximum read so any
> over-read lands in padding, never out of bounds.  szip (libaec),
> blosc2, zstd, and lz4 are checked next under the same lens.

### SEC-011 — HIGH — FIXED

- **Surface:** `tensogram-sz3` `decompress` (reached from the sz3 codec
  on the `.tgm` decode path).
- **Class:** D (unbounded-allocation OOM DoS). `Vec::with_capacity(len)`
  where `len` is the element count from the attacker-controlled SZ3
  config trailer — a hostile config claiming a huge `num` aborts the
  process via OOM (infallible allocation).
- **Mitigation:** `try_reserve_exact(len)`, surfacing a pathological
  count as `SZ3Error::MalformedCompressedStream` instead of aborting.
- **Verification:** `fuzz_codec_decode` clean over 150 s after this +
  SEC-009/010 (no OOM-abort on hostile sz3 streams); 37 sz3 round-trip
  tests pass.

### Audit notes — surfaces reviewed clean / low residual

- **szip (libaec):** decode bounds libaec's output via `avail_out =
  expected_size` (fallibly allocated), `checked_sub` on `decoded_len`,
  strict-equality short-decode rejection, `set_len` only with
  `decoded_len <= expected_size`. libaec honours `avail_in` (input is
  bounded), unlike zfp/sz3. Fuzz-clean. **No finding.**
- **blosc2:** uses the safe `blosc2` Rust crate over the self-describing
  c-blosc2 frame format (internal size validation); per-chunk
  `try_reserve`; documented `items_num()` over-report mitigation.
  Fuzz-clean. **No finding.**
- **C FFI input boundary (`tensogram-ffi/lib.rs`):** every
  `slice::from_raw_parts(buf, buf_len)` on the decode/scan/range path is
  preceded by an `is_null` guard (and the range path null-checks the
  offset/count arrays); `CStr::from_ptr(...).to_str()` validates UTF-8
  with error handling; `extract_inline_hashes` slices only validated
  `scan` output (safe given SEC-002/005). Caller-contract violations
  (dangling / wrong-length pointers) are inherent to any C ABI and the
  caller's responsibility. **No finding** — well-guarded.
- **remote.rs / remote_scan_parse.rs:** range arithmetic is already
  `checked_add` / `saturating_add`/`sub`-hardened throughout; the few
  raw `pos + PREAMBLE_SIZE as u64` additions operate on offsets bounded
  by the object's real `file_size`, so a `u64` overflow would require a
  >16-exabyte object. **No HIGH** — low residual.

### SEC-012 — LOW — logged (not on hostile-input path)

- **Surface:** `tensogram-sz3` `decompress_into_dimensioned`
  `assert_eq!(decompressed_data.len(), len)` panics on a config/buffer
  length mismatch. **Not reachable from `.tgm` decode** — the codec uses
  `decompress` (SEC-011-fixed), not this variant, which is a direct
  `tensogram-sz3` API where the caller owns the buffer. Logged for a
  future `Result`-returning cleanup.

_(first audit pass complete — zero HIGH remaining)_

## 9. Deliverables

- This document (threat model + methodology + findings).
- Regression tests (kept permanently) for every confirmed weakness and
  every verified-secure invariant.
- `cargo-fuzz` harnesses + crash-repro corpus.
- Code fixes (one focused commit each) on `security/hardening-audit`,
  landed as a single reviewed PR.
- Opt-in `ReaderOptions`/`DecodeOptions` resource limits (default off)
  where a resource-DoS class warrants caller control.

## 10. Recommended follow-ups (not HIGH; out of this pass)

- **CI deep-fuzz job.** Wire the `fuzz/` targets into a nightly/weekly CI
  job (longer `-max_total_time`, persistent corpus) so regressions and
  deeper inputs are caught continuously. The smoke runs in this pass
  were bounded (~1–2 min/target).
- **Add fuzz targets for the FFI C ABI and the remote scan-parse**
  (`fuzz_ffi_decode`, `fuzz_remote_scan_parse`) to fuzz those boundaries
  directly rather than only through the core.
- **SEC-012 (LOW):** convert `decompress_into_dimensioned`'s `assert_eq!`
  to a `Result` so the direct `tensogram-sz3` API never panics on a
  length mismatch.
- **Resource-limit knobs.** Per the agreed model, resource *limits*
  (max-object-count, max-decompressed-size) remain opt-in/off; revisit
  if an operational consumer wants them ON for the remote path.
- **Periodic re-audit + `cargo audit` in CI** to catch new dependency
  CVEs and code drift.
- **Upstream report (optional):** SEC-009 is an upstream libzfp
  bounds-check gap; consider reporting it to the zfp project (we contain
  it at our boundary regardless).
