# Security Analysis & Hardening Plan

> **Status:** In progress. This document is the durable record of the
> security threat model, the iterative audit methodology, the findings
> (with severity), and their mitigations. It is updated as the audit
> proceeds.

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

_(none recorded yet — audit starting)_

## 9. Deliverables

- This document (threat model + methodology + findings).
- Regression tests (kept permanently) for every confirmed weakness and
  every verified-secure invariant.
- `cargo-fuzz` harnesses + crash-repro corpus.
- Code fixes (one focused commit each) on `security/hardening-audit`,
  landed as a single reviewed PR.
- Opt-in `ReaderOptions`/`DecodeOptions` resource limits (default off)
  where a resource-DoS class warrants caller control.
