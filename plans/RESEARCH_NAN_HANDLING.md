# NaN Handling — Research Memo

> **Status**: research, not accepted for implementation. This memo
> surveys every path through the Tensogram library where NaN (and by
> extension Inf) values can appear on the encode side, catalogues the
> gaps and surprising corners in the current behaviour, and proposes
> directions for future work.
>
> Nothing here is committed. Items that mature move to `TODO.md`;
> items that stay speculative move to `IDEAS.md`. The memo exists so
> that a future engineer (human or agent) picking up any of these
> threads has the full context without re-reading the codebase.
>
> Related docs: `DESIGN.md` (one line on simple_packing rejecting NaN),
> `docs/src/encodings/simple-packing.md`, `docs/src/edge-cases.md`
> (§"NaN in Simple Packing", §"compute_common() NaN Handling"),
> `docs/src/guide/error-handling.md` (§"NaN in Simple Packing"),
> `docs/src/guide/convert-netcdf.md` (§"Simple packing with NaN").

## Contents

- [1. Why this matters](#1-why-this-matters)
- [2. Current behaviour — the full pipeline map](#2-current-behaviour--the-full-pipeline-map)
- [3. Gaps and gotchas — catalogue](#3-gaps-and-gotchas--catalogue)
- [4. Proposed directions](#4-proposed-directions)
- [5. Open design questions](#5-open-design-questions)
- [6. Test matrix for future work](#6-test-matrix-for-future-work)
- [7. Verification approach](#7-verification-approach)
- [8. Out of scope](#8-out-of-scope)
- [9. Code and test pointers](#9-code-and-test-pointers)

---

## 1. Why this matters

NaN is not an exotic corner in scientific data. It appears in
Tensogram's target workloads in several common ways:

- **Missing-value substitution.** The NetCDF converter deliberately
  replaces `_FillValue` / `missing_value` sentinels with `f64::NAN`
  during CF unpacking (`rust/tensogram-netcdf/src/converter.rs::read_and_unpack`).
  GRIB bitmaps have a similar need, though GRIB handles missingness
  out-of-band rather than by NaN substitution.
- **Arithmetic propagation.** Any downstream computation on a model
  output that includes `0.0/0.0`, `Inf - Inf`, or a square root of a
  negative produces NaN.
- **Sensor failures.** Observation pipelines use NaN to flag rejected
  readings.
- **ML outputs.** Models given degenerate inputs (empty batches,
  mis-shapen tensors, gradient blow-ups) can emit NaN-laced tensors.
- **CF convention.** `_FillValue` → NaN is a documented convention,
  not an accident.

The library handles these cases correctly in some pipeline
configurations, rejects them hard in others, and has **uncovered
corners** where behaviour is delegated to a third-party library with
no wrapper-level tests. The goal of this memo is to produce a single
source of truth for where the system stands today and what a coherent
design would look like.

Tensogram's design principle is "design errors out of existence" (`plans/STYLE.md`).
NaN handling is one corner where the current surface area does not
meet that bar — there are three distinct classes of gotcha where a
user's reasonable expectation diverges from actual behaviour.

## 2. Current behaviour — the full pipeline map

The encoder runs `raw bytes → encoding → filter → compression → (inline
hash) → frame` with each stage independently configurable. NaN
behaviour depends on which stages are active.

### 2.1 `encoding="none"` — bit-exact passthrough (tested)

With `encoding="none"`, the pipeline never materialises the payload
as typed floats — it is just bytes. NaN bit patterns survive the
encode → decode round-trip exactly.

**Verified by**:

- `rust/tensogram/tests/edge_cases.rs::float32_nan_inf_neg_zero_bit_exact_roundtrip`
- `rust/tensogram/tests/edge_cases.rs::float64_nan_inf_neg_zero_bit_exact_roundtrip`

Both tests encode `[NaN, +Inf, -Inf, -0.0, +0.0]`, round-trip, and
assert `orig.to_bits() == decoded.to_bits()`. Bit-level
equality — not `==` which would fail for NaN. The `-0.0` vs `+0.0`
distinction is also pinned.

**Gap**: both tests use the default pipeline (`filter="none"`,
`compression="none"`). See §3.2 for the untested-but-assumed corners.

### 2.2 `encoding="simple_packing"` — NaN is rejected at four points

Simple packing is GRIB-style lossy quantisation:
`value = reference + 2^E × 10^(-D) × packed_integer`. The formula has
no slot for NaN, so the library rejects it. Four rejection points,
all surfaced as `TensogramError::Encoding` (Rust), `ValueError`
(Python), `EncodingError` (TypeScript), `TGM_ERROR_ENCODING` (C FFI),
or `tensogram::encoding_error` (C++).

| # | Site | File | Semantics |
|---|------|------|-----------|
| 1 | `compute_params()` → `scan_min_max()` | `rust/tensogram-encodings/src/simple_packing.rs` | Returns `PackingError::NanValue(index)`. Sequential: index is globally first NaN. Parallel (`threads ≥ 2` and `values.len() ≥ 8192`): index is the first NaN the current rayon worker sees — **may not be globally first**. |
| 2 | `encode()` pre-check (sequential) | `simple_packing.rs::encode_with_threads` | `values.iter().position(\|v\| v.is_nan())` before packing. |
| 3 | `encode()` per-chunk check (parallel) | `simple_packing.rs::encode_aligned_par` / `encode_generic_par` | NaN check fused into the packing loop; each worker short-circuits on its first NaN. Same first-worker-to-see caveat. |
| 4 | Descriptor `reference_value` validation | `rust/tensogram/src/encode.rs::extract_simple_packing_params` | Rejects **both** NaN and Inf. Catches the case where `reference_value` was externally supplied (e.g. `encode_pre_encoded`) and is non-finite. |

The `encode_pipeline_f64()` typed-input variant (used by the GRIB /
NetCDF converters) goes through the same `encode_with_threads` path
and therefore the same rejection.

### 2.3 Converters — soft downgrade with a stderr warning

`rust/tensogram/src/pipeline.rs::apply_pipeline` is shared by
`tensogram-grib` and `tensogram-netcdf`. If the user requests
`--encoding simple_packing` and the variable contains NaN,
`compute_params()` fails, and the shared helper:

```rust
match simple_packing::compute_params(values, bits, 0) {
    Ok(params) => { desc.encoding = "simple_packing".into(); ... }
    Err(e) => {
        eprintln!("warning: skipping simple_packing for {var_label}: {e}");
        // desc.encoding stays "none" — conversion continues
    }
}
```

Rationale (inline comment): "Common cause: NaN values from unpacked
fill_value."

The NetCDF converter's own `read_and_unpack()` deliberately **inserts**
NaNs during unpacking, so a variable that was packable in the source
NetCDF routinely arrives at `apply_pipeline` with NaNs. Soft downgrade
avoids failing an entire conversion over a single variable.

Tested: `rust/tensogram/src/pipeline.rs::simple_packing_with_nan_values_skips_with_warning`,
`rust/tensogram-netcdf/tests/integration.rs::attr_type_variants_missing_value_replaced_with_nan`.

### 2.4 CBOR metadata — NaN is preserved but equality is bitwise

NaN can appear in metadata as a `ciborium::Value::Float(NaN)`, for
example in a MARS-like attribute.

- **Serialisation**: encoded verbatim with its bit pattern. RFC 8949
  §4.2 canonical ordering places the map key deterministically.
- **`compute_common()`** (`rust/tensogram/src/metadata.rs`): uses
  `cbor_values_equal()`, which for `Float` compares via
  `f64::to_bits()`. Two `Float(NaN)` entries with **identical bit
  patterns** are treated as equal — deliberately diverging from IEEE 754
  equality so that a metadata key whose value is NaN in every `base[i]`
  entry gets classified as common. Tested in
  `test_compute_common_nan_values_treated_as_equal`.

Documented in `docs/src/edge-cases.md` §"compute_common() NaN Handling".

### 2.5 Validate (`ValidationLevel::Fidelity`, `--full`) — NaN is an error

`rust/tensogram/src/validate/fidelity.rs` scans decoded float payloads
per element for NaN and Inf and reports them as **errors**
(`IssueCode::NanDetected` / `InfDetected`). Handles every float type:

- `float32` / `float64`: `val.is_nan()` / `val.is_infinite()`.
- `float16`: bit-level inspection (`exp == 0x1F` + `mantissa != 0`).
- `bfloat16`: bit-level (`exp == 0xFF`).
- `complex64` / `complex128`: scans real and imag independently;
  reports "real component" vs "imaginary component".
- integers, bitmask: skipped (no NaN possible).

A NaN found at Level 4 forces `hash_verified = false`. Default
validation (no flag or `--quick`, `--checksum`) does **not** run Level
4, so NaN-bearing messages pass unless the user opts in with `--full`.

Tested in `rust/tensogram/src/validate/mod.rs::{full_mode_nan_float64_detected,
full_mode_inf_float64_detected, full_mode_float16_nan, full_mode_bfloat16_nan,
full_mode_complex64_real_nan, full_mode_complex128_imag_inf, …}`.

### 2.6 Lossy float codecs — delegated, uncovered

`compression="zfp"` and `compression="sz3"` operate on typed `f64`
values. Wrapper code in `rust/tensogram-encodings/src/compression/{zfp,sz3}.rs`:

- Deserialises the payload bytes to `Vec<f64>` via `bytes_to_f64_native`.
- Hands the values to the upstream library (`zfp-sys-cc` or
  `tensogram-sz3` → `tensogram-sz3-sys` C++ shim).
- Serialises the decompressed values back to bytes.

**Neither wrapper scans for NaN or Inf.** Behaviour on NaN input is
whatever the upstream library produces. The unit tests in those files
use smooth sinusoidal data only — there is no coverage for NaN
survival (or lack thereof).

This is §3.3 below.

### 2.7 Lossless bulk codecs — assumed transparent, untested

`compression ∈ {lz4, zstd, blosc2, szip}` and `filter=shuffle` operate
on the byte level and are lossless by construction, so NaN bit patterns
**should** survive. There is no explicit test pinning this with NaN
inputs. See §3.2.

### 2.8 Integrity hashing

xxh3-64 is computed over the encoded payload bytes. Different NaN bit
payloads (IEEE 754 allows ~2^51 distinct NaN bit patterns for double)
produce different hashes. Canonical CBOR does not normalise NaN. Two
"semantically identical" messages that differ only in their NaN bit
payloads have different xxh3 hashes and different CBOR bytes. See §3.7.

### 2.9 Compact matrix

| Pipeline | NaN in data | Inf in data | Bit-exact roundtrip | Notes |
|----------|:---:|:---:|:---:|-------|
| `none` + `none` + `none` | ✅ preserved | ✅ preserved | ✅ tested | bit-for-bit |
| `none` + `shuffle` + `none` | (✅) | (✅) | ❌ untested | lossless permutation, should preserve |
| `none` + `none` + `lz4`/`zstd`/`blosc2` | (✅) | (✅) | ❌ untested | byte-level lossless |
| `none` + `none` + `szip` | (✅) | (✅) | ❌ untested | integer codec, byte-level |
| `none` + `none` + `zfp` | ❓ **undefined** | ❓ **undefined** | ❌ untested | delegated to C lib |
| `none` + `none` + `sz3` | ❓ **undefined** | ❓ **undefined** | ❌ untested | delegated to C++ lib |
| `simple_packing` (any) | ❌ rejected (EncodingError) | ⚠️ silently produces bad params (§3.1) | n/a | |
| `simple_packing` + shuffle / any compressor | ❌ rejected | ⚠️ silent corruption | n/a | |
| `encode_pre_encoded` with `reference_value=NaN/Inf` | ❌ MetadataError | ❌ MetadataError | n/a | descriptor-level guard |

Parenthesised ✅ entries are inferred from codec semantics rather than
pinned by a regression test. See §3.2.

## 3. Gaps and gotchas — catalogue

Numbered so proposals in §4 can cite them directly.

### 3.1 Inf in simple_packing data silently corrupts output

The biggest data-integrity issue in the current behaviour.
`simple_packing::compute_params` scans for NaN but **not** for Inf.
If a caller passes `[1.0, Inf, 3.0]` to `compute_params`:

- `range = max - min = Inf - 1.0 = +Inf`.
- `binary_scale_factor = (Inf / max_packed).log2().ceil() as i32`.
  `Inf.log2().ceil() = Inf`; `Inf as i32` saturates to `i32::MAX`
  (stable Rust saturating cast, since 1.45).
- `SimplePackingParams` carries `reference_value = 1.0,
  binary_scale_factor = 2_147_483_647`.

Then during `encode()`:

- `scale = 10^D × 2^(-2^31)` underflows to `0.0`.
- `((value - ref) * 0).round() as u64 = 0` for finite values;
  `((Inf - 1) * 0).round()` = `NaN.round()` → `NaN as u64 = 0` for Inf.
- All packed values are `0`.

During `decode()`:

- `inv_scale = 2^(2^31) × 10^(-D)` overflows to `+Inf`.
- `value = ref + Inf × 0 = ref + NaN = NaN` for every element.

**Net effect**: every decoded value becomes NaN silently. The only
safety net is the `extract_simple_packing_params` check at the
descriptor-validation stage of the high-level `encode()` — that
rejects `reference_value.is_infinite()`. But:

- This net triggers only if `reference_value` itself is non-finite.
  For data like `[1.0, Inf, 3.0]` the derived `reference_value` is
  `1.0`, which **passes** the finite check. The corruption is from the
  `binary_scale_factor`, which the descriptor validator does not check.
- For data like `[1.0, -Inf]` the derived `reference_value` is `-Inf`
  and the net catches it. Asymmetric.
- For data like `[Inf, Inf]` the derived `reference_value` is `+Inf`
  and the net catches it.

**So the specific failure mode "mixed finite + Inf data"** is a silent
corruption. Python's `compute_packing_params` inherits this — no Inf
check in the binding either.

### 3.2 Bit-exact roundtrip tests cover only passthrough, not codec combos

The `float{32,64}_nan_inf_neg_zero_bit_exact_roundtrip` tests use the
default pipeline (`encoding=none`, `filter=none`, `compression=none`).
A future change to:

- the shuffle filter (e.g., a byte-alignment tweak);
- a compression codec (e.g., blosc2 adding a `BLOSC_SPECIAL_NAN`
  optimisation that compresses NaN-only chunks specially);
- the xxh3 input scope;

could silently break NaN bit-pattern preservation on other pipelines
without tripping any existing test. The combinatorial matrix is small
enough to cover exhaustively.

### 3.3 `zfp` and `sz3` with NaN: undefined behaviour

Neither wrapper scans for NaN. Neither wrapper has a NaN round-trip
test. Behaviour is delegated to:

- **zfp** (via `zfp-sys-cc`): the upstream C library documents
  undefined behaviour for NaN/Inf in fixed-rate mode. In
  fixed-accuracy and fixed-precision modes it may partially succeed
  but with unpredictable values.
- **sz3** (via `tensogram-sz3-sys`): the C++ SZ3 library has limited
  NaN handling; its predictors (Lorenzo, interpolation, regression)
  are numerical and NaN-propagating.

The library silently forwards the caller's NaN and returns whatever
the codec produced. The decoded output may:

- contain NaN (NaN-propagation through the predictor).
- contain garbage finite values at NaN positions.
- silently corrupt neighbouring finite values (predictors are
  multi-element).
- fail at decompress time with an opaque C-level error.

All four outcomes have been observed in upstream bug reports against
zfp and SZ3 over the years. We have no pinned test; behaviour depends
on the exact upstream version.

### 3.4 Parallel NaN index non-determinism

`compute_params_with_threads` and `encode_with_threads` short-circuit
on the first NaN that any rayon worker sees. The reported index in
`PackingError::NanValue(index)` is therefore **not the globally first
NaN** when `threads ≥ 2` and `values.len() ≥ 8192`. This is
documented in the doc comments of both functions and is an accepted
trade-off.

User-visible surprise: running the same code twice on the same
NaN-containing data, with `threads=8` in both runs, can report
different NaN indices. Automated error-matching tests that look for
the "first NaN index" will be flaky.

### 3.5 `compute_params` + `encode` vs `extract_simple_packing_params` — API asymmetry

When a user goes through the **high-level** `tensogram::encode()`
flow:

```
user-provided descriptor (with simple_packing params)
  → extract_simple_packing_params (validates reference_value finite)
  → compute pipeline config
  → encode_pipeline
```

The finite check at `extract_simple_packing_params` catches
non-finite `reference_value` regardless of whether it came from the
caller or from an earlier `compute_params` call. Safety net present.

When a user goes through the **standalone**
`tensogram_encodings::simple_packing::compute_params()` → `encode()`
API (e.g. from Python via `compute_packing_params`):

```
compute_params (no Inf check, no finite check on output)
  → encode (no Inf check)
  → caller uses bytes
```

There is no safety net. Callers who use the standalone API — which is
explicitly exposed as a public Python function — are on their own.

### 3.6 Default validate does not catch NaN

`tensogram validate` without flags runs Levels 1-3 (structure,
metadata, integrity). Level 4 (Fidelity, `--full`) is the only level
that reports NaN/Inf. A user running `tensogram validate forecast.tgm`
on a NaN-bearing file gets a clean report.

The `ValidationReport::is_ok()` method returns `true` for
NaN-bearing messages decoded at the default level. This is correct
(NaN is not a structural error) but is a surprise for users who
expect "valid = no NaN".

### 3.7 Hash asymmetry over NaN bit patterns

xxh3-64 hashes raw bytes. IEEE 754 allows ~2^51 distinct NaN
bit patterns for `f64`. Two "logically equal" messages that differ
only in their NaN bit payloads produce:

- Different encoded bytes.
- Different xxh3 hashes.
- Different golden-file comparisons.

This means:

- A message constructed from `f64::NAN` (the Rust canonical quiet NaN,
  bit pattern `0x7FF8000000000000`) and a message constructed from,
  say, a signalling NaN produced by hardware after an invalid
  operation will not compare equal even though both are "NaN".
- Deduplication systems keying on the hash will correctly identify
  byte-level duplicates but will treat different-NaN-payload messages
  as distinct objects.

This is probably intended (the wire format is byte-level) but is not
documented and is a surprise. See §5 Q7 for the design question.

### 3.8 No descriptor-level NaN policy

The `DataObjectDescriptor` has no field that says "this tensor is
expected to contain NaN" or "this tensor is guaranteed NaN-free". As
a result:

- A producer cannot communicate missingness semantics downstream.
- The converters' "soft downgrade to `encoding=none` on NaN" decision
  is invisible to the consumer — the consumer sees
  `encoding=none` with no signal that `simple_packing` was attempted
  and skipped.
- Consumers building against a schema cannot enforce "this field must
  be NaN-free" statically.

The "NaN bitmask companion object" idea in `IDEAS.md` addresses part
of this (signal presence via a reserved preamble flag + companion
bitmask data object) but does not address the descriptor-level policy.

### 3.9 GRIB vs NetCDF missingness divergence

- **NetCDF converter**: substitutes `_FillValue` / `missing_value`
  with `f64::NAN` during unpacking. The NaN-bearing variable then
  takes the soft-downgrade path if `--encoding simple_packing` is
  requested.
- **GRIB converter**: reads the GRIB bitmap separately (via ecCodes'
  `bitmapPresent` / `missingValue` keys). Missingness is represented
  out-of-band in GRIB itself; `tensogram-grib` currently **does not
  produce NaNs** in the payload — missing values come through with
  whatever the GRIB encoder used as the sentinel (often a specific
  large finite value depending on the packing used, e.g.
  `9999` or `3.4028234663852886e+38`).

This means:

- The same logical "missing value" round-trips as NaN in the NetCDF
  path and as a sentinel value in the GRIB path. Consumers can't
  treat them uniformly.
- Neither path propagates "this field has missing values" as
  structured metadata; downstream tools have to guess.

### 3.10 f16 / bf16 / complex at encode time

Simple packing is f64-only by construction — its `bits_per_value` is
interpreted against `compute_params`'s f64 domain. f16/bf16/complex
dtypes therefore cannot take the simple_packing path at all, so
encode-time NaN detection for these dtypes is a no-op (they always go
through `encoding=none` and preserve bits).

Validate Level 4 **does** perform bit-level NaN/Inf detection for
f16/bf16 (via exponent+mantissa bitpatterns) and for complex64/128
(via per-component f32/f64 checks). Asymmetric: no encode-time
detection, but decode-time validation available.

This is mostly fine — bit-exact preservation is the right default
for types without a packing strategy. But a future lossy codec for
f16 (quantising to u8, say) would need to re-do the NaN rejection at
its own entry point.

### 3.11 Error message index is flat-array, not multi-D

`PackingError::NanValue(index)` carries a flat-array index. Users
decoding the error for a multi-dimensional tensor have to compute
`(i, j, k, …)` from the flat index using the descriptor's
strides. Not surprising for library-level APIs, but is worth
documenting in the user-facing error text.

### 3.12 `verify_hash` does not catch payload NaN corruption

If a message was encoded with NaN in the data and `encoding=none`,
and during transport a single NaN-bit becomes a different NaN-bit
(e.g. via a float-normalisation step in a middlebox — rare but
observed in some GPU-to-CPU transfer paths), the hash will mismatch.
That's good.

But if the transport zeros out a NaN and replaces it with a canonical
NaN pattern (e.g. some JIT compilers do this for NaN stability), the
hash also mismatches. Again good — this is detected.

The subtler case: if a middleware layer decodes the message, scans
for NaN, replaces with a sentinel, re-encodes with a new xxh3, the
message still validates structurally but has been altered. This is
outside the library's responsibility; noted here so it doesn't
surprise readers of this document.

### 3.13 Streaming encoder inherits all of the above

`StreamingEncoder::write_object` calls the same `encode_pipeline`
code path. NaN rejection for `simple_packing` works as expected. NaN
preservation for `encoding=none` works as expected. None of the
streaming-specific machinery (preceder frames, footer index
construction) touches payload bytes — all the gotchas above apply
identically in streaming mode.

### 3.14 TypeScript / WASM surface

The WASM crate uses `tensogram` with `szip-pure`, `zstd-pure`, `lz4`
features only. `zfp`, `sz3`, `blosc2` are not available in WASM.
**So §3.3 does not apply to the TypeScript/browser path.** The
WASM-accessible codecs are all byte-level lossless (plus szip, which
is integer-level but still byte-transparent for NaN bit patterns),
so NaN round-trips bit-exactly in every supported WASM pipeline.

Worth calling out because it changes the risk posture for the browser
path: zfp/sz3 NaN risk is a native-only concern.

### 3.15 Async / threaded decode does not re-check NaN

On the decode side, there is no NaN detection at any pipeline stage.
Bad data produced by §3.1 (silent Inf corruption) round-trips through
`decode()` and produces NaN output without error. The only way to
detect it is `validate --full` after the fact.

## 4. Proposed directions

Tiered by cost and priority. Items that mature can be promoted to
`TODO.md`; items that stay speculative move to `IDEAS.md`.

### 4.1 Tier 1 — low-cost fixes (recommend promoting to TODO)

**4.1.1 Reject Inf in `simple_packing::compute_params` and `encode`.**

Fixes §3.1 (the silent-corruption gotcha). Parallel to the existing
NaN rejection. Add:

```rust
pub enum PackingError {
    NanValue(usize),
    InfValue(usize),          // new
    ...
}
```

Modify `scan_min_max` to short-circuit on both NaN and Inf.
Update `encode_with_threads`'s pre-check to call out Inf as well. The
reported index semantics match NaN (globally first for sequential,
worker-first for parallel — same documented caveat).

Cost: ~50 LOC + tests + doc update. No wire-format change.

**4.1.2 Extend the bit-exact roundtrip tests to cover codec combos.**

Fixes §3.2. Parametrise the existing
`float{32,64}_nan_inf_neg_zero_bit_exact_roundtrip` tests over:

- All combinations of `filter ∈ {none, shuffle}` and
  `compression ∈ {none, lz4, zstd, blosc2, szip}` (15 combos).
- For each dtype {float16, bfloat16, float32, float64, complex64,
  complex128}, construct a buffer with representative NaN payloads
  (canonical quiet NaN, signalling NaN, negative NaN, both Infs,
  ±0) and round-trip.

Cost: ~200 LOC test code. No production code changes.

Expected outcome: all 15 combos preserve bits exactly. Any that
don't are actual bugs to investigate before shipping the test.

**4.1.3 Document the NaN policy in one place.**

Fixes §3.6 (surprise about default-validate not catching NaN) and
§3.7 (surprise about hash asymmetry).

Create `docs/src/reference/nan-handling.md` with the full catalogue
below plus the compact §2.9 matrix, and link it from:

- `docs/src/encodings/simple-packing.md`
- `docs/src/guide/error-handling.md`
- `docs/src/edge-cases.md`
- `docs/src/cli/validate.md`

Cost: documentation only.

### 4.2 Tier 2 — medium-cost design work (candidates for TODO)

**4.2.1 `zfp` / `sz3` NaN scan at the wrapper level.**

Addresses §3.3. Two sub-options:

- **4.2.1a Reject NaN/Inf at codec entry, matching `simple_packing`.**
  Add a pre-compress scan in `ZfpCompressor::compress` and
  `Sz3Compressor::compress`. Returns `CompressionError::Nan(index)`
  mapped to `TensogramError::Encoding`.
  - Pros: consistent behaviour across all float codecs; eliminates
    undefined behaviour.
  - Cons: blocks callers who legitimately want "best-effort
    compression with whatever the codec produces" for NaN.
    Conservative, but loses functionality that some users rely on.

- **4.2.1b Document behaviour by running pinned tests against
  upstream.** Write round-trip tests that characterise what zfp and
  sz3 do to NaN today, pin those outcomes, and break loudly if
  upstream changes. Do not reject; trust the caller.
  - Pros: preserves current functionality.
  - Cons: relies on upstream stability; test output is "undefined"
    captured at one version.

- **4.2.1c Hybrid — opt-in flag.** Add `strict_float: bool` to
  `ZfpCompressor` / `Sz3Compressor` (exposed via
  `PipelineConfig`). Default `false` (preserves behaviour);
  `true` runs the pre-compress scan and rejects NaN/Inf.
  - Pros: backwards-compatible, gives users the choice.
  - Cons: another option to document; more test matrix to maintain.

Recommendation: prefer (c). It's the most conservative, it mirrors
how `verify_hash` is opt-in, and it gives users an explicit contract.

Cost: ~100 LOC + tests + docs per codec.

**4.2.2 Converter downgrade behaviour — IMPLEMENTED as hard-fail.**

> **Status**: ✅ LANDED (0.17, commit `2410c162`).  The implementation
> departed from the original "provenance-note" proposal below.  In
> discussion with the maintainer, the soft-downgrade was deemed to
> hide data-quality problems rather than surface them, so the
> converter now **hard-fails** when `--encoding simple_packing` is
> requested on data containing NaN or Inf.  See the
> [convert-netcdf guide](../docs/src/guide/convert-netcdf.md#encoding-pipeline-flags)
> for the user-facing contract.

*Original proposal (kept for historical context — SUPERSEDED by the
hard-fail above):*

Addresses §3.8. When the converters (GRIB / NetCDF, via
`apply_pipeline`) soft-downgrade a variable from `simple_packing` to
`none` because of NaN, emit a provenance note in `base[i]["_extra_"]`
or under a reserved key (pending §5 Q5 on `_reserved_` extensions).
Concretely:

```json
{
  "base": [{
    "mars": { ... },
    "_extra_": {
      "encoding_downgraded_from": "simple_packing",
      "encoding_downgraded_reason": "NaN values present"
    }
  }]
}
```

**4.2.3 Standalone-API safety net — IMPLEMENTED.**

> **Status**: ✅ LANDED (0.17, commit `281a99cc`).  See the
> [strict-finite guide](../docs/src/guide/strict-finite.md#simple_packing-params-safety-net-always-on)
> for the user-facing contract.

Addresses §3.5. `simple_packing::encode_with_threads` now validates
its [`SimplePackingParams`] input unconditionally at the top of every
encode call:

- `reference_value.is_finite()` else
  `PackingError::InvalidParams { field: "reference_value", .. }`.
- `|binary_scale_factor| <= MAX_REASONABLE_BINARY_SCALE` (public const,
  value `256`) else
  `PackingError::InvalidParams { field: "binary_scale_factor", .. }`.
- `bits_per_value` extremes unchanged: `> 64` caught by the
  pre-existing `BitsPerValueTooLarge`; `= 0` intentionally accepted
  (legitimate constant-field encoding).

Cross-language parity: Rust unit + integration tests, Python, TS,
C++.  Fires on every encode path including the high-level
`tensogram::encode()` via delegation.

*Original proposal (for posterity):* Have `simple_packing::compute_params`
additionally validate that its output is itself finite and sane. The
final implementation lives on `encode_with_threads` instead (per
maintainer's Q6 answer) so hand-crafted or mutated params are caught
regardless of whether `compute_params` was ever called.

### 4.3 Tier 3 — wire-format / API changes (candidates for IDEAS)

**4.3.1 Descriptor-level `nan_policy` field.**

Addresses §3.8. Add to `DataObjectDescriptor`:

```rust
enum NanPolicy {
    NaNFree,       // encoder guarantees no NaN; decoder may assert.
    Tolerated,     // NaN may be present; consumers should handle.
    Bitmask,       // see 4.3.2 — NaN positions in a companion object.
}
```

Producers set the policy; consumers can react. This is a
wire-format-visible descriptor field, so it counts as an API change
— not a wire-format version bump (new keys in a CBOR map are
silently ignored by old decoders) but does require cross-language
plumbing.

**4.3.2 NaN bitmask companion object** (already in `IDEAS.md`).

Use a reserved preamble flag (new bit on `MessageFlags`) to signal
"the next data object's NaN positions are carried in the immediately
following bitmask object". Encoders that would otherwise fail
`simple_packing` on NaN can emit two objects: the NaN-substituted
data + the bitmask, with the consumer reconstructing NaN positions
at decode time.

Interaction with §4.3.1: a descriptor with
`nan_policy = Bitmask` indicates this pattern is in use.

This is a genuine wire-format extension, unlike §4.3.1. Requires
agreement across all language bindings. Pre-conditions: §4.3.1
promoted, §4.2.2 in place.

**4.3.3 Hash policy over NaN bit patterns.**

Addresses §3.7. Three options:

- **Status quo**: raw byte hash over whatever NaN payload the producer
  wrote. Differentiable across NaN payloads. Current behaviour.
- **Canonical NaN normalisation**: before hashing, substitute every
  NaN with a canonical bit pattern (e.g. `f64::NAN.to_bits() =
  0x7FF8000000000000`). Hashes become stable across NaN-payload
  variations.
- **Hash-aware NaN policy on the descriptor**: let the producer
  signal "treat all NaN as equivalent for hashing purposes".

Recommendation: probably status quo — the cost of normalisation
(extra scan on encode, plus a subtlety in the hash-while-encoding
design in `HASH_WHILE_ENCODING.md`) isn't worth it unless a concrete
dedup use case requires it. Worth writing up for completeness, not
worth building right now.

## 5. Open design questions

These need answers before any of §4 is implemented.

**Q1.** Should the "reject Inf in simple_packing" fix (§4.1.1) be
back-ported to the standalone `simple_packing::compute_params`
function, or only to the high-level `encode()` flow?

  - Back-porting to standalone: closes the footgun, may break
    callers who currently pass Inf-bearing arrays and rely on
    undefined behaviour being "good enough".
  - Only high-level: leaves standalone as a footgun, but that API is
    explicitly "low-level" and documented as such.

  Current recommendation: **back-port to standalone**. The current
  behaviour is not a feature; it's an oversight.

**Q2.** Threshold for "unreasonable" `SimplePackingParams` in §4.2.3.
What value of `binary_scale_factor` is unambiguously wrong? The
working range in production weather data is `[-60, 30]`. A threshold
of `abs() > 256` captures all real-world data with plenty of headroom
and catches `i32::MAX`-level corruption.

**Q3.** For §4.2.1 (zfp/sz3), if option (c) is chosen, what is the
default value of `strict_float`? `false` (preserve behaviour) or
`true` (fail loud)?

  Current recommendation: **`true`**. The current behaviour is
  undefined, which is worse than a caller-visible error. A one-line
  opt-out (`strict_float: false`) gives users who really want the
  upstream behaviour a way to get it.

**Q4.** Should the validate command get a `--nan-as-warning` flag
to demote NaN/Inf from errors to warnings in the Level 4 report?
Some users want to see "your data has N NaNs" as an informational
output, not a validation failure.

**Q5.** Is the "soft downgrade" provenance note (§4.2.2) in
`_extra_` or in a new `_reserved_` sub-key? `_extra_` is
client-writable; `_reserved_` is library-only. Library-emitted
provenance notes feel like they belong in `_reserved_`, but the
current convention puts all encoder provenance under
`_reserved_.encoder`, not `_reserved_.downgrades`. This may warrant
widening the `_reserved_` convention.

**Q6.** For complex64/128 at encode time, does "NaN" mean "either
component is NaN" or "both components are NaN"? The validator (§2.5)
reports either component; a future encode-time scan should match.

**Q7.** Is canonical-NaN normalisation for hashing (§4.3.3) a
future concern worth reserving design space for, or a settled "no"?

**Q8.** Should `Dtype::Bitmask` get first-class support for
representing NaN positions alongside float arrays? (`Bitmask` as a
sibling object to a float array, semantically meaning "true at
positions where the float is NaN".) This is one of the patterns
callers use today, but without tooling support.

**Q9.** GRIB missingness (§3.9): should the `tensogram-grib`
converter, when reading a GRIB bitmap, substitute missing values
with NaN (matching NetCDF) for consistency? Or keep the ecCodes
sentinel convention and document the difference?

**Q10.** Should the NaN rejection error include a field with the
dtype in the message (`"NaN value encountered at index 42 of
float32 array"`)? Currently it's just `"NaN value encountered at
index 42"`. Useful for logs where the caller has lost track of
which variable hit the error.

## 6. Test matrix for future work

Every proposed change in §4 should come with behaviour-driven tests.
Specific test shapes below. Apply TDD — tests fail first, then the
implementation lands.

### For §4.1.1 (reject Inf in simple_packing)

```rust
#[test]
fn simple_packing_rejects_positive_inf_in_compute_params() {
    let values = vec![1.0, f64::INFINITY, 3.0];
    let err = simple_packing::compute_params(&values, 16, 0).unwrap_err();
    assert!(matches!(err, PackingError::InfValue(1)));
}

#[test]
fn simple_packing_rejects_negative_inf_in_compute_params() {
    let values = vec![f64::NEG_INFINITY, 2.0];
    let err = simple_packing::compute_params(&values, 16, 0).unwrap_err();
    assert!(matches!(err, PackingError::InfValue(0)));
}

#[test]
fn simple_packing_rejects_inf_in_encode() {
    let params = SimplePackingParams {
        reference_value: 1.0, binary_scale_factor: 0,
        decimal_scale_factor: 0, bits_per_value: 16,
    };
    let err = simple_packing::encode(&[1.0, f64::INFINITY], &params).unwrap_err();
    assert!(matches!(err, PackingError::InfValue(1)));
}

#[test]
fn simple_packing_rejects_inf_mixed_with_nan_reports_first() {
    // Whichever is first in the sequential scan wins.
    let values = vec![1.0, f64::INFINITY, f64::NAN, 4.0];
    let err = simple_packing::compute_params(&values, 16, 0).unwrap_err();
    match err {
        PackingError::NanValue(i) | PackingError::InfValue(i) => assert_eq!(i, 1),
        _ => panic!(),
    }
}

#[cfg(feature = "threads")]
#[test]
fn simple_packing_rejects_inf_parallel() {
    let mut values = vec![1.0; 100_000];
    values[50_000] = f64::INFINITY;
    let err = simple_packing::compute_params_with_threads(&values, 16, 0, 4).unwrap_err();
    assert!(matches!(err, PackingError::InfValue(_)));
}
```

### For §4.1.2 (bit-exact roundtrip across codec combos)

```rust
fn codec_combos() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("none", "none", "none"),
        ("none", "shuffle", "none"),
        ("none", "none", "lz4"),
        ("none", "shuffle", "lz4"),
        ("none", "none", "zstd"),
        ("none", "shuffle", "zstd"),
        ("none", "none", "blosc2"),
        ("none", "shuffle", "blosc2"),
        ("none", "none", "szip"),
        // (no simple_packing combos — NaN is rejected there)
    ]
}

#[test]
fn nan_bit_exact_roundtrip_f32_all_codecs() {
    let specials: [u32; 6] = [
        0x7FC00000,  // canonical quiet NaN
        0x7F800001,  // signalling NaN
        0xFFC00000,  // negative quiet NaN
        0x7F800000,  // +Inf
        0xFF800000,  // -Inf
        0x80000000,  // -0.0
    ];
    let data: Vec<u8> = specials.iter()
        .flat_map(|bits| bits.to_ne_bytes())
        .collect();

    for (enc, filt, comp) in codec_combos() {
        let desc = make_descriptor(vec![6], Dtype::Float32, enc, filt, comp);
        let (_, objects) = encode_roundtrip(&desc, &data);
        let got_bits: Vec<u32> = objects[0].1.chunks_exact(4)
            .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            got_bits, specials,
            "NaN/Inf bit-exact roundtrip failed for {enc}+{filt}+{comp}"
        );
    }
}

// Parallel f64, f16, bf16, complex64, complex128 tests with the
// matching special bit patterns per dtype.
```

### For §4.2.1 (zfp/sz3 strict_float)

```rust
#[test]
fn zfp_rejects_nan_when_strict_float_true() {
    let compressor = ZfpCompressor {
        mode: ZfpMode::FixedRate { rate: 16.0 },
        num_values: 4,
        byte_order: ByteOrder::Little,
        strict_float: true,
    };
    let mut values = vec![1.0, 2.0, 3.0, 4.0];
    values[2] = f64::NAN;
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert!(matches!(
        compressor.compress(&data),
        Err(CompressionError::Nan { index: 2 })
    ));
}

#[test]
fn zfp_tolerates_nan_when_strict_float_false() {
    // Behaviour documented but not asserted — pin with a round-trip
    // that at minimum does not panic.
    let compressor = ZfpCompressor {
        strict_float: false, ..
    };
    // ... compress + decompress, no panic.
}

// Analogous tests for Sz3Compressor.
```

### For §4.2.2 (converter downgrade provenance)

```rust
#[test]
fn netcdf_simple_packing_downgrade_notes_reason() {
    // build a NetCDF file with a variable containing NaN after unpacking
    // run convert with --encoding simple_packing
    // decode the output, assert base[i]["_extra_"]["encoding_downgraded_from"] exists
}
```

### Regression pin

Keep the existing `float{32,64}_nan_inf_neg_zero_bit_exact_roundtrip`
tests unchanged. Add the new parameterised tests alongside them so
any drift in either catches early.

## 7. Verification approach

Behaviour-driven: each gap in §3 has a "given / when / then" test. No
production change lands without its test failing first.

For §4.1.1 (Inf rejection):

- **Given** a f64 slice containing +Inf at index N,
- **when** `compute_params` is called with any bits,
- **then** the error is `InfValue(N)` (sequential) or `InfValue(_)`
  (parallel with `N ≥ 8192`).

For §4.1.2 (codec combo roundtrips):

- **Given** each codec combo × each dtype with NaN/Inf/±0 specials,
- **when** encoded and decoded,
- **then** the output bytes match the input byte-for-byte.

For §4.2.1 (zfp/sz3 strict_float):

- **Given** `strict_float = true` and NaN-bearing input,
- **when** compressed,
- **then** `CompressionError::Nan` is returned.
- **Given** `strict_float = false` and NaN-bearing input,
- **when** compressed and decompressed,
- **then** the operation does not panic and the output length matches
  input length (the values themselves are whatever upstream produced;
  we pin the shape, not the values).

Standard verification checklist before promotion:

```
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
# For zfp/sz3 tests:
cargo test -p tensogram-encodings --features zfp,sz3
# For C++ parity, if the fix is visible through the FFI:
cmake --build build && ctest --test-dir build
# Python parity:
(cd python/bindings && maturin develop)
python -m pytest python/tests/
```

## 8. Out of scope

- **Cryptographic signing over NaN-stable representations.** Discussed
  in `BRAINSTORMING.md` §B2 but unrelated to NaN handling per se.
- **GPU-side NaN handling for §IDEAS GPU encoding/decoding.** The
  current memo restricts itself to the CPU pipeline.
- **Re-designing `simple_packing` to represent NaN natively.** GRIB
  doesn't either — NaN is represented out-of-band via bitmaps. Any
  "native NaN in simple_packing" work is a much bigger wire-format
  change than §4.3.2 and would break GRIB-compatibility.
- **IEEE 754 signalling-NaN semantics.** The library treats all NaN
  bit patterns uniformly at the logical level (all are "a NaN"). The
  bit-exact roundtrip guarantee preserves whatever payload the
  producer wrote, including signalling NaNs, but the library itself
  does not trap on sNaN.

## 10. Empirical findings — `zfp` / `sz3` NaN/Inf behaviour

> **Status**: pinned by
> `rust/tensogram-encodings/tests/lossy_float_nan_characterisation.rs`.
> Upstream versions pinned via `Cargo.lock`:
> `zfp-sys-cc = 0.2`, `tensogram-sz3 = 0.15`.  A future upstream bump
> that changes any of these outcomes will trip the characterisation
> tests.

Outcome legend:
- **N (preserved)** — NaN/Inf survives at the original position with
  the same kind; finite neighbours within the codec's error bound.
- **R (bad-slot silently replaced)** — NaN/Inf at the original
  position silently decodes to a finite value; neighbours unaffected.
- **C (neighbour corruption)** — neighbours drift beyond tolerance
  or become NaN.
- **G (garbage)** — >50% of finite positions damaged, or output
  length mismatch.
- **E (encode / decode error)** — codec self-defends.

### ZFP: always class R

| Mode | Input shape | Outcome | Observed replacement |
|---|---|---|---|
| `fixed_rate(16)` | NaN at index 32 | R | slot → `0.0` |
| `fixed_rate(16)` | +Inf at index 16 | R | slot → `-2.0` |
| `fixed_rate(16)` | -Inf at index 40 | R | slot → `-2.0` |
| `fixed_rate(16)` | all-NaN | R | all slots → `0.0` |
| `fixed_precision(32)` | NaN at index 10 | R | slot → `0.0` |
| `fixed_precision(32)` | +Inf at index 20 | R | slot → `-2.0` |
| `fixed_accuracy(1e-4)` | NaN at index 5 | R | slot → `0.0` |
| `fixed_accuracy(1e-4)` | +Inf at index 45 | R | slot → `-2.0` |

**Interpretation.** ZFP never errors on NaN/Inf input and never
propagates the bad value into neighbours.  It silently substitutes a
finite value at the bad slot.  Consumers of zfp-decoded data lose the
"non-finite" signal at those positions — e.g. a "missing value" marker
encoded as NaN is indistinguishable from `0.0` after round-trip.

**Severity.** Medium — localised but undetectable without an
external bitmask or a comparison against the pre-compress input.

### SZ3: always class N (NaN/Inf-preserving)

| Mode | Input shape | Outcome | Max neighbour error |
|---|---|---|---|
| `absolute(1e-4)` | NaN at index 128 | N | `4.4e-16` |
| `absolute(1e-4)` | +Inf at index 64 | N | `4.4e-16` |
| `absolute(1e-4)` | -Inf at index 192 | N | `1.8e-15` |
| `absolute(1e-4)` | NaN + both Infs mixed | N | `1.3e-15` |
| `relative(1e-3)` | NaN at index 100 | N | `1.3e-2` |
| `relative(1e-3)` | +Inf at index 50 | N | `0.0` |
| `psnr(60)` | NaN at index 77 | N | `3.4e-3` |

**Interpretation.** SZ3 — at least in the version we pin — handles
NaN/Inf surprisingly cleanly.  The bit pattern is preserved at the
bad position and neighbours come through within the declared error
bound.  This is better than the memo's pre-characterisation
expectation (§3.3 guessed that SZ3's predictors would propagate NaN
numerically).

**Severity.** Low — behaviour matches the contract a reasonable
caller would expect.

### Decision matrix — what next?

With the empirical outcomes in hand, the §5 Q8 question ("which
defence?") narrows to:

- **D1 (always-on pre-compress scan in the wrappers).**
  - zfp: fixes the class-R silent replacement.
  - sz3: **unnecessary** — already class N.
  - Downside: locks users out of any future zfp version that might
    handle NaN better, without a release of Tensogram.
  - ❓ Probably too aggressive.

- **D2 (opt-in `strict_float` on `ZfpCompressor`, default true).**
  - Closes the zfp class-R gap by default for users who construct
    compressors directly.
  - But most users construct compressors via
    `encode_pipeline()` from a descriptor, where there's no
    ergonomic path to set this flag.
  - ❓ Inconsistent surface area.

- **D3 (no wrapper-level defence; W1 is the escape hatch).**
  - The new `reject_nan` / `reject_inf` `EncodeOptions` flags
    already catch NaN/Inf **before** zfp/sz3 sees them.
  - Users who care about non-finite preservation have a
    documented path forward.
  - Users who don't care keep the current behaviour (and the sz3
    class-N finding documents it's not a silent surprise).
  - Users who explicitly want to let zfp eat their NaNs (unlikely
    but possible) can still do so.
  - ✅ **Clean separation of concerns, no new knobs.**

- **D4 (auto-enable `reject_nan` when compression is zfp/sz3).**
  - Would change the default behaviour of existing users using
    zfp without flags — breaking change.
  - For sz3 it would be unnecessary (no corruption to prevent).
  - ❌ **Rejected** given empirical sz3 finding.

### Recommendation

Prefer **D3**.  The empirical finding that sz3 preserves cleanly
removes one of the main arguments for codec-level defence.  The
remaining zfp gap is covered by W1's `reject_nan` / `reject_inf`
flags, which are pipeline-independent and language-wide.

**Documentation follow-up if we go with D3:**
- Add a table in `docs/src/encodings/compression.md` summarising the
  findings above so callers know what to expect.
- Reference the characterisation tests from the encodings guide so
  the test file is treated as the authoritative behaviour record.

### Open question for maintainer

1. Accept recommendation D3?  Or prefer D1 / D2 on the grounds that
   the class-R zfp behaviour is a latent footgun even with W1
   available as an escape hatch?
2. If D3: pin zfp-sys-cc and tensogram-sz3 to specific minor versions
   in `Cargo.toml` (not just the lock file) so the characterisation
   findings remain the authoritative record across contributors?

## 9. Code and test pointers

### Core encoding paths

- `rust/tensogram-encodings/src/simple_packing.rs`
  - `scan_min_max` (sequential + rayon) — NaN-only check.
  - `compute_params`, `compute_params_with_threads`.
  - `encode`, `encode_with_threads`, `encode_aligned_par`,
    `encode_generic_par`.
- `rust/tensogram/src/encode.rs::extract_simple_packing_params` —
  finite check on `reference_value`.
- `rust/tensogram/src/pipeline.rs::apply_pipeline` — converter
  shared helper; soft downgrade on NaN.
- `rust/tensogram-encodings/src/compression/{zfp,sz3}.rs` — no NaN
  handling.

### Validation

- `rust/tensogram/src/validate/fidelity.rs` — Level 4 scanners for
  f32, f64, f16, bf16, complex64, complex128.
- `rust/tensogram/src/validate/types.rs::IssueCode::{NanDetected,
  InfDetected}`.

### Metadata

- `rust/tensogram/src/metadata.rs::cbor_values_equal` — NaN-safe
  bitwise equality for CBOR floats.
- `rust/tensogram/src/metadata.rs::compute_common` — uses
  `cbor_values_equal` for common-key detection across `base` entries.

### Converters

- `rust/tensogram-netcdf/src/converter.rs::read_and_unpack` —
  inserts NaN when `_FillValue` / `missing_value` matches.
- `rust/tensogram-grib/src/converter.rs` — GRIB bitmap handling
  (does not produce NaN; leaves sentinel values).

### Tests (currently passing, to keep passing)

- `rust/tensogram/tests/edge_cases.rs::float{32,64}_nan_inf_neg_zero_bit_exact_roundtrip`
- `rust/tensogram/tests/edge_cases.rs::nan_reference_value_rejected`
- `rust/tensogram/tests/edge_cases.rs::infinity_reference_value_rejected`
- `rust/tensogram-encodings/src/simple_packing.rs::test_nan_rejection_in_compute_params`
- `rust/tensogram-encodings/src/simple_packing.rs::test_nan_rejection_in_encode`
- `rust/tensogram-encodings/src/simple_packing.rs::test_compute_params_nan_detection`
- `rust/tensogram-encodings/src/simple_packing.rs::test_compute_params_nan_at_start`
- `rust/tensogram/src/pipeline.rs::simple_packing_with_nan_values_skips_with_warning`
- `rust/tensogram/src/metadata.rs::test_compute_common_nan_values_treated_as_equal`
- `rust/tensogram/src/metadata.rs::test_compute_common_nested_maps_with_nan`
- `rust/tensogram/src/validate/mod.rs::full_mode_nan_float64_detected`
- `rust/tensogram/src/validate/mod.rs::full_mode_inf_float64_detected`
- `rust/tensogram/src/validate/mod.rs::full_mode_float32_le_nan_detected`
- `rust/tensogram/src/validate/mod.rs::full_mode_float16_nan`
- `rust/tensogram/src/validate/mod.rs::full_mode_float16_inf`
- `rust/tensogram/src/validate/mod.rs::full_mode_bfloat16_nan`
- `rust/tensogram/src/validate/mod.rs::full_mode_complex64_real_nan`
- `rust/tensogram/src/validate/mod.rs::full_mode_complex128_imag_inf`
- `rust/tensogram/src/validate/mod.rs::full_mode_hash_verified_false_on_nan`
- `rust/tensogram/src/validate/mod.rs::full_mode_negative_zero_passes`
- `rust/tensogram/src/validate/mod.rs::full_mode_subnormal_passes`
- `rust/tensogram-netcdf/tests/integration.rs::attr_type_variants_missing_value_replaced_with_nan`
- `rust/tensogram-netcdf/tests/integration.rs::multi_dtype_nan_values_in_f64_with_nan`

### User-facing docs

- `docs/src/encodings/simple-packing.md` §"NaN is Rejected"
- `docs/src/edge-cases.md` §"NaN in Simple Packing",
  §"NaN/Infinity in Simple Packing Parameters",
  §"compute_common() NaN Handling"
- `docs/src/guide/error-handling.md` §"NaN in Simple Packing"
- `docs/src/guide/convert-netcdf.md` §"Simple packing with NaN"
- `docs/src/cli/validate.md` — Level 4 explanation
- `docs/src/guide/python-api.md` — NaN in `ValueError`

### Related plans documents

- `DESIGN.md` — one-line mention of `simple_packing` rejecting NaN.
- `IDEAS.md` — NaN bitmask companion object (see §4.3.2).
- `BRAINSTORMING.md` §C2 (mixed precision / region of interest) —
  tangentially related; the "bitmask indicating coverage" pattern
  generalises to the NaN bitmask idea.
- `HASH_WHILE_ENCODING.md` — hash is computed over raw encoded bytes
  including NaN payloads; relevant for §3.7 and §4.3.3.

### External references

- RFC 8949 (CBOR) §4.2 — canonical encoding; does not normalise NaN.
- IEEE 754-2019 §6.2 — NaN payload semantics.
- GRIB-2 manual — bitmap handling as out-of-band missingness.
- ZFP documentation — fixed-rate mode and NaN (undefined).
- SZ3 paper — error-bounded compression assumes finite predictors.
- CF conventions §2.5.1 — `_FillValue` semantics and NaN convention.

---

**Next step before implementing any of this**: resolve Q1, Q2, Q3, Q5,
Q9 in §5 with the maintainer. Tier 1 items (§4.1) are the cheapest
place to start and carry the most user value per LOC.
