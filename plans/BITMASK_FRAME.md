# NaN / Inf Bitmask Companion Frame — Design

**Status**: accepted for implementation, 0.17 wire-format change.
**Supersedes**: `RESEARCH_NAN_HANDLING.md` §3.8 / §4.3.2 (NaN bitmask
companion object — promoted from IDEAS to active design).

## 1. Motivation

The strict-finite work that landed in 0.16 (see `RESEARCH_NAN_HANDLING.md`
§4.1 IMPLEMENTED) gave callers opt-in pipeline-independent rejection via
`reject_nan` / `reject_inf`.  It never produced usable encoded messages
from NaN/Inf-bearing input — callers had to pre-process.

The bitmask companion frame closes that gap by making NaN/Inf a
**first-class wire-format concept**: the encoder substitutes non-finite
values with `0.0` and records their positions in one or more compressed
bitmasks stored inside the same data-object frame.  The decoder
reconstructs the original positions on demand.

Semantically identical workflows from upstream (NetCDF `_FillValue` →
NaN substitution, sensor-failure propagation, sparse ML outputs) become
representable in `simple_packing` and every other pipeline without
pre-processing, at the cost of a small, well-compressed mask blob.

## 2. Default behaviour change — BREAKING

**Pre-0.17**: default library behaviour was to pass NaN / Inf bits
through the pipeline verbatim under `encoding="none"`, and to reject
them only inside `simple_packing::compute_params`.  Opt-in
`reject_nan` / `reject_inf` flags upgraded the rejection to be
pipeline-independent.

**0.17+**: the default flips.  Any NaN / Inf in the input is a hard
error at the encode entry point.  Two new opt-in flags **allow the
values through by substituting with 0 and recording them in bitmasks**:

- `EncodeOptions.allow_nan: bool` (default `false`)
- `EncodeOptions.allow_inf: bool` (default `false`)

The removed fields are `reject_nan` and `reject_inf`.  The new flags
express the same policy inverted: reject is now the default, allow is
the opt-in.

### User-visible impact

Every encode path that used to silently pass NaN / Inf under default
options now errors.  Callers who relied on the pass-through behaviour
must either:
1. **Pre-process** the data (replace NaN / Inf with a sentinel they
   control) — preserves the pre-0.17 behaviour.
2. **Opt in to masking** via `allow_nan=true` / `allow_inf=true` — lets
   the library handle it via the new mask frame.

The pre-existing `reject_nan` / `reject_inf` API surface is removed.

## 3. Frame format

### 3.1 Frame type registry

| Type | Name | Status |
|-----:|---|---|
| 1 | `HeaderMetadata` | unchanged |
| 2 | `HeaderIndex` | unchanged |
| 3 | `HeaderHash` | unchanged |
| 4 | `NTensorFrame` (renamed from `DataObject`) | **legacy**, read-only; new encoders do not emit |
| 5 | `FooterHash` | unchanged |
| 6 | `FooterIndex` | unchanged |
| 7 | `FooterMetadata` | unchanged |
| 8 | `PrecederMetadata` | unchanged |
| 9 | **`NTensorMaskedFrame`** (new) | all new encoders emit this |

### 3.2 Layout (type 9)

```
FR | u16 type=9 | u16 version | u16 flags | u64 total_length
  ┃
  ┠─ [payload] encoded n-tensor bytes            (NaN/Inf substituted with 0.0)
  ┠─ [mask_nan]  bitmask blob       (optional; bit i = 1 iff element i was NaN)
  ┠─ [mask_inf+] bitmask blob       (optional; bit i = 1 iff element i was +Inf)
  ┠─ [mask_inf-] bitmask blob       (optional; bit i = 1 iff element i was -Inf)
  ┠─ [cbor]   CBOR descriptor (with "masks" sub-map if any mask present)
  ┠─ u64 cbor_offset                (relative to payload region start)
ENDF
```

Each section is written contiguously.  No padding between sections.
All byte offsets in the CBOR's `masks` sub-map are **relative to the
start of the payload region** (the first byte after the frame
header).

Wire-format version stays at `2`.  Old decoders that don't recognise
type 9 skip the frame (existing frame-scan logic).

### 3.3 CBOR schema

The descriptor gains an optional `masks` sub-map.  Absence of `masks`
is the "no non-finite values present" case — a `NTensorMaskedFrame`
with no `masks` sub-map is byte-semantically identical to the legacy
`NTensorFrame` payload.

```cbor
{
  ...standard DataObjectDescriptor fields (obj_type, ndim, shape, dtype,
     encoding, filter, compression, params, ...)...
  "masks": {                   ; optional top-level key
    "nan":  { ... },           ; optional; at least one of nan/inf+/inf- when "masks" present
    "inf+": { ... },
    "inf-": { ... }
  }
}
```

Each mask sub-map:

```cbor
{
  "method": "roaring",         ; tstr — one of "rle" | "roaring" | "blosc2" | "zstd" | "lz4" | "none"
  "offset": 800000,            ; uint — byte offset from start of payload region
  "length": 512,               ; uint — byte length of the (compressed) mask blob
  "params": { ... }            ; optional map — method-specific parameters
}
```

`params` by method:
- `rle` — no params.
- `roaring` — no params.  (Serialized Roaring format embeds its own metadata.)
- `blosc2` — `{ "codec": "lz4" | "zstd", "level": int }`.  Defaults `codec="lz4"`, `level=5`.
- `zstd` — `{ "level": int }` (optional, default 3).
- `lz4` — no params.
- `none` — no params.

Canonical CBOR sort order for `masks`' keys: `inf+` < `inf-` < `nan`
(byte-lex).  The writer emits in this order deterministically.

### 3.4 Reading legacy type-4 frames

Decoders read type 4 (`NTensorFrame`) by interpreting it as a type-9
frame with no `masks` sub-map.  All field layouts are identical; only
the type number differs.  This preserves read-compatibility for files
produced by pre-0.17 encoders.

## 4. Bit packing

Raw bit layout (before compression):

- MSB-first, matching the existing `Dtype::Bitmask` convention.
- `ceil(N / 8)` bytes for `N` elements.
- Trailing bits in the last byte are **zero-filled** for
  determinism (required for stable hashing).
- Bit `i` (element index, 0-based) lives at byte `i / 8`, bit
  position `7 - (i % 8)`.

`1` = the element at that index was the specific non-finite kind being
masked.  `0` = the element is finite (or a different non-finite kind).

### Priority for simultaneous classifications

A single element cannot simultaneously be "NaN" and "Inf".  For
**complex** dtypes (c64 / c128), where real and imag are independent,
the priority rule is:

1. **NaN** wins over any Inf (either component being NaN → nan mask).
2. **+Inf** wins over -Inf (real is +Inf or imag is +Inf while neither
   is NaN → inf+ mask).
3. **-Inf** otherwise (applies when the only non-finiteness is -Inf).

After substitution (both components set to `0.0 + 0.0i`), decode
restores with the canonical bit pattern of the mask's kind
(`f64::NAN` / `f64::INFINITY` / `f64::NEG_INFINITY`) written to
**both** real and imag.  See §7 for the lossy-reconstruction caveat.

A future `c-real/c-imag` pair of masks (doubling to 6 masks) is not
in scope — when the lossy round-trip becomes a blocker, we add a new
frame type rather than complicating this one.

## 5. Mask compression methods

Six methods, selectable per-mask.  All are byte-output format; no
format-specific framing inside the mask blob beyond what each
algorithm's standard serialisation provides.

| `method` | Implementation | Default? | Notes |
|---|---|---|---|
| `"rle"` | new pure-Rust impl (`tensogram-encodings/src/bitmask/rle.rs`) | — | bandwidth-bound; best on clustered masks |
| `"roaring"` | `roaring = "0.11"` crate | **default** on all platforms (incl. wasm32, verified) | hybrid array/bitmap/RLE containers |
| `"blosc2"` | reuse existing blosc2 path + `BLOSC_BITSHUFFLE` filter | — | feature-gated `blosc2` as today |
| `"zstd"` | reuse existing zstd codec, zero-copy on packed bytes | — | always available |
| `"lz4"` | reuse existing lz4 codec, zero-copy on packed bytes | — | always available |
| `"none"` | raw packed bytes, no compression | auto-fallback | used when uncompressed size ≤ `small_mask_threshold_bytes` (default 128) |

### Small-mask fallback

For tiny masks, compression overhead exceeds the mask itself.  When
the uncompressed mask byte-count is `≤ small_mask_threshold_bytes`
(default 128, configurable via `EncodeOptions.small_mask_threshold_bytes`
— single threshold across all three masks), the encoder writes the
mask as `"none"` regardless of the user-requested method.  The
descriptor's `method` field reflects what was actually written.

### RLE on-wire format

```
[u8 start_bit] [varint run_1] [varint run_2] ... [varint run_k]
```

- `start_bit`: `0x00` or `0x01` — the value of the first run.
- Each `run_i`: unsigned LEB128 (ULEB128) — count of consecutive bits
  of the alternating value.  Minimum run length 1.  Sum of runs must
  equal exactly `N` (descriptor's total element count); decoder errors
  otherwise.
- Total length of the serialised form is naturally self-delimiting via
  the `length` field in the CBOR descriptor.

Edge cases:

- All-zero mask (shouldn't happen in practice — the mask wouldn't be
  written at all): `[0x00, varint(N)]`.
- All-one mask (every element is the kind): `[0x01, varint(N)]`.
- Alternating bits: worst case for RLE — `[bit, 1, 1, 1, 1, ...]`
  which for typical N can inflate.  The small-mask fallback and the
  per-mask method override let the caller avoid this.

### Roaring on-wire format

Use the standard [Roaring Portable Serialization Format](https://github.com/RoaringBitmap/RoaringFormatSpec).
The `roaring` Rust crate's `serialize_into` produces this format by
default.  We store the raw bytes as-is — no Tensogram-specific
framing.

## 6. Encoder

### 6.1 Pre-pipeline substitution stage

A new stage runs before the encoding pipeline in every encode path.
Input: a slice of float payload bytes; outputs: possibly-modified
bytes + up to three bitmasks.

```
fn substitute_and_mask(
    data: &[u8],
    dtype: Dtype,
    byte_order: ByteOrder,
    allow_nan: bool,
    allow_inf: bool,
) -> Result<(Cow<'_, [u8]>, MaskSet), TensogramError>
```

Behaviour:

- For non-float dtypes: returns `(Cow::Borrowed(data), MaskSet::empty())`
  immediately.  Zero cost.
- For float dtypes: single forward scan over `data`.
  - On each element, classify as Finite / NaN / +Inf / -Inf via
    `validate_sample`-style checks (plus the per-dtype bit-level
    handling for f16 / bf16 already present in `strict_finite`).
  - **Before any substitution** happens, if a classification is
    non-finite and the relevant `allow_*` flag is false: return
    `Err(TensogramError::Encoding("..."))` with element index,
    dtype, and an actionable hint pointing at `--allow-nan` /
    `--allow-inf`.
  - When allowed: lazily allocate the per-kind mask on first hit;
    set the corresponding bit; write `0.0` (dtype-specific zero) at
    the element position in the output buffer.
- For complex c64 / c128: follow priority rule (§4) — classify once
  per element, emit to at most one of the three masks.
- Output buffer: `Cow::Borrowed` if no substitutions (the zero-allocation
  fast path); `Cow::Owned(cleaned_bytes)` otherwise.

Small-mask fallback decision happens at mask-serialisation time, not
here — the substitution stage is responsible for producing raw
bitmasks; the mask codec layer decides whether to compress or emit
`"none"`.

Threading: scan parallelisable the same way `strict_finite::scan` is
today (64 KiB chunks via rayon when `threads > 0` and payload ≥
threshold).  Parallel mask assembly requires per-chunk local masks
that get OR-reduced at the end; correct and deterministic.

### 6.2 Encoder integration

In `encode_one_object` (`rust/tensogram/src/encode.rs`):

1. Validate descriptor and options.
2. Run substitute_and_mask on float dtypes (no-op on others).
3. Pass the (possibly Cow::Owned) cleaned bytes through the existing
   encoding → filter → compression → hash pipeline, unchanged.
4. Compress each mask via the user-specified method (per-mask options
   on `EncodeOptions`).
5. Emit a type-9 `NTensorMaskedFrame` with payload + masks + CBOR.

When `allow_nan == false && allow_inf == false`, the substitution stage
is still invoked (for rejection), but never produces masks.  In that
case the type-9 frame has no `masks` sub-map.

### 6.3 `EncodeOptions` additions

```rust
pub struct EncodeOptions {
    // ...existing fields: hash_algorithm, emit_preceders,
    //    compression_backend, threads, parallel_threshold_bytes...

    // REMOVED: reject_nan, reject_inf

    pub allow_nan: bool,                       // default false
    pub allow_inf: bool,                       // default false
    pub nan_mask_method: MaskMethod,           // default MaskMethod::Roaring
    pub pos_inf_mask_method: MaskMethod,       // default MaskMethod::Roaring
    pub neg_inf_mask_method: MaskMethod,       // default MaskMethod::Roaring
    pub small_mask_threshold_bytes: usize,     // default 128
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskMethod {
    Rle,
    Roaring,          // DEFAULT
    Blosc2 { codec: Blosc2Codec, level: i32 },
    Zstd { level: Option<i32> },
    Lz4,
    None,             // uncompressed; also auto-fallback for small masks
}
```

## 7. Decoder

### 7.1 Lossy reconstruction — documented limitation

The encoder replaces the original float bits (the NaN / Inf value at
position `i`) with `0.0` and records just the position.  The decoder
restores using the **canonical** bit pattern for the kind:

- NaN → `f64::NAN` bit pattern (`0x7FF8000000000000`; quiet NaN).
- +Inf → `f64::INFINITY` bit pattern (`0x7FF0000000000000`).
- -Inf → `f64::NEG_INFINITY` bit pattern (`0xFFF0000000000000`).

For f16 / bf16 / f32 / c64 / c128, the dtype-specific canonical
patterns of the same concepts are used.

**Implication**: specific NaN payloads (signalling NaN, custom
payload bits) are **not preserved** through an `allow_nan=true`
encode.  Callers who need bit-exact NaN payload preservation must
either:
1. Pre-process and encode with `allow_nan=false` + a pass-through
   encoding, pushing the semantics into the caller's data layer, or
2. Use a future wire-format extension (not planned).

This is a deliberate design trade-off: supporting bit-exact NaN
payloads through a bitmask companion would require extending each
mask with an auxiliary per-element payload store, defeating the
compression advantage.  Callers working with signalling NaNs are an
edge case; we optimise for the common case (missing data encoded as
NaN) and document the limitation.

### 7.2 `decode` path

```
fn decode(buf: &[u8], opts: &DecodeOptions) -> Result<(GlobalMetadata, Vec<DecodedObject>)>
```

Unchanged signature.  For each object:
1. Decode frame header, descriptor, payload.
2. If `descriptor.masks` is present:
   a. Decompress each mask via its `method`.
   b. If `opts.restore_non_finite` (default `true`): iterate mask
      bits, write canonical NaN/+Inf/-Inf at each `1` position in the
      decoded payload.
   c. If `opts.restore_non_finite == false`: leave payload as-is
      (user sees the 0-substituted bytes).  Masks are dropped on
      this path — callers who want the masks use `decode_with_masks`.

### 7.3 `decode_with_masks` (new API)

```rust
fn decode_with_masks(
    buf: &[u8],
    opts: &DecodeOptions,
) -> Result<(GlobalMetadata, Vec<DecodedObjectWithMasks>)>

pub struct DecodedObjectWithMasks {
    pub descriptor: DataObjectDescriptor,
    pub payload: Vec<u8>,      // always 0-substituted (masks not applied)
    pub masks: MaskSet,        // may be empty
}

pub struct MaskSet {
    pub nan:  Option<Bitmask>,  // decompressed bits, convenience accessors
    pub pos_inf: Option<Bitmask>,
    pub neg_inf: Option<Bitmask>,
}
```

Advanced callers can apply masks manually, aggregate across kinds,
convert to their own domain types.  Primary consumer is the
`decode_range` machinery below.

### 7.4 `decode_range` with masks

`decode_range` reads sub-ranges of an object.  With masks:

```
decode_range(buf, obj_idx, &[(offset_0, count_0), (offset_1, count_1)], opts)
  ┃
  ▼
1. Decode requested payload sub-ranges via existing pipeline.
2. If masks present AND opts.restore_non_finite:
   a. Decompress the full mask (all three kinds).
   b. For each (offset_i, count_i), slice the mask to cover that range.
   c. Write canonical NaN / +Inf / -Inf at `1` positions in the decoded sub-range.
3. Return the ranges.
```

Future optimisation (post-initial-landing): the Roaring format
supports efficient range queries (`RoaringBitmap::iter().advance_to()`
+ range slicing); blosc2 BitShuffle supports per-block random access;
both could avoid decompressing the full mask.  Not in scope for the
first pass — full-decompress is fine for typical mask sizes.

## 8. Validate — mask-aware Fidelity level

`tensogram validate --full` (Fidelity level in `ValidateOptions`):
currently flags any NaN / Inf in decoded float arrays as
`NanDetected` / `InfDetected` errors.

With masks:

- At each bit set in the NaN mask, NaN is **expected** — no error.
- At each bit set in the +Inf mask, +Inf is expected — no error.
- At each bit set in the -Inf mask, -Inf is expected — no error.
- At any **other** position, NaN / Inf is still an error (indicates
  data corruption or an incorrect mask).

The validator reads the masks on Level 4 and cross-checks against the
reconstructed decoded output.

## 9. CLI

### 9.1 New global flags

```bash
tensogram --allow-nan \                          # also env: TENSOGRAM_ALLOW_NAN
          --allow-inf \                          # also env: TENSOGRAM_ALLOW_INF
          --nan-mask-method roaring \            # also env: TENSOGRAM_NAN_MASK_METHOD
          --pos-inf-mask-method rle \            # also env: TENSOGRAM_POS_INF_MASK_METHOD
          --neg-inf-mask-method rle \            # also env: TENSOGRAM_NEG_INF_MASK_METHOD
          --small-mask-threshold 128 \           # also env: TENSOGRAM_SMALL_MASK_THRESHOLD
          <subcommand> ...
```

`--allow-inf` enables detection of BOTH `+Inf` and `-Inf` (single
flag for the semantic pair).  Per-kind method flags remain independent.

`--reject-nan` / `--reject-inf` (the 0.16 flags): **removed**.
Passing them on the command line produces the standard clap
"unrecognized flag" error.  This is intentional per the
pre-1.0-no-backwards-compat stance.

### 9.2 Env-var bool parsing

Same `clap::builder::BoolishValueParser` convention as the old
strict-finite env vars: `1`/`true`/`yes`/`on` → `true`,
`0`/`false`/`no`/`off`/unset → `false`.

### 9.3 Subcommand coverage

Global flags apply to every encoding-capable subcommand: `merge`,
`split`, `reshuffle`, `convert-grib`, `convert-netcdf`.  `copy` (byte
copy) and `set` (metadata only) are no-ops under these flags.

## 10. Cross-language parity

| Binding | `allow_*` flags | `*_mask_method` options | Status |
|---|---|---|---|
| Rust (core) | kwarg on `EncodeOptions` | field on `EncodeOptions` | implemented in Commit 5 |
| Python (PyO3) | kwarg on every encode fn + `convert_*` | string-valued kwarg | Commit 9 |
| TypeScript (WASM) | field on `EncodeOptions` | string-valued field | Commit 10 |
| C FFI | `bool` params on `tgm_encode`, `tgm_file_append`, `tgm_streaming_encoder_create` | string param | Commit 11 |
| C++ | field on `encode_options` struct | string field | Commit 11 |
| CLI | global flag | global flag | Commit 12 |

All mask methods (`rle`, `roaring`, `blosc2`, `zstd`, `lz4`, `none`)
available from every binding.  `blosc2` errors cleanly at runtime on
bindings built without the `blosc2` feature.

## 11. Test matrix

### 11.1 Exhaustive — f64 and c64

For each of `f64` and `c64`, exhaustively cover:

```
input_shape            ∈ {no-nonfinite, nan-only, pos_inf-only, neg_inf-only, all-three-kinds}
encoding               ∈ {none, simple_packing}
filter                 ∈ {none, shuffle}
compression            ∈ {none, lz4, zstd, blosc2, szip, zfp, sz3}
nan_mask_method        ∈ {rle, roaring, blosc2, zstd, lz4, none}
pos_inf_mask_method    ∈ {rle, roaring, blosc2, zstd, lz4, none}
neg_inf_mask_method    ∈ {rle, roaring, blosc2, zstd, lz4, none}
decode_path            ∈ {decode, decode_with_masks, decode_range(single), decode_range(multi)}
```

Not every combination: exhaustive over the **single-kind** axis for
each dtype (5 input-shapes × 2 encodings × 2 filters × 7 compressions
× 6 methods × 4 decode paths = ~3,360 per dtype).  Constrain to one
sensible combination per dimension except the mask-method axis, which
is exercised fully.

### 11.2 Sampled — other dtypes

One representative combination per dtype for `f16`, `bf16`, `f32`,
`c128`.

### 11.3 Property tests

`proptest` round-trips for RLE and Roaring codecs on random bitmasks
up to 4096 bits.  Invariant: `decode(encode(mask)) == mask`.

### 11.4 Cross-language parity

Each binding has a smoke test covering: (a) encode with `allow_nan=true`,
(b) decode restores NaN at the right index, (c) `method="roaring"` and
`method="rle"` both work.

### 11.5 Regression goldens

5–10 canonical `.tgm` fixtures covering combinations above; byte-level
diffed on every commit.

## 12. Documentation

- **This file** (`plans/BITMASK_FRAME.md`) — source of truth for the
  design.
- `plans/WIRE_FORMAT.md` — add the new frame type to the registry table
  and document the CBOR `masks` schema.
- `docs/src/format/wire-format.md` — user-facing wire-format reference;
  add type 9 section.
- `docs/src/format/cbor-metadata.md` — document the `masks` sub-map.
- Rewrite `docs/src/guide/strict-finite.md` → `docs/src/guide/nan-inf-handling.md`
  describing reject-by-default + allow-with-mask.
- Cross-link from `simple-packing.md`, `convert-netcdf.md`, etc.
- `CHANGELOG.md` — `[Unreleased]` entry with `### Changed — BREAKING`
  for the default-behaviour flip and the API removal.

## 13. Rollout ordering (implementation commits)

Each commit compiles + tests green + clippy clean before the next.
All commits on `feature/improve-nan-handling` branch.

| # | Commit | What |
|--:|---|---|
| 0 | Clean-break | Remove `reject_nan` / `reject_inf` from every surface.  Delete strict-finite tests.  Default encode rejects NaN/Inf across all pipelines.  No masks yet; this commit leaves the library in a fully-strict state. |
| 1 | Frame types | Rename `DataObject` → `NTensorFrame` (type 4).  Add `NTensorMaskedFrame` (type 9).  Back-compat read path.  Golden fixtures updated. |
| 2 | Bit-pack + RLE | `tensogram-encodings/src/bitmask/{mod,packing,rle}.rs`.  `MaskMethod` enum.  Unit + proptest coverage. |
| 3 | Roaring | `roaring` crate dep.  Codec module.  Unit + proptest coverage.  WASM build verified. |
| 4 | Other mask codecs | blosc2, zstd, lz4, none.  Reuse existing codec infrastructure. |
| 5 | Substitution + `allow_*` + encoder integration | `substitute_and_mask` stage.  `EncodeOptions` additions.  End-to-end `allow_nan=true` encode path produces type-9 frame with mask. |
| 6 | Decode + `decode_with_masks` | Reconstruction on decode.  New advanced API.  Both paths tested. |
| 7 | `decode_range` with masks | Exhaustive f64+c64 range tests. |
| 8 | Validate mask-aware | Level 4 cross-checks masks against reconstructed payload. |
| 9 | Python bindings | kwargs on `encode`, `append`, `StreamingEncoder`, `convert_*`.  Tests. |
| 10 | TypeScript + WASM | TS fields + WASM plumbing.  Tests. |
| 11 | C FFI + C++ | FFI params + cbindgen regen.  C++ wrapper fields.  Tests. |
| 12 | CLI | Global flags + env vars.  Integration tests. |
| 13 | Docs | Wire format, CBOR schema, user guide rewrite, CHANGELOG BREAKING section. |
| 14 | Exhaustive test matrix | f64/c64 exhaustive combinations.  Cross-language smoke. |

## 14. Open / deferred design decisions

Deferred to later iterations (not in scope for initial landing):

- **Mixed real/imag masks for complex** — two more masks (6 total) to
  preserve mixed cases exactly.  Blocked by a concrete use-case.
- **Range-optimised mask decode** — Roaring and blosc2 support
  per-range decompression.  Current impl decompresses the full mask
  on first access.
- **Streaming mask write** in `StreamingEncoder` — initial impl
  buffers masks in memory alongside the payload.  Fine for typical
  object sizes; streaming masks would require a more complex frame
  structure.
- **Inf-aware `simple_packing` where Inf **wants** to be preserved at
  the bit level** — out of scope.  Mask + substitution is the only
  first-class path.

---

*End of design.*
