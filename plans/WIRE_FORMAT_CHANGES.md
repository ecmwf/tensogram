# Wire Format Changes — v2 → v3 (phased plan)

> **Status.** DRAFT — sign-off required at Phase 0 before any
> implementation lands.  Target release: `0.17.0` (currently under
> `[Unreleased]` in `CHANGELOG.md`).
>
> **Companion spec.** [`plans/WIRE_FORMAT.md`](WIRE_FORMAT.md) —
> updated to v3, describing the end-state.  Read that first.
>
> **Related work already in-flight on this branch.**
> [`plans/BITMASK_FRAME.md`](BITMASK_FRAME.md) introduced frame
> type 9 (`NTensorMaskedFrame`) and the bitmask companion design.
> v3 is the consolidation + expansion of that work into a full
> wire-format breaking release.

---

## Objectives

The v2 → v3 transition bundles five wire-format changes into one
clean break:

1. **Version bump** — preamble `version` field goes `2 → 3`.  v1/v2
   messages are rejected at the decoder.  All existing golden
   fixtures are regenerated.
2. **Self-locating postamble** — postamble grows from 16 B to 24 B
   by mirroring the preamble's `total_length`, enabling backward
   and bidirectional scan.
3. **Inline per-frame hash slot** — every frame ends with
   `[hash u64][ENDF]`, preceded by any type-specific footer
   fields (e.g. `cbor_offset` on `NTensorFrame`).  The hash slot
   is populated message-wide by the `HASHES_PRESENT` preamble flag.
   Hash scope covers the frame *body* only — nothing in the
   header or footer is hashed.
4. **Frame registry cleanup** — type 4 (obsolete v2 `NTensorFrame`)
   removed + reserved.  Type 9 renamed from `NTensorMaskedFrame`
   to `NTensorFrame` and becomes the first concrete data-object
   type under the generic data-object scheme.  New data-object
   types can be added at fresh type numbers without bumping the
   wire version.
5. **Bitmask codecs as first-class compression methods** — `rle`
   and `roaring` (from the bitmask companion design) become
   selectable `compression` values for `dtype = bitmask` tensors,
   with a dtype guard that errors on misuse.

Plus documentation fixes (hash frames / index frames previously
undocumented in `WIRE_FORMAT.md`).

No backward compatibility is preserved.  All pre-existing
`.tgm` test fixtures are wiped and regenerated from the v3
encoders.

---

## Design decisions (all confirmed)

The two points that were left ambiguous during the planning round
have been resolved before implementation starts:

- **Hash slot position for data-object frames** (was [PD-1]).
  **Option X unified** — hash always at `frame_end − 12` for
  every frame type.  `NTensorFrame` footer is
  `[cbor_offset][hash][ENDF]` (20 B total).  The hash scope
  covers only the frame *body*; neither the header nor any
  byte of the footer (including `cbor_offset`) is hashed.
  This gives a single uniform validator rule while keeping
  `cbor_offset` as a pure locator for the CBOR descriptor.
- **Per-frame `HAS_HASH` flag** (was [PD-2]).  **Not added.**
  Only the preamble bit 7 `HASHES_PRESENT` controls whether
  hash slots are populated across the whole message.  No
  per-frame / per-object selective hashing in v3.  Adding it
  later is non-breaking (reuse an unused bit of
  `DataObjectFlags` on `NTensorFrame`).

---

## Phase order and dependencies

```
Phase 0 ──► Phase 1 ──┬─► Phase 2 ──► Phase 3
                      │
                      ├─► Phase 4
                      │
                      ├─► Phase 5 ──► Phase 6
                      │
                      └─► Phase 7
                                    ▼
                                  Phase 8 ──► Phase 9 ──► Phase 10
```

Phases 2–5 are largely independent and can be parallelised after
Phase 1 lands.  Phase 8 (cross-language parity) pulls from all of
2–7.  Phase 9 (docs) pulls from 2–8.  Phase 10 (golden fixtures
+ CHANGELOG + release prep) is last.

Every phase ends with a **green-quad** invariant:

```
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
mdbook build docs/
```

plus the language-specific test suites that a given phase touches
(pytest for Python, vitest for TypeScript, GoogleTest for C++,
wasm-pack test for WASM).

---

## Phase 0 — Spec sign-off ✅

**Goal.** Align on the wire-format decisions before writing code.

**Status.** **Complete.**  The maintainer has signed off on both
historical open points (hash slot position = Option X unified;
no per-frame `HAS_HASH` flag), and on the follow-up corrections
(hashes exclude all footer bytes including `cbor_offset`; type 9
renamed to `NTensorFrame` without a version suffix; the format
explicitly supports generic data-object types in the body phase).

**Deliverables (all done).**

- [`plans/WIRE_FORMAT.md`](WIRE_FORMAT.md) updated to v3.
- This plan document.
- Design-decisions block above reflects the signed-off state.

**Gate released.** Phase 1 may begin on user's go-ahead.

**Risk retired.** Getting the postamble / hash-slot byte layout
wrong would have meant a second wire-format bump; the spec
document nailed the layout before implementation.

---

## Phase 1 — Version bump (v2 → v3) + hard reject v2

**Goal.** A minimal, stand-alone commit that makes every future
phase land against a clean v3 baseline.  After this phase, every
v2 golden fixture fails to decode and every new encode produces
`version = 3` in the preamble — even though no other structural
changes have been made yet.  This is a failure-inducing phase on
purpose: the v2 tests that fail here are the exact ones that will
be rewritten to v3 in subsequent phases.

**Scope (Rust core).**

- `rust/tensogram/src/wire.rs`:
  - Bump the default `Preamble.version` emission from `2` to `3`.
  - `Preamble::read_from` rejects any `version != 3` with
    `FramingError("unsupported message version {n}, required = 3")`.
  - Drop the legacy `version < 2` rejection branch (folded into
    the new exact-match check).
- `rust/tensogram/src/types.rs`:
  - `GlobalMetadata::default()` emits `version: 3`.
- `rust/tensogram/src/framing.rs`:
  - All places that hard-code `version: 2` in the test fixtures
    switch to `3`.  Tests that assert `version == 2` update.
- `rust/tensogram-wasm`, `rust/tensogram-ffi`, `rust/tensogram-cli`:
  - No code changes — they read the version from the preamble
    struct.

**Test updates.**

- `rust/tensogram/tests/adversarial.rs`: add
  `message_with_v2_preamble_is_rejected`.
- Existing tests that encode a message and decode assert
  `version == 3`.
- Golden fixtures (`rust/tensogram/tests/golden/*.tgm`) — do **not**
  regenerate yet; they stay broken until Phase 10.  Mark the
  `golden_files.rs` test with `#[ignore = "regenerated in Phase 10"]`
  so CI stays green.  Remove the `#[ignore]` in Phase 10.

**Acceptance.**

- `cargo test -p tensogram` passes with the golden tests ignored.
- A manually-constructed v2 preamble (`version = 2`) fails to
  decode with the new exact-match error.
- `cargo clippy` clean.

**Dependencies.** Phase 0 sign-off.

**Post-conditions.** The wire is now tagged v3 but otherwise
unchanged.  Phases 2–7 all build on this.

---

## Phase 2 — Postamble extension (16 → 24 B)

**Goal.** Add the mirrored `total_length` field to the postamble so
backward scan becomes possible.  Streaming encoders back-fill when
the sink is seekable; otherwise they leave it `0`.

**Scope (Rust core).**

- `rust/tensogram/src/wire.rs`:
  - `POSTAMBLE_SIZE: 16 → 24`.
  - `Postamble` struct gains `pub total_length: u64`.
  - `Postamble::read_from` reads 24 bytes: `first_footer_offset`
    at 0, `total_length` at 8, `END_MAGIC` at 16.  Returns
    `FramingError` if `END_MAGIC` is missing.
  - `Postamble::write_to` writes all three fields.
- `rust/tensogram/src/framing.rs`:
  - `assemble_message` writes the new 24-byte postamble with the
    real `total_length`.
  - `decode_message` reads the 24-byte postamble and optionally
    cross-checks `postamble.total_length` against
    `preamble.total_length` (both must agree if both non-zero;
    mismatch → `FramingError`).
- `rust/tensogram/src/streaming.rs`:
  - `StreamingEncoder` takes a new `seekable: bool` flag at
    construction (default `true` when the sink is `&mut File`,
    `false` for write-only sinks).
  - `finish()`:
    - Writes a 24-byte postamble with `total_length = 0` if not
      seekable.
    - If seekable: back-fills both the preamble's and postamble's
      `total_length` (ties into B4b — back-fill both).  The
      streaming encoder records the preamble offset at start; on
      finish it seeks to that offset, writes the real
      `total_length`, seeks forward past the just-written
      postamble, and is done.

**Test updates.**

- `rust/tensogram/src/wire.rs::tests::test_postamble_round_trip`:
  round-trip a `Postamble { first_footer_offset, total_length }`.
- `rust/tensogram/tests/integration.rs`:
  - Encode a buffered message; assert postamble's `total_length`
    equals the buffer length.
  - Encode streaming to a `Vec<u8>` wrapped in `Cursor` (seekable);
    assert both total_length slots are back-filled and equal.
  - Encode streaming to a pipe-like sink (non-seekable); assert
    both total_length slots are `0`.
- `rust/tensogram/tests/adversarial.rs`:
  - `postamble_total_length_mismatch_fails` — a hand-crafted
    message where preamble says 100 and postamble says 200 fails
    decode with `FramingError`.

**FFI / Python / TS / WASM / CLI.** No API changes; the size of
`Postamble` is an internal detail.  All language bindings continue
to call `decode_message` / `encode_message` unchanged.  The
`seekable` flag on `StreamingEncoder` does need to be plumbed:

- Python: `StreamingEncoder.__init__` gains `seekable: bool = True`.
- TS: `StreamingEncoder` constructor option `seekable?: boolean`.
- C FFI: `tgm_streaming_encoder_create` gains `bool seekable` arg.
- C++ wrapper: `streaming_encoder::options.seekable`.
- CLI: `--no-seekable-output` flag when targeting stdout pipes
  (default is seekable-to-file).

**Acceptance.**

- Round-trip on every codec × buffered/streaming × seekable/non.
- Backward read from postamble locates `TENSOGRM` correctly for
  multi-message files.

**Dependencies.** Phase 1.

---

## Phase 3 — Bidirectional `scan_file` / `scan`

**Goal.** Exploit the new postamble `total_length` to halve the
hop count on cold-open scans of multi-message files.

**Scope (Rust core).**

- `rust/tensogram/src/framing.rs`:
  - Rewrite `scan` and `scan_file` to use a pair of walkers:
    - Forward walker: starts at offset 0, uses
      `preamble.total_length` per message.
    - Backward walker: starts at EOF, uses
      `postamble.total_length` per message.
    - Meet in the middle; merge the two result lists.
  - Add `ScanOptions { bidirectional: bool, max_message_size: u64 }`
    with sensible defaults (`bidirectional = true`,
    `max_message_size = 4 GiB`).
  - When either walker hits a `total_length = 0`
    (streaming-non-seekable message), the backward walker yields
    and the forward walker completes the scan alone.
- `rust/tensogram/src/remote.rs`:
  - `scan_remote_http` gains bidirectional support via a single
    additional HTTP range request (the last 24 bytes to read the
    postamble + the max-size backward search).  Gated by
    `ScanOptions.bidirectional` — default on for HTTP (where RTTs
    dominate), off for local files (where seq-scan is already
    cheap).
- `rust/tensogram/src/file.rs`:
  - `TensogramFile` uses the new `scan_file` on first access.  The
    bidirectional path cuts first-`message_count()` latency on
    large multi-message files.

**Test updates.**

- `rust/tensogram/tests/integration.rs`:
  - Build a 10-message file; verify `scan_file` (bidir) and `scan`
    (bidir) return the same `(offset, length)` list as the old
    forward-only path.
  - Build a file where message 5 has `total_length = 0`
    (non-seekable streaming); verify bidir scan falls back to
    forward-only with a single tracing event.
- `rust/tensogram/tests/remote_http.rs`: bidir scan against a mock
  HTTP server; assert a 2× reduction in HTTP calls vs the current
  forward-only path.

**FFI / Python / TS / WASM / CLI.** No API surface change —
`scan` / `scan_file` keep the same signatures; new option struct is
additive.  Language bindings pick up the speed-up for free.

**Acceptance.**

- Scans of 100-message files complete in half the hops.
- Streaming messages still scanned correctly under fallback.

**Dependencies.** Phase 2.

---

## Phase 4 — Remove obsolete type 4, rename type 9 → `NTensorFrame`

**Goal.** Consolidate the body phase on a single canonical
data-object frame type (for now), free up the code path from the
two-variant check that survived from the bitmask-frame work, and
name the concrete v3 frame cleanly (`NTensorFrame`, no version
suffix — future data-object types will slot in at fresh type
numbers rather than by bumping the name).

**Scope (Rust core).**

- `rust/tensogram/src/wire.rs`:
  - Remove the old `FrameType::NTensorFrame = 4` variant entirely.
  - Rename `FrameType::NTensorMaskedFrame = 9 → FrameType::NTensorFrame = 9`.
  - `FrameType::from_u16(4)` returns
    `FramingError("reserved frame type 4 (obsolete v2 NTensorFrame) not supported in v3")`.
  - `FrameType::is_data_object` simplifies to
    `matches!(self, FrameType::NTensorFrame)` — structured to
    accept additional data-object variants if/when they appear
    without changing the match ergonomics (document the future-
    extension intent in the doc comment).
- `rust/tensogram/src/framing.rs`:
  - `encode_data_object_frame` emits type 9 (`NTensorFrame`).
  - `decode_data_object_frame` accepts only type 9; type 4 is an
    error.
  - The "type 4 = type 9 with no masks" compatibility path is
    removed.
- `rust/tensogram/src/types.rs`:
  - Doc comments on `DataObjectDescriptor` reference
    `NTensorFrame` and explain that it is the concrete v3
    instantiation of the generic data-object concept.
- `rust/tensogram-ffi`, `rust/tensogram-wasm`, `rust/tensogram-cli`:
  - Internal references to `NTensorMaskedFrame` rename to
    `NTensorFrame`.  No public C API change — frame type is not
    directly exposed.

**Test updates.**

- `rust/tensogram/src/wire.rs::tests::test_frame_type_parse`:
  - `from_u16(4)` now errors.
  - `from_u16(9)` returns `NTensorFrame`.
- `rust/tensogram/tests/adversarial.rs`:
  - `frame_type_4_is_rejected` — a hand-constructed type-4 frame
    fails decode.
- Any test that explicitly references `NTensorMaskedFrame`
  updates to `NTensorFrame`.

**FFI / Python / TS / WASM / CLI.** No observable API change.  This
is a naming cleanup.

**Acceptance.** No references to `NTensorMaskedFrame` remain in
the codebase.  No references to the old type-4 `NTensorFrame`
accept-path remain.

**Dependencies.** Phase 1.

---

## Phase 5 — Inline hash slot + hash scope change + optional hashing

**Goal.** Ship the core integrity change: every frame gets an
inline 8-byte hash slot at `frame_end − 12`, the hash covers only
the *frame body* (strictly between header and type-specific
footer), and hashing becomes a message-level opt-in via
`HASHES_PRESENT`.  This is the biggest single phase.

**Scope (Rust core).**

- `rust/tensogram/src/wire.rs`:
  - Add `MessageFlags::HASHES_PRESENT = 1 << 7`.
  - `FrameHeader` unchanged (hash slot is in the footer, not the
    header).
  - Add `FRAME_HASH_TAIL_SIZE = 12` = `[hash u64 + ENDF]`; this
    is the **common tail** at the end of every frame's footer.
  - Rename `DATA_OBJECT_FOOTER_SIZE` semantic to reflect v3:
    the full data-object footer is
    `FRAME_HASH_TAIL_SIZE (12) + cbor_offset (8) = 20 B`.  Either
    keep the name `DATA_OBJECT_FOOTER_SIZE` with the new value
    (20) or split it into two constants; pick whichever reads
    more cleanly when touched.
  - Add a helper `footer_size_for(ft: FrameType) -> usize`:
    returns 20 for `NTensorFrame`, 12 for everything else.
    This is the single function the hash code consults.
- `rust/tensogram/src/types.rs`:
  - **Remove** `HashDescriptor` struct entirely.
  - **Remove** `DataObjectDescriptor.hash: Option<HashDescriptor>`.
  - Drop the `#[serde(skip_serializing_if = ...)]` wrapper.
- `rust/tensogram/src/hash.rs`:
  - Add `hash_frame_body(frame_bytes: &[u8], frame_type: FrameType) -> u64`
    that computes xxh3-64 over
    `frame_bytes[16 .. frame_bytes.len() - footer_size_for(ft)]`.
    Single source of truth for the scope definition.
  - Add `verify_frame_hash(frame_bytes: &[u8], ft: FrameType) -> Result<()>`
    used by validate.
- `rust/tensogram/src/framing.rs`:
  - `write_frame` (for non-data-object frames): after writing the
    payload, reserves 8 bytes for the hash slot, writes ENDF,
    then back-fills the hash (if `HASHES_PRESENT`) by hashing
    `bytes[16 .. end - 12)`.  Zeros when flag is off.
  - `encode_data_object_frame`: writes payload + masks + CBOR,
    then the 8-byte cbor_offset, then reserves the hash slot,
    then ENDF.  Hash back-fill hashes
    `bytes[16 .. end - 20)` — i.e. payload + masks + CBOR only;
    the cbor_offset is footer and is **not** in scope.
  - `decode_data_object_frame`: reads hash slot at
    `frame_end − 12`, reads cbor_offset at `frame_end − 20`.
    Validates `cbor_offset ∈ [16, total_length − 20]`.
- `rust/tensogram/src/encode.rs`:
  - `EncodeOptions.hash_algorithm: Option<HashAlgorithm>`
    (already there).  When `Some(Xxh3)`, sets `HASHES_PRESENT`
    in the preamble and populates every frame's hash slot.
    When `None`, clears the flag and writes zero slots.
  - Remove `emit_hashes` / `hash_scope` / any leftover options
    relating to per-descriptor hash.
  - Remove the descriptor `hash` population step.
- `rust/tensogram/src/streaming.rs`:
  - `StreamingEncoder.write_object` computes the inline hash as
    part of the single-pass framing (continues the
    hash-while-encoding pattern documented in `plans/DONE.md`).
  - Non-data-object frames the streaming encoder writes
    (`HeaderMetadata` at start, footer frames at `finish()`) get
    their hashes computed inline too, with the 12-byte footer
    scope.
- `rust/tensogram/src/validate/`:
  - `integrity.rs`: switch to the inline slot using
    `footer_size_for()` for scope lookup.  When the preamble
    says `HASHES_PRESENT = 0`, emit
    `ValidationCode::InlineHashesAbsent` at *warning* level on
    the default `--quick` path and at *error* level on
    `--checksum` (user explicitly asked for checksum verification).
  - Add the fast scan-checksum path: walk frame-by-frame via
    `total_length`, compute `xxh3(bytes[16 .. end - footer_size))`,
    compare to the u64 at `end - 12`.  No CBOR parsing on the
    happy path.

**CBOR schema change.**

- `DataObjectDescriptor` no longer has a `hash` key.  Golden
  fixtures regenerated in Phase 10 will not contain it.  Any test
  that introspects CBOR output updates.

**Test updates.**

- `rust/tensogram/tests/integration.rs`:
  - Encode with `hash_algorithm = Some(Xxh3)`, decode, re-hash
    every frame using `footer_size_for()` scope, assert inline
    slots match.
  - Encode with `hash_algorithm = None`, assert every frame's
    inline slot is `0x00…00`.
- `rust/tensogram/tests/adversarial.rs`:
  - `frame_hash_mismatch_detected` — flip a byte in a
    data-object frame's CBOR descriptor (inside the hash scope);
    decode succeeds (CBOR parses fine) but `validate --checksum`
    flags the hash mismatch with `HashMismatch { computed, stored }`.
  - `frame_hash_stable_under_cbor_offset_change` — overwrite
    the `cbor_offset` bytes (which are in the footer, outside
    the hash scope).  The re-hashed slot is unchanged — the
    test pins the hash-scope rule "cbor_offset is NOT hashed".
    Decode will still fail (cbor_offset now points to an
    invalid location) but that error path is `FramingError`,
    *not* `HashMismatch`.
  - `frame_hash_excludes_endf_and_slot` — overwrite the hash
    slot itself to a random value; the computed hash from the
    scope bytes is still stable (only the stored value drifted).
    Validator flags `HashMismatch` but the scope bytes are
    untouched — exercises scope isolation.
- `rust/tensogram-encodings/tests/hash_while_encoding.rs`:
  - Streaming and buffered hashes agree for every codec (existing
    test, updated for new hash scope).

**FFI / Python / TS / WASM / CLI.** Mechanical changes:

- Python: `encode(hash_algorithm="xxh3" | None)` — already present.
- TS: `encode(opts)` already has `hash` option.
- C FFI: `tgm_encode(hash_algorithm, ...)` — already has it.
- C++ wrapper: `encode_options.hash_algorithm` — already has it.
- Per-language: remove any `hash` field on the per-descriptor Python
  / TS / C struct surface (C6a: remove from CBOR).

**Acceptance.**

- All buffered encode → decode → re-verify round-trips pass.
- `tensogram validate --checksum <file>` walks every frame and
  verifies the inline slot without CBOR parsing.
- On a message with `HASHES_PRESENT = 0`, `validate --checksum`
  fails with `InlineHashesAbsent`.

**Dependencies.** Phase 1.  Parallelisable with 2, 3, 4.

---

## Phase 6 — Header / Footer hash frame auto-population + schema update

**Goal.** Drive the message-level `HeaderHash` / `FooterHash`
frames off the inline slots.  Rename the CBOR key `hash_type` →
`algorithm`.  Drop redundant `object_count`.

**Scope (Rust core).**

- `rust/tensogram/src/types.rs`:
  - `HashFrame` renames `hash_type: String` → `algorithm: String`.
  - Drop `HashFrame.object_count` (derived from `hashes.len()`).
  - `IndexFrame` likewise: drop `object_count` (derived from
    `offsets.len()`).
- `rust/tensogram/src/metadata.rs`:
  - `hash_frame_to_cbor` emits the new key.
  - `cbor_to_hash_frame` accepts the new key; reading the old
    `hash_type` is a `MetadataError` (no backwards read).
  - Same for `index_to_cbor` / `cbor_to_index` — drop the
    `object_count` key.
- `rust/tensogram/src/encode.rs`:
  - Remove old `emit_hashes` option (if still present from v2
    cleanup).
  - Add `EncodeOptions.create_header_hashes: bool` and
    `create_footer_hashes: bool`.
  - Buffered mode defaults: `create_header_hashes = true`,
    `create_footer_hashes = false`.
  - Validate at construction: `create_header_hashes = true` while
    in streaming mode is an `EncodingError::InvalidOption`.
- `rust/tensogram/src/streaming.rs`:
  - Defaults: `create_footer_hashes = true`.
  - At `finish()`, populates a `FooterHash` frame from the
    accumulated per-object inline hashes.
- `rust/tensogram/src/framing.rs`:
  - `build_hash_frame_cbor` reads hashes from each object's
    inline slot rather than `descriptor.hash` (which no longer
    exists).

**Test updates.**

- Hash-frame round trip uses `algorithm: "xxh3"` and the new
  shape.
- Cross-check between inline slot and hash frame hex entry:
  `hex(u64_slot) == hash_frame.hashes[i]`.
- Streaming + `create_header_hashes = true` fails at encoder
  construction.

**FFI / Python / TS / WASM / CLI.** Match the Rust option surface:

- Python: `encode(create_header_hashes=True, create_footer_hashes=False)`,
  same for `StreamingEncoder`.
- TS: `EncodeOptions.createHeaderHashes`, `createFooterHashes`.
- C FFI: extended options struct; cbindgen re-generates header.
- C++ wrapper: fields on `encode_options`.
- CLI: `--create-header-hashes` / `--no-create-header-hashes` on
  every encoding-capable subcommand (`merge`, `split`, `reshuffle`,
  `convert-grib`, `convert-netcdf`).

**Acceptance.**

- Buffered encode → `HeaderHash` frame present, content matches
  inline slots.
- Streaming encode → `FooterHash` frame present, content matches.
- Buffered encode with both flags true emits both frames with
  identical CBOR.

**Dependencies.** Phase 5.

---

## Phase 7 — `rle` + `roaring` as compression codecs

**Goal.** Promote the two bitmask-specific mask codecs from
`tensogram-encodings/src/bitmask/` into first-class compression
methods selectable via `DataObjectDescriptor.compression`.

**Scope (Rust encodings).**

- `rust/tensogram-encodings/src/compression/`:
  - New module `rle.rs` exposing a `RleCompressor: Compressor`
    that wraps the existing `bitmask/rle.rs` codec.
  - New module `roaring.rs` for `RoaringCompressor: Compressor`.
  - Both compressors' `decompress_range` return
    `CompressionError::RangeNotSupported`.
- `rust/tensogram-encodings/src/pipeline.rs`:
  - `build_compressor(desc: &DataObjectDescriptor)`:
    - When `desc.compression ∈ {"rle", "roaring"}`, require
      `desc.dtype == Dtype::Bitmask`; otherwise return
      `EncodingError::IncompatibleDtype { codec, dtype }`.
  - Pipeline dispatch for these codecs is otherwise the same as
    `zstd` / `lz4` (no block offsets, no shuffle filter
    interaction).

**Test updates.**

- `rust/tensogram-encodings/tests/compression_rle.rs`:
  round-trip random bitmasks at various lengths.
- `rust/tensogram-encodings/tests/compression_roaring.rs`: ditto.
- `rust/tensogram-encodings/tests/compression_dtype_guard.rs`:
  attempt `compression = "rle"` with `dtype = "float32"` →
  `IncompatibleDtype` error.
- Integration: full encode → decode through `tensogram::encode` for
  a `bitmask` dtype message with `compression = "rle"` and
  `compression = "roaring"`.

**FFI / Python / TS / WASM / CLI.** Mechanical:

- Python: string value `compression="rle"` / `"roaring"` accepted.
- TS: `compression: "rle" | "roaring"` accepted.
- C FFI: same — `compression` is a `char*` already.
- C++ wrapper: same.
- CLI: `--compression rle` / `roaring` on encoding-capable
  subcommands.  `clap::PossibleValuesParser` updated to include
  the new values so mistyped codec names fail early with
  "did-you-mean" suggestions.
- WASM: `tensogram-wasm` Cargo features already include pure-Rust
  bitmask codecs (verified from `bitmask/mod.rs` — roaring and
  rle are both pure Rust).  Add them to the WASM feature set
  gate-check in `rust/tensogram-wasm/Cargo.toml`.

**Acceptance.**

- Round-trip a 1 M-element bitmask with each of `rle` and
  `roaring` under every language binding.
- Compression failure tests fire with the correct error.

**Dependencies.** Phase 1 (only needs v3 tagging).

---

## Phase 8 — Cross-language parity

**Goal.** Bring every language binding's test matrix onto v3.  No
new features; just making sure every binding produces / consumes
the new wire format correctly.

**Scope.**

- **Python (PyO3)**: update all encode / decode / validate tests.
  Add tests for `create_header_hashes` / `create_footer_hashes`
  options.  Verify `compression ∈ {"rle", "roaring"}` with
  `dtype = "bitmask"`.  Update `python/tests/test_validate.py` for
  the new `InlineHashesAbsent` code.
- **TypeScript (WASM)**: update `typescript/tests/*`.  Exercise
  backward-scan via `TensogramFile.fromUrl` (Range backend already
  in place).  Add tests for the new codecs.
- **C FFI + C++ wrapper**: update `cpp/tests/*` — GoogleTest suite
  for hash-slot verification, new codecs, postamble `total_length`
  parity.
- **Golden-file cross-check**: add a TS/Python/C++ test that
  decodes a Rust-produced v3 golden fixture and verifies:
  - preamble `version == 3`,
  - postamble `total_length` matches file size,
  - every frame's inline hash slot agrees with
    `xxh3(frame[16..end-12))`,
  - `HashFrame.algorithm == "xxh3"`.

**Dependencies.** All of Phases 2–7.

**Acceptance.** Full-matrix CI green on all language-specific
suites.

---

## Phase 9 — Documentation polish

**Goal.** Every user-facing doc page reflects v3.

**Scope.**

- `docs/src/format/wire-format.md` — rewritten from the new
  `plans/WIRE_FORMAT.md`.
- `docs/src/format/cbor-metadata.md` — updated schemas for
  `HashFrame` (algorithm rename), `IndexFrame` (object_count
  drop), `DataObjectDescriptor` (hash removal).
- `docs/src/cli/validate.md` — `--checksum` now means
  "fast inline-hash scan"; document the `InlineHashesAbsent` error
  and how to regenerate with hashing enabled.
- `docs/src/guide/encoding.md` — `create_header_hashes`,
  `create_footer_hashes`.
- `docs/src/encodings/compression.md` — section on the two
  dtype-restricted codecs (`rle`, `roaring`).
- `docs/src/guide/decoding.md` — bidirectional scan mention.
- `docs/src/guide/remote-access.md` — note the one-request-saved
  win from the new postamble `total_length`.
- `docs/src/internals.md` — updated frame footer layout diagram.
- `plans/DONE.md` — new "Wire Format v3" section.
- `plans/IDEAS.md` — close the "hash-in-frame-footer" idea (it
  shipped).
- `CHANGELOG.md` — consolidated `[Unreleased] → 0.17.0` entry
  covering all of Phases 1–7.

**Dependencies.** Phases 1–8.

**Acceptance.** `mdbook build docs/` clean; CI `docs` job green.

---

## Phase 10 — Golden fixture regeneration + release prep

**Goal.** Re-baseline every byte-level test fixture and cut the
0.17.0 release.

**Scope.**

- `rust/tensogram/tests/golden/` — regenerate all 5 (or
  however many are shipped) `.tgm` files from v3 encoders.
  Remove the `#[ignore]` added in Phase 1.
- Cross-language golden tests re-record:
  - `typescript/tests/golden.test.ts`
  - `python/tests/test_golden.py`
  - `cpp/tests/test_golden.cpp`
- Run `make-release 0.17.0` — the existing command handles
  version bumps across all manifests.
- Release-preflight workflow runs and must be green.
- `cargo publish --dry-run` for the publishable crates (excluded
  for the grib/netcdf/wasm crates per their workspace-exclusion).

**Dependencies.** All previous phases.

**Acceptance.**

- `cargo test --workspace` green with golden tests re-enabled.
- `pytest`, `vitest`, `ctest` all green.
- `mdbook build docs/` clean.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean.

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Hash scope accidentally includes footer bytes** (`cbor_offset`, hash slot itself, ENDF) — a single off-by-one in `hash_frame_body` would silently change wire output and poison every fixture | High | Single source of truth in `wire.rs::footer_size_for()`; three explicit tests in Phase 5 (`frame_hash_mismatch_detected`, `frame_hash_stable_under_cbor_offset_change`, `frame_hash_excludes_endf_and_slot`) pin the scope rule from every direction. |
| **Streaming encoder forgets to back-fill `total_length` on seekable sinks** | Medium | Integration test in Phase 2 encodes to a `Cursor<Vec<u8>>`, verifies both slots are non-zero post-finish. |
| **Golden fixture drift between Rust and TS/C++ producers** | Medium | Phase 8 cross-language golden-file tests use the **Rust** output as source of truth; every other binding verifies byte-identical decode of Rust-produced files.  Writers that can't round-trip byte-identical are caught early. |
| **Streaming non-seekable sinks writing `total_length = 0`** confuses readers | Low | The fallback path (forward scan) is documented in `WIRE_FORMAT.md` §9 and exercised in tests.  No older readers exist — v3 is a clean break. |
| **`--checksum` semantic change** breaks existing scripts that expect the old hash-descriptor check | Low | Documented in CHANGELOG under BREAKING.  Old behaviour replaced by the faster inline path, which is strictly more informative (detects corruption across the whole frame body, not just the encoded payload). |
| **Bitmask codec dtype guard surfaces an error the user doesn't expect** | Low | Error text explicitly names the codec and the incompatible dtype; mirrored in the docs page on compression. |
| **Future non-tensor data-object type bumps `footer_size_for()` default** | Low | `footer_size_for` is the pinch point; adding a new data-object type means extending the match arm there plus the §2.2 footer-sizes table in `WIRE_FORMAT.md`.  Documented in the type-4-reserved comment and the generic-data-object note in §1. |

---

## Work allocation estimate

| Phase | Scope | Rough LOC | Estimated session count |
|-------|-------|-----------|-----------------------|
| 0 | Spec + sign-off | 0 (docs only) | Gate |
| 1 | Version bump | 50 | 0.5 |
| 2 | Postamble ext. | 300 | 1 |
| 3 | Bidir scan | 250 | 1 |
| 4 | Type 4 removal | 150 | 0.5 |
| 5 | Inline hash slot | 600 | 2 |
| 6 | Hash frame auto | 200 | 1 |
| 7 | Bitmask codecs | 400 | 1 |
| 8 | Cross-language | 800 | 2 |
| 9 | Docs | 400 | 1 |
| 10 | Goldens + release | 150 | 0.5 |
| **Total** | | **~3.3 k LOC** | **~10 sessions** |

Session counts assume ~300 LOC + matching tests per session, with
full green-quad verification at each boundary.  Parallelisable
phases can reduce wall time if multiple agents collaborate.

---

## Open follow-ups (not blocking 0.17.0)

- Python `compute_common` surface — currently Rust + TS only.
  Mechanical, but out of scope here (tracked in `plans/TODO.md`).
- Backward-scan support in the remote `object_store` backend.
  Gated behind `ScanOptions.bidirectional` so it's safe to add
  incrementally.
- `tensogram validate --scan-checksum` as a separate flag to keep
  the existing `--checksum` semantics for users who want it.
  Current plan repurposes `--checksum`; if that breaks too many
  users in practice, split the flag.
- Hash algorithm registry (`"xxh3-128"`, `"blake3"`): reserve
  space in the inline slot for variable-length digests via a new
  frame-header flag; deferred until a concrete need arises.
- Python / TypeScript / C++ wrapper test-suite runs.  The Rust
  surface they wrap is v3-complete (phases 5–7 cascaded through
  the bindings at the source level); the bindings' own test
  suites require local `pytest` / `npm` / `cmake` environments
  that were not available in the implementation session.  CI
  picks these up automatically on the next run.

---

## Implementation status (2026-04-20)

| Phase | Status | Notes |
|------:|:------:|-------|
| 0 — spec sign-off              | ✅ | Plans committed; maintainer sign-off recorded in the "Design decisions" block above. |
| 1 — version bump               | ✅ | `WIRE_VERSION = 3`; v1/v2 preambles hard-fail. |
| 2 — postamble 24 B             | ✅ | Mirrored `total_length`; `finish_with_backfill()` on seekable `StreamingEncoder`. |
| 3 — bidirectional scan         | ✅ | `ScanOptions { bidirectional, max_message_size }`; default on.  Pass 3 wired `max_message_size` into both in-memory and file-based walkers. |
| 4 — type 4 removal             | ✅ | Obsolete `NTensorFrame` (type 4) reserved; `NTensorMaskedFrame` (type 9) renamed to `NTensorFrame`. |
| 5 — inline hash slot           | ✅ | `[hash u64][ENDF]` common tail on every frame; hash scope = body only; `DataObjectDescriptor.hash` removed. |
| 6 — HashFrame auto-populate    | ✅ | `HashFrame { algorithm, hashes }` (renamed from `hash_type`); `IndexFrame { offsets, lengths }` (no `object_count`); `create_header_hashes` / `create_footer_hashes` on `EncodeOptions`. |
| 7 — rle / roaring codecs       | ✅ | First-class compression codecs; bitmask-dtype guard at pipeline-build time; pure-Rust (no feature gate). |
| 8 — cross-language parity      | ⚠️ partial | Rust workspace (lib + cli + ffi + wasm) v3-complete and all tests pass.  Python / TypeScript / C++ wrapper suites pick up changes via source inheritance; dev-env test runs deferred to CI.  FFI tests re-enabled in pass 2 after adding `extract_inline_hashes` to surface the inline slot through `tgm_payload_has_hash` / `tgm_object_hash_*`. |
| 9 — documentation              | ✅ | `CHANGELOG.md` `[Unreleased]` section covers every wire change; `plans/WIRE_FORMAT.md` is the v3 canonical spec. |
| 10 — golden fixtures + release | ✅ | Five `.tgm` fixtures regenerated; `test_golden_*` re-enabled and passing. |

**Pass 2 / Pass 3 addenda.** A follow-up review pass reconciled
all ten v3-related `#[ignore]` tests (rewriting 7 against the v3
design and collapsing 3 that had become structurally impossible),
added the missing `max_message_size` plumbing in both scan
walkers, replaced `try_into().unwrap()` patterns in the bitmask
compressors with panic-free `split_first_chunk`, and tightened
error messages in `hash_frame_body` / `verify_frame_hash` with
size values and remediation hints.
