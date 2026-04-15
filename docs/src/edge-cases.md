# Edge Cases

A collection of non-obvious situations and how the library handles them.

## Corrupted Messages

**What happens:** The scanner (`scan()`) searches for `TENSOGRM` magic bytes and validates the postamble (last 8 bytes should be `39277777`). If `total_length` is set, the scanner checks for the end magic at the expected position.

**Recovery:** If a message fails validation, the scanner skips one byte and resumes searching. A single corrupted message in a multi-message file does not prevent reading the others.

```rust
let offsets = scan(&file_bytes);
// offsets only contains valid (start, length) pairs
// Corrupted regions are silently skipped
```

**Edge case within edge case:** If a random byte sequence inside a valid payload happens to match `TENSOGRM`, the scanner might try to parse a "message" starting mid-payload. The postamble cross-check catches this: the false start's postamble won't contain the expected `39277777` end magic.

## NaN in Simple Packing

Simple packing cannot represent NaN. The quantization formula maps the range `[min, max]` onto integers, and NaN has no defined place in this range.

**What happens:** `compute_params()` returns `PackingError::NanValue(index)` if any value is NaN. The `encode()` function also rejects NaN inputs before packing.

**Solution:** Replace NaN values with a sentinel (e.g. the minimum representable value, or a separate bitmask object) before encoding.

## Decode Range on Compressed Data

`decode_range()` supports partial range decode for compressors that have random access capability: szip (via RSI block offsets), blosc2 (via chunk-based access), and zfp fixed-rate mode. Stream compressors (zstd, lz4, sz3) return `CompressionError::RangeNotSupported`.

**Workaround for stream compressors:** Decode the full object with `decode_object()` and slice the result in memory.

## Bitmask Byte Width

`Dtype::Bitmask` returns `0` from `byte_width()`. This is a sentinel, not a real byte width.

**Why:** A bitmask of N elements occupies `ceil(N / 8)` bytes. The library cannot infer N from the byte width alone, so the "element size" concept doesn't apply. Callers that need the payload size must compute it from the element count.

```rust
let num_elements: u64 = descriptor.shape.iter().product();
let payload_bytes = if descriptor.dtype == Dtype::Bitmask {
    let n = usize::try_from(num_elements)?;
    (n + 7) / 8
} else {
    let n = usize::try_from(num_elements)?;
    n * descriptor.dtype.byte_width()
};
```

## verify_hash on Messages Without Hashes

If a message was encoded with `hash_algorithm: None` (no hash), and you decode it with `verify_hash: true`, the decoder silently skips hash verification for that object. No error is returned.

**Rationale:** The absence of a hash is not an error. The decoder cannot verify what was never stored. If you need to enforce that all messages have hashes, check `descriptor.hash.is_some()` after decoding.

## Constant-Value Fields with simple_packing

If all values in a field are identical (range = 0), `compute_params()` sets `binary_scale_factor` such that all packed integers are 0, and the full value is recovered from `reference_value` alone. This is correct and handled without special cases.

## Very Short Buffers

Passing a buffer shorter than the preamble size (24 bytes) to any decode function returns `TensogramError::Framing("buffer too short ...")`. No panic.

## Object Index Out of Range

`decode_object(&message, 99, &options)` when the message has fewer than 100 objects returns `TensogramError::Object("object index N out of range")`.

## Empty Files

`TensogramFile::message_count()` returns `0`. `read_message(0)` returns an error.

## CBOR Key Ordering

The library uses canonical CBOR key ordering (RFC 8949 §4.2). If you construct a `GlobalMetadata` struct with keys in one order and then check the CBOR bytes, the bytes may not match your insertion order. This is intentional and correct — it ensures deterministic output.

If you need to compare metadata across languages or implementations, always compare the decoded values, not the raw CBOR bytes from different encoders.

You can verify that any CBOR output is canonical using the `verify_canonical_cbor()` utility:

```rust
use tensogram_core::verify_canonical_cbor;

let cbor_bytes = /* ... */;
verify_canonical_cbor(&cbor_bytes)?; // Returns Ok(()) if canonical, Err if not
```

## Frame Ordering Violations

The decoder validates that frames appear in the expected order: header frames first, then data object frames, then footer frames. A message with frames out of order (e.g. a header metadata frame appearing after a data object frame) is rejected with `TensogramError::Framing`.

This catches malformed or tampered messages. Valid messages produced by the encoder always have correct ordering.

## Streaming Mode (total_length = 0)

When encoding for a non-seekable output (e.g. TCP socket), the preamble's `total_length` is set to 0. In this mode:

- Header index and header hash frames are omitted (the encoder doesn't know the data object count or offsets upfront).
- The footer must contain at least the metadata frame.
- The `first_footer_offset` in the postamble points to the first footer frame.

Decoders that encounter `total_length = 0` should read from the postamble backward to find the footer frames, then use the footer index (if present) for random access to data objects.

## first_footer_offset is Never Zero

The postamble's `first_footer_offset` field always points to a valid position:
- If footer frames exist: it points to the start of the first footer frame.
- If no footer frames exist: it points to the start of the postamble itself.

This invariant means decoders can always seek to `first_footer_offset` and determine whether they've landed on a footer frame or the postamble.

## Inter-Frame Padding

The encoder may insert padding bytes between frames for memory alignment (e.g. 64-bit alignment). Padding appears between the `ENDF` marker of one frame and the `FR` marker of the next. Decoders should scan for the `FR` marker rather than assuming frames are contiguous.

## Zero-Element Tensors

Shapes containing zero dimensions are valid: `shape: [0]`, `shape: [3, 0, 5]`. This matches numpy and PyTorch semantics where zero-element tensors are legitimate objects (e.g. an empty batch). The encoded payload for a zero-element tensor is zero bytes.

## Scalar Tensors

`shape: []` (empty shape, `ndim: 0`) represents a scalar tensor containing exactly one element. The payload size equals `dtype.byte_width()` bytes.

## Metadata-Only Messages

A message with zero data objects is valid. This can be used to transmit metadata without any tensor data (e.g. coordination signals, timestamps, provenance records). Both `encode()` with an empty descriptors slice and `StreamingEncoder` with no `write_object()` calls produce valid messages.

## Mixed Dtypes in One Message

Multiple data objects in the same message may have different dtypes. For example, a `Float32` tensor paired with a `Bitmask` object used as a missing-data mask. Each object's pipeline (encoding, filter, compression) is configured independently.

## Bitmask with Encoding/Compression

Bitmask data is internally packed into `uint8` bytes. Any encoding or compression pipeline that supports `uint8` should work with bitmask data. The total bit count must be stored separately (in the shape) since the byte count `ceil(N / 8)` may not equal `N` exactly.

## Strides Validation

Strides are validated for length: `strides.len()` must match `shape.len()`. Non-contiguous strides (e.g. `shape: [4, 4], strides: [8, 1]`) are accepted — they indicate a view into a larger array and are semantically valid.

## Version Constraints

- `version: 0` and `version: 1` are deprecated and must be rejected by the decoder.
- `version: 2` is the current version.
- Higher versions (3+) are reserved for future use and will be valid once defined.

## NaN/Infinity in Simple Packing Parameters

If `reference_value` is NaN or Infinity, encoding fails immediately with a clear error. This value is used in the quantization formula and would produce corrupt output. (`binary_scale_factor` and `decimal_scale_factor` are integers and cannot be NaN/Infinity.)

## Duplicate CBOR Keys

Duplicate keys at the same level in a CBOR map are never accepted. The library uses canonical CBOR (RFC 8949 §4.2) which inherently rejects duplicate keys. Same-name keys at different nesting levels are acceptable: `base[0]["foo"]` and `_extra_["foo"]` are distinct keys.

## Unknown Hash Algorithm on Decode

If a message contains a hash with an algorithm the decoder doesn't recognize (e.g. `"sha256"` when only `xxh3` is implemented), `verify_hash: true` issues a warning and skips verification rather than returning an error. This ensures forward compatibility when new hash algorithms are added.

## decode_range with Empty Ranges

Calling `decode_range()` with an empty `ranges` slice (`&[]`) returns `(descriptor, vec![])` — the parts vector is empty. This is not an error.

## Preceder Metadata Error Paths

The decoder validates PrecederMetadata frames strictly:

| Condition | Error type | Message |
|-----------|-----------|---------|
| Consecutive preceders without DataObject | `Framing` | "PrecederMetadata must be followed by a DataObject frame, got {type}" |
| Dangling preceder (no DataObject follows) | `Framing` | "dangling PrecederMetadata: no DataObject frame followed" |
| Base has 0 or 2+ entries | `Metadata` | "PrecederMetadata base must have exactly 1 entry, got {n}" |
| Metadata base entries > data objects | `Metadata` | "metadata base has {n} entries but message contains {m} objects" |

On the encoder side:
- `StreamingEncoder::write_preceder()` errors if called twice without an intervening `write_object()`.
- `StreamingEncoder::finish()` errors if a preceder was written without a following `write_object()`.
- `encode()` (buffered mode) errors if `emit_preceders: true` — use `StreamingEncoder::write_preceder()` instead.

## File Concatenation

Tensogram is a message format, not a file format. Multiple `.tgm` files can be concatenated:

```bash
cat 1.tgm 2.tgm > all.tgm
```

The resulting file is valid. `scan()` and `TensogramFile` will find all messages from both source files.

## xarray Layer Edge Cases

### meta.base Out-of-Range

If a message has more data objects than `meta.base` entries (e.g. 3 objects but `base` has only 1 entry), the xarray layer logs a warning and treats the missing base entries as empty dicts. The objects are still decoded — they just have no per-object metadata attributes.

This can happen when a message is encoded with an incomplete `base` array, or when objects are appended to a message without updating `base`. The warning helps diagnose silent metadata loss:

```
WARNING: meta.base has 1 entries but object index 2 requested;
         per-object metadata will be empty for this object
```

### Empty or Missing base Attribute

A message with `base: []` or no `base` key at all is valid. All objects get empty per-object metadata and are named `object_0`, `object_1`, etc. The `_reserved_` key (auto-populated by the encoder in each base entry) is always filtered out — it never appears in user-facing variable attributes.

### Variable Naming with Dot Paths

When `variable_key="mars.param"` is used, the `resolve_variable_name()` function traverses the nested dict path. If any segment is missing, the function falls back to the generic `object_<index>` name. The `obj_index` used is the object's position in the message (not its position among data variables), so a file with objects 0 (coord), 1 (data), 2 (data) would produce names like `"object_1"` and `"object_2"` for the data variables.

### Coordinate Name Case Insensitivity

Coordinate detection (`detect_coords`) is case-insensitive: `"LATITUDE"`, `"Lat"`, and `"latitude"` all match the known coordinate name `"latitude"`. The canonical dimension name is always lowercase (e.g. `"latitude"`, not `"LATITUDE"`).

### Ambiguous Dimension Size Matching

When two coordinate arrays have the same size (e.g. latitude with 5 points and depth with 5 points), the dimension resolution assigns the first matching coord to the first axis that matches the size, and the second to the next axis. If the data variable is 2D [5, 5], one axis gets `"latitude"` and the other gets `"depth"`. When no coord has the matching size, the axis gets a generic `"dim_N"` name.

### Multi-Message Merge with Different Keys

When `open_datasets()` merges multiple messages, objects whose base entries have different key sets are handled as follows:
- Keys present in all objects with identical values become Dataset attributes (constant).
- Keys present in all objects with varying values become outer dimensions (if they form a hypercube) or separate variables.
- Keys present in some objects but not others are treated as varying with `None` for missing entries.

### _reserved_ Filtering Consistency

The `_reserved_` key is filtered at every access point:
- `TensogramDataStore._get_per_object_meta()` (store.py)
- `_base_entry_from_meta()` (scanner.py)
- `_filter_reserved()` (zarr store.py)

This ensures the encoder's auto-populated tensor info (ndim, shape, strides, dtype) never leaks into user-facing metadata.

## Zarr Layer Edge Cases

### Group Attributes from meta.extra

Group-level attributes in the root `zarr.json` come from `meta.extra` (message-level annotations). If `meta.extra` is empty or absent, the group `zarr.json` only contains internal attributes (`_tensogram_version`, `_tensogram_variables`).

### Per-Array Attributes from meta.base[i]

Per-array attributes come from `meta.base[i]` with the `_reserved_` key filtered out. Descriptor encoding params are stored under `_tensogram_params` to avoid namespace collisions.

### Variable Name Resolution — No Extra Fallback

Variable names are resolved exclusively from `per_object_meta` (from `meta.base[i]`). The `common_meta` (from `meta.extra`) is **not** searched for variable naming. This prevents all objects in a message from sharing the same name when a name key exists only at the message level.

This is consistent across both xarray and zarr layers.

### Zarr Metadata Key Collision

If a base entry has keys like `"zarr"`, `"chunks"`, or `"shape"`, they go into the Zarr array's `attributes` dict — not the top-level metadata. There is no collision with Zarr's own `shape`, `chunk_grid`, etc. fields.

### Write Path: _reserved_ Filtering

When writing through `TensogramStore`, user-set array attributes are written into `base[i]` entries. The `_reserved_` key is explicitly filtered from these entries to prevent collision with the encoder's auto-populated `_reserved_.tensor` info.

### Write Path: Group Attributes

Group attributes set via Zarr become unknown top-level keys in GlobalMetadata, which the encoder preserves as `_extra_`. On re-read, they appear in `meta.extra`. Internal keys (starting with `_tensogram_`) and reserved structural keys (`version`, `base`, `_extra_`, `_reserved_`) are excluded.

### Empty TGM File

A `.tgm` file with zero messages produces a root group `zarr.json` with no arrays. A message with zero data objects produces a root group with the message's extra metadata but no arrays.

### Variable Name Deduplication

When multiple objects resolve to the same name, suffixes `_1`, `_2`, etc. are appended. For example, three objects named `"x"` become `"x"`, `"x_1"`, `"x_2"`.

### Variable Name Sanitization

Slashes and backslashes in resolved variable names are replaced with underscores to prevent spurious directory nesting in the Zarr virtual key space. Empty names are replaced with `"_"`.

## GRIB Converter Edge Cases

### Single GRIB to base[0] Has ALL MARS Keys

In `OneToOne` mode, each GRIB message becomes one Tensogram message. All MARS namespace keys (plus `gridType` as `"grid"`) go into `base[0]["mars"]`. When `--all-keys` is enabled, non-MARS namespace keys (geography, time, vertical, parameter, statistics) go into `base[0]["grib"]`.

### MergeAll with N Fields

In `MergeAll` mode, N GRIB fields become one Tensogram message with N data objects. Each `base[i]` holds ALL metadata for that object independently — there is no common/varying partitioning at encode time. This means metadata keys are duplicated across base entries.

**Performance note:** With 1000 GRIB fields, this means 1000 copies of common keys (class, type, stream, expver, date, time, etc.). This is by design — the wire format prioritizes simplicity and independent object access over byte savings. Use `tensogram_core::compute_common()` at display/merge time to extract shared keys.

### Different Grid Types in MergeAll

GRIB fields with different grid types (e.g. `regular_ll` and `reduced_gg`) can be merged into the same Tensogram message. Each `base[i]["mars"]["grid"]` independently records its grid type. Downstream consumers (xarray, zarr) must handle the structural differences (e.g. different shapes).

### GRIB Shape from Ni/Nj

The shape is derived from ecCodes `Ni` and `Nj` keys (row-major: [Nj, Ni]). If either is zero or missing (e.g. reduced Gaussian grids), the shape falls back to `[numberOfPoints]` (1-D).

### Empty params in DataObjectDescriptor

GRIB-converted data objects have empty `desc.params` — all metadata lives in `base[i]["mars"]` and `base[i]["grib"]`, not in the per-object descriptor. This is by design: the descriptor carries only what's needed to decode the payload (shape, dtype, encoding pipeline).

## Metadata Model Edge Cases (base / _reserved_ / _extra_)

The v2 metadata model has three sections: `base` (per-object), `_reserved_` (library internals), and `_extra_` (client annotations). These create several non-obvious edge cases.

### _reserved_ is Protected

Client code **must not** set `_reserved_` in any context:
- Python: `tensogram.encode({"version": 2, "_reserved_": {...}})` raises `ValueError`.
- Python: `encode({"version": 2, "base": [{"_reserved_": {...}}]})` raises `ValueError`.
- FFI: JSON with `"base": [{"_reserved_": {...}}]` returns `TgmError::Metadata`.
- CLI: `set -s _reserved_.tensor.ndim=5` returns an error.

The encoder auto-populates `_reserved_.tensor` in each base entry (ndim, shape, strides, dtype) and `_reserved_` at the message level (encoder, time, uuid).

### Metadata Lookup Semantics (base first-match)

All lookup functions (`__getitem__` in Python, `tgm_metadata_get_string` in FFI, `lookup_key` in CLI) use first-match semantics:

1. Search `base[0]`, then `base[1]`, ..., skipping the `_reserved_` key within each entry.
2. If not found in any base entry, search `_extra_`.
3. If not found → `None` (FFI/CLI) or `KeyError` (Python).

**Implication:** If `base[0]` has `mars.param=2t` and `base[1]` has `mars.param=msl`, lookups return `"2t"` (the first match). This is message-level lookup, not per-object.

### _reserved_ is Hidden from Dict Access

- `meta["_reserved_"]` → `KeyError` (Python). The key is skipped during base entry iteration.
- `"_reserved_" in meta` → `False`.
- `tgm_metadata_get_string(meta, "_reserved_.tensor")` → `NULL` (FFI). The path is blocked.
- To read `_reserved_` data, use `meta.reserved` (Python) or read the base entry directly via `meta.base[i]["_reserved_"]`.

### Explicit _extra_ / extra Prefix

The CLI and FFI support explicit `_extra_.key` or `extra.key` prefixes to target the `_extra_` map directly, bypassing the base search:

```bash
# CLI: write to _extra_ map
tensogram set -s "extra.custom=value" input.tgm output.tgm
tensogram set -s "_extra_.custom=value" input.tgm output.tgm

# CLI: read from _extra_ map
tensogram get -p "_extra_.custom" input.tgm
```

Without the prefix, `set` writes to all base entries. With the prefix, it writes to `_extra_` specifically.

### Empty Key String

An empty key `""` returns `None` (FFI/CLI) or raises `KeyError` (Python). This is not an error — it simply finds no match.

### base vs Descriptor Count

The `base` array length should match the number of data objects. The encoder auto-extends base entries (adding `_reserved_.tensor`) for each object. If the user provides fewer base entries than objects, the encoder creates entries for the missing ones. If the user provides more base entries than objects, the encoder returns an error.

### tgm_metadata_num_objects (FFI)

`tgm_metadata_num_objects()` returns `base.len()`, which is the number of per-object metadata entries. After encoding, this matches the actual data object count because the encoder populates one base entry per object.

### set Command on Zero-Object Messages

The CLI `set` command redirects mutations to `_extra_` when the message has zero data objects. This is because base entries must align 1:1 with descriptors, and a zero-object message has no descriptors.

### Both _extra_ and extra in Python Dict

When both `"_extra_"` and `"extra"` are present in a Python metadata dict, `_extra_` takes precedence (it's the wire-format name). The `"extra"` key is treated as a convenience alias and only used if `"_extra_"` is absent.

### Filter Matching with Multi-Object Messages

CLI where-clause filters (`-w mars.param=2t`) match at the **message** level. If `base[0]` has `mars.param=2t` and `base[1]` has `mars.param=msl`, the filter matches `"2t"` (first base entry match). To filter by per-object values, split the message first.

### Split Preserves Per-Object Metadata

When splitting a multi-object message, the CLI `split` command assigns each object its own base entry from the original message. The `_reserved_` key is stripped from each entry (the encoder regenerates it). Extra metadata is copied to all split messages.

### Merge Concatenates Base Arrays

When merging messages, the CLI `merge` command concatenates all base arrays. The merge strategy (`first`/`last`/`error`) only applies to `_extra_` key conflicts. The `_reserved_` section is cleared and regenerated by the encoder.

### Deeply Nested Paths

Dot-notation paths support arbitrary nesting depth: `grib.geography.Ni`, `a.b.c.d.e`. The recursive resolver walks through CBOR Map values at each level. If a non-Map value is encountered before the path is fully resolved, the lookup returns `None`.

### JSON Output Structure

CLI `dump -j` and `ls -j` output uses the wire-format structure:

```json
{
  "version": 2,
  "base": [{"mars": {"param": "2t"}, "_reserved_": {"tensor": {"ndim": 1}}}],
  "extra": {"custom": "value"}
}
```

The `_reserved_` keys within base entries are included in JSON output for transparency.

---

## Metadata Refactor: Detailed Edge Cases

The following edge cases were identified during systematic review of the Rust core crate (`tensogram-core`) after the metadata refactor.

### base Array Count Validation

| Scenario | Behaviour |
|----------|-----------|
| `base.len() < descriptors.len()` | Auto-extended with empty entries. `_reserved_.tensor` is inserted in each. |
| `base.len() == descriptors.len()` | Normal path. Pre-existing application keys preserved. |
| `base.len() > descriptors.len()` | **Error**: "metadata base has N entries but only M descriptors provided; extra base entries would be discarded". |

**Rationale:** Silently truncating excess base entries would lose user data. Auto-extending is safe because the library adds `_reserved_.tensor` to each new entry.

### `_reserved_.tensor` After Encode

After encoding, each `base[i]["_reserved_"]["tensor"]` always contains exactly four keys:

| Key | Value | Example |
|-----|-------|---------|
| `ndim` | CBOR integer | `0` for scalar, `2` for matrix |
| `shape` | CBOR array of integers | `[]` for scalar, `[10, 20]` for matrix |
| `strides` | CBOR array of integers | `[]` for scalar, `[20, 1]` for matrix |
| `dtype` | CBOR text | `"float32"`, `"int64"`, etc. |

For scalar tensors (`ndim: 0`), `shape` and `strides` are empty arrays `[]`.

### Preceder `_reserved_` Protection

**Encoder side:** `StreamingEncoder::write_preceder()` rejects any metadata map containing a `_reserved_` key. Error: "client code must not write '_reserved_' in preceder metadata".

**Decoder side:** When the decoder encounters a `_reserved_` key in a preceder's `base[0]`, it **strips** the key rather than rejecting the message. This is permissive — the data may come from a non-standard producer. The encoder-populated `_reserved_.tensor` from the footer metadata is preserved.

**Merge order in `finish()`:** Footer metadata is populated first (`_reserved_.tensor`), then preceder payloads are merged on top. Since the decoder strips `_reserved_` from preceders, there is no risk of preceder `_reserved_` clobbering the encoder's `_reserved_.tensor`.

### Backward Compatibility with Old CBOR Keys

| Old key | Behaviour on decode |
|---------|---------------------|
| `"common"` (v2 pre-refactor) | Silently ignored (unknown CBOR key). |
| `"payload"` (v2 pre-refactor) | Silently ignored. |
| `"reserved"` (old name) | Silently ignored — only `"_reserved_"` is recognized. |
| Both `"reserved"` and `"_reserved_"` | Only `"_reserved_"` is captured; `"reserved"` is ignored. |

`GlobalMetadata` does not use `#[serde(deny_unknown_fields)]`, so serde drops unrecognized keys.

### compute_common() Key Selection

`compute_common()` only examines keys from the **first** base entry as candidates for common keys. Keys present in later entries but absent from the first entry are never promoted to common.

Example: if entry 0 has keys `{a, b}` and entry 1 has `{b, c}`, only `b` is a candidate (and becomes common if values match). Key `c` appears only in entry 1's remaining set.

### compute_common() NaN Handling

CBOR `Float(NaN)` values with identical bit patterns are treated as equal by `cbor_values_equal()`, using `f64::to_bits()` comparison. This means NaN values are classified as common when all entries share the same NaN bit pattern. Standard CBOR equality (`PartialEq`) would fail because `NaN != NaN`.

### compute_common() CBOR Map Ordering

`cbor_values_equal()` compares CBOR maps positionally (entry-by-entry). Two maps with the same keys and values in different order are NOT equal. This is correct because canonical CBOR encoding ensures all maps are always sorted — different-order maps can only arise from non-canonical input.

### Shape Product Overflow

All shape-product computations use `checked_mul` to detect overflow. This applies to `encode()`, `decode()`, `ObjectIter::next()`, and `decode_range()`. If the product overflows `u64`, a `TensogramError::Metadata("shape product overflow")` is returned. No silent wraparound.

### `_extra_` Scope Independence

`_extra_` is message-level, while `base[i]` entries are per-object. Keys with the same name can exist in both:

```rust
meta.base[0].insert("mars".into(), ...);  // per-object
meta.extra.insert("mars".into(), ...);     // message-level
// Both preserved after encode/decode round-trip
```

### Empty `_extra_` in CBOR

An empty `_extra_` map is omitted from CBOR output via `skip_serializing_if = "BTreeMap::is_empty"`. On decode, a missing `_extra_` key is deserialized as an empty `BTreeMap`. Round-trips correctly.

### Deeply Nested `_reserved_` in base Entries

Only the **top-level** `_reserved_` key in `base[i]` is rejected by the encoder. Deeply nested `_reserved_` keys (like `{"foo": {"_reserved_": ...}}`) are allowed and preserved. The encoder only checks `entry.contains_key("_reserved_")`.

### CLI `set` on Zero-Object Messages

When `tensogram set` modifies a zero-object message, keys that would normally go into `base` are redirected to `_extra_` instead (since `base` entries must align 1:1 with data objects, and there are none).

---

## Error Handling Reference

This section documents all error types, how they propagate across languages, and what messages users can expect.

### TensogramError Variants (Rust)

The core library defines seven error variants in `TensogramError`:

| Variant | When it occurs | Example message |
|---------|---------------|-----------------|
| `Framing(String)` | Invalid wire format — magic bytes, postamble, frame ordering | `"buffer too short (12 bytes, need >= 24)"` |
| `Metadata(String)` | Metadata validation failures — version, base count, CBOR parse | `"metadata base has 3 entries but only 2 descriptors provided"` |
| `Encoding(String)` | Encoding pipeline errors — simple_packing NaN, bit-width | `"NaN value at index 42"` |
| `Compression(String)` | Compression/decompression failures — codec errors, range access | `"RangeNotSupported: zstd does not support partial decode"` |
| `Object(String)` | Per-object errors — index out of range, shape overflow | `"object index 99 out of range (num_objects=2)"` |
| `Io(io::Error)` | File system errors — open, read, write, seek | `"data.tgm: No such file or directory"` |
| `HashMismatch { expected, actual }` | Integrity check failure | `"hash mismatch: expected=abc123, actual=def456"` |

### Python Exception Mapping

The Python bindings convert `TensogramError` to Python exceptions:

| Rust variant | Python exception | Prefix in message |
|-------------|-----------------|-------------------|
| `Framing` | `ValueError` | `FramingError:` |
| `Metadata` | `ValueError` | `MetadataError:` |
| `Encoding` | `ValueError` | `EncodingError:` |
| `Compression` | `ValueError` | `CompressionError:` |
| `Object` | `ValueError` | `ObjectError:` |
| `Io` | `IOError` | *(raw io message)* |
| `HashMismatch` | `RuntimeError` | `HashMismatch:` |

Additional Python-side exceptions:

| Function | Exception | Condition |
|----------|-----------|-----------|
| `encode()` | `ValueError` | Missing `version` key, `_reserved_` in dict, unknown dtype |
| `decode()` | `ValueError` | Corrupted buffer, invalid CBOR |
| `Metadata.__getitem__()` | `KeyError` | Key not found in base or extra |
| `Metadata.__getitem__("_reserved_")` | `KeyError` | `_reserved_` is always hidden from dict access |
| `TensogramFile.__getitem__()` | `IndexError` | Message index out of range |
| `TensogramFile.__getitem__()` | `TypeError` | Non-integer, non-slice index |
| `compute_packing_params()` | `ValueError` | NaN in input array |
| `encode(hash="sha256")` | `ValueError` | `"unknown hash: sha256"` |

**Example: handling errors in Python:**

```python
import tensogram

# File not found
try:
    with tensogram.TensogramFile.open("missing.tgm") as f:
        pass
except IOError as e:
    print(f"File error: {e}")
    # → "File error: file not found: missing.tgm"

# Corrupted buffer
try:
    tensogram.decode(b"garbage")
except ValueError as e:
    print(f"Decode error: {e}")
    # → "Decode error: FramingError: buffer too short ..."

# Hash verification failure
try:
    meta, objects = tensogram.decode(buf, verify_hash=True)
except RuntimeError as e:
    print(f"Integrity error: {e}")
    # → "Integrity error: HashMismatch: expected=..., actual=..."

# Missing metadata key
meta, objects = tensogram.decode(buf)
try:
    val = meta["nonexistent"]
except KeyError:
    print("Key not found")

# Index out of range
with tensogram.TensogramFile.open("data.tgm") as f:
    try:
        msg = f[999]
    except IndexError as e:
        print(f"Index error: {e}")
        # → "message index 999 out of range for file with 2 messages"
```

### CLI Error Handling

All CLI commands:
- Print errors to stderr with `error:` prefix
- Show the full error chain (nested causes)
- Exit with code 1 on any error
- Exit with code 0 on success

**Common CLI error scenarios:**

```bash
# File not found
$ tensogram ls nonexistent.tgm
error: file not found: nonexistent.tgm

# Invalid where clause
$ tensogram ls -w "bad-clause" data.tgm
error: invalid where clause: invalid where-clause: bad-clause (expected key=value or key!=value)

# Missing key in strict get
$ tensogram get -p "nonexistent" data.tgm
error: key not found: nonexistent

# Protected namespace
$ tensogram set -s "_reserved_.tensor.ndim=5" input.tgm output.tgm
error: cannot modify '_reserved_' — this namespace is managed by the library

# Immutable descriptor key
$ tensogram set -s "shape=broken" input.tgm output.tgm
error: cannot modify immutable key: shape

# Merge conflict with error strategy
$ tensogram merge --strategy error a.tgm b.tgm -o merged.tgm
error: conflicting values for key 'param' (use --strategy first or last to resolve)

# Invalid merge strategy
$ tensogram merge --strategy unknown a.tgm b.tgm -o merged.tgm
error: unknown merge strategy 'unknown': expected first, last, or error

# Message index out of range (via file.read_message)
$ tensogram dump corrupt.tgm
error: framing error: buffer too short ...
```

### xarray Backend Error Handling

| Scenario | Behaviour |
|----------|-----------|
| File not found | `IOError` from `tensogram.TensogramFile.open()` |
| Corrupt file | `ValueError` from `tensogram.decode_descriptors()` |
| `message_index` out of range | `ValueError` from `TensogramFile.read_message()` |
| `message_index < 0` | `ValueError("message_index must be >= 0, got -1")` |
| `meta.base` shorter than objects | Warning logged; missing entries treated as empty dicts |
| Unsupported dtype | `TypeError("unsupported tensogram dtype ...")` |
| `dim_names` count mismatch | `ValueError("dim_names has N entries but tensor has M dimensions")` |
| `decode_range` failure | Warning logged; falls back to full `decode_object()` |
| File with zero messages + `merge_objects=True` | Returns empty `xr.Dataset()` |

### Zarr Store Error Handling

| Scenario | Behaviour |
|----------|-----------|
| File not found | `OSError("failed to open TGM file ...")` wrapping the original error |
| Corrupt message | `ValueError("failed to decode message ...")` wrapping the original error |
| Failed object decode | `ValueError("failed to decode object N ...")` wrapping the original error |
| `message_index` out of range | `IndexError("message_index N out of range (file has M message(s))")` |
| `message_index < 0` | `ValueError("message_index must be >= 0, got -1")` |
| Invalid mode | `ValueError("invalid mode 'x'; expected 'r', 'w', or 'a'")` |
| Empty path | `ValueError("path must be a non-empty string, got ''")` |
| Store already open | `ValueError("store is already open")` |
| Write to read-only store | Raises from Zarr base class |
| Flush failure during exception | Warning logged; original exception preserved |
| Unsupported dtype on write | `ValueError("unsupported dtype for variable ...")` |
| Chunk size mismatch on write | `ValueError("chunk data for 'var': expected N bytes ... got M")` |
| Multiple chunks per variable | `ValueError("variable 'var' has N chunk keys; TensogramStore only supports single-chunk arrays")` |
| Unsupported `ByteRequest` type | `TypeError("unsupported ByteRequest type: ...")` |
| Zero messages in file | Root group zarr.json with empty attributes; no arrays |

### IO Error Path Context

All file I/O errors include the file path in the error message. This applies to:
- `TensogramFile::open()` — `"file not found: /path/to/file.tgm"`
- `TensogramFile::create()` — `"cannot create /path/to/file.tgm: Permission denied"`
- Internal re-opens (scan, read, append) — `"/path/to/file.tgm: No such file or directory"`

This ensures that when errors propagate through multiple layers (e.g. Rust → Python → xarray), the original file path is always visible in the error message.
