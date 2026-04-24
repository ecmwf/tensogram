# Metadata

Metadata in Tensogram is stored as **CBOR** -- Concise Binary Object Representation (RFC 8949). Think of it as a compact, binary version of JSON. It supports the same types (strings, integers, floats, booleans, arrays, maps), but is smaller and faster to parse.

## Metadata Locations

In v3, metadata lives in two distinct places:

| Level | Where it lives | What it contains |
|---|---|---|
| **Global** | Header or footer metadata frame | `GlobalMetadata`: `base` (per-object metadata array) + `_reserved_` (library internals) + `_extra_` (client annotations). The **wire-format version lives in the preamble**, not in the CBOR metadata frame. |
| **Per-object** | Each data object frame's CBOR descriptor | `DataObjectDescriptor`: tensor shape, encoding pipeline, plus `params` for encoding parameters |

Each data object carries its own descriptor inline within its frame.

## GlobalMetadata

The global metadata frame contains a `GlobalMetadata` struct with three named sections:

```rust
GlobalMetadata {
    base: Vec::new(),              // one BTreeMap per data object (independent entries)
    reserved: BTreeMap::new(),     // library internals (_reserved_ in CBOR)
    extra: BTreeMap::new(),        // client-writable catch-all (_extra_ in CBOR)
}
```

In CBOR, this looks like (using ECMWF MARS keys as one concrete example
vocabulary):

```json
{
  "base": [
    {
      "mars": {
        "class": "od", "type": "fc",
        "date": "20260401", "time": "1200", "param": "2t"
      }
    }
  ],
  "_extra_": {
    "source": "ifs-cycle49r2"
  }
}
```

The same mechanism works for any application vocabulary. A neuroimaging
pipeline might use a BIDS namespace:

```json
{
  "base": [{
    "bids": { "subject": "sub-01", "session": "ses-01",
              "task": "rest", "run": 1 }
  }]
}
```

A materials-simulation pipeline might use a custom namespace:

```json
{
  "base": [{
    "material": { "composition": "Fe3O4", "lattice": "cubic", "T_K": 300.0 }
  }]
}
```

The library does not know or care which vocabulary is used — it simply
stores, serialises, and returns the keys you supply.

**There are no required top-level keys.**  The CBOR metadata frame is
**fully free-form** — only `base`, `_reserved_`, and `_extra_` are
library-interpreted.  Any other top-level key the caller supplies
(including a stray legacy `"version"` from pre-0.17 producers) is
routed into `_extra_` on decode so the data round-trips cleanly.
`_extra_` itself is a **free-form** catch-all — you can add any key
using any CBOR value type.  The library does not interpret or validate
these keys.  Your application layer assigns meaning.

> **Reading the wire version.** The wire-format version is carried in
> the preamble (see `../format/wire-format.md` §3).  Rust callers use
> `tensogram::WIRE_VERSION`; Python uses `tensogram.WIRE_VERSION`;
> TypeScript uses `WIRE_VERSION` from `@ecmwf.int/tensogram`; FFI /
> C++ callers call `tgm_message_version` / `msg.version()`.  All of
> these resolve to the constant `3` in v3.

## Per-Object Metadata in `base`

The `base` section is a CBOR **array of maps** — one entry per data object. Each entry holds ALL structured metadata for that object independently. Entries are self-contained — there is no tracking of which keys are common across objects.

The encoder auto-populates `_reserved_.tensor` (with ndim, shape, strides, dtype) in each entry when you call `encode()` or `StreamingEncoder::finish()`. Application keys are preserved:

```json
{
  "base": [
    {
      "mars": { "class": "od", "type": "fc", "param": "2t", "levtype": "sfc" },
      "_reserved_": {
        "tensor": { "ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float64" }
      }
    },
    {
      "mars": { "class": "od", "type": "fc", "param": "10u", "levtype": "sfc" },
      "_reserved_": {
        "tensor": { "ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float64" }
      }
    }
  ]
}
```

This lets readers discover the shape, type, and per-object metadata of every object by reading only the global metadata frame — without opening each data object frame.

> **No common/varying split:** Every `base[i]` entry is self-contained. MARS keys shared across all objects (e.g. `class`, `type`) are simply repeated in each entry. If you need to extract commonalities (e.g. for display or merges), use the `compute_common()` utility in software after decoding.

## DataObjectDescriptor

The `params` field of each `DataObjectDescriptor` is a `BTreeMap<String, ciborium::Value>` for **encoding parameters only** (e.g. `sp_reference_value`, `sp_bits_per_value`). These are flattened into the CBOR descriptor alongside the fixed tensor fields.

For example, a data object's CBOR descriptor might look like:

```json
{
  "type": "ntensor",
  "ndim": 2,
  "shape": [721, 1440],
  "strides": [1440, 1],
  "dtype": "float32",
  "byte_order": "big",
  "encoding": "simple_packing",
  "filter": "none",
  "compression": "szip",
  "sp_reference_value": 230.5,
  "sp_bits_per_value": 16
}
```

> In v3 the per-object payload hash lives in the frame footer's
> inline `[hash u64]` slot (see `../format/wire-format.md` §2.2),
> not in the CBOR descriptor.

Here, `sp_reference_value` and `sp_bits_per_value` live in the `params` map. Application metadata such as MARS keys belongs in `base[i]["mars"]` in the global metadata.

## Namespaced Keys

Convention: application-layer keys are grouped under a **namespace** key, so
that multiple vocabularies can coexist in the same message. For example, ECMWF's
MARS vocabulary lives under `"mars"`:

```json
{
  "base": [
    {
      "mars": {
        "class": "od", "type": "fc",
        "param": "2t", "date": "20260401", "step": 6
      }
    }
  ]
}
```

Other pipelines use other namespaces — `"cf"` for CF conventions, `"bids"` for
neuroimaging, `"dicom"` for medical imaging, or anything your application
defines. This convention applies at both levels — global metadata and
per-object params.

## Filtering with the CLI

The `-w` flag on `ls`, `dump`, `get`, and `copy` uses dot-notation to filter
messages on any namespace. The examples below use the MARS vocabulary, but the
same syntax works with any application namespace (e.g. `bids.subject`,
`dicom.Modality`, `product.name`):

```bash
# Only messages where mars.param equals "2t" or "10u"
tensogram ls data.tgm -w "mars.param=2t/10u"

# Exclude messages where mars.class equals "od"
tensogram ls data.tgm -w "mars.class!=od"
```

The `/` character separates OR values. Key lookup searches `base[i]` entries first (skipping `_reserved_`, first match across entries), then `_extra_` for backwards compatibility.

## Preceder Metadata Frames

In streaming mode, per-object metadata is normally only available in the footer metadata frame (written after all objects). A **Preceder Metadata Frame** (frame type 8) allows producers to send per-object metadata *before* the data object, without waiting for the footer.

A preceder carries a `GlobalMetadata` CBOR with a single-entry `base` array for the next data object:

```json
{
  "base": [{"product": {"name": "temperature"}, "units": "K"}]
}
```

**Merge rule:** On decode, preceder keys override footer `base[i]` keys on conflict. Structural keys auto-populated by the encoder (in `_reserved_.tensor`: ndim, shape, strides, dtype) are preserved from the footer when absent from the preceder. The consumer sees a unified `GlobalMetadata.base` — the preceder/footer distinction is transparent.

Use `StreamingEncoder::write_preceder()` before `write_object()` to emit a preceder frame. Preceders are optional per-object: some objects may have them, others may not.

## Value Type Rules

Keys must be **text strings**. Values must be JSON-compatible CBOR types: string, integer, float, boolean, null, array, or map. Byte strings, CBOR tags, undefined, and half-precision floats are not allowed. See [Metadata Value Types](../format/metadata-values.md) for the full rules and rationale.

## Deterministic Encoding

When Tensogram encodes metadata to CBOR, it **sorts all map keys** by their CBOR byte representation (RFC 8949 Section 4.2 canonical form). This guarantees that the same metadata always produces the same bytes, regardless of the order you inserted keys in your application code. This matters for hashing and reproducibility.

> **Edge case:** Nested maps are also sorted recursively. Even metadata stored inside a CBOR map value (like the `"mars"` namespace) gets canonical ordering.
