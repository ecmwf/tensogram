# Metadata

Metadata in Tensogram is stored as **CBOR** -- Concise Binary Object Representation (RFC 8949). Think of it as a compact, binary version of JSON. It supports the same types (strings, integers, floats, booleans, arrays, maps), but is smaller and faster to parse.

## Two Levels of Metadata

In v2, metadata lives in two distinct places:

| Level | Where it lives | What it contains |
|---|---|---|
| **Global** | Header or footer metadata frame | `GlobalMetadata`: version + structured sections (common/payload/reserved) + free-form extra keys |
| **Per-object** | Each data object frame's CBOR descriptor | `DataObjectDescriptor`: tensor shape, encoding pipeline, hash, plus `params` for extra per-object keys |

Each data object carries its own descriptor inline within its frame.

## GlobalMetadata

The global metadata frame contains a `GlobalMetadata` struct with three named sections:

```rust
GlobalMetadata {
    version: 2,
    common: BTreeMap::new(),   // keys shared across all objects
    payload: BTreeMap::new(),  // keys describing the collection
    reserved: BTreeMap::new(), // reserved for future use
    extra: BTreeMap::new(),    // free-form application-level keys
}
```

In CBOR, this looks like:

```json
{
  "version": 2,
  "mars": {
    "class": "od",
    "type": "fc",
    "date": "20260401",
    "time": "1200"
  }
}
```

The `version` field is required (u16). Everything else in `extra` is **free-form** -- you can add any key using any CBOR value type. The library does not interpret or validate these keys. Your application layer assigns meaning.

## Per-Object Metadata

Per-object metadata lives in the `params` field of each `DataObjectDescriptor`. This is a `BTreeMap<String, ciborium::Value>` that gets flattened into the CBOR descriptor alongside the tensor and encoding fields.

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
  "reference_value": 230.5,
  "bits_per_value": 16,
  "mars": { "param": "2t" },
  "units": "K",
  "hash": { "type": "xxh3", "value": "a1b2c3d4e5f6..." }
}
```

Here, `reference_value`, `bits_per_value`, `mars`, and `units` all live in the `params` map. They are flattened into the same CBOR map as the fixed fields.

## Namespaced Keys

Convention: application-layer keys are grouped under a **namespace** key. ECMWF's MARS vocabulary lives under `"mars"`:

```json
{
  "version": 2,
  "mars": {
    "class": "od",
    "type": "fc",
    "param": "2t",
    "date": "20260401",
    "step": 6
  }
}
```

This convention applies at both levels -- global metadata and per-object params.

## Filtering with the CLI

The `-w` flag on `ls`, `dump`, `get`, and `copy` uses dot-notation to filter messages:

```bash
# Only messages where mars.param equals "2t" or "10u"
tensogram ls forecast.tgm -w "mars.param=2t/10u"

# Exclude messages where mars.class equals "od"
tensogram ls forecast.tgm -w "mars.class!=od"
```

The `/` character separates OR values. Key lookup checks global metadata first, then per-object params (returning the first match).

## Deterministic Encoding

When Tensogram encodes metadata to CBOR, it **sorts all map keys** by their CBOR byte representation (RFC 8949 Section 4.2 canonical form). This guarantees that the same metadata always produces the same bytes, regardless of the order you inserted keys in your application code. This matters for hashing and reproducibility.

> **Edge case:** Nested maps are also sorted recursively. Even metadata stored inside a CBOR map value (like the `"mars"` namespace) gets canonical ordering.
