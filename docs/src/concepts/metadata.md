# Metadata

Metadata in Tensogram is stored as **CBOR** — Concise Binary Object Representation (RFC 8949). Think of it as a compact, binary version of JSON. It supports the same types (strings, integers, floats, booleans, arrays, maps), but is smaller and faster to parse.

## Top-Level Structure

Every message's metadata has this shape:

```json
{
  "version": 1,
  "objects": [ ...one descriptor per tensor... ],
  "payload": [ ...one descriptor per tensor... ],
  "mars": { "class": "od", "type": "fc", "date": "20260401" }
}
```

The `version`, `objects`, and `payload` fields are fixed. Everything else is **free-form** — you can add any key at the top level or inside an object descriptor, using any CBOR value type.

## Namespaced Keys

Convention: application-layer keys are grouped under a **namespace** key. ECMWF's MARS vocabulary lives under `"mars"`:

```json
{
  "version": 1,
  "mars": {
    "class": "od",
    "type": "fc",
    "param": "2t",
    "date": "20260401",
    "step": 6
  },
  "objects": [...],
  "payload": [...]
}
```

The library does not interpret or validate these keys. Your application layer assigns meaning.

## Per-Object vs Message-Level Metadata

You can attach metadata at two levels:

| Level | Where it goes | Typical use |
|---|---|---|
| **Message-level** | Top-level `extra` fields | Forecast date, run type, domain |
| **Per-object** | `objects[i].extra` fields | Parameter name, level, units |

Both use the same namespace convention:

```rust
// Message-level: mars.class = "od"
extra.insert("mars", Value::Map(vec![
    (Value::Text("class"), Value::Text("od")),
]));

// Per-object: mars.param = "2t"
obj_extra.insert("mars", Value::Map(vec![
    (Value::Text("param"), Value::Text("2t")),
]));
```

## Filtering with the CLI

The `-w` flag on `ls`, `dump`, `get`, and `copy` uses dot-notation to filter messages:

```bash
# Only messages where mars.param equals "2t" or "10u"
tensogram ls forecast.tgm -w "mars.param=2t/10u"

# Exclude messages where mars.class equals "od"
tensogram ls forecast.tgm -w "mars.class!=od"
```

The `/` character separates OR values. Key lookup checks message-level metadata first, then object-level (returning the first match).

## Deterministic Encoding

When Tensogram encodes metadata to CBOR, it **sorts all map keys** by their CBOR byte representation (RFC 8949 §4.2 canonical form). This guarantees that the same metadata always produces the same bytes, regardless of the order you inserted keys in your application code. This matters for hashing and reproducibility.

> **Edge case:** Nested maps are also sorted recursively. Even metadata stored inside a CBOR map value (like the `"mars"` namespace) gets canonical ordering.
