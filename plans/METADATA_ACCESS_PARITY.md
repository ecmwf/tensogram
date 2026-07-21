# Metadata Access Parity — Design

> **Status: PLAN — not yet implemented.** This document is the proposed
> design for making CBOR-metadata access **symmetric across every binding**
> (Rust, C, C++, Python, TypeScript/WASM, Fortran), with **Python as the
> capability benchmark**. The JSON exporters (`*_to_json`) are **not** an
> access path — they remain serialization/debug utilities only.

## 1. Problem

Metadata lookup capability is currently uneven across bindings. A caller
cannot reliably answer *"does this key exist?"* the same way everywhere, and
the scalar getters conflate **absent**, **wrong-type**, and **a real value
equal to the default**.

| Surface | Existence check | Missing-key signal | Default hides missing key? | Enumerate keys | Nested / array access |
|---|---|---|---|---|---|
| **Rust** (core) | ✅ `BTreeMap::contains_key` / `get -> Option` | `Option::None` | No | ✅ native | ✅ native (`ciborium::Value`) |
| **Python** (benchmark) | ✅ `key in meta` (`__contains__`) | `KeyError` / `.get()→None` | No | ✅ real dicts | ✅ real dict/list |
| **TypeScript/JS** | ✅ `getMetaKey→undefined`, `'k' in entry` | `undefined` | No | ✅ plain objects | ✅ plain objects |
| **C FFI** | ⚠️ partial (`get_string→NULL`) | NULL / caller default | **Yes** (int/float) | ❌ | ❌ (JSON only) |
| **C++** | ❌ | `""` / default (0) | **Yes** (worse: absent == `""`) | ❌ | ❌ (JSON only) |
| **Fortran** | ❌ | `''` / default | **Yes** | ❌ | ❌ (no JSON either) |

The C-ABI family (C/C++/Fortran) is the bottleneck: scalars cannot express
"absent vs present" without a sentinel, and arrays/maps cannot cross the ABI
at all except by serializing to JSON — which we explicitly reject as an
access mechanism.

## 2. Goal

One **shared access model** with identical semantics in all six bindings,
matching what a Python caller can do today against `meta` / `meta.base[i]` /
`meta.extra`:

1. **Existence** — distinguishable from value (and from wrong-type / null).
2. **Get-optional** — a single lookup that yields *either* nothing (absent)
   *or* a rich value handle; no sentinel that can collide with real data.
3. **Typed extraction** — string / int / float / bool / bytes, where
   "not that type" is distinct from "absent".
4. **Enumeration** — walk the keys of `base[i]`, `_extra_`, `_reserved_`.
5. **Structural navigation** — descend nested maps and index arrays
   *without* JSON.
6. **Per-object scoping** — everything available message-level *and* scoped
   to a single `base[i]`.

**Efficiency:** lookups are **zero-copy borrows** into the parsed metadata;
strings/bytes are returned as `ptr + len` (no allocation, no interior-NUL
limitation). **Friction:** the common path is a single call returning an
optional value handle — not `has` *then* `get`.

## 3. Decisions (confirmed — design review 2026-07-20)

| # | Decision |
|---|---|
| Model | A **borrowed value cursor** (`tgm_value_t` in C; `meta_value` / `tensogram_value` wrappers) is the single primitive. Existence, typed get, enumeration and navigation are all expressed through it. |
| Primary API shape | **`get(...) → optional<value>`** is the ergonomic default (present ⇔ has-value). `has`/`contains` and `try_get_<T>` are thin conveniences layered on it, not the core. ✅ |
| No JSON access path | `*_to_json` stay as **serialization/logging** helpers only. Never required to answer existence/typed/enumeration queries. |
| Single source of truth | The dot-path + first-match + `_extra_` fallback + `_reserved_` hiding logic is **consolidated in `tensogram` core** and reused by the FFI, the CLI, and mirrored by TS. No more three copies. |
| Zero-copy | Value handles borrow `&ciborium::Value`; strings/bytes returned `ptr+len`; all valid until `tgm_metadata_free`. |
| Coercion | **Precise — no cross-type coercion.** `as_string` = text only; `as_i64` = integer only; `as_f64` = float **or** integer-widened. The legacy coercing getters keep their old behaviour. ✅ |
| Backward compat | **Purely additive.** Every existing symbol/function keeps its exact current behaviour. Legacy default-based getters are **retained + doc-deprecated** (pointer to the precise API), scheduled for removal in a future MAJOR (with approval). ✅ |
| Versioning | New surface ⇒ **MINOR** bump. No wire-format change. No MAJOR. |
| Path grammar | Dotted segments navigate **maps only**; **array elements are reached via the cursor** (index), matching Python (`meta["a"]["b"]` then `[0]`). ✅ |
| Reserved visibility | `_reserved_` is **hidden from path getters** at the first segment (unchanged), but **visible when enumerating** a `base[i]` section (parity with Python `meta.base[i]` and the JSON export). ✅ |
| Integer typing | **Single `TGM_VALUE_TYPE_INT`** (CBOR integers are i128-backed); range is resolved by the extractors `as_i64` / `as_u64`, each reporting success. No separate `UINT` type. ✅ |
| Vocabulary | `has` / `has_at` (C), `contains` / `contains_at` (C++ & Python), `get` / `get_at`, `try_get_<T>`; handle types `tgm_value_t` (C) / `meta_value` (C++) / `tensogram_value` (Fortran). ✅ |

## 4. Non-goals

- No wire-format / CBOR schema change.
- No removal or behavioural change of existing functions (no MAJOR).
- JSON as an access/parity mechanism (explicitly rejected).
- Mutating metadata through the cursor (read-only access; encoding is
  unchanged and still goes through the existing `*_json` / builder paths).
- Numeric-string path segments indexing arrays (arrays use the cursor).

## 5. Shared semantics (one spec, all bindings)

- **Scoping**
  - *Message-level* (`get` / `has`): first match across `base[0..]`
    (skipping `_reserved_` at the first segment), then fall back to
    `_extra_`. Explicit `extra.` / `_extra_.` prefix targets `_extra_`
    directly. The `version` pseudo-key resolves to the preamble wire
    version (unchanged).
  - *Per-object* (`get_at(i, …)` / `has_at`): scoped to `base[i]` only —
    no cross-object first-match, no `_extra_` fallback, `_reserved_` hidden
    at the first segment.
- **Absent vs type vs null** — the crux, and what today's defaults destroy:
  - `has` / `get()→optional` answer **presence** (any type, including CBOR
    null).
  - a present value's type is inspected via `value_type()`; typed
    extractors return "wrong type" **distinctly** from "absent".
  - So a stored `0` / `""` / null is now unambiguously distinguishable from
    a missing key in *every* language.
- **Types** — `NULL, BOOL, INT, FLOAT, STRING, BYTES, ARRAY, MAP`. Integers
  are i128-backed in CBOR and surface as a **single `INT`** type; extraction
  offers `as_i64` **and** `as_u64`, each reporting range success.
- **Lifetimes (C/C++/Fortran)** — value handles and returned `ptr+len`
  string/bytes buffers are **borrowed**, valid until `tgm_metadata_free`,
  never freed by the caller. C++ `meta_value` and Fortran `tensogram_value`
  are non-owning and must not outlive their parent `metadata`.
- **Strings** — UTF-8 `ptr+len`; may contain interior NUL; **not** required
  to be NUL-terminated (fixes the current interior-NUL drop).

## 6. Layered design

### 6.1 Rust core (`tensogram::metadata`) — single source of truth

New public, borrowing accessors on `GlobalMetadata` plus a thin value view.
The FFI and CLI are refactored to call these (dedupe the three copies of the
path walker in `tensogram-ffi`, `tensogram-cli/filter.rs`, and the
TS mirror).

```rust
impl GlobalMetadata {
    pub fn get(&self, path: &str) -> Option<MetaValue<'_>>;            // message-level
    pub fn get_at(&self, obj: usize, path: &str) -> Option<MetaValue<'_>>;
    pub fn contains(&self, path: &str) -> bool;
    pub fn contains_at(&self, obj: usize, path: &str) -> bool;
    pub fn object(&self, obj: usize) -> Option<MetaValue<'_>>;        // base[i] as a map view
    pub fn extra_view(&self) -> MetaValue<'_>;                         // _extra_ as a map view
    pub fn reserved_view(&self) -> MetaValue<'_>;                      // _reserved_ as a map view
}

pub enum MetaType { Null, Bool, Int, Float, String, Bytes, Array, Map }

pub struct MetaValue<'a>(/* &Value or &BTreeMap section */);
impl<'a> MetaValue<'a> {
    pub fn value_type(&self) -> MetaType;
    pub fn as_bool(&self) -> Option<bool>;
    pub fn as_i64(&self) -> Option<i64>;
    pub fn as_u64(&self) -> Option<u64>;
    pub fn as_f64(&self) -> Option<f64>;      // float or int-widened
    pub fn as_str(&self) -> Option<&'a str>;
    pub fn as_bytes(&self) -> Option<&'a [u8]>;
    // arrays
    pub fn len(&self) -> usize;               // array/map element count
    pub fn get_index(&self, i: usize) -> Option<MetaValue<'a>>;
    // maps
    pub fn get_key(&self, key: &str) -> Option<MetaValue<'a>>;
    pub fn key_at(&self, i: usize) -> Option<&'a str>;
    pub fn value_at(&self, i: usize) -> Option<MetaValue<'a>>;
}
```

*Internal representation* (implementation detail): `MetaValue` wraps
`enum Ref<'a> { Cbor(&'a Value), Section(&'a BTreeMap<String, Value>) }` so
that `base[i]` / `_extra_` / `_reserved_` present as maps **without cloning**.

### 6.2 C ABI (`tensogram-ffi`) — the value cursor

Opaque borrowed handle + enum (registered in `cbindgen.toml`
`[export.rename]`: `TgmValue → tgm_value_t`, `TgmValueType → tgm_value_type`).
Handles are borrowed views owned by the `tgm_metadata_t`; an append-only
arena inside the handle keeps them alive until `tgm_metadata_free`.

```c
typedef struct tgm_value tgm_value_t;                 /* borrowed; never freed by caller */
typedef enum tgm_value_type {
  TGM_VALUE_TYPE_NULL, TGM_VALUE_TYPE_BOOL, TGM_VALUE_TYPE_INT,
  TGM_VALUE_TYPE_FLOAT, TGM_VALUE_TYPE_STRING, TGM_VALUE_TYPE_BYTES,
  TGM_VALUE_TYPE_ARRAY, TGM_VALUE_TYPE_MAP,
} tgm_value_type;

/* lookup (NULL ⇒ absent / out-of-range) */
const tgm_value_t* tgm_metadata_get   (const tgm_metadata_t*, const char* path);
const tgm_value_t* tgm_metadata_get_at(const tgm_metadata_t*, size_t obj, const char* path);
/* existence (cheap, no handle) */
bool tgm_metadata_has   (const tgm_metadata_t*, const char* path);
bool tgm_metadata_has_at(const tgm_metadata_t*, size_t obj, const char* path);
/* sections as map values (for enumeration) */
const tgm_value_t* tgm_metadata_object  (const tgm_metadata_t*, size_t obj);  /* base[i] (incl _reserved_) */
const tgm_value_t* tgm_metadata_extra   (const tgm_metadata_t*);
const tgm_value_t* tgm_metadata_reserved(const tgm_metadata_t*);

/* value inspection */
tgm_value_type tgm_value_get_type(const tgm_value_t*);
bool tgm_value_as_bool  (const tgm_value_t*, bool*     out);
bool tgm_value_as_i64   (const tgm_value_t*, int64_t*  out);
bool tgm_value_as_u64   (const tgm_value_t*, uint64_t* out);
bool tgm_value_as_f64   (const tgm_value_t*, double*   out);
const char*    tgm_value_as_string(const tgm_value_t*, size_t* len_out);  /* borrowed ptr+len */
const uint8_t* tgm_value_as_bytes (const tgm_value_t*, size_t* len_out);  /* borrowed ptr+len */

/* arrays */
size_t             tgm_value_array_len(const tgm_value_t*);
const tgm_value_t* tgm_value_array_get(const tgm_value_t*, size_t index);
/* maps */
size_t             tgm_value_map_len     (const tgm_value_t*);
const char*        tgm_value_map_key_at  (const tgm_value_t*, size_t index, size_t* len_out);
const tgm_value_t* tgm_value_map_value_at(const tgm_value_t*, size_t index);
const tgm_value_t* tgm_value_map_get     (const tgm_value_t*, const char* key); /* single segment */
```

Legacy `tgm_metadata_get_string/int/float(_at)` and `*_to_json` are retained
unchanged. `tgm_metadata_num_objects` already exists.

### 6.3 C++ (`tensogram::metadata` + new `tensogram::meta_value`)

Near-Python ergonomics; `meta_value` is non-owning (tied to the parent
`metadata`). Old `get_string/get_int/get_float(_at)` retained.

```cpp
bool                       metadata::contains(std::string_view path) const;
bool                       metadata::contains_at(std::size_t obj, std::string_view path) const;
std::optional<meta_value>  metadata::get(std::string_view path) const;
std::optional<meta_value>  metadata::get_at(std::size_t obj, std::string_view path) const;
std::optional<std::int64_t>   metadata::try_get_int   (std::string_view path) const;
std::optional<double>         metadata::try_get_double(std::string_view path) const;
std::optional<std::string_view> metadata::try_get_string(std::string_view path) const;
std::optional<bool>           metadata::try_get_bool  (std::string_view path) const;
meta_value                 metadata::object(std::size_t obj) const;   // base[i] map
meta_value                 metadata::extra() const;
meta_value                 metadata::reserved() const;

class meta_value {                    // value_type(), is_map()/is_array()/...
  std::optional<std::int64_t>     as_int()    const;
  std::optional<double>           as_double() const;
  std::optional<std::string_view> as_string() const;
  std::optional<bool>             as_bool()   const;
  std::optional<std::span<const std::uint8_t>> as_bytes() const;
  // array:  size(), operator[](size_t)->meta_value, begin()/end()
  // map:    size(), contains(key), get(key)->optional<meta_value>,
  //         keys(), begin()/end() over (string_view, meta_value)
};
```
Example: `meta.object(0).get("mars")->get("class")->as_string()`, or
`for (auto& v : meta.get("grib_repro")->as_array()) ...`.

### 6.4 Fortran (`tensogram_metadata_*` + new `tensogram_value`)

Full parity — existence, typed try-get, enumeration, nested/array
navigation — with no JSON. Also binds the currently-missing
`tensogram_metadata_num_objects`. Old getters retained.

```fortran
logical :: tensogram_metadata_has(meta, key [, obj_index])
logical :: tensogram_metadata_try_get_int   (meta, key, value [, obj_index])  ! result=found
logical :: tensogram_metadata_try_get_float (meta, key, value [, obj_index])
logical :: tensogram_metadata_try_get_string(meta, key, value [, obj_index])
logical :: tensogram_metadata_try_get_bool  (meta, key, value [, obj_index])

type(tensogram_value) :: v            ! opaque, non-owning
v = tensogram_metadata_get(meta, key [, obj_index])   ! v%is_present()
!   v%kind()  -> integer(TGM_VALUE_TYPE_*)
!   v%as_int(x)/as_float(x)/as_string(s)/as_bool(b) -> logical found
!   arrays:  v%len(),  v%elem(i) -> tensogram_value
!   maps:    v%len(),  v%key(i) -> string,  v%value(i) -> tensogram_value,  v%get(key)
tensogram_metadata_object(meta, i)    ! base[i] map view
```

### 6.5 Python (benchmark — round out to a full Mapping + shared path grammar)

Python already has the capability; two gaps remain vs the *shared grammar*:
it exposes only **flat** keys (no dot-path) and is not a complete `Mapping`.

- Make `PyMetadata` a full `collections.abc.Mapping`: add `get(key,
  default=None)`, `keys()`, `values()`, `items()`, `__iter__`, `__len__`
  (message-level, first-match+extra — consistent with `in`).
- Add dot-path helpers matching the other bindings:
  `meta.get_path("mars.class", default=None)`, `meta.has_path(path)`,
  `meta.get_path_at(i, "geometry.gridType")`, `meta.has_path_at(i, path)`.
- `meta.base[i]`, `meta.extra`, `meta.reserved` stay real dicts (unchanged).

### 6.6 TypeScript/JS (already native — add symmetry helpers + per-object)

- `hasMetaKey(meta, path)`, `hasMetaKeyAt(meta, obj, path)`.
- `getMetaKeyAt(meta, obj, path)` (per-object; today only message-level
  `getMetaKey` exists).
- Typed conveniences for symmetry: `getMetaString/Int/Float/Bool(meta, path)`
  returning `T | undefined`.
- Section access stays native (`meta.base[i]`, `meta._extra_`). Keep the
  existing "mirrors Rust/Python/CLI semantics" invariant under test.

## 7. Testing — parity as a first-class artifact

- **Shared fixture**: one `.tgm` multi-object message whose metadata
  exercises every kind — string, negative + huge int, float + NaN, bool,
  bytes, nested map, array, `_reserved_.tensor`, `_extra_`, a key present in
  one object but absent in another, an **empty-string** value, and a value
  **equal to a common default (0)** — the last two prove the
  default-hiding/absent-vs-empty bugs are gone.
- **Capability matrix test** in every language asserting identical answers
  for: present/absent, each typed get, type-mismatch, enumeration order +
  count, nested + array navigation, per-object scoping, reserved-visibility
  rules. Rust (core + FFI), C (extend the cargo-c smoke), C++ (ctest),
  Python (pytest — the oracle), TS (vitest), Fortran (its harness).
- **Differential test**: random metadata trees — Python & TS (native) vs
  C/C++/Fortran (cursor) must agree on enumeration + values.
- **Fuzz**: extend the FFI fuzz target to walk the value cursor over
  arbitrary CBOR (memory-safety of navigation on adversarial input).
- **Gates**: `make version-check` unaffected; cargo-c header-drift +
  C-API smoke in `release-preflight` cover the regenerated `tensogram.h`.

## 8. Rollout (phased, each independently mergeable)

1. **Core** — `get/get_at/contains/contains_at`, `MetaValue`/`MetaType`;
   refactor FFI + CLI onto it (no external behaviour change). Tests.
2. **C ABI** — `tgm_value_t` + `tgm_value_type` + lookups/`has`/`as_*`/
   array+map nav/section accessors; regen `tensogram.h`; cbindgen renames;
   extend C smoke; fuzz. (Legacy untouched.)
3. **C++** — `meta_value` + `metadata` methods + iterators; ctest.
4. **Fortran** — `tensogram_value` + `has`/`try_get_*`/enumeration/nav +
   `num_objects`; Fortran tests.
5. **Python** — full Mapping + dot-path helpers; pytest (parity oracle).
6. **TS** — `hasMetaKey(_at)`, `getMetaKeyAt`, typed getters; vitest.
7. **Docs** — new `docs/src/guide/metadata.md` (one spec + per-language
   snippets); update `c-api.md`/`cpp-api.md`/`fortran-api.md`/
   `python-api.md`/`typescript-api.md`; CHANGELOG; doc-deprecate the
   ambiguous default getters (not removed).
8. **Parity matrix** wired into `make all` / `release-preflight`.

## 9. Resolved decisions (design review 2026-07-20)

All Phase-1 forks are settled (see §3):

1. **API shape** — ✅ `get()→optional<value>` is primary; `has` / `try_get_<T>`
   are thin conveniences over it.
2. **Coercion** — ✅ precise (no cross-type coercion) in the new API; legacy
   getters keep their coercion.
3. **Integer typing** — ✅ single `INT` type; range via `as_i64` / `as_u64`.
4. **Reserved visibility** — ✅ visible on `object(i)` enumeration, hidden
   from path getters at the first segment.
5. **Deprecation posture** — ✅ keep + doc-deprecate the default-based
   getters; schedule removal for a future MAJOR (with approval).
6. **Vocabulary** — ✅ `has`/`has_at` (C), `contains`/`contains_at`
   (C++ & Python), `get`/`get_at`, `try_get_<T>`; `tgm_value_t` /
   `meta_value` / `tensogram_value`.

**Ready for Phase 1** (core accessor + `MetaValue`/`MetaType`, refactor FFI +
CLI onto it). No blocking questions remain.
