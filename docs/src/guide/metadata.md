# Reading Metadata

Every tensogram message carries a CBOR **metadata frame**. This page is the
single, language-neutral contract for reading it. The same capabilities —
existence checks, typed access, key enumeration, and nested/array navigation —
are available in **Rust, C, C++, Python, TypeScript, and Fortran**, with
matching semantics.

## The model

Metadata has three sections:

- **`base`** — a list, one entry per data object. `base[i]` holds *all*
  metadata for the *i*-th object, including the library-managed
  `_reserved_` block (tensor shape/dtype/provenance).
- **`_extra_`** — client-writable, message-level annotations.
- **`_reserved_`** — library internals (read-only).

Access comes in two shapes:

- **Get-optional** — a single lookup that returns *either* nothing (the key is
  absent) *or* a **value cursor**. Because "absent" is a distinct answer, a
  stored `0` or `""` is never confused with a missing key — unlike the older
  default-returning getters.
- **A value cursor** carries the value's *type* and typed extractors, and can
  descend into nested maps and index arrays.

### Path grammar and scoping

Lookups take a **dot-path** (`"mars.class"`, `"geometry.gridType"`). Dotted
segments navigate **maps**; array elements are reached through the cursor
(by index), not by numeric path segments.

| Scope | Behaviour |
|---|---|
| **Message-level** (`get` / `has`) | First match across `base[0]`, `base[1]`, … (skipping `_reserved_` at the first segment), then fall back to `_extra_`. An explicit `extra.` / `_extra_.` prefix targets `_extra_`. |
| **Per-object** (`get_at` / `has_at`, index `i`) | Scoped to `base[i]` only — no cross-object match, no `_extra_` fallback, `_reserved_` hidden at the first segment. |

`_reserved_` is **hidden from path getters** at the first segment, but **visible
when you enumerate** a `base[i]` section (so a viewer can show everything).

### Type contract (no coercion)

The precise accessors do **not** coerce across types:

- string accessor → text values only,
- integer accessor → integer values only (with `i64` / `u64` range checks),
- float accessor → float, or an integer widened to `f64`,
- bool / bytes → exact type only.

"Wrong type" is reported **distinctly** from "absent". (The older
`get_string` / `get_int` / `get_float` getters keep their historical coercion
and are documented as convenience shortcuts.)

> **Note — the `version` pseudo-key.** The wire-format version lives in the
> message preamble, not the CBOR frame. Use the dedicated version accessor
> (`meta.version`, `tgm_metadata_version`, …), not a metadata lookup.

## By language

All examples answer the same questions against a decoded metadata handle:
*does `mars.class` exist?*, *read it as a string*, *walk object 0's keys*,
*index an array*.

### Rust

```rust
use tensogram::MetaType;

// meta: &tensogram::GlobalMetadata
if meta.contains("mars.class") {
    if let Some(v) = meta.get("mars.class") {
        let class = v.as_str();          // Option<&str>, no coercion
    }
}

// per-object, scoped to base[0]
let grid = meta.get_at(0, "geometry.gridType").and_then(|v| v.as_str());

// enumerate object 0 (includes _reserved_)
if let Some(obj) = meta.object(0) {
    for i in 0..obj.len() {
        let key = obj.key_at(i);         // Option<&str>
    }
}

// arrays — reached through the cursor (dot-paths navigate maps only)
if let Some(a) = meta.get_at(0, "levels") {
    if a.value_type() == MetaType::Array {
        let first = a.get_index(0).and_then(|v| v.as_i64());
    }
}

// _reserved_ is hidden from path getters; reach it by enumerating object(0)
if let Some(shape) = meta.object(0)
    .and_then(|o| o.get_key("_reserved_"))
    .and_then(|r| r.get_key("tensor"))
    .and_then(|t| t.get_key("shape"))
{
    let first_dim = shape.get_index(0).and_then(|v| v.as_i64());
}
```

### C

Value handles are **borrowed** — never freed by the caller, valid until
`tgm_metadata_free`. Strings/bytes come back as `ptr + len` (not
NUL-terminated).

```c
if (tgm_metadata_has(meta, "mars.class")) {
    const tgm_value_t *v = tgm_metadata_get(meta, "mars.class");
    size_t len = 0;
    const char *s = tgm_value_as_string(v, &len);   /* NULL if not a string */
    /* use s[0..len) */
}

int64_t out;
const tgm_value_t *level = tgm_metadata_get_at(meta, 0, "level");  /* fetch once */
if (level && tgm_value_as_i64(level, &out)) {
    /* out is set only on success */
}

/* enumerate object 0 (includes _reserved_) */
const tgm_value_t *obj = tgm_metadata_object(meta, 0);
for (size_t i = 0; i < tgm_value_map_len(obj); i++) {
    size_t klen = 0;
    const char *key = tgm_value_map_key_at(obj, i, &klen);
    const tgm_value_t *val = tgm_value_map_value_at(obj, i);
}
```

### C++

```cpp
if (meta.contains("mars.class")) {
    if (auto cls = meta.try_get_string("mars.class"))   // std::optional<std::string_view>
        use(*cls);
}

// cursor navigation + range-for over arrays
if (auto levels = meta.get_at(0, "levels"))
    for (auto e : *levels)
        if (auto n = e.as_int()) use(*n);

// enumerate object 0
auto obj = meta.object(0);
for (std::size_t i = 0; i < obj.size(); i++)
    auto key = obj.key_at(i);       // std::optional<std::string_view>
```

### Python

`Metadata` is a `collections.abc.Mapping`; `base` / `extra` / `reserved` are
plain `dict`s.

```python
meta = tensogram.decode_metadata(buf)

"mars" in meta                       # membership (message-level, first-match)
meta.get("mars")                     # None if absent
meta.get_path("mars.class")          # dot-path, None if absent
meta.has_path_at(0, "geometry.gridType")
for key in meta:                     # iterate message-level keys
    ...

# native dict/list access to a single object (includes _reserved_)
obj0 = meta.base[0]
shape = obj0["_reserved_"]["tensor"]["shape"]   # a list
```

### TypeScript

Works on the decoded `GlobalMetadata` object; absent is `undefined`.

```ts
import { hasMetaKey, getMetaKey, getMetaKeyAt, getMetaString } from '@ecmwf.int/tensogram';

hasMetaKey(meta, 'mars.class');                 // boolean
getMetaString(meta, 'mars.class');              // string | undefined (no coercion)
getMetaKeyAt(meta, 0, 'geometry.gridType');     // per-object scoped

// native access
const shape = meta.base?.[0]?._reserved_?.tensor?.shape;
```

### Fortran

```fortran
type(tensogram_value) :: v, cls
integer(c_int64_t) :: lvl
character(len=:), allocatable :: str
integer :: i
logical :: ok

if (tensogram_metadata_has(meta, "mars.class")) then
   ok = tensogram_metadata_try_get_string(meta, "mars.class", str)   ! ok = found
end if

! per-object (optional obj_index; 1-based)
ok = tensogram_metadata_try_get_int(meta, "level", lvl, obj_index=1)

! cursor navigation (chain via intermediate variables)
v = tensogram_metadata_get(meta, "mars")
cls = v%get("class")
ok = cls%as_string(str)

! enumerate object 1 (includes _reserved_)
v = tensogram_metadata_object(meta, 1)
do i = 1, v%len()
   print *, v%key(i)
end do
```

## Legacy getters

The earlier accessors remain for compatibility but **cannot distinguish a
missing key from a real value equal to the default**, so prefer the accessors
above:

- C: `tgm_metadata_get_string/int/float(_at)`
- C++: `metadata::get_string/get_int/get_float(_at)`
- Fortran: `tensogram_metadata_get_string/int/float`

They keep their historical string coercion and default-return behaviour and are
scheduled for removal in a future major release.
