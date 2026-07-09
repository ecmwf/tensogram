# Porting `tensogram-earthkit` to earthkit-data 1.0

Status: **investigation complete, port in progress.** The plugin is currently
pinned to `earthkit-data>=0.19,<1.0` (see `pyproject.toml`); this document
records what 1.0 changed and the concrete migration required to lift the cap.

## Why it is not a mechanical change

earthkit-data 1.0 replaced the flat, dict-based metadata model (`UserMetadata`
+ `ArrayField`) with a **strict, structured field-component model**. The APIs
the plugin depends on were removed or moved:

| 0.x import | 1.0 status |
|---|---|
| `earthkit.data.indexing.fieldlist.SimpleFieldList` | moved → `earthkit.data.indexing.simple` |
| `earthkit.data.sources.array_list.ArrayField` | **removed** |
| `earthkit.data.utils.metadata.dict.UserMetadata` | **removed** (whole `utils.metadata` package gone) |
| `earthkit.data.wrappers.get_wrapper` | **removed** (whole `wrappers` package gone) |
| `earthkit.data.readers.Reader` | present |
| `earthkit.data.sources.file.FileSource` | present |
| `earthkit.data.encoders.{Encoder,EncodedData}` | present |

## The hard part: flat MARS metadata → structured components

A field is now built with `Field.from_components(values=..., time=, parameter=,
geography=, vertical=, ensemble=, proc=, labels=)`, each component a small
typed object (or a `dict` the component parses). Empirically (earthkit-data
1.0.2):

- `parameter` accepts `{"param": ...}` but rejects `shortName` / `paramId` /
  `id` / `name`.
- `time` needs `{"base_datetime": ISO, "step": int}` — not MARS `date`/`time`/
  `step`.
- `vertical` rejected every MARS key tried (`levtype`, `level`, `type`).
- `ensemble` needs `{"member": ...}` — not MARS `number`.
- `geography` accepts `{"shape": ...}`.
- `labels` accepts an arbitrary dict, **but** keys placed there are **not**
  retrievable via `field.metadata("param")` (it raises `KeyError:
  metadata.param not found`), so `labels` is not a drop-in for the old flat
  metadata — `sel(param=...)` / `metadata("param")` would not work.
- Even for accepted construction keys, the *read-back* key names differ from the
  construction keys (e.g. `parameter={"param":"2t"}` does not read back via
  `metadata("param")`), so both directions of the mapping must be worked out
  against the component classes.

There is currently **no** documented flat-dict path that preserves
`metadata(key)` / `sel(...)` semantics, which the plugin (and its 140 tests)
rely on.

## Migration plan (per file)

1. **`mars.py`** — replace `base_entry_to_usermetadata` with a builder that maps
   the flat MARS dict into the structured components, and rewrite
   `field_to_base_entry` to read them back. Needs an explicit, tested MARS↔
   component key table derived from the component classes
   (`earthkit.data.field.component.{parameter,time,vertical,ensemble,geography}`).
   Keys with no component home (`class`, `stream`, `expver`, `type`, `domain`,
   …) need a decided carrier (labels, or a MARS-namespace metadata object) that
   still round-trips.
2. **`fieldlist.py`** — swap the `SimpleFieldList` import to
   `earthkit.data.indexing.simple`; replace `ArrayField(arr, metadata)` with
   `Field.from_components(values=arr, **components)`; keep the
   `TensogramSimpleFieldList` subclass + `to_xarray` override.
3. **`encoder.py`** — replace `earthkit.data.wrappers.get_wrapper` (removed);
   confirm the `Encoder`/`EncodedData` 1.0 signatures still match; adapt
   `field_to_base_entry` usage to the new metadata read interface.
4. **`readers/file.py`, `source.py`** — `Reader` / `FileSource` still exist;
   verify their 1.0 constructor/override surface is unchanged (likely
   low-touch).
5. **`pyproject.toml` + CI** — lift the `<1.0` cap and drop the `earthkit-data
   <1.0` pins in `.github/workflows/ci.yml` (Linux + macOS earthkit steps).

## Recommended next step

Derive the MARS↔component key table by reading the component classes, land
`mars.py` + `fieldlist.py` first behind the existing tests, then the encoder.
Because 1.0's metadata semantics differ, expect some tests to need updating (not
just the code) — those changes should be reviewed against what MARS consumers
actually need. Consider confirming the intended flat-metadata→component mapping
with the earthkit-data maintainers to avoid guessing at the private component
key names.
