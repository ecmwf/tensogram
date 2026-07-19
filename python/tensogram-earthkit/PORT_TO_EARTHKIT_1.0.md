# `tensogram-earthkit` on earthkit-data 1.x

Status: **ported.** The plugin requires `earthkit-data>=1.0.2` and uses the
1.x structured-component Field model throughout.  This document records what
1.0 changed and how the port maps tensogram's flat MARS metadata onto it.

## What earthkit-data 1.0 changed

1.0 replaced the flat, dict-based metadata model (`UserMetadata` +
`ArrayField`) with a **typed field-component model**: a `Field` is built from
`data` / `time` / `parameter` / `geography` / `vertical` / `ensemble` /
`labels` components, and metadata is accessed with **namespaced keys**
(`field.get("parameter.variable")`, `sel(**{"parameter.variable": "2t"})`).

| 0.x API used by the plugin | 1.0 replacement used by the port |
|---|---|
| `indexing.fieldlist.SimpleFieldList` | `indexing.simple.SimpleFieldList` |
| `sources.array_list.ArrayField(arr, md)` | `Field(data=ArrayDataFieldComponentHandler(arr), …)` |
| `utils.metadata.dict.UserMetadata(flat, shape=…)` | Mars component builders + `SimpleLabels({"mars": flat})` + `EmptyGeography(shape=…)` |
| `wrappers.get_wrapper` (encoder dispatch) | `data.wrappers.from_object` |
| `EncodedData.metadata(key)` | `EncodedData.get(key, default)` (abstract in 1.0) |
| `field.metadata("param")` | `field.get("labels.mars")["param"]` / `field.get("parameter.variable")` |
| `fl.sel(param="2t")` | `fl.sel(**{"parameter.variable": "2t"})` |
| `Reader` (concrete) | `Reader` gained an abstract `_encode_default(encoder)` |

## The mapping (reader direction)

`mars.base_entry_to_field(base_entry, values, shape)` flattens the base entry
(`extract_mars_keys`: `mars` sub-map wins over siblings) and builds:

- **parameter / vertical / ensemble** via earthkit's own
  `field.mars.{MarsParameterBuilder, MarsVerticalBuilder, MarsEnsembleBuilder}`
  — the same builders `create_mars_field` (used by gribjump) relies on — but
  **guarded**: a missing `date`, a missing/unknown `levtype`, or an unusable
  `param` simply omits that component instead of raising, because tensogram
  MARS maps are not guaranteed complete.
- **time** with a lenient date parse: GRIB-style (`20250101` / `"0000"`) via
  `datetime_from_grib`, falling back to ISO (`"2025-01-01"`) via
  `datetime_from_date_and_time`, honouring `hdate` precedence.
- **geography** as `EmptyGeography(shape=…)` so `field.shape` / `to_numpy()`
  report the grid extent (1.0's `Field.values` is flat by design).
- **labels** as `SimpleLabels({"mars": request})` — the **full flat request is
  preserved** and retrievable via `field.get("labels.mars")`, so nothing is
  lost even when a component could not be built.

We do not call `create_mars_field` directly because it raises on partial
requests (unconditional `datetime_from_grib(date, …)` and
`MarsVerticalBuilder` raising for unknown `levtype`).

## The mapping (encoder direction)

`mars.field_to_base_entry(field)`:

1. `field.get("labels.mars")` — fields built by this plugin (and by earthkit's
   own MARS machinery, e.g. gribjump) carry the full request here.
2. Fallback: the field's raw metadata (`field.get("metadata.<key>")`) for each
   canonical MARS key — covers GRIB-backed fields.

Canonical keys route to `base[i]["mars"]`, everything else to `_extra_` —
unchanged from 0.x.

## Other 1.0 integration points

- `TensogramFileReader` implements the new abstract `_encode_default`
  (FieldList for MARS files, xarray otherwise), sets `_format = "tensogram"`,
  and **holds a strong reference to its source**: 1.x `from_source` returns
  `reader.to_data_object()` and drops the source, while the base `Reader`
  only weak-refs it — without the strong ref, a bytes-backed source's temp
  file would be unlinked by its finaliser while still in use.
- `from_source("tensogram", …)` therefore returns the `TensogramData` wrapper,
  which now carries the FieldList conveniences (`sel` / `order_by` / `get` /
  `len` / `iter`), `storage_options`, and the `close()` lifecycle.
- `TensogramEncoder` implements the two additional abstract hooks
  (`_encode_featurelist`, `_encode_path` — both reject) and dispatches via
  `earthkit.data.data.wrappers.from_object`.
