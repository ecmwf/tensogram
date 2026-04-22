# Vocabularies

Tensogram is **vocabulary-agnostic**: the library never interprets metadata
keys. The same message can carry any combination of application-defined
namespaces alongside the auto-populated library-reserved keys. This page
collects example vocabularies that have been (or could naturally be) used
with Tensogram, so you can pick a convention that matches your domain —
or invent your own.

## How metadata is structured

A Tensogram message's per-object metadata lives in `base[i]`, a
`BTreeMap<String, ciborium::Value>`. By convention, each application
vocabulary sits under its own top-level **namespace key** so that multiple
vocabularies can coexist without collision:

```json
{
  "version": 3,
  "base": [{
    "mars":   { "class": "od", "param": "2t" },
    "cf":     { "standard_name": "air_temperature", "units": "K" },
    "custom": { "experiment": "run-042" }
  }]
}
```

All three namespaces above are valid, visible to tooling, and survive
round-trip. The library never reads or validates their contents.

## Example vocabularies

### MARS (ECMWF, weather forecasting)

Used internally at ECMWF and by downstream consumers of ECMWF's MARS archive.
Keys describe the operational provenance of a forecast field: class, stream,
type, parameter, level, date/time, step, etc.

```json
{
  "mars": {
    "class": "od", "stream": "oper", "type": "fc",
    "date": "20260401", "time": "1200", "step": 6,
    "param": "2t", "levtype": "sfc"
  }
}
```

The GRIB importer (`tensogram convert-grib`) automatically populates this
namespace from GRIB MARS keys. See
[MARS Key Mapping](../grib/metadata-mapping.md) for the full key list.

### CF Conventions (climate, ocean, atmospheric)

[CF Conventions](https://cfconventions.org/) are the standard attribute
vocabulary for climate and forecast data in NetCDF. The NetCDF importer
(`tensogram convert-netcdf --cf`) lifts the CF allow-list into a `"cf"`
sub-map. See [NetCDF CF Metadata Mapping](../reference/netcdf-cf-mapping.md).

```json
{
  "cf": {
    "standard_name": "air_temperature",
    "long_name": "2 metre temperature",
    "units": "K",
    "cell_methods": "time: mean"
  }
}
```

### BIDS (neuroimaging)

The [Brain Imaging Data Structure](https://bids.neuroimaging.io/) organises
neuroimaging datasets with entity-level metadata. A natural fit for
Tensogram messages carrying fMRI, dMRI, or EEG tensors.

```json
{
  "bids": {
    "subject": "sub-01", "session": "ses-01",
    "task": "rest", "run": 1, "acq": "hires"
  }
}
```

### DICOM (medical imaging)

[DICOM](https://www.dicomstandard.org/) tags are the standard descriptors
for medical imaging studies. They can be mapped into a `"dicom"` namespace
for use with Tensogram messages carrying imaging volumes, time-series, or
segmentation masks.

```json
{
  "dicom": {
    "Modality": "MR", "SeriesDescription": "T2_FLAIR",
    "SliceThickness": 1.0, "RepetitionTime": 8000
  }
}
```

### Zarr attributes (generic)

Zarr v3 attribute maps are generic key-value stores. When using the Zarr
backend (`tensogram-zarr`), group-level and array-level attributes are
surfaced through `_extra_` and per-array descriptor params.

### Custom namespaces

For any domain that does not have an established vocabulary, or when a
pipeline wants to carry bespoke fields alongside a standard namespace,
invent your own:

```json
{
  "experiment": {
    "id": "run-042",
    "operator": "alice",
    "hypothesis": "beam stability",
    "started_at": "2026-04-18T10:30:00Z"
  }
}
```

Suggested conventions for custom namespaces:

- Use a **short, lowercase** namespace key (`"product"`, `"instrument"`,
  `"run"`, `"experiment"`, `"device"`).
- Group related fields under a single namespace rather than scattering them
  at the top level of `base[i]`.
- Prefer **ISO 8601** timestamps, **SI units** in `units` fields, and
  **UTF-8 text** for identifiers.
- Document your namespace schema somewhere versioned (a README, a JSON
  schema, a wiki page) so downstream consumers can interpret it consistently.

## Multiple vocabularies in one message

You can freely mix vocabularies in the same `base[i]` entry — the library
preserves all of them:

```json
{
  "base": [{
    "mars":       { "param": "2t", "levtype": "sfc" },
    "cf":         { "standard_name": "air_temperature", "units": "K" },
    "provenance": { "pipeline_id": "pp-17", "stage": "post-process" }
  }]
}
```

This lets one team's producers emit messages that are simultaneously
interpretable by tools expecting MARS, CF-aware tooling, and an internal
provenance tracker.

## Looking up keys

The dotted-path helpers exposed by each binding vary. The CLI, the C FFI
(`tgm_metadata_get_string` / `_get_int` / `_get_float`), the C++ wrapper
(`metadata::get_string` / `get_int` / `get_float`), and the TypeScript
package (`getMetaKey`) all accept a full dotted path. The Rust crate and
the Python package do not expose a dotted-path helper at this time; use
direct nested access instead.

### TypeScript — dotted path

```ts
import { getMetaKey } from '@ecmwf.int/tensogram';

const param   = getMetaKey(meta, 'mars.param');
const subject = getMetaKey(meta, 'bids.subject');
```

### CLI — dotted path

```bash
# Filter messages on a namespaced key
tensogram ls data.tgm -w "mars.param=2t/10u"
tensogram ls data.tgm -w "bids.subject=sub-01"

# Print specific keys
tensogram get -p "cf.standard_name,cf.units" data.tgm
```

### Python — dict-style nested access

```python
# Metadata.__getitem__ does a top-level search across base[i] (skipping
# _reserved_) and falls back to the message-level _extra_ map. The returned
# value is a plain Python dict, so the next lookup is standard dict access.
param   = meta["mars"]["param"]
subject = meta["bids"]["subject"]

# meta.base[i], meta.reserved, and meta.extra are also available directly
# if you want the raw per-object / reserved / extra dicts.
first_base = meta.base[0]
```

### Rust — pattern-match on `ciborium::Value`

```rust
use ciborium::Value;
use tensogram::GlobalMetadata;

// `meta.base` is `Vec<BTreeMap<String, Value>>`. Find the namespace on
// the first-matching base entry, then pull a text field from the nested
// map. Falls back to `meta.extra` for message-level annotations.
fn get_text<'a>(meta: &'a GlobalMetadata,
                namespace: &str, field: &str) -> Option<&'a str> {
    let pull = |map: &'a [(Value, Value)]| -> Option<&'a str> {
        map.iter().find_map(|(k, v)| match (k, v) {
            (Value::Text(k), Value::Text(v)) if k == field => Some(v.as_str()),
            _ => None,
        })
    };
    for entry in &meta.base {
        if let Some(Value::Map(items)) = entry.get(namespace)
            && let Some(val) = pull(items)
        {
            return Some(val);
        }
    }
    if let Some(Value::Map(items)) = meta.extra.get(namespace) {
        return pull(items);
    }
    None
}

let param = get_text(&meta, "mars", "param");
```

Tensogram keeps the Rust surface small on purpose. If your pipeline needs
dotted-path lookup in Rust, wrap the snippet above in a helper of your
own, or call out to the CLI.

### Lookup semantics (all bindings that support dotted paths)

First match across `base[0]`, `base[1]`, … (skipping `_reserved_` within
each entry), then fall back to the message-level `_extra_` map. An
explicit `_extra_.key` (or `extra.key`) prefix bypasses the base search.

## See also

- [Metadata concepts](../concepts/metadata.md) — how the `base`, `_reserved_`,
  and `_extra_` sections fit together.
- [CBOR Metadata Schema](../format/cbor-metadata.md) — field-level reference.
- [Metadata Value Types](../format/metadata-values.md) — which CBOR types are
  allowed inside metadata.
- [GRIB MARS Key Mapping](../grib/metadata-mapping.md) — what the GRIB
  importer produces.
- [NetCDF CF Metadata Mapping](../reference/netcdf-cf-mapping.md) — what the
  NetCDF importer produces with `--cf`.
