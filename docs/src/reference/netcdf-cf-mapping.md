# NetCDF CF Metadata Mapping

When `tensogram convert-netcdf --cf` is set, the importer walks each
NetCDF variable and lifts a fixed set of 16 [CF Conventions
v1.10](https://cfconventions.org/cf-conventions/v1.10/cf-conventions.html)
attributes into a `cf` sub-map under the corresponding `base[i]` entry. The
attributes are also still present in the verbose `netcdf` map alongside
every other variable attribute — the `cf` map is a curated, schema-stable
view that CF-aware tooling can rely on.

The allow-list lives in [`rust/tensogram-netcdf/src/metadata.rs`](https://github.com/ecmwf/tensogram/blob/main/rust/tensogram-netcdf/src/metadata.rs)
as the constant `CF_ATTRIBUTES`. If you change the list, update this page
to match.

## Attributes lifted by `--cf`

| CF Attribute | Tensogram Key | Notes |
|---|---|---|
| `standard_name` | `base[i]["cf"]["standard_name"]` | CF standard name from the [CF Standard Name Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html), e.g. `"air_temperature"`, `"eastward_wind"`. |
| `long_name` | `base[i]["cf"]["long_name"]` | Free-form descriptive label, e.g. `"2 metre temperature"`. |
| `units` | `base[i]["cf"]["units"]` | UDUNITS-compliant string, e.g. `"K"`, `"m s-1"`, `"days since 1970-01-01"`. |
| `calendar` | `base[i]["cf"]["calendar"]` | Calendar for time coordinate variables, e.g. `"gregorian"`, `"noleap"`, `"360_day"`. |
| `cell_methods` | `base[i]["cf"]["cell_methods"]` | Aggregation description, e.g. `"time: mean"`, `"area: sum"`. |
| `coordinates` | `base[i]["cf"]["coordinates"]` | Space-separated list of auxiliary coordinate variable names, e.g. `"lon lat"`. |
| `axis` | `base[i]["cf"]["axis"]` | Dimension role flag: `"X"`, `"Y"`, `"Z"`, or `"T"`. |
| `positive` | `base[i]["cf"]["positive"]` | Direction of vertical coordinate: `"up"` (altitude) or `"down"` (depth/pressure). |
| `valid_min` | `base[i]["cf"]["valid_min"]` | Minimum valid value for QA/range checks. |
| `valid_max` | `base[i]["cf"]["valid_max"]` | Maximum valid value for QA/range checks. |
| `valid_range` | `base[i]["cf"]["valid_range"]` | Two-element array `[min, max]` — alternative to `valid_min`/`valid_max`. |
| `bounds` | `base[i]["cf"]["bounds"]` | Name of an associated cell-bounds variable (irregular grids). |
| `grid_mapping` | `base[i]["cf"]["grid_mapping"]` | Name of an associated coordinate reference system variable. |
| `ancillary_variables` | `base[i]["cf"]["ancillary_variables"]` | Space-separated list of related ancillary variable names (uncertainty, QA flags, etc.). |
| `flag_values` | `base[i]["cf"]["flag_values"]` | Array of integer flag values for categorical variables. |
| `flag_meanings` | `base[i]["cf"]["flag_meanings"]` | Space-separated list of meanings, paired with `flag_values`. |

That's 16 attributes — the full CF allow-list as of v0.7.0.

## Storage layout

For a CF-compliant temperature variable, the `--cf` flag produces:

```text
base[0]:
  name: "temperature"
  netcdf:
    units: "K"
    long_name: "2 metre temperature"
    standard_name: "air_temperature"
    _FillValue: -32768
    add_offset: 273.15
    scale_factor: 0.01
    cell_methods: "time: mean"
    _global:
      Conventions: "CF-1.10"
      title: "ERA5 reanalysis"
  cf:
    units: "K"
    long_name: "2 metre temperature"
    standard_name: "air_temperature"
    cell_methods: "time: mean"
```

The `netcdf` map is a verbatim dump of every variable attribute (the
`_global` sub-map carries the file-level attributes). The `cf` map is a
filtered slice containing only the allow-listed keys, in the order they
appear on the variable.

## What is **not** extracted

The allow-list is intentionally narrow. The following CF concepts are out
of scope for v0.7.0 — they are accessible via the verbose `netcdf` map but
not surfaced under `cf`:

- **Grid mapping variable contents** — only the `grid_mapping` *reference*
  is lifted, not the projection parameters of the referenced variable.
- **Coordinate variable contents** — coordinate variables are converted
  to their own data objects, not inlined into other variables' metadata.
- **Bounds variable contents** — only the `bounds` reference is lifted.
- **Cell measures** — `cell_measures` is not in the allow-list.
- **Climatology bounds** — `climatology` is not lifted.
- **Geometry containers** — CF 1.8+ geometries are out of scope.
- **Labels and string-valued auxiliary coordinates** — not in the allow-list.
- **Compound coordinates / `compress`** — ragged-array support is out of scope.

If you need these, read the raw NetCDF metadata from `base[i]["netcdf"]`
instead — every original attribute is preserved there, byte-for-byte.

## Why a curated allow-list?

Two reasons:

1. **Schema stability.** Downstream tooling (xarray engines, dashboards,
   indexers) wants to rely on a small, fixed key set without having to
   inspect every NetCDF file's variable-attribute zoo. The `cf` map gives
   them that contract.
2. **Interop friendliness.** The 16 allow-listed attributes are the ones
   that show up in essentially every CF-compliant climate or weather
   dataset. They are the lingua franca that makes CF data interoperable.

If you have a strong case for adding an attribute, file an issue on the
GitHub project and we'll evaluate it.

## Related

- [CF Conventions §3](https://cfconventions.org/cf-conventions/v1.10/cf-conventions.html#description-of-the-data) — variable attributes.
- [CF Conventions §8](https://cfconventions.org/cf-conventions/v1.10/cf-conventions.html#reduction-of-dataset-size) — packed data, scale_factor / add_offset.
- [CF Standard Name Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html) — the controlled vocabulary referenced by `standard_name`.
- [NetCDF Import](../guide/convert-netcdf.md) — main user guide for
  `tensogram convert-netcdf`.
