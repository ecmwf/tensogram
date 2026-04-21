# anemoi-inference Integration

The `tensogram-anemoi` package provides a plug-and-play output for
[anemoi-inference](https://github.com/ecmwf/anemoi-inference), the ECMWF framework
for running AI-based weather forecast models.  Once installed, anemoi-inference
automatically discovers the plugin via Python entry points — no code changes to
anemoi-inference are required.

## Installation

```bash
pip install tensogram-anemoi
```

Or from source:

```bash
pip install -e python/tensogram-anemoi/
```

## Usage

In an anemoi-inference run config, specify `tensogram` as the output:

```yaml
output:
  tensogram:
    path: forecast.tgm
```

All forecast steps are written to a single `.tgm` file as they are produced.
Remote destinations (S3, GCS, Azure, ...) are supported via fsspec:

```yaml
output:
  tensogram:
    path: s3://my-bucket/forecast.tgm
    storage_options:
      key: ...
      secret: ...
```

## Configuration options

All options after `path` must be supplied as keyword arguments.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | `str` | — | Destination file path or remote URL |
| `encoding` | `str` | `"none"` | `"none"` or `"simple_packing"` |
| `bits` | `int` | `None` | Bits per value (required when `encoding="simple_packing"`) |
| `compression` | `str` | `"zstd"` | `"none"`, `"zstd"`, `"lz4"`, `"szip"`, `"blosc2"` |
| `dtype` | `str` | `"float32"` | Field array dtype: `"float32"` or `"float64"` |
| `storage_options` | `dict` | `{}` | Forwarded to fsspec for remote paths |
| `stack_pressure_levels` | `bool` | `False` | Stack pressure-level fields into 2-D objects |
| `variables` | `list[str]` | `None` | Restrict output to a subset of variables |
| `output_frequency` | `int` | `None` | Write every N steps |
| `write_initial_state` | `bool` | `None` | Whether to write step 0 |

## Pressure-level stacking

When `stack_pressure_levels=True`, all fields sharing the same GRIB `param`
are merged into a single 2-D object of shape `(n_grid, n_levels)`, sorted by
level ascending.  The `"mars"` namespace carries `"levelist": [500, 850, ...]`
instead of a scalar `"level"` (following standard MARS convention).
Non-pressure-level fields are always written as individual 1-D objects.

```yaml
output:
  tensogram:
    path: forecast.tgm
    stack_pressure_levels: true
```

## Simple packing

For compact storage, use `simple_packing` with a `bits` value:

```yaml
output:
  tensogram:
    path: forecast.tgm
    encoding: simple_packing
    bits: 16
    compression: zstd
```

Coordinate arrays (lat/lon) are never lossy-encoded; only field arrays are packed.

---

## Metadata reference

Each `.tgm` file produced by `tensogram-anemoi` contains one message per forecast
step.  This section documents exactly what is stored in each message and how to
read it with the raw tensogram Python API.

### Opening a file

```python
import tensogram

tgm = tensogram.TensogramFile.open("forecast.tgm")
print(len(tgm), "steps")

meta, objects = tgm[0]   # first step
```

`meta` is the decoded message metadata.  `objects` is a list of
`(descriptor, array)` pairs, one entry per object in the message.

### Object layout

Every message has the following fixed layout:

| Index | `base[i]["name"]` | Content |
|-------|-------------------|---------|
| 0 | `"grid_latitude"` | Latitude coordinates, float64, shape `(n_grid,)` |
| 1 | `"grid_longitude"` | Longitude coordinates, float64, shape `(n_grid,)` |
| 2 … N | variable name or param name | Field data |

```python
meta, objects = tgm[0]

lat_desc, lat_arr = objects[0]   # latitudes
lon_desc, lon_arr = objects[1]   # longitudes
fld_desc, fld_arr = objects[2]   # first field
```

The coordinate names `"grid_latitude"` and `"grid_longitude"` are intentionally
distinct from the standard `"latitude"` / `"longitude"` names so that all objects
in a message share a single flat grid dimension rather than each coordinate
spawning its own dimension.

### `base[i]` — per-object metadata

Each object has a corresponding entry in `meta.base`:

```python
for i, entry in enumerate(meta.base):
    print(i, entry)
```

Every entry contains:

| Key | Type | Present on | Description |
|-----|------|-----------|-------------|
| `"name"` | `str` | all objects | Variable or coordinate name |
| `"anemoi"` | `dict` | all objects | anemoi-specific metadata (see below) |
| `"mars"` | `dict` | field objects only | MARS metadata (see below) |

#### `"anemoi"` namespace

| Key | Type | Present on | Description |
|-----|------|-----------|-------------|
| `"variable"` | `str` | all objects | Internal anemoi-inference variable name |

For coordinates, `"variable"` is `"latitude"` or `"longitude"` (the canonical
name, not the `"grid_*"` name stored in `"name"`):

```python
assert meta.base[0]["name"] == "grid_latitude"
assert meta.base[0]["anemoi"]["variable"] == "latitude"

assert meta.base[1]["name"] == "grid_longitude"
assert meta.base[1]["anemoi"]["variable"] == "longitude"
```

For fields, `"variable"` is the internal anemoi-inference name (e.g. `"t500"`
for 500 hPa temperature, `"2t"` for 2 m temperature):

```python
assert meta.base[2]["anemoi"]["variable"] == "2t"
```

#### `"mars"` namespace

Coordinate objects carry no `"mars"` key.  Every field object carries a `"mars"`
dict combining keys from the anemoi-inference checkpoint with the temporal keys
derived from the forecast state:

**Temporal keys** (present on every field object):

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `"date"` | `str` | Analysis/base date (`YYYYMMDD`) | `"20240101"` |
| `"time"` | `str` | Analysis/base time (`HHMM`) | `"0000"` |
| `"step"` | `int` or `float` | Forecast lead time in hours | `6`, `1.5` |

**Checkpoint keys** (present when available in the model checkpoint):

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `"param"` | `str` | GRIB parameter short name | `"2t"`, `"t"`, `"u"` |
| `"levtype"` | `str` | Level type | `"sfc"`, `"pl"`, `"ml"` |
| `"level"` | `int` | Pressure level (unstacked fields only) | `500` |
| `"levelist"` | `list[int]` | Pressure levels (stacked fields only) | `[500, 850, 1000]` |

Reading field metadata:

```python
meta, objects = tgm[0]

# Surface field (e.g. 2 m temperature)
entry = meta.base[2]
print(entry["name"])                    # "2t"
print(entry["anemoi"]["variable"])      # "2t"
print(entry["mars"]["param"])           # "2t"
print(entry["mars"]["date"])            # "20240101"
print(entry["mars"]["time"])            # "0000"
print(entry["mars"]["step"])            # 6

# Pressure-level field (unstacked)
entry = meta.base[3]
print(entry["mars"]["param"])           # "t"
print(entry["mars"]["levtype"])         # "pl"
print(entry["mars"]["level"])           # 500
```

With `stack_pressure_levels=True`, the pressure-level group has `"levelist"`
instead of `"level"`, and the array is 2-D:

```python
entry = meta.base[2]                    # stacked t group
print(entry["mars"]["levelist"])        # [500, 850, 1000]
print(entry["mars"]["param"])           # "t"

desc, arr = objects[2]
print(arr.shape)                        # (n_grid, 3)  — columns sorted by level
```

### `meta.extra` — message-level metadata

`meta.extra` carries metadata that applies to the whole message rather than
individual objects.

#### `"dim_names"` — axis-size hints

```python
dim_names = meta.extra["dim_names"]
# e.g. {"21600": "values"}
# or   {"21600": "values", "3": "level"}  (with stack_pressure_levels=True)
```

`dim_names` maps the string representation of an axis length to a semantic
name.  It exists to allow downstream tools to assign meaningful axis names
without requiring any anemoi-specific knowledge.  The grid axis is always
labelled `"values"`; when pressure-level stacking is enabled, each unique
level-axis size is labelled `"level"`.

### Object descriptors

Each `(descriptor, array)` pair returned by `objects[i]` gives low-level
encoding detail:

```python
desc, arr = objects[2]

print(desc.dtype)        # "float32" or "float64"
print(desc.shape)        # [n_grid] for flat, [n_grid, n_levels] for stacked
print(desc.encoding)     # "none" or "simple_packing"
print(desc.compression)  # "zstd", "lz4", etc.
```

Coordinate arrays are always `float64` regardless of the `dtype` setting.
Field arrays use the configured `dtype` (`"float32"` by default), promoted to
`float64` automatically when `encoding="simple_packing"`.

### Full inspection example

```python
import tensogram

tgm = tensogram.TensogramFile.open("forecast.tgm")

for step_idx, (meta, objects) in enumerate(tgm):
    print(f"\n--- step {step_idx} ---")

    # Dimension hints
    print("dim_names:", meta.extra.get("dim_names", {}))

    for i, entry in enumerate(meta.base):
        desc, arr = objects[i]
        anemoi = entry.get("anemoi", {})
        mars = entry.get("mars", {})

        print(
            f"  [{i}] name={entry['name']!r:20s}"
            f"  variable={anemoi.get('variable')!r:10s}"
            f"  shape={arr.shape}"
            f"  dtype={desc.dtype}"
            + (f"  step={mars.get('step')}" if mars else "")
        )
```

Example output for a single step with surface fields and stacked pressure levels:

```
--- step 0 ---
dim_names: {'21600': 'values', '3': 'level'}
  [0] name='grid_latitude'    variable='latitude'   shape=(21600,)  dtype=float64
  [1] name='grid_longitude'   variable='longitude'  shape=(21600,)  dtype=float64
  [2] name='2t'               variable='2t'         shape=(21600,)  dtype=float32  step=6
  [3] name='t'                variable='t'          shape=(21600, 3)  dtype=float32  step=6
  [4] name='u'                variable='u'          shape=(21600, 3)  dtype=float32  step=6
```
