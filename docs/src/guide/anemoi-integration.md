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

All forecast steps are appended to a single `.tgm` file as they are produced.
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

## Message layout

Each forecast step is written as one tensogram message containing:

1. A **lat** coordinate object (`name: "grid_latitude"`)
2. A **lon** coordinate object (`name: "grid_longitude"`)
3. One **field object per variable** (or one stacked object per parameter when
   `stack_pressure_levels=True`)

Per-object metadata is stored under the `"anemoi"` namespace and the
`"mars"` namespace (MARS keys from the checkpoint, plus `date`, `time`,
`step` derived from the forecast state).

Dimension-name hints are written to `_extra_["dim_names"]` so the
`tensogram-xarray` backend can resolve axis names without explicit configuration:

```python
import tensogram_xarray  # noqa: F401
import xarray as xr

ds = xr.open_dataset("forecast.tgm", engine="tensogram")
```

## Pressure-level stacking

When `stack_pressure_levels=True`, all fields sharing the same GRIB `param`
are merged into a single 2-D object of shape `(n_grid, n_levels)`, sorted by
level ascending.  The `"anemoi"` namespace carries `"levelist": [500, 850, ...]`
instead of a scalar `"level"`.  Non-pressure-level fields are always written
as individual 1-D objects.

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

Coordinate arrays (lat/lon) are never lossy-encoded; only field arrays
are packed.
