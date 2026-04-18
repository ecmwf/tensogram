# NetCDF Import

Tensogram ships `tensogram-netcdf`, a dedicated crate for importing NetCDF
(both Classic and NetCDF-4) files into Tensogram messages. NetCDF is widely
used in climate, ocean, atmospheric, and Earth-observation science, but the
importer treats any NetCDF file the same way — the mapping is structural, not
domain-specific.

The crate is exposed through the CLI as `tensogram convert-netcdf` and through
a thin Rust library API. Conversion is one-way: NetCDF → Tensogram. There is
no Tensogram → NetCDF writer.

## System requirement

The NetCDF C library must be installed on your system:

```bash
brew install netcdf            # macOS
apt install libnetcdf-dev      # Debian/Ubuntu
```

The crate transitively pulls in HDF5 (used internally by NetCDF-4 files), so
on Debian-family distros you also want `libhdf5-dev`.

## Building

The `tensogram-netcdf` crate is excluded from the default workspace build to
avoid forcing libnetcdf on every contributor. Build it explicitly:

```bash
# Library
cargo build --manifest-path rust/tensogram-netcdf/Cargo.toml

# CLI with NetCDF support
cargo build -p tensogram-cli --features netcdf
```

The binary then exposes the new subcommand:

```bash
tensogram convert-netcdf --help
```

## Quick example

```bash
# Convert one file
tensogram convert-netcdf input.nc -o output.tgm

# Convert multiple files into a single output
tensogram convert-netcdf jan.nc feb.nc mar.nc -o q1.tgm

# Stream to stdout (useful for piping)
tensogram convert-netcdf input.nc | tensogram info /dev/stdin
```

## Command-line options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output PATH` | stdout | Where to write the Tensogram file. |
| `--split-by MODE` | `file` | Grouping mode: `file`, `variable`, or `record`. See [Splitting modes](#splitting-modes). |
| `--cf` | off | Extract the CF attribute allow-list into `base[i]["cf"]`. See [CF metadata mapping](reference/netcdf-cf-mapping.md). |
| `--encoding ENC` | `none` | `none` or `simple_packing`. |
| `--bits N` | auto (16) | Bits per value for `simple_packing` (1–64). |
| `--filter FILTER` | `none` | `none` or `shuffle`. |
| `--compression CODEC` | `none` | `none`, `zstd`, `lz4`, `blosc2`, or `szip`. |
| `--compression-level N` | codec default | Level for `zstd` (1–22) and `blosc2` (0–9). |

The `--encoding`/`--bits`/`--filter`/`--compression`/`--compression-level`
flags are the same set used by `tensogram convert-grib`. Both importers share
a `PipelineArgs` struct so the two commands stay symmetric.

## How variables become objects

Each numeric NetCDF variable in the root group is mapped 1:1 to a Tensogram
data object. The variable's name is stored under `base[i]["name"]`, the dtype
and shape come from the NetCDF type and dimension list, and the raw bytes
become the object payload (always little-endian).

### Dtype matrix

| NetCDF type | Tensogram `Dtype` |
|---|---|
| `byte` | `Int8` |
| `ubyte` | `Uint8` |
| `short` | `Int16` |
| `ushort` | `Uint16` |
| `int` | `Int32` |
| `uint` | `Uint32` |
| `int64` | `Int64` |
| `uint64` | `Uint64` |
| `float` | `Float32` |
| `double` | `Float64` |

`char` and `string` variables, as well as the NetCDF-4 enhanced types
(`compound`, `vlen`, `enum`, `opaque`), are skipped with a warning. They have
no clean tensor representation.

### Scalar variables

A NetCDF scalar (zero dimensions) becomes an object with `ndim = 0`,
`shape = []`, and a single value in the payload.

## Packed data

Variables with `scale_factor` or `add_offset` attributes are *unpacked* during
conversion: the raw integer values are read, multiplied by the scale, offset
applied, and the result stored as `Float64` regardless of the on-disk dtype.
This matches the convention used by xarray and most netCDF tooling.

The fill value (`_FillValue` or `missing_value`) is replaced with `NaN` in the
unpacked output. The original sentinel is preserved under
`base[i]["netcdf"]["_FillValue"]` so consumers can recover it.

## Time coordinates

Time coordinate variables are stored as numeric values (typically `Float64`)
exactly as they appear in the file — Tensogram does not convert them to
calendar dates. The CF `units` string (`"days since 1970-01-01"`) and
`calendar` (`"gregorian"`, `"noleap"`, etc.) are preserved under
`base[i]["netcdf"]` so a consumer can decode them on demand.

## NetCDF-4 groups

Tensogram extracts only the **root group** of a NetCDF-4 file. If sub-groups
are detected the importer prints a warning to stderr and continues with the
root variables. Sub-group support is intentionally out of scope for v1 — most
operational datasets keep their data variables at the root anyway.

## Splitting modes

The `--split-by` flag controls how variables are grouped into Tensogram
messages.

### `--split-by=file` (default)

All variables from one input file are bundled into a single Tensogram
message containing N data objects. This is the most compact representation
and is the right choice when you want to keep a NetCDF file as a single
logical unit.

```bash
tensogram convert-netcdf forecast.nc -o forecast.tgm
# 1 message with N objects
```

### `--split-by=variable`

Each variable becomes its own one-object Tensogram message. Useful when
downstream consumers want to fetch individual variables without decoding the
whole file.

```bash
tensogram convert-netcdf forecast.nc -o forecast.tgm --split-by variable
# N messages with 1 object each
```

### `--split-by=record`

Splits along the unlimited (record) dimension. Each step along the unlimited
dimension produces a separate message. The unlimited dimension is detected
automatically; passing this mode against a file without one is a hard error
(`NoUnlimitedDimension`).

Variables that don't depend on the unlimited dimension (e.g. a static `mask`
variable) are still included in *every* output message — that way each
record is fully self-describing.

```bash
tensogram convert-netcdf timeseries.nc -o timeseries.tgm --split-by record
# 1 message per record
```

## Encoding pipeline flags

The pipeline flags are applied per data object before encoding into the
wire format. They use the same names and semantics as `convert-grib`:

| Stage | Flag | Notes |
|---|---|---|
| Encoding | `--encoding simple_packing --bits N` | Lossy quantization. **Float64 only** — non-`f64` variables in the same file are skipped (with a warning) and pass through unencoded so mixed files convert cleanly. |
| Filter | `--filter shuffle` | Byte-shuffle filter, sets `shuffle_element_size` to the post-encoding byte width. |
| Compression | `--compression zstd --compression-level 3` | `zstd_level` defaults to 3. |
| Compression | `--compression lz4` | No params. |
| Compression | `--compression blosc2 --compression-level 9` | Uses `blosc2_codec=lz4` by default. |
| Compression | `--compression szip` | Sets `szip_rsi=128`, `szip_block_size=16`, `szip_flags=8`. **Requires** preceding `simple_packing` or `shuffle` because libaec szip caps at 32 bits per sample (raw `f64` is 64 bits). |

Variables that contain `NaN` (typically from unpacked fill values) cannot be
`simple_packed` because the algorithm rejects NaN inputs. Such variables are
skipped (with a warning) and pass through with `encoding=none`.

```bash
# Pack temperature to 24-bit + zstd
tensogram convert-netcdf --encoding simple_packing --bits 24 \
  --compression zstd --compression-level 3 \
  era5_t2m.nc -o era5_t2m.tgm

# Shuffle + szip on a multi-variable file
tensogram convert-netcdf --filter shuffle --compression szip \
  forecast.nc -o forecast.tgm
```

## CF metadata mapping

NetCDF attributes are always extracted into a `netcdf` sub-map under each
base entry:

```text
base[0]:
  name: "temperature"
  netcdf:
    units: "K"
    long_name: "Air Temperature"
    standard_name: "air_temperature"
    _FillValue: -32768
    add_offset: 273.15
    scale_factor: 0.01
    _global:
      Conventions: "CF-1.10"
      title: "..."
      institution: "..."
```

When `--cf` is set, an additional `cf` sub-map is added containing only the
[16 CF allow-list attributes](reference/netcdf-cf-mapping.md). This duplicate
copy makes CF-aware tooling cheaper because it can ignore the verbose
`netcdf` map and rely on a stable, standardised key set.

## Limitations

- **No NetCDF writer.** Conversion is one-way only.
- **No string or char variables.** They are skipped with a warning.
- **No NetCDF-4 enhanced types** (`compound`, `vlen`, `enum`, `opaque`).
- **Root group only.** Sub-groups are skipped with a warning.
- **No `tensogram-python` bindings.** The Python ecosystem talks to
  `convert-netcdf` through `subprocess`. The library API is Rust-only in v1.
- **`simple_packing` is `f64`-only.** Mixed-dtype files convert cleanly but
  only `f64` variables get packed.

## Library API

If you'd rather call the importer directly from Rust:

```rust
use std::path::Path;
use tensogram_netcdf::{convert_netcdf_file, ConvertOptions, DataPipeline, SplitBy};

let options = ConvertOptions {
    split_by: SplitBy::Variable,
    cf: true,
    pipeline: DataPipeline {
        encoding: "simple_packing".to_string(),
        bits: Some(24),
        compression: "zstd".to_string(),
        compression_level: Some(3),
        ..Default::default()
    },
    ..Default::default()
};

let messages = convert_netcdf_file(Path::new("forecast.nc"), &options)?;
// messages: Vec<Vec<u8>> — each element is a complete wire-format message
```

**Note:** `DataPipeline` is defined in `tensogram::pipeline` and
re-exported from both `tensogram_netcdf` and `tensogram_grib`. The
underlying `apply_pipeline` helper is the same for both importers,
guaranteeing that `convert-grib` and `convert-netcdf` produce
byte-identical descriptor fields for equivalent flag combinations.

## See also

- [GRIB Import](../grib/overview.md) — sister importer with the same
  pipeline-flag semantics.
- [Simple Packing](../encodings/simple-packing.md), [Shuffle](../encodings/shuffle.md), [Compression](../encodings/compression.md) — the encoding stages applied to each object.
- [CF Metadata Mapping](../reference/netcdf-cf-mapping.md) — full table of the
  16 attributes lifted by `--cf`.
