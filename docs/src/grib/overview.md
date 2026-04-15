# GRIB Conversion

Tensogram provides `tensogram-grib`, a dedicated crate for converting GRIB (GRIdded Binary) messages to Tensogram format. This enables importing existing GRIB data into the Tensogram ecosystem while preserving all MARS namespace metadata.

## System Requirement

The ecCodes C library must be installed:

```bash
brew install eccodes       # macOS
apt install libeccodes-dev # Debian/Ubuntu
```

## Building

The `tensogram-grib` crate is excluded from the default workspace build to avoid requiring ecCodes on all machines.

```bash
# Build the library
cd rust/tensogram-grib && cargo build

# Build CLI with GRIB support
cargo build -p tensogram-cli --features grib
```

## Conversion Modes

### Merge All (default)

All GRIB messages are combined into a single Tensogram message with N data objects. ALL MARS keys for each GRIB message are placed into the corresponding `base[i]` entry independently — there is no common/varying partitioning in the output.

```bash
tensogram convert-grib forecast.grib -o forecast.tgm
```

### One-to-One (split)

Each GRIB message becomes a separate Tensogram message with one data object. All MARS keys go into `base[0]`.

```bash
tensogram convert-grib forecast.grib -o forecast.tgm --split
```

## Rust API

```rust
use std::path::Path;
use tensogram_grib::{convert_grib_file, ConvertOptions, Grouping};

let options = ConvertOptions {
    grouping: Grouping::MergeAll,
    ..Default::default()
};

let messages = convert_grib_file(Path::new("forecast.grib"), &options)?;
// messages is Vec<Vec<u8>> — each element is a complete Tensogram wire-format message
```

## Data Mapping

| Source (GRIB) | Target (Tensogram) |
|---------------|-------------------|
| Grid values (`values` key) | Data object payload (float64, little-endian) |
| Grid dimensions (Ni, Nj) | `DataObjectDescriptor.shape` as `[Nj, Ni]` |
| Reduced Gaussian grids (Ni=0) | Shape `[numberOfPoints]` (1D) |
| MARS keys (all, per message) | `GlobalMetadata.base[i]["mars"]` (each entry independent) |

## Scope

Currently only GRIB → Tensogram conversion is supported. Tensogram → GRIB is out of scope.

## See also

- [NetCDF Conversion](../guide/convert-netcdf.md) — sister converter for
  NetCDF files; shares the `--encoding`/`--bits`/`--filter`/`--compression`
  pipeline flags with `convert-grib`.
