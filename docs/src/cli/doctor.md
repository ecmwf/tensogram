# tensogram doctor

Report environment diagnostics: compiled-in features, backend library versions,
and a self-test of the encode/decode pipeline plus the GRIB and NetCDF converters.

## Usage

```bash
tensogram doctor [OPTIONS]
```

## All flags

| Flag | Description |
|------|-------------|
| `--json` | Machine-parseable JSON output |
| `-h, --help` | Print help |

## Output sections

### Build

Compile-time metadata: crate version, wire-format version, target triple, and
build profile (`debug` or `release`).

### Compiled-in features

One row per known feature.  Features that were not compiled in show `off`.
Features that are on show the backend library name, linkage model (`FFI` or
`pure-Rust`), and version string.

| Feature | Backend | Notes |
|---------|---------|-------|
| `szip` | libaec (FFI) | GRIB szip compression |
| `szip-pure` | tensogram-szip (pure-Rust) | Pure-Rust szip fallback |
| `zstd` | libzstd (FFI) | Zstandard compression |
| `zstd-pure` | ruzstd (pure-Rust) | Pure-Rust zstd decompressor |
| `lz4` | lz4_flex (pure-Rust) | LZ4 compression |
| `blosc2` | libblosc2 (FFI) | Blosc2 compression |
| `zfp` | libzfp (FFI) | ZFP lossy compression |
| `sz3` | SZ3 (FFI) | SZ3 lossy compression |
| `threads` | rayon (pure-Rust) | Multi-threaded pipeline |
| `remote` | object_store (pure-Rust) | Remote object store I/O |
| `mmap` | memmap2 (pure-Rust) | Memory-mapped file I/O |
| `async` | tokio (pure-Rust) | Async I/O runtime |
| `grib` | libeccodes (FFI) | GRIB converter |
| `netcdf` | libnetcdf (FFI) | NetCDF converter |

### Self-test

Runs a suite of encode/decode round-trips and converter smoke tests.  Each row
shows a label and one of:

- `ok` — test passed
- `FAILED (reason)` — test failed; the reason is shown inline
- `skipped (reason)` — feature not compiled in

## Human-readable output (default)

```
tensogram doctor
================

Build
  tensogram             0.19.0
  wire-format           3
  target                aarch64-apple-darwin
  profile               release

Compiled-in features
  szip                  on  (libaec FFI 1.1.4)
  szip-pure             off
  zstd                  on  (libzstd FFI 1.5.5)
  zstd-pure             off
  lz4                   on  (lz4_flex pure-Rust 0.11.3)
  blosc2                on  (libblosc2 FFI 2.15.0)
  zfp                   on  (libzfp FFI 1.0.1)
  sz3                   off
  threads               on  (rayon pure-Rust 1.10.0)
  remote                on  (object_store pure-Rust 0.13.0)
  mmap                  on  (memmap2 pure-Rust 0.9.5)
  async                 on  (tokio pure-Rust 1.44.2)
  grib                  off
  netcdf                off

Self-test
  encode/decode  none/none/none            ok
  decode_metadata round-trip               ok
  scan multi-message buffer                ok
  hash xxh3 verify                         ok
  pipeline       simple_packing+szip       ok
  pipeline       shuffle+zstd              ok
  pipeline       lz4                       ok
  pipeline       blosc2                    ok
  pipeline       zfp (fixed-rate)          ok
  pipeline       sz3 (absolute error)      skipped (feature 'sz3' not built in)
  convert        grib    (sanity.grib2)    skipped (feature 'grib' not built in)
  convert        netcdf3 (sanity-classic.nc) skipped (feature 'netcdf' not built in)
  convert        netcdf4 (sanity-hdf5.nc)  skipped (feature 'netcdf' not built in)

Status: HEALTHY
```

## JSON output (`--json`)

```json
{
  "build": {
    "version": "0.19.0",
    "wire_version": 3,
    "target": "aarch64-apple-darwin",
    "profile": "release"
  },
  "features": [
    {
      "name": "szip",
      "kind": "compression",
      "state": "on",
      "backend": "libaec",
      "linkage": "ffi",
      "version": "1.1.4"
    },
    {
      "name": "zstd-pure",
      "kind": "compression",
      "state": "off"
    }
  ],
  "self_test": [
    {
      "label": "encode/decode  none/none/none",
      "outcome": "ok"
    },
    {
      "label": "pipeline       shuffle+zstd",
      "outcome": "ok"
    },
    {
      "label": "convert        grib    (sanity.grib2)",
      "outcome": "skipped",
      "reason": "feature 'grib' not built in"
    }
  ]
}
```

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | All self-tests passed (or were skipped) |
| `1` | One or more self-tests failed |

A `Skipped` row does **not** cause exit code 1.

## Library API

The same diagnostics are available programmatically from every Tensogram
language binding.  All of them produce the JSON shape documented above —
the only differences are the native return type and the entry-point name.

### Rust

```rust
use tensogram::doctor::{run_diagnostics, DoctorReport};

let report: DoctorReport = run_diagnostics();
println!("version: {}", report.build.version);
for feat in &report.features {
    println!("{}: {:?}", feat.name, feat.state);
}
for row in &report.self_test {
    println!("{}: {:?}", row.label, row.outcome);
}
```

### Python

```python
import tensogram

report = tensogram.doctor()  # returns a dict
print(report["build"]["version"])
for feat in report["features"]:
    print(feat["name"], feat["state"])
```

### TypeScript / WebAssembly

```typescript
import init, { doctor } from "@ecmwf.int/tensogram";
await init();
const report = doctor();  // returns DoctorReport
console.log(report.build.version);
for (const feat of report.features) {
    console.log(feat.name, feat.state);
}
```

### C / C++

```c
#include "tensogram.h"
#include <stdio.h>

tgm_bytes_t report = {0};
if (tgm_doctor_to_json(&report) == TGM_ERROR_OK) {
    fwrite(report.data, 1, report.len, stdout);  // UTF-8 JSON
    tgm_bytes_free(report);
} else {
    fprintf(stderr, "doctor failed: %s\n", tgm_last_error());
}
```

The C FFI returns the report as a (non-null-terminated) UTF-8 JSON byte
buffer so callers can parse it with their own JSON library
(`json_loads`, `nlohmann::json::parse`, `cJSON_Parse`, etc.).

## Examples

```bash
# Default human-readable output
tensogram doctor

# Machine-parseable JSON (pipe to python for pretty-printing)
tensogram doctor --json | python3 -m json.tool

# With GRIB and NetCDF converters compiled in
cargo run -p tensogram-cli --features grib,netcdf -- doctor
```
