# Rust Examples

Runnable Rust examples for the [`tensogram`](../../rust/tensogram) crate
and its workspace siblings.  Each example is a stand-alone `[[bin]]`
target in this crate (`tensogram-rust-examples`) — clone the repo,
build once, run each one directly.

## Quick start

```bash
# From the repo root. Default feature set (no netcdf / no remote).
cargo run --release -p tensogram-rust-examples --bin 01_encode_decode
```

For the two feature-gated examples:

```bash
# 12_convert_netcdf requires system libnetcdf.
cargo run --release -p tensogram-rust-examples --features netcdf --bin 12_convert_netcdf

# 14_remote_access requires the `remote` feature (object_store + reqwest).
cargo run --release -p tensogram-rust-examples --features remote  --bin 14_remote_access
```

To build every example at once:

```bash
cargo build --release -p tensogram-rust-examples --features netcdf,remote
```

## Examples

| File | Topic |
|------|-------|
| `01_encode_decode.rs` | Basic encode/decode round-trip of a 2-D `f32` grid |
| `02_mars_metadata.rs` | Per-object metadata using the ECMWF MARS vocabulary |
| `02b_generic_metadata.rs` | Per-object metadata with a custom application namespace |
| `03_simple_packing.rs` | Simple-packing encoding for lossy integer quantisation (GRIB-style) |
| `04_shuffle_filter.rs` | Byte shuffle filter for better float compressibility |
| `05_multi_object.rs` | Multiple tensors in one message, each with its own descriptor |
| `06_hash_verification.rs` | Inline hash slots and `validate_message` for corruption detection |
| `07_scan_buffer.rs` | Scanning a multi-message buffer with `scan()` |
| `08_decode_variants.rs` | `decode` / `decode_metadata` / `decode_object` / `decode_range` |
| `09_file_api.rs` | `TensogramFile` — create, append, open, index, iterate |
| `10_iterators.rs` | Iterator APIs: `messages()`, file iteration, object iteration |
| `11_encode_pre_encoded.rs` | Pre-encoded payloads (GPU pipeline pattern) |
| `11_streaming.rs` | `StreamingEncoder` — progressive encode to any `io::Write` |
| `12_convert_netcdf.rs` | NetCDF → Tensogram via `tensogram-netcdf` (requires `--features netcdf`) |
| `13_validate.rs` | Structural, integrity, and fidelity validation at four levels |
| `14_remote_access.rs` | Opening a `.tgm` file over HTTP with a self-contained Range-capable server (requires `--features remote`) |
| `16_multi_threaded_pipeline.rs` | Caller-controlled `threads=N` encode/decode with determinism invariants |
| `18_remote_scan_trace.rs` | Subscribe to `tensogram::remote_scan` tracing events while running forward-only and bidirectional walkers (requires `--features remote`) |

> Two bins share the `11_` prefix (`11_encode_pre_encoded` for the
> pre-encoded payload API, `11_streaming` for the progressive
> streaming encoder).  This mirrors the two unrelated topics covered
> at the same slot in the Python and TypeScript example sets.

For **narrative walk-throughs** with live plots and prose explanations,
see the companion notebooks under `../jupyter/`.

## System dependencies

- **`netcdf` feature**: `libnetcdf` + `libhdf5`.
  `apt install libnetcdf-dev` on Debian/Ubuntu; `brew install netcdf hdf5` on macOS.
- **`remote` feature**: no system dependencies — the object_store + rustls stack is pure Rust.

## Next steps

- Full API reference: <https://sites.ecmwf.int/docs/tensogram/main/>
- Rust user guide: <https://sites.ecmwf.int/docs/tensogram/main/guide/rust-api.html>
- Wire-format spec: [`plans/WIRE_FORMAT.md`](../../plans/WIRE_FORMAT.md)
