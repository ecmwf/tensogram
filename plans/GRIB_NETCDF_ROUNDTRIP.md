# GRIB & NetCDF Round-Trip — Design

> **Status: PLAN — not yet implemented.** This document is the accepted
> design for making `tensogram` a *native, reversible* representation of
> GRIB and NetCDF. The committed work items live in `TODO.md`; the test
> methodology is in `TEST.md` ("Round-trip conversion testing").

## 1. Goal

Turn the current one-way importers (`tensogram-grib`, `tensogram-netcdf`)
into **round-trippable** bridges:

```
GRIB   → tensogram → GRIB    (tensogram convert-grib   … then  tensogram to-grib)
NetCDF → tensogram → NetCDF  (tensogram convert-netcdf … then  tensogram to-netcdf)
```

The tensogram message must be a **native, usable tensogram tensor** (not an
opaque container), and the round-trip must reproduce the source **without
loss of data or metadata** — to the fidelity the source encoding allows.

## 2. Decisions (from design review)

| # | Decision |
|---|---|
| Primary goal | **Native tensogram tensors**, not a byte-exact container. |
| Byte/MD5 identicality | **Not a hard requirement.** Round-trip fidelity is *measured* (max abs / max rel error) and *reported*, not guaranteed at the byte level. (Verification shows it is often lossless anyway — see §4.) |
| Encoding | **Keep the origin codec when tensogram supports it** (GRIB `grid_simple`→`simple_packing`, `grid_ccsds`→`szip`). Otherwise decode to values and use a **lossless default**; the user may opt into a lossy codec via the existing pipeline flags. |
| Values | Flow through **f64** only for *lossy-packed* representations (GRIB packing, CF `scale_factor`/`add_offset`); keep **native dtype** everywhere else. |
| Reconstruction | **Rebuild from captured key-values**, never from retained byte streams. Store everything in CBOR. |
| NetCDF-4 | **Reconstruct** via libnetcdf; classic *may* land byte-identical (bonus), nc4 is best-effort (accepted non-MD5). **No `netcdf-sys` unsafe FFI** — safe crate only. |
| Error policy | **Warnings only.** If it can be encoded, encode it — even with precision loss. |
| Structural fidelity | Preserve GRIB `mars` + all reconstruct keys + **local sections** + **bitmap/missing**; NetCDF **dims + coordinate variables + groups + exact attribute types**. |
| Delivery | GRIB and NetCDF **in parallel** on a shared harness. |

## 3. Non-goals (v1)
- Byte/MD5-identical reproduction as a contract (explicitly dropped).
- Opaque byte-stream storage / scaffolds / retained source files.
- `netcdf-sys` FFI for HDF5 filter introspection/reproduction.
- GRIB→tensogram carrying of **compressed chunks verbatim** (we go through values).

## 4. Verification evidence (informs feasibility)

- **GRIB value round-trip is lossless when packing params are preserved.**
  Re-packing a real CCSDS field (`2t.grib2`, 1,038,240 values, `bitsPerValue=12`)
  through a decode→re-encode cycle gave `max_abs=0`, `max_rel=0`,
  bit-identical values. So the f64 pipeline loses nothing in the common case;
  error is non-zero only if a lossy tensogram codec is chosen or `bitsPerValue`
  is changed.
- **GRIB reconstruct-from-keys works.** Setting grid/product keys on the
  `regular_ll_sfc_grib2.tmpl` sample rebuilt the geometry exactly;
  `numberOfValues` resolves once the values array is injected (API path).
- **ecCodes 0.14** exposes all we need: full key iteration incl. `Bytes`
  (local sections), `values`/`codedValues`, `KeyWrite::write_key_unchecked`,
  `try_clone`, `write_to_file`. Real ECMWF fixtures are all `grid_ccsds`.
- **NetCDF**: `nccopy` reproduces **classic** byte-identical; **nc4** not
  (HDF5 non-determinism). The **safe** `netcdf` 0.12 crate exposes dtype,
  `endianness()`, `chunking()`, `set_compression(deflate, shuffle)`, groups,
  dims, coord vars, and exact attribute types — sufficient without FFI.

Residual GRIB risk: switching `packingType` (e.g. onto `grid_ccsds`) and
product-template / local-section / bitmap **key ordering** are finicky in
ecCodes — to be hardened during implementation (§14).

## 5. Architecture & module boundaries (clean design)

**All GRIB-specific code stays in `tensogram-grib`; all NetCDF-specific code
stays in `tensogram-netcdf`. The core `tensogram` crate remains
format-agnostic and gains no GRIB/NetCDF knowledge.** Metadata is carried
through the existing generic `base[i]` CBOR channel; no core wire-format
change is required.

```
tensogram-grib     import: convert_grib_{file,buffer}()   (exists)
                   export: to_grib_message() / to_grib_file()   (NEW)
                   — all ecCodes usage confined here

tensogram-netcdf   import: convert_netcdf_file()           (exists)
                   export: to_netcdf_file()                (NEW)
                   — all libnetcdf usage confined here

tensogram (core)   unchanged; format-agnostic
tensogram-cli      thin, feature-gated wiring of the subcommands
```

## 6. CLI surface (feature-gated on availability)

Mirrors the existing importers exactly:

| Subcommand | Feature | Direction |
|---|---|---|
| `convert-grib`   (exists) | `grib`   | GRIB → tensogram |
| `convert-netcdf` (exists) | `netcdf` | NetCDF → tensogram |
| **`to-grib`**   (new) | `grib`   | tensogram → GRIB |
| **`to-netcdf`** (new) | `netcdf` | tensogram → NetCDF |

`to-grib` / `to-netcdf` take the **same flag vocabulary** as their
`convert-*` counterparts where it makes sense (grouping/split, pipeline
selection, `--cf`, `--all-keys`). Gated behind the same `grib` / `netcdf`
cargo features (which require `libeccodes` / `libnetcdf`).

## 7. Import design (source → tensogram)

**GRIB** (`tensogram-grib`): per message, capture the **full reconstruct
key-set** as typed CBOR — `mars` (as today) + all grid / product / level /
packing keys + **local sections** (incl. array/`Bytes` keys — stop dropping
them) + **bitmap** (`bitmapPresent` + missing → f64 NaN via tensogram NaN
masks). Decode values → f64. Encode the tensor **keeping the origin codec**:
`grid_ccsds`→`szip` (source `bitsPerValue`), `grid_simple`→`simple_packing`,
`grid_ieee`→`none`; unknown packing → lossless default + **warning**.

**NetCDF** (`tensogram-netcdf`): capture dims (names/lengths/unlimited),
**coordinate variables**, **groups**, variables with **native dtype** +
`endianness` + `chunking` + `set_compression` state, **exact attribute
types**, global attrs. CF `scale_factor`/`add_offset` is the only "packed"
case → unpack to f64 (native dtype otherwise), attrs retained for re-pack.

## 8. Export design (tensogram → source)

**GRIB**: `try_clone` a sample for the captured edition/templates →
`write_key_unchecked` the captured keys (careful `packingType` / local-section
/ bitmap ordering) → set f64 values → `write_to_file` → concat messages.

**NetCDF** (safe crate only): `create` (classic vs nc4 per captured format) →
add dims (preserving order) → add variables (dtype, endianness, chunking,
`set_compression` best-effort) + exact-typed attributes → put values (re-pack
CF using captured `scale_factor`/`add_offset`). Classic aims byte-identical;
nc4 is best-effort.

## 9. Encoding bridge (keep the codec where tensogram supports it)

| Source encoding | tensogram codec | Round-trip |
|---|---|---|
| GRIB `grid_simple` | `simple_packing` (reuse `bitsPerValue`/`decimalScaleFactor`) | lossless |
| GRIB `grid_ccsds` | `szip` (+ simple packing) | lossless (verified) |
| GRIB `grid_ieee` | `none` (raw float) | lossless |
| GRIB `grid_jpeg` / `grid_complex` | lossless default + **warning** | lossy re-pack, reported |
| NetCDF native int/float | `none`, native dtype + endianness | lossless (classic byte-identical bonus) |
| NetCDF CF scale/offset | unpack→f64; re-pack with captured scale/offset | lossy re-pack, reported |
| NetCDF deflate/shuffle | `set_compression` on export (lossless container) | values lossless; nc4 best-effort |

## 10. Metadata schema (CBOR, no new frame type)

- `base[i]["grib"]` — complete reconstruct keys (typed) + `_local` + `_bitmap`;
  `base[i]["mars"]` / `["cf"]` additive.
- `base[i]["netcdf"]` — attrs (exact types) + `_dims` + `_group` + `_coord`
  + `_storage` (endianness/chunking/compression) + `_global`.
- `_extra_["source"]` — `{ format, version, per-object { codec, abs_err,
  rel_err, lossy } }`.

## 11. Fidelity model

Per-object **max-abs / max-rel error** + a metadata-exact diff. **Warnings,
never hard errors.** Lossless is the default, so the reported error is
typically zero; it is non-zero only when the user opts into a lossy codec
(`sz3`/`zfp`) or changes `bitsPerValue`.

## 12. Round-trip test methodology (compare in the source format)

A round-trip test **does not** compare tensogram internals; it compares the
**re-emitted source file against the original, using that format's own
tools** (see `TEST.md`):

```
tensogram convert-FORMAT in.FMT -o mid.tgm
tensogram to-FORMAT      mid.tgm -o out.FMT
# compare in.FMT vs out.FMT with FORMAT-native tools:
#   GRIB   → grib_compare (ships with ecCodes; key + value compare, tolerance-aware)
#   NetCDF → nccmp -dmgf if installed, else ncdump -s structural diff + values compare
#            (nccmp is a separate package, often absent)  [+ MD5 for classic]
```

Assertions: metadata equal (per the tool), values within the reported
abs/rel tolerance, and (classic NetCDF only) MD5 identical as a bonus.

## 13. Phasing (two parallel workstreams, shared harness)

- **Shared**: fidelity report types + `verify_roundtrip` harness + the
  `convert→to→native-compare` test scaffold.
- **GRIB workstream** (`tensogram-grib`): full typed key capture → codec-
  preserving f64 tensor → ecCodes reconstruct (`to-grib`) → report. Order:
  simple/IEEE (lossless) → CCSDS→szip → local sections + bitmap →
  `packingType`-set hardening.
- **NetCDF workstream** (`tensogram-netcdf`): classic (byte-identical target)
  → nc4 + groups + coord vars + exact attr types → CF unpack/re-pack →
  deflate/shuffle re-apply.

## 14. Residual implementation risks (validate during build)

1. GRIB `packingType` setting onto a sample (esp. `grid_ccsds`) — finicky;
   may need a pre-set order or a ccsds-preconfigured sample.
2. Full reconstruct key-set for exotic product templates + local sections +
   bitmap, end-to-end with injected values.
3. NetCDF classic byte-identical from decomposed tensors needs exact def-order
   / attr-order / fill-mode replay.

## 15. Fixtures

**Sourced** (generated with ecCodes / nccopy and added to the crates'
`testdata/`):
- `tensogram-grib/testdata/2t_simple.grib2` — `grid_simple`, 12-bit.
- `tensogram-grib/testdata/2t_ieee.grib2` — `grid_ieee`.
- `tensogram-grib/testdata/lsm_bitmap.grib2` — `grid_ccsds` with a bitmap
  section (`bitmapPresent=1`).
- `tensogram-netcdf/testdata/cf_temperature_deflate.nc` — nc4 with
  deflate(5) + shuffle + chunking.

**Still to source externally** (need real data / extra libs):
- GRIB `grid_jpeg` (JPEG2000) and `grid_complex` / `grid_second_order`.
- A GRIB with **genuine missing values** (the generated bitmap is all-present).

Note: `nccmp` is not installed in the reference environment; the NetCDF
round-trip suite falls back to an `ncdump`-based comparison (see §12).
