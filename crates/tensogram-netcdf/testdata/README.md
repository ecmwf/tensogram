# tensogram-netcdf test fixtures

Fourteen small NetCDF files used by the integration test suite
(8 baseline fixtures + 6 coverage fixtures added in v0.7.0).

## Regenerating

```bash
source .venv/bin/activate   # from repo root
pip install "netCDF4<2" numpy
cd crates/tensogram-netcdf/testdata
python generate.py
python verify.py             # should print "All fixtures verified."
```

Pinned dependency versions used to generate the committed fixtures:
- netCDF4 1.7.4
- numpy 2.4.4

**CI does NOT run `generate.py`.** The `.nc` files are binary artifacts committed to the repo.

## Baseline fixtures (Tasks 6–12)

| File | Format | Purpose |
|------|--------|---------|
| `simple_2d.nc` | NETCDF3_CLASSIC | Minimal float64 variable, no special attrs |
| `cf_temperature.nc` | NETCDF4 | CF-1.8 compliant; packed int16 with scale/offset/fill; time + lat + lon coords |
| `multi_var.nc` | NETCDF4 | Three float32 vars + one char (S1) variable sharing dims |
| `multi_dtype.nc` | NETCDF4 | One var per dtype (i8/i16/i32/i64/u8/u16/u32/u64/f32/f64); scalar var; NaN-containing var |
| `unlimited_time.nc` | NETCDF4 | Unlimited time dimension; `temp(time,y,x)` + static `mask(y,x)` |
| `nc4_groups.nc` | NETCDF4 | Root group var + sub-group (`forecast/predicted`) for sub-group warning test |
| `nc3_classic.nc` | NETCDF3_CLASSIC | Float32 temperature; endianness test |
| `empty_file.nc` | NETCDF3_CLASSIC | Global attributes only; zero variables |

## Coverage fixtures (v0.7.0 code-coverage pass)

Added to push `tensogram-netcdf` line coverage from ~78% to ~95%.
Each one targets a specific uncovered code path.

| File | Format | Exercises |
|------|--------|-----------|
| `record_multi_dtype.nc` | NETCDF4 | Every dtype arm of `read_native_extents` (i8/u8/i16/u16/i32/u32/i64/u64/f32/f64) via record-split |
| `attr_type_variants.nc` | NETCDF4 | Every numeric arm of `get_f64_attr` (Float/Int/Short/Longlong scale\_factor); `missing_value` fallback; `_ => None` fallback on a non-numeric (String) `scale_factor` |
| `empty_unlimited.nc` | NETCDF4 | `record_count == 0` early-return in `encode_by_record` |
| `complex_types.nc` | NETCDF4 | Enum-typed variable alongside a normal float → Compound/Opaque/Enum/Vlen rejection in `extract_variable` |
| `complex_types_unlimited.nc` | NETCDF4 | Same, but along the unlimited dim → rejection in `extract_variable_record`; also carries global attrs so the `_global` injection path is exercised |
| `record_with_char.nc` | NETCDF4 | Char variable sharing the unlimited dim → Char/String rejection inside `extract_variable_record` |
