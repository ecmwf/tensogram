# tensogram-netcdf test fixtures

Eight small NetCDF files used by the integration test suite.

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

## Fixtures

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
