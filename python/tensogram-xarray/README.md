# Tensogram xarray Backend

xarray backend engine for Tensogram `.tgm` files — open a Tensogram
message or file as an `xarray.Dataset` with the native `engine="tensogram"`
hook, including Dask-backed lazy loading.

## Installation

```bash
pip install tensogram-xarray
# or, with Dask-backed lazy reads
pip install "tensogram-xarray[dask]"
```

The `tensogram` native package is pulled in automatically.

## Usage

```python
import xarray as xr

ds = xr.open_dataset("forecast.tgm", engine="tensogram")
print(ds)                    # coordinates, variables, CF attributes
print(ds["temperature"])     # standard xarray DataArray
```

With Dask for out-of-core reads:

```python
ds = xr.open_dataset("forecast.tgm", engine="tensogram", chunks="auto")
```

Multi-message files open as a single merged Dataset; coordinates and
dimensions are inferred from CF-style metadata when present.

## Documentation

- Full guide: https://sites.ecmwf.int/docs/tensogram/main/guide/xarray-integration.html
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0.
