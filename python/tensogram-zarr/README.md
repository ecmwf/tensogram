# Tensogram Zarr v3 Store Backend

Zarr v3 store backend for Tensogram `.tgm` files — read and write Tensogram
as if it were a Zarr v3 hierarchy.  Works with `zarr`, `xarray`, and any
consumer that speaks the Zarr v3 store protocol.

## Installation

```bash
pip install tensogram-zarr
```

The `tensogram` native package and `zarr>=3.0` are pulled in automatically.

## Usage

```python
import zarr
from tensogram_zarr import TensogramStore

store = TensogramStore("forecast.tgm")
root  = zarr.open(store, mode="r")
arr   = root["temperature"]
print(arr.shape, arr.dtype)
print(arr[0, :10, :10])      # Zarr-style chunk-aware access
```

`TensogramStore` is a standard `zarr.abc.store.Store`, so the full Zarr
surface (groups, arrays, async reads, region I/O) applies.

## Documentation

- Full guide: https://sites.ecmwf.int/docs/tensogram/main/guide/zarr-backend.html
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0.
