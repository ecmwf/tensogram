# Tensogram Python Bindings

Python bindings for the Tensogram N-tensor message format.

Native extension built with PyO3 + maturin. Supports NumPy arrays,
async I/O, and GIL-free operation on free-threaded Python (3.13t / 3.14t).

## Installation

```bash
pip install tensogram
# or, with the xarray and Zarr backends:
pip install tensogram[all]
```

## Usage

```python
import numpy as np
import tensogram

data = np.random.randn(100, 200).astype(np.float32)
msg = tensogram.encode(
    {"version": 3},
    [(
        {"type": "ntensor", "shape": [100, 200], "dtype": "float32",
         "compression": "szip"},
        data,
    )],
)
result = tensogram.decode(msg)
arr = result.objects[0][1]   # numpy array
```

## Features

- NumPy integration across every supported dtype (float / complex /
  int / uint, plus `bitmask` and `bfloat16`)
- Sync and async file APIs (`TensogramFile` / `AsyncTensogramFile`)
- GIL-free parallel encode / decode on free-threaded Python
- Partial-range decode (`decode_range`)
- Full codec support: szip, zstd, lz4, blosc2, zfp, sz3
- Validation (`tensogram.validate`, `tensogram.validate_file`)
- GRIB / NetCDF conversion (when the wheel is built with the matching
  feature)

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Python user guide: <https://sites.ecmwf.int/docs/tensogram/main/guide/python-api.html>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.
