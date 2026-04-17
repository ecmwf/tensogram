# Tensogram Python Bindings

Python bindings for Tensogram N-tensor message format.

Native extension with PyO3 and maturin. Supports NumPy arrays, async I/O, and GIL-free operation on Python 3.13t.

## Installation

```bash
pip install tensogram
```

## Usage

```python
import numpy as np
import tensogram

data = np.random.randn(100, 200).astype(np.float32)
msg = tensogram.encode(
    {"version": 2},
    [({"type": "ntensor", "shape": [100, 200], "dtype": "float32",
       "compression": "szip"}, data)],
)
result = tensogram.decode(msg)
arr = result.objects[0][1]
```

## Features

NumPy integration, async I/O, GIL-free parallel ops, partial decode, full codec support (szip, zstd, lz4, blosc2, zfp, sz3).

## Documentation

- Full docs: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.
