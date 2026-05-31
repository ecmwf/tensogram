# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Cross-language parity producer, Python (NumPy, row-major) side.

Pairs with ``xlang_fortran fread``. Both agree on a logical field
F(i,j) = i*1000 + j (1-based). Python encodes a C/row-major ``[NJ, NI]``
array with ``arr[r, c] = F(c+1, r+1)``; the Fortran reader decodes it into
``out(ni, nj)`` and asserts ``out(i,j) == F(i,j)`` — the column-major
contract from the Python→Fortran direction (PLAN_FORTRAN.md §8). Lossless
zstd compression keeps it bit-exact.
"""

import sys

import numpy as np

import tensogram

NI, NJ = 5, 3


def main(path: str) -> None:
    arr = np.empty((NJ, NI), dtype=np.float32)
    for r in range(NJ):
        for c in range(NI):
            arr[r, c] = float((c + 1) * 1000 + (r + 1))

    descriptor = {
        "type": "ntensor",
        "shape": [NJ, NI],
        "dtype": "float32",
        "encoding": "none",
        "filter": "none",
        "compression": "zstd",
    }
    msg = tensogram.encode({}, [(descriptor, arr)])
    with open(path, "wb") as fh:
        fh.write(bytes(msg))
    print(f"parity_py_write: wrote {len(bytes(msg))} bytes")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: parity_py_write.py <out.tgm>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
