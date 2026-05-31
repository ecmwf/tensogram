# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Cross-language parity consumer for the Fortran binding.

Decodes the ``.tgm`` written by ``parity_write.f90`` and asserts the
column-major contract (PLAN_FORTRAN.md §5.1, §8): a Fortran ``a(ni, nj)`` is
encoded with the on-wire shape reversed, so NumPy sees the transpose
``(nj, ni)`` with ``arr[j-1, i-1] == field(i, j)``.
"""

import sys

import numpy as np

import tensogram

NI, NJ = 5, 3


def main(path: str) -> None:
    with open(path, "rb") as fh:
        raw = fh.read()

    result = tensogram.decode(raw)
    arr = result.objects[0][1]

    assert arr.dtype == np.float32, f"dtype {arr.dtype} != float32"
    assert arr.shape == (NJ, NI), (
        f"shape {arr.shape} != {(NJ, NI)} (transpose of Fortran)"
    )

    for i in range(1, NI + 1):
        for j in range(1, NJ + 1):
            expected = float(i * 1000 + j)
            got = float(arr[j - 1, i - 1])
            assert got == expected, (
                f"arr[{j - 1},{i - 1}]={got} != field({i},{j})={expected}"
            )

    print("parity_check: PASS (Fortran a(ni,nj) -> Python sees transpose (nj,ni))")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: parity_check.py <in.tgm>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
