#!/usr/bin/env python
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""earthkit-data ↔ Tensogram round-trip example.

Demonstrates:

1. Encoding a synthetic MARS-keyed tensogram message from scratch.
2. Opening it with earthkit-data via the ``tensogram`` source.
3. FieldList selection / metadata access (MARS path).
4. Delegating ``.to_xarray()`` to the tensogram-xarray backend.
5. Writing a FieldList back out with the ``tensogram`` encoder.
6. Array-namespace interop (torch, if installed).
7. A plain non-MARS tensogram going through the xarray-only path.

Run with::

    python examples/python/18_earthkit_integration.py

Requires: ``tensogram``, ``tensogram-xarray``, ``tensogram-earthkit``,
``earthkit-data``, ``numpy``, ``xarray`` (``torch`` optional).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import tensogram

import earthkit.data as ekd


def _write_synthetic_mars_tgm(path: Path) -> None:
    """Create a two-field MARS tensogram for the demo."""
    lat = np.linspace(60.0, 30.0, 4, dtype=np.float32)
    lon = np.linspace(-10.0, 20.0, 6, dtype=np.float32)
    field_2t = np.full((4, 6), 273.15, dtype=np.float32)
    field_tp = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(4, 6)

    descriptors = [
        {"type": "ntensor", "dtype": "float32", "ndim": 1, "shape": [4],
         "strides": [1], "encoding": "none", "filter": "none", "compression": "none"},
        {"type": "ntensor", "dtype": "float32", "ndim": 1, "shape": [6],
         "strides": [1], "encoding": "none", "filter": "none", "compression": "none"},
        {"type": "ntensor", "dtype": "float32", "ndim": 2, "shape": [4, 6],
         "strides": [6, 1], "encoding": "none", "filter": "none", "compression": "none"},
        {"type": "ntensor", "dtype": "float32", "ndim": 2, "shape": [4, 6],
         "strides": [6, 1], "encoding": "none", "filter": "none", "compression": "none"},
    ]
    base = [
        {"name": "latitude"},
        {"name": "longitude"},
        {"mars": {"class": "od", "type": "fc", "param": "2t",
                  "date": "2025-01-01", "time": "0000", "step": 0, "levtype": "sfc"}},
        {"mars": {"class": "od", "type": "fc", "param": "tp",
                  "date": "2025-01-01", "time": "0000", "step": 6, "levtype": "sfc"}},
    ]
    meta = {"base": base, "_extra_": {"title": "demo MARS tensogram"}}
    objects = [
        (descriptors[0], lat), (descriptors[1], lon),
        (descriptors[2], field_2t), (descriptors[3], field_tp),
    ]
    path.write_bytes(tensogram.encode(meta, objects))


def _write_synthetic_nonmars_tgm(path: Path) -> None:
    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    desc = {"type": "ntensor", "dtype": "float64", "ndim": 3,
            "shape": [2, 3, 4], "strides": [12, 4, 1],
            "encoding": "none", "filter": "none", "compression": "none"}
    meta = {"base": [{"name": "generic_cube"}],
            "_extra_": {"title": "demo non-MARS tensogram"}}
    path.write_bytes(tensogram.encode(meta, [(desc, arr)]))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        mars_path = tmpdir / "mars.tgm"
        nonmars_path = tmpdir / "generic.tgm"
        _write_synthetic_mars_tgm(mars_path)
        _write_synthetic_nonmars_tgm(nonmars_path)

        # 1. Open MARS file via earthkit
        print("--- 1. Open MARS tensogram via earthkit-data ---")
        data = ekd.from_source("tensogram", str(mars_path))
        print(f"source: {type(data).__name__}")

        # 2. FieldList path with MARS selection
        print("\n--- 2. FieldList access ---")
        fl = data.to_fieldlist()
        print(f"fields: {len(fl)}")
        for f in fl:
            print(f"  param={f.metadata('param')!r} "
                  f"step={f.metadata('step')!r} shape={f.shape}")

        print("\n--- 3. Select by MARS param ---")
        subset = fl.sel(param="2t")
        print(f"2t fields: {len(subset)}")

        # 4. Delegate to_xarray to tensogram-xarray
        print("\n--- 4. FieldList.to_xarray() via tensogram-xarray ---")
        ds = fl.to_xarray()
        print(ds)

        # 5. Round-trip: write FieldList back as tensogram
        print("\n--- 5. Encoder round-trip ---")
        out_path = tmpdir / "roundtrip.tgm"
        fl.to_target("file", str(out_path), encoder="tensogram")
        restored = ekd.from_source("tensogram", str(out_path)).to_fieldlist()
        print(f"restored fields: {len(restored)}")
        np.testing.assert_array_equal(fl[0].to_numpy(), restored[0].to_numpy())
        print("values round-trip OK")

        # 6. Array namespace (torch optional)
        print("\n--- 6. Array-namespace interop ---")
        np_arr = fl[0].to_array(array_namespace="numpy")
        print(f"numpy: {np_arr.dtype} {np_arr.shape}")
        try:
            import torch  # noqa: F401

            t_arr = fl[0].to_array(array_namespace="torch", device="cpu")
            print(f"torch: {type(t_arr).__name__} {t_arr.dtype} {t_arr.shape}")
        except ImportError:
            print("torch not installed — skipping")

        # 7. Non-MARS / xarray-only path
        print("\n--- 7. Non-MARS tensogram → xarray-only path ---")
        gen = ekd.from_source("tensogram", str(nonmars_path))
        ds2 = gen.to_xarray()
        print(ds2)
        try:
            gen.to_fieldlist()
        except NotImplementedError as exc:
            print(f"to_fieldlist correctly rejected: {exc}")


if __name__ == "__main__":
    main()
