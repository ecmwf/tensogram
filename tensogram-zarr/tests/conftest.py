# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Shared fixtures for tensogram-zarr tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensogram


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def simple_tgm(tmp_dir: Path) -> str:
    """Create a single-message .tgm file with one float32 array (6x10)."""
    path = str(tmp_dir / "simple.tgm")
    data = np.arange(60, dtype=np.float32).reshape(6, 10)
    meta = {"version": 2}
    desc = {"type": "ntensor", "shape": [6, 10], "dtype": "float32"}
    with tensogram.TensogramFile.create(path) as f:
        f.append(meta, [(desc, data)])
    return path


@pytest.fixture
def multi_object_tgm(tmp_dir: Path) -> str:
    """Create a .tgm file with one message containing 3 named objects."""
    path = str(tmp_dir / "multi.tgm")
    temp = np.random.rand(4, 8).astype(np.float64)
    pressure = np.random.rand(4, 8).astype(np.float64)
    humidity = np.random.rand(4, 8).astype(np.float32)

    meta = {
        "version": 2,
        "base": [
            {"mars": {"param": "2t"}},
            {"mars": {"param": "sp"}},
            {"mars": {"param": "q"}},
        ],
    }
    descs_and_data = [
        ({"type": "ntensor", "shape": [4, 8], "dtype": "float64"}, temp),
        ({"type": "ntensor", "shape": [4, 8], "dtype": "float64"}, pressure),
        ({"type": "ntensor", "shape": [4, 8], "dtype": "float32"}, humidity),
    ]
    with tensogram.TensogramFile.create(path) as f:
        f.append(meta, descs_and_data)
    return path


@pytest.fixture
def mars_metadata_tgm(tmp_dir: Path) -> str:
    """Create a .tgm file with rich MARS metadata."""
    path = str(tmp_dir / "mars.tgm")
    data = np.ones((3, 5), dtype=np.float32) * 273.15
    meta = {
        "version": 2,
        # Message-level metadata goes into extra (unknown top-level keys)
        "mars": {
            "class": "od",
            "type": "fc",
            "stream": "oper",
            "expver": "0001",
            "date": "20260401",
            "time": "1200",
        },
        "base": [
            {"mars": {"param": "2t", "levtype": "sfc"}},
        ],
    }
    desc = {"type": "ntensor", "shape": [3, 5], "dtype": "float32"}
    with tensogram.TensogramFile.create(path) as f:
        f.append(meta, [(desc, data)])
    return path


@pytest.fixture
def int_types_tgm(tmp_dir: Path) -> str:
    """Create a .tgm file with integer dtype arrays."""
    path = str(tmp_dir / "ints.tgm")
    i32 = np.array([1, 2, 3, 4], dtype=np.int32)
    u16 = np.array([10, 20, 30], dtype=np.uint16)
    meta = {
        "version": 2,
        "base": [
            {"name": "counts"},
            {"name": "flags"},
        ],
    }
    descs_and_data = [
        ({"type": "ntensor", "shape": [4], "dtype": "int32"}, i32),
        ({"type": "ntensor", "shape": [3], "dtype": "uint16"}, u16),
    ]
    with tensogram.TensogramFile.create(path) as f:
        f.append(meta, descs_and_data)
    return path


@pytest.fixture
def empty_tgm(tmp_dir: Path) -> str:
    """Create an empty .tgm file (no messages)."""
    path = str(tmp_dir / "empty.tgm")
    with tensogram.TensogramFile.create(path):
        pass  # no messages appended
    return path


@pytest.fixture
def output_path(tmp_dir: Path) -> str:
    """Return a path for writing output .tgm files."""
    return str(tmp_dir / "output.tgm")
