# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Shared fixtures for tensogram-xarray tests.

All fixtures create temporary ``.tgm`` files with known data using the
tensogram Python bindings.

Per-object metadata is stored in ``base`` entries (one dict per object).
Extra keys in the descriptor dict are accessible via ``desc.params``
after decode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensogram


def _make_meta(version: int = 2, **extra: Any) -> dict[str, Any]:
    return {"version": version, **extra}


def _make_desc(
    shape: list[int],
    dtype: str = "float32",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": dtype,
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
        **extra,
    }


def _make_base_entry(**fields: Any) -> dict[str, Any]:
    """Build a per-object base entry from keyword arguments."""
    return dict(fields)


# ---------------------------------------------------------------------------
# Single-message fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_tgm(tmp_path: Path) -> Path:
    """Single message, single 2D float32 object, no metadata."""
    data = np.arange(60, dtype=np.float32).reshape(6, 10)
    meta = _make_meta()
    desc = _make_desc([6, 10])
    path = tmp_path / "simple.tgm"
    with tensogram.TensogramFile.create(str(path)) as f:
        f.append(meta, [(desc, data)])
    return path


@pytest.fixture
def simple_data() -> np.ndarray:
    """The data array matching ``simple_tgm``."""
    return np.arange(60, dtype=np.float32).reshape(6, 10)


@pytest.fixture
def tgm_with_coords(tmp_path: Path) -> Path:
    """Message with 3 objects: lat array, lon array, temperature field.

    Per-object metadata (name) is stored in base entries.
    """
    lat = np.linspace(-90, 90, 5, dtype=np.float64)
    lon = np.linspace(0, 360, 8, endpoint=False, dtype=np.float64)
    temp = np.random.default_rng(42).random((5, 8), dtype=np.float32).astype(np.float32)

    meta = _make_meta(
        base=[
            _make_base_entry(name="latitude"),
            _make_base_entry(name="longitude"),
            _make_base_entry(name="temperature"),
        ],
    )
    descs = [
        _make_desc([5], dtype="float64"),
        _make_desc([8], dtype="float64"),
        _make_desc([5, 8], dtype="float32"),
    ]
    path = tmp_path / "with_coords.tgm"
    with tensogram.TensogramFile.create(str(path)) as f:
        f.append(meta, [(descs[0], lat), (descs[1], lon), (descs[2], temp)])
    return path


@pytest.fixture
def coord_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lat, lon, temp arrays matching ``tgm_with_coords``."""
    lat = np.linspace(-90, 90, 5, dtype=np.float64)
    lon = np.linspace(0, 360, 8, endpoint=False, dtype=np.float64)
    temp = np.random.default_rng(42).random((5, 8), dtype=np.float32).astype(np.float32)
    return lat, lon, temp


@pytest.fixture
def tgm_with_mars(tmp_path: Path) -> Path:
    """Message with 2 objects using MARS-style metadata.

    MARS keys are stored in base entries.
    """
    t2m = np.ones((3, 4), dtype=np.float32) * 273.15
    u10 = np.ones((3, 4), dtype=np.float32) * 5.0

    meta = _make_meta(
        base=[
            _make_base_entry(mars={"param": "2t", "levtype": "sfc"}),
            _make_base_entry(mars={"param": "10u", "levtype": "sfc"}),
        ],
    )
    descs = [
        _make_desc([3, 4]),
        _make_desc([3, 4]),
    ]
    path = tmp_path / "mars.tgm"
    with tensogram.TensogramFile.create(str(path)) as f:
        f.append(meta, [(descs[0], t2m), (descs[1], u10)])
    return path


# ---------------------------------------------------------------------------
# Multi-message fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_msg_tgm(tmp_path: Path) -> Path:
    """File with 4 messages, each with 1 object [3,4].

    Metadata varies on: param (2t, 10u) x date (20260401, 20260402).
    Per-object metadata stored in base entries.
    """
    path = tmp_path / "multi.tgm"
    rng = np.random.default_rng(99)

    with tensogram.TensogramFile.create(str(path)) as f:
        for param in ["2t", "10u"]:
            for date in ["20260401", "20260402"]:
                data = rng.random((3, 4), dtype=np.float32).astype(np.float32)
                meta = _make_meta(
                    base=[
                        _make_base_entry(mars={"param": param, "date": date}),
                    ],
                )
                desc = _make_desc([3, 4])
                f.append(meta, [(desc, data)])

    return path


@pytest.fixture
def heterogeneous_tgm(tmp_path: Path) -> Path:
    """File with 3 messages of different shapes/dtypes.

    Message 0: [3,4] float32
    Message 1: [3,4] float32
    Message 2: [5]   int32
    """
    path = tmp_path / "hetero.tgm"
    with tensogram.TensogramFile.create(str(path)) as f:
        # Message 0
        f.append(
            _make_meta(base=[_make_base_entry(name="temp")]),
            [(_make_desc([3, 4]), np.ones((3, 4), dtype=np.float32))],
        )

        # Message 1
        f.append(
            _make_meta(base=[_make_base_entry(name="wind")]),
            [(_make_desc([3, 4]), np.ones((3, 4), dtype=np.float32) * 2)],
        )

        # Message 2 -- different shape and dtype
        f.append(
            _make_meta(base=[_make_base_entry(name="counts")]),
            [
                (
                    _make_desc([5], dtype="int32"),
                    np.array([1, 2, 3, 4, 5], dtype=np.int32),
                )
            ],
        )

    return path
