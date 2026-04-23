# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Shared pytest fixtures for the tensogram-earthkit test suite.

Produces real ``.tgm`` files (not golden fixtures) by calling the bundled
``tensogram`` Python bindings.  Two flavours:

* :func:`mars_tensogram_file` — a multi-object message whose ``base[i]``
  entries carry MARS keys.  Drives the :class:`FieldList` path.
* :func:`nonmars_tensogram_file` — a message with no MARS metadata.
  Drives the xarray-only path.

The fixtures are session-scoped because the encode cost is non-trivial
and none of the consumer tests mutate the files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensogram")
import tensogram

# ---------------------------------------------------------------------------
# Synthetic MARS tensogram
# ---------------------------------------------------------------------------


def _build_mars_message() -> bytes:
    """Two 2-D fields that look like 2 GRIB messages re-encoded as tensogram.

    Each field is a 4 by 6 float32 grid with a MARS namespace describing it,
    plus one coordinate object per axis.  A MARS-aware consumer can
    reconstruct time / parameter / geography components from this.
    """
    lat = np.linspace(60.0, 30.0, 4, dtype=np.float32)
    lon = np.linspace(-10.0, 20.0, 6, dtype=np.float32)
    field_2t = np.full((4, 6), 273.15, dtype=np.float32) + np.arange(24, dtype=np.float32).reshape(
        4, 6
    )
    field_tp = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(4, 6)

    descriptors = [
        {
            "type": "ntensor",
            "dtype": "float32",
            "ndim": 1,
            "shape": [4],
            "strides": [1],
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        },
        {
            "type": "ntensor",
            "dtype": "float32",
            "ndim": 1,
            "shape": [6],
            "strides": [1],
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        },
        {
            "type": "ntensor",
            "dtype": "float32",
            "ndim": 2,
            "shape": [4, 6],
            "strides": [6, 1],
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        },
        {
            "type": "ntensor",
            "dtype": "float32",
            "ndim": 2,
            "shape": [4, 6],
            "strides": [6, 1],
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        },
    ]
    base = [
        {"name": "latitude"},
        {"name": "longitude"},
        {
            "mars": {
                "class": "od",
                "stream": "oper",
                "type": "fc",
                "param": "2t",
                "date": "2025-01-01",
                "time": "0000",
                "step": 0,
                "levtype": "sfc",
            }
        },
        {
            "mars": {
                "class": "od",
                "stream": "oper",
                "type": "fc",
                "param": "tp",
                "date": "2025-01-01",
                "time": "0000",
                "step": 6,
                "levtype": "sfc",
            }
        },
    ]
    meta = {"base": base, "_extra_": {"title": "synthetic MARS tensogram"}}
    objects = [
        (descriptors[0], lat),
        (descriptors[1], lon),
        (descriptors[2], field_2t),
        (descriptors[3], field_tp),
    ]
    return tensogram.encode(meta, objects)


def _build_nonmars_message() -> bytes:
    """A single 3-D float64 object with no MARS keys — generic N-tensor."""
    shape = (2, 3, 4)
    arr = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    descriptor = {
        "type": "ntensor",
        "dtype": "float64",
        "ndim": 3,
        "shape": list(shape),
        "strides": [12, 4, 1],
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }
    meta = {
        "base": [{"name": "generic_cube"}],
        "_extra_": {"title": "synthetic non-MARS tensogram"},
    }
    return tensogram.encode(meta, [(descriptor, arr)])


# ---------------------------------------------------------------------------
# File fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mars_tensogram_bytes() -> bytes:
    """Raw bytes of a MARS-flavoured tensogram message."""
    return _build_mars_message()


@pytest.fixture(scope="session")
def nonmars_tensogram_bytes() -> bytes:
    """Raw bytes of a non-MARS tensogram message."""
    return _build_nonmars_message()


@pytest.fixture(scope="session")
def mars_tensogram_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped ``.tgm`` file with MARS metadata."""
    path = tmp_path_factory.mktemp("mars") / "mars.tgm"
    path.write_bytes(_build_mars_message())
    return path


@pytest.fixture(scope="session")
def nonmars_tensogram_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped ``.tgm`` file with no MARS metadata."""
    path = tmp_path_factory.mktemp("nonmars") / "generic.tgm"
    path.write_bytes(_build_nonmars_message())
    return path
