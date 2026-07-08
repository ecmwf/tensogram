# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Python API tests for the reverse conversions `to_grib` / `to_netcdf`.

These exercise the PyO3 bindings added alongside `convert_grib` /
`convert_netcdf`.  To avoid a hard dependency on ``netCDF4`` / ``eccodes``
Python packages, the round-trips go *through tensogram's own* converters:

    convert_netcdf(src.nc)  -> messages -> to_netcdf(messages, out.nc)
                            -> convert_netcdf(out.nc) -> compare
    convert_grib(src.grib2) -> messages -> to_grib(messages) (bytes)
                            -> convert_grib_buffer(bytes)     -> compare

The source fixtures are the ones shipped for the Rust crates.  Each test
skips cleanly when the wheel was built without the relevant feature (the
`__has_grib__` / `__has_netcdf__` probes), matching `test_convert_netcdf.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tensogram = pytest.importorskip("tensogram")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _nc_fixture(name: str) -> Path:
    return _repo_root() / "rust" / "tensogram-netcdf" / "testdata" / name


def _grib_fixture(name: str) -> Path:
    return _repo_root() / "rust" / "tensogram-grib" / "testdata" / name


def _has(attr: str) -> bool:
    return bool(getattr(tensogram, attr, False))


requires_netcdf = pytest.mark.skipif(
    not _has("__has_netcdf__"), reason="tensogram built without the 'netcdf' feature"
)
requires_grib = pytest.mark.skipif(
    not _has("__has_grib__"), reason="tensogram built without the 'grib' feature"
)


# ── NetCDF export ────────────────────────────────────────────────────────────


@requires_netcdf
def test_to_netcdf_roundtrip(tmp_path: Path) -> None:
    """convert_netcdf → to_netcdf → convert_netcdf preserves the payload."""
    src = _nc_fixture("simple_2d.nc")
    messages = tensogram.convert_netcdf(str(src))
    out = tmp_path / "out.nc"

    tensogram.to_netcdf(messages, str(out))
    assert out.exists()

    back = tensogram.convert_netcdf(str(out))
    _, objs_a = tensogram.decode(messages[0])
    _, objs_b = tensogram.decode(back[0])
    np.testing.assert_array_equal(objs_a[0][1], objs_b[0][1])


@requires_netcdf
def test_to_netcdf_reassembles_variable_split(tmp_path: Path) -> None:
    """A variable-split conversion (N messages) reassembles into one file."""
    src = _nc_fixture("multi_var.nc")
    messages = tensogram.convert_netcdf(str(src), split_by="variable")
    assert len(messages) >= 3, "multi_var.nc has ≥3 numeric variables"

    out = tmp_path / "out.nc"
    tensogram.to_netcdf(messages, str(out))

    # Default file-split re-read → one message carrying every variable.
    back = tensogram.convert_netcdf(str(out))
    assert len(back) == 1
    meta, _objs = tensogram.decode(back[0])
    names = {entry.get("name") for entry in meta.base}
    assert {"temperature", "humidity", "pressure"} <= names


@requires_netcdf
def test_to_netcdf_empty_list_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tensogram.to_netcdf([], str(tmp_path / "out.nc"))


@requires_netcdf
def test_to_netcdf_failure_preserves_existing(tmp_path: Path) -> None:
    """An atomic write must not clobber a pre-existing file when it fails."""
    out = tmp_path / "out.nc"
    out.write_bytes(b"SENTINEL")
    with pytest.raises((ValueError, RuntimeError, OSError)):
        tensogram.to_netcdf([b"not a tensogram message"], str(out))
    assert out.read_bytes() == b"SENTINEL"


# ── GRIB export ──────────────────────────────────────────────────────────────


@requires_grib
def test_to_grib_roundtrip() -> None:
    """convert_grib → to_grib (bytes) → convert_grib_buffer preserves values."""
    src = _grib_fixture("2t_ieee.grib2")
    messages = tensogram.convert_grib(str(src))

    grib_bytes = tensogram.to_grib(messages)
    assert isinstance(grib_bytes, (bytes, bytearray))
    assert len(grib_bytes) > 0

    back = tensogram.convert_grib_buffer(bytes(grib_bytes))
    _, objs_a = tensogram.decode(messages[0])
    _, objs_b = tensogram.decode(back[0])
    np.testing.assert_allclose(objs_a[0][1], objs_b[0][1], atol=1e-2)


@requires_grib
def test_to_grib_rejects_non_grib_message() -> None:
    """A message without the grib_repro key-set is a ValueError."""
    with pytest.raises(ValueError):
        tensogram.to_grib([b"not a tensogram message"])


# ── Feature-disabled stubs ───────────────────────────────────────────────────


def test_to_netcdf_stub_when_feature_missing() -> None:
    if _has("__has_netcdf__"):
        pytest.skip("netcdf feature enabled in this build")
    with pytest.raises(RuntimeError, match="built without NetCDF support"):
        tensogram.to_netcdf([b""], "unused.nc")


def test_to_grib_stub_when_feature_missing() -> None:
    if _has("__has_grib__"):
        pytest.skip("grib feature enabled in this build")
    with pytest.raises(RuntimeError, match="built without GRIB support"):
        tensogram.to_grib([b""])
