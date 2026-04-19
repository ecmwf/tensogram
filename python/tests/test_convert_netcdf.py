# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Python end-to-end round-trip tests for `tensogram convert-netcdf`.

These tests build a small NetCDF input file with `netCDF4`, run the
`tensogram` CLI binary as a subprocess (the v1 pattern, since the
Python bindings do NOT expose `convert_netcdf_file()` directly),
and then decode the output through the `tensogram` Python bindings
to verify the round-trip preserves data and metadata.

Prerequisites:
    cargo build -p tensogram-cli --features netcdf
    uv pip install netCDF4

The fixture `tensogram_binary` looks under `target/{debug,release}/`
relative to the repository root, and skips all tests cleanly when
either the binary or `netCDF4` is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

# Skip the whole module if netCDF4 isn't available — keeps the rest of
# the python suite green on machines without libnetcdf.
nc4 = pytest.importorskip("netCDF4")
tensogram = pytest.importorskip("tensogram")


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tensogram_binary() -> str:
    """Locate a tensogram CLI binary *that was built with the `netcdf` feature*.

    A binary without the ``netcdf`` feature does not expose the
    ``convert-netcdf`` subcommand — probe with ``binary --help`` and
    skip cleanly when it is missing, instead of letting every test in
    this module fail with a cryptic "unrecognized subcommand" error.
    """
    root = _repo_root()
    candidates = [
        root / "target" / "debug" / "tensogram",
        root / "target" / "release" / "tensogram",
    ]
    paths: list[str] = [str(c) for c in candidates if c.exists() and os.access(c, os.X_OK)]
    found = shutil.which("tensogram")
    if found:
        paths.append(found)

    for path in paths:
        # `binary help <subcommand>` is a clap standard: exit-0 when the
        # subcommand is compiled in, exit-nonzero when it is not. More
        # precise than scanning `--help` text (the `--threads` flag's
        # own docstring mentions `convert-netcdf` by name).
        try:
            result = subprocess.run(
                [path, "help", "convert-netcdf"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode == 0:
            return path

    pytest.skip(
        "tensogram binary with `netcdf` feature not found. "
        "Run: cargo build -p tensogram-cli --features netcdf"
    )


def _write_simple_f64(path: Path) -> np.ndarray:
    """Write a 5x4 f64 variable named `data` and return the source array."""
    rng = np.random.default_rng(seed=42)
    arr = rng.standard_normal((5, 4)).astype(np.float64)
    with nc4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", 5)
        ds.createDimension("x", 4)
        var = ds.createVariable("data", "f8", ("y", "x"))
        var[:] = arr
    return arr


def _write_packed_temperature(path: Path) -> np.ndarray:
    """Write a packed int16 temperature variable. Returns the unpacked
    f64 view (i.e. what the converter should produce).

    netCDF4 auto-scales on write when `scale_factor` / `add_offset` are
    set on the variable: assigning a float array stores it as packed
    integers. We compute the post-quantization expected values
    explicitly so the round-trip assertion has the right ground truth.
    """
    rng = np.random.default_rng(seed=123)
    nlat, nlon = 8, 12
    kelvin = rng.uniform(240.0, 310.0, size=(nlat, nlon)).astype(np.float64)

    add_offset = 273.15
    scale_factor = 0.01

    # What netCDF4 will actually store on disk — round-trip through the
    # int16 quantization so the expected values match what tensogram
    # reads back.
    packed = np.rint((kelvin - add_offset) / scale_factor).astype(np.int16)
    unpacked = packed.astype(np.float64) * scale_factor + add_offset

    with nc4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.10"
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)
        temp = ds.createVariable("temperature", "i2", ("lat", "lon"), fill_value=-32768)
        temp.standard_name = "air_temperature"
        temp.long_name = "Air Temperature"
        temp.units = "K"
        temp.scale_factor = scale_factor
        temp.add_offset = add_offset
        temp.cell_methods = "time: mean"
        # Assign the unpacked Kelvin values and let netCDF4 pack them
        # through scale_factor/add_offset. This matches how CF writers
        # normally produce packed files.
        temp[:] = unpacked

    return unpacked


def _write_three_variables(path: Path) -> None:
    with nc4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", 4)
        ds.createDimension("x", 3)
        for name in ("alpha", "beta", "gamma"):
            v = ds.createVariable(name, "f8", ("y", "x"))
            v[:] = np.full((4, 3), 1.0, dtype=np.float64)


def _write_with_missing_values(path: Path) -> None:
    """Write an int16 variable with a ``_FillValue`` that netCDF4 turns
    into NaN on CF unpacking.

    Used by the strict-finite parity tests to exercise the rejection
    path that the converter otherwise soft-downgrades (today) or
    hard-fails (after Workstream B).
    """
    with nc4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.10"
        ds.createDimension("y", 4)
        ds.createDimension("x", 3)
        var = ds.createVariable("sparse", "i2", ("y", "x"), fill_value=-32768)
        var.scale_factor = 0.1
        var.add_offset = 0.0
        # Masked array — netCDF4 replaces the masked cells with the
        # fill value on disk.  On read with CF unpacking, the fill
        # value becomes NaN in the f64 output.
        arr = np.ma.array(
            np.full((4, 3), 1.0, dtype=np.float64),
            mask=[[False, True, False], [False, False, False], [True, False, False], [False, False, False]],
        )
        var[:] = arr


def _run_convert(
    binary: str, *args: str, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, "convert-netcdf", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _decode_messages(tgm_path: Path) -> Iterator[tuple]:
    """Yield (metadata, objects) tuples from a tgm file."""
    with tensogram.TensogramFile.open(str(tgm_path)) as f:
        for msg in f:
            yield msg.metadata, msg.objects


# ── Tests ────────────────────────────────────────────────────────────────────


def test_simple_roundtrip(tmp_path: Path, tensogram_binary: str) -> None:
    """f64 input → tensogram → decode → byte-equal."""
    nc_path = tmp_path / "simple.nc"
    tgm_path = tmp_path / "simple.tgm"
    expected = _write_simple_f64(nc_path)

    result = _run_convert(tensogram_binary, str(nc_path), "-o", str(tgm_path))
    assert result.returncode == 0, (
        f"convert-netcdf failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    msgs = list(_decode_messages(tgm_path))
    assert len(msgs) == 1, "default --split-by=file → one message"
    _metadata, objects = msgs[0]
    assert len(objects) == 1, "simple.nc has one variable"
    _desc, arr = objects[0]
    assert arr.dtype == np.float64
    assert arr.shape == expected.shape
    np.testing.assert_array_equal(arr, expected)


def test_packed_unpacks_to_float64(tmp_path: Path, tensogram_binary: str) -> None:
    """int16 + scale_factor + add_offset → unpacked f64 in output."""
    nc_path = tmp_path / "packed.nc"
    tgm_path = tmp_path / "packed.tgm"
    expected = _write_packed_temperature(nc_path)

    result = _run_convert(tensogram_binary, str(nc_path), "-o", str(tgm_path))
    assert result.returncode == 0, result.stderr

    _metadata, objects = next(_decode_messages(tgm_path))
    _desc, arr = objects[0]
    assert arr.dtype == np.float64, "packed var should be unpacked to f64"
    assert arr.shape == expected.shape
    # Quantization tolerance: scale_factor = 0.01.
    np.testing.assert_allclose(arr, expected, atol=0.02)


def test_cf_flag_extracts_standard_name(tmp_path: Path, tensogram_binary: str) -> None:
    """--cf populates base[i]['cf']['standard_name']."""
    nc_path = tmp_path / "cf.nc"
    tgm_path = tmp_path / "cf.tgm"
    _write_packed_temperature(nc_path)

    result = _run_convert(tensogram_binary, "--cf", str(nc_path), "-o", str(tgm_path))
    assert result.returncode == 0, result.stderr

    metadata, _objects = next(_decode_messages(tgm_path))
    cf_entries = [entry for entry in metadata.base if isinstance(entry.get("cf"), dict)]
    assert cf_entries, "--cf should populate at least one cf sub-map"
    cf_map = cf_entries[0]["cf"]
    assert cf_map.get("standard_name") == "air_temperature"
    assert cf_map.get("units") == "K"


def test_no_cf_flag_omits_cf_key(tmp_path: Path, tensogram_binary: str) -> None:
    """Without --cf, the cf sub-map must not appear."""
    nc_path = tmp_path / "no_cf.nc"
    tgm_path = tmp_path / "no_cf.tgm"
    _write_packed_temperature(nc_path)

    result = _run_convert(tensogram_binary, str(nc_path), "-o", str(tgm_path))
    assert result.returncode == 0, result.stderr

    metadata, _ = next(_decode_messages(tgm_path))
    for entry in metadata.base:
        assert "cf" not in entry, "without --cf, no entry should have a 'cf' key"


def test_split_by_variable(tmp_path: Path, tensogram_binary: str) -> None:
    """--split-by variable produces N messages for N numeric variables."""
    nc_path = tmp_path / "three.nc"
    tgm_path = tmp_path / "three.tgm"
    _write_three_variables(nc_path)

    result = _run_convert(
        tensogram_binary,
        "--split-by",
        "variable",
        str(nc_path),
        "-o",
        str(tgm_path),
    )
    assert result.returncode == 0, result.stderr

    msgs = list(_decode_messages(tgm_path))
    assert len(msgs) == 3, "three numeric vars → three messages"
    for _meta, objects in msgs:
        assert len(objects) == 1, "each split message has one object"


def test_compression_zstd_actually_compresses(tmp_path: Path, tensogram_binary: str) -> None:
    """zstd output should be smaller than uncompressed for predictable
    data with low entropy."""
    nc_path = tmp_path / "compressible.nc"
    plain_path = tmp_path / "plain.tgm"
    zstd_path = tmp_path / "zstd.tgm"

    # Highly compressible: a constant grid.
    with nc4.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", 64)
        ds.createDimension("x", 64)
        v = ds.createVariable("data", "f8", ("y", "x"))
        v[:] = np.full((64, 64), 42.0, dtype=np.float64)

    r1 = _run_convert(tensogram_binary, str(nc_path), "-o", str(plain_path))
    assert r1.returncode == 0, r1.stderr

    r2 = _run_convert(
        tensogram_binary,
        "--compression",
        "zstd",
        str(nc_path),
        "-o",
        str(zstd_path),
    )
    assert r2.returncode == 0, r2.stderr

    plain_size = plain_path.stat().st_size
    zstd_size = zstd_path.stat().st_size
    assert zstd_size < plain_size, (
        f"zstd output ({zstd_size}) should be smaller than "
        f"uncompressed ({plain_size}) for a constant grid"
    )


def test_record_split_requires_unlimited(tmp_path: Path, tensogram_binary: str) -> None:
    """--split-by record on a file without an unlimited dim is a hard error."""
    nc_path = tmp_path / "no_unlimited.nc"
    _write_simple_f64(nc_path)

    result = _run_convert(tensogram_binary, "--split-by", "record", str(nc_path))
    assert result.returncode != 0, "should fail without unlimited dim"
    combined = (result.stdout + result.stderr).lower()
    assert "unlimited" in combined, f"error message should mention 'unlimited', got: {combined!r}"


def test_simple_packing_flag_round_trips(tmp_path: Path, tensogram_binary: str) -> None:
    """--encoding simple_packing on a pure-f64 file produces a packed
    descriptor whose decoded values match the input within the
    quantization tolerance."""
    nc_path = tmp_path / "packed.nc"
    tgm_path = tmp_path / "packed.tgm"
    expected = _write_simple_f64(nc_path)

    result = _run_convert(
        tensogram_binary,
        "--encoding",
        "simple_packing",
        "--bits",
        "24",
        str(nc_path),
        "-o",
        str(tgm_path),
    )
    assert result.returncode == 0, result.stderr

    _meta, objects = next(_decode_messages(tgm_path))
    desc, arr = objects[0]
    assert desc.encoding == "simple_packing"
    assert arr.dtype == np.float64
    # 24-bit packing on data with std≈1 has quantization step << 1e-5,
    # so the decoded values should be very close to the originals.
    np.testing.assert_allclose(arr, expected, atol=1e-5)


# ── Python API tests: tensogram.convert_netcdf(...) ──────────────────────────
#
# The tests above exercise the CLI binary. The tests below exercise the
# PyO3 binding added in v0.15.  When the wheel was built without the
# `netcdf` feature, `convert_netcdf` raises RuntimeError — we skip the
# whole block in that case.  When the feature is built in but libnetcdf
# is otherwise broken, the import-time linker would have failed already.


def _has_netcdf_feature() -> bool:
    """True when the bindings were compiled with `--features netcdf`."""
    return bool(getattr(tensogram, "__has_netcdf__", False))


requires_netcdf = pytest.mark.skipif(
    not _has_netcdf_feature(),
    reason="tensogram was built without the 'netcdf' feature",
)


@requires_netcdf
def test_py_api_simple_roundtrip(tmp_path: Path) -> None:
    """convert_netcdf() returns bytes that round-trip through decode()."""
    nc_path = tmp_path / "simple_api.nc"
    expected = _write_simple_f64(nc_path)

    messages = tensogram.convert_netcdf(str(nc_path))
    assert isinstance(messages, list)
    assert len(messages) == 1, "default split_by='file' → single message"
    assert isinstance(messages[0], bytes)

    _meta, objects = tensogram.decode(messages[0])
    assert len(objects) == 1
    _desc, arr = objects[0]
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, expected)


@requires_netcdf
def test_py_api_packed_unpacks_to_float64(tmp_path: Path) -> None:
    """int16 + scale_factor + add_offset → unpacked f64 through the Python API."""
    nc_path = tmp_path / "packed_api.nc"
    expected = _write_packed_temperature(nc_path)

    messages = tensogram.convert_netcdf(str(nc_path))
    _meta, objects = tensogram.decode(messages[0])
    _desc, arr = objects[0]
    assert arr.dtype == np.float64
    np.testing.assert_allclose(arr, expected, atol=0.02)


@requires_netcdf
def test_py_api_cf_flag(tmp_path: Path) -> None:
    """cf=True populates base[i]['cf']."""
    nc_path = tmp_path / "cf_api.nc"
    _write_packed_temperature(nc_path)

    messages = tensogram.convert_netcdf(str(nc_path), cf=True)
    meta, _ = tensogram.decode(messages[0])
    cf_entries = [e for e in meta.base if isinstance(e.get("cf"), dict)]
    assert cf_entries
    assert cf_entries[0]["cf"].get("standard_name") == "air_temperature"


@requires_netcdf
def test_py_api_split_by_variable(tmp_path: Path) -> None:
    """split_by='variable' produces one message per variable."""
    nc_path = tmp_path / "three_api.nc"
    _write_three_variables(nc_path)

    messages = tensogram.convert_netcdf(str(nc_path), split_by="variable")
    assert len(messages) == 3


@requires_netcdf
def test_py_api_pipeline_arguments(tmp_path: Path) -> None:
    """The full encoding pipeline is plumbed through the Python API."""
    nc_path = tmp_path / "pipe_api.nc"
    expected = _write_simple_f64(nc_path)

    messages = tensogram.convert_netcdf(
        str(nc_path),
        encoding="simple_packing",
        bits=24,
        compression="zstd",
    )
    _, objects = tensogram.decode(messages[0])
    desc, arr = objects[0]
    assert desc.encoding == "simple_packing"
    assert desc.compression == "zstd"
    np.testing.assert_allclose(arr, expected, atol=1e-5)


@requires_netcdf
def test_py_api_record_split_requires_unlimited(tmp_path: Path) -> None:
    """split_by='record' raises :class:`ValueError` when no unlimited dimension.

    This is a caller-input mismatch (requested ``split_by="record"``
    against a file that does not support it), hence ``ValueError`` and
    not ``RuntimeError``.
    """
    nc_path = tmp_path / "no_unlimited_api.nc"
    _write_simple_f64(nc_path)

    with pytest.raises(ValueError, match="unlimited"):
        tensogram.convert_netcdf(str(nc_path), split_by="record")


@requires_netcdf
def test_py_api_invalid_split_by() -> None:
    """Unknown split_by value raises ValueError before touching the C lib."""
    with pytest.raises(ValueError, match="split_by"):
        tensogram.convert_netcdf("/does/not/matter.nc", split_by="nonsense")


@requires_netcdf
def test_py_api_invalid_hash() -> None:
    """Unknown hash algorithm raises ValueError."""
    nc_path = Path("/does/not/matter.nc")
    with pytest.raises(ValueError, match="hash"):
        tensogram.convert_netcdf(str(nc_path), hash="md5")


@requires_netcdf
def test_py_api_missing_file(tmp_path: Path) -> None:
    """A missing file raises :class:`FileNotFoundError`.

    Matches the Pythonic convention that missing paths are ``OSError``
    subclasses, not opaque ``RuntimeError``.
    """
    with pytest.raises(FileNotFoundError):
        tensogram.convert_netcdf(str(tmp_path / "does_not_exist.nc"))


# ── Strict-finite flag parity with CLI convert-netcdf ───────────────────────


@requires_netcdf
def test_py_api_reject_nan_catches_fill_value_substitution(tmp_path: Path) -> None:
    """``reject_nan=True`` fires when CF unpacking substitutes ``_FillValue``
    with NaN — the canonical NetCDF source of NaN in converted data."""
    nc_path = tmp_path / "sparse.nc"
    _write_with_missing_values(nc_path)
    with pytest.raises(ValueError, match=r"(?i)nan"):
        tensogram.convert_netcdf(str(nc_path), reject_nan=True)


@requires_netcdf
def test_py_api_reject_nan_off_by_default(tmp_path: Path) -> None:
    """Default behaviour unchanged: NaN-bearing variables convert successfully."""
    nc_path = tmp_path / "sparse_default.nc"
    _write_with_missing_values(nc_path)
    # Default encoding="none": NaN bits pass through byte-exactly.
    messages = tensogram.convert_netcdf(str(nc_path))
    assert isinstance(messages, list)
    assert messages


@requires_netcdf
def test_py_api_reject_inf_accepts_kwarg(tmp_path: Path) -> None:
    """``reject_inf=True`` is accepted and plumbed through.

    NetCDF fixtures rarely contain Inf (CF uses fill-values, not
    Inf), so the scan does not fire here; we just verify conversion
    still completes with the kwarg set.
    """
    nc_path = tmp_path / "simple.nc"
    _write_simple_f64(nc_path)
    messages = tensogram.convert_netcdf(str(nc_path), reject_inf=True)
    assert isinstance(messages, list)
    assert messages


@requires_netcdf
def test_py_api_reject_nan_and_inf_together(tmp_path: Path) -> None:
    """Both flags together on NaN-bearing data still rejects with ValueError."""
    nc_path = tmp_path / "sparse_both.nc"
    _write_with_missing_values(nc_path)
    with pytest.raises(ValueError, match=r"(?i)nan"):
        tensogram.convert_netcdf(
            str(nc_path),
            reject_nan=True,
            reject_inf=True,
        )


def test_py_api_stub_when_feature_missing() -> None:
    """If the feature is off, the stub raises RuntimeError with clear guidance."""
    if _has_netcdf_feature():
        pytest.skip("feature is enabled in this build")
    with pytest.raises(RuntimeError, match="built without NetCDF support"):
        tensogram.convert_netcdf("anything.nc")
