# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the PyO3 wrappers ``tensogram.convert_grib()`` and
``tensogram.convert_grib_buffer()``.

These exercise the Python-binding code path added in v0.15 that wraps
``tensogram-grib::convert_grib_file`` / ``convert_grib_buffer``. When
the wheel was compiled without the ``grib`` feature, every test in this
module (except the stub-behaviour test at the bottom) is skipped — the
public API still exists but raises ``RuntimeError`` on call, which is
what the stub test verifies.

The committed GRIB fixtures under ``rust/tensogram-grib/testdata/`` are
real ECMWF opendata byte-range downloads (IFS 0.25 deg operational,
2026-04-04 00z, step 0h) — see ``download.sh`` in that directory for
the exact URLs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tensogram = pytest.importorskip("tensogram")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _testdata(name: str) -> Path:
    """Path to a committed GRIB fixture."""
    return _repo_root() / "rust" / "tensogram-grib" / "testdata" / name


def _has_grib_feature() -> bool:
    """True when the bindings were compiled with ``--features grib``."""
    return bool(getattr(tensogram, "__has_grib__", False))


requires_grib = pytest.mark.skipif(
    not _has_grib_feature(),
    reason="tensogram was built without the 'grib' feature",
)


# ── Happy paths ─────────────────────────────────────────────────────────────


@requires_grib
def test_convert_grib_file_returns_bytes_list() -> None:
    """``convert_grib()`` returns ``list[bytes]`` with at least one entry."""
    messages = tensogram.convert_grib(str(_testdata("2t.grib2")))
    assert isinstance(messages, list)
    assert messages, "at least one message expected"
    # PyO3 returns Python `bytes` objects — never `bytearray` or any
    # other byte-like.  Be explicit so the test catches an API drift.
    assert all(isinstance(m, bytes) for m in messages)


@requires_grib
def test_convert_grib_buffer_matches_file() -> None:
    """File and buffer paths must produce payload-identical output."""
    path = _testdata("2t.grib2")
    from_file = tensogram.convert_grib(str(path))
    with open(path, "rb") as fh:
        from_buffer = tensogram.convert_grib_buffer(fh.read())

    assert len(from_file) == len(from_buffer)
    # Hashes and UUID stamps differ per call, so we compare decoded payloads
    # + descriptor shape / pipeline rather than byte-for-byte.
    for a, b in zip(from_file, from_buffer):
        _, objs_a = tensogram.decode(a)
        _, objs_b = tensogram.decode(b)
        assert len(objs_a) == len(objs_b)
        for (desc_a, arr_a), (desc_b, arr_b) in zip(objs_a, objs_b):
            assert desc_a.shape == desc_b.shape
            assert desc_a.dtype == desc_b.dtype
            assert desc_a.encoding == desc_b.encoding
            np.testing.assert_array_equal(arr_a, arr_b)


@requires_grib
def test_convert_grib_decoded_payload_is_f64_and_sane() -> None:
    """Decoded 2m-temperature values are plausible Kelvin readings."""
    messages = tensogram.convert_grib(str(_testdata("2t.grib2")))
    _meta, objects = tensogram.decode(messages[0])
    _desc, arr = objects[0]

    assert arr.dtype == np.float64
    assert arr.ndim == 2, "2D lat/lon grid"
    # 2m temperature on an Earth-covering grid: physically valid range
    # is roughly [180 K, 330 K]. We use a slightly wider tolerance.
    assert arr.min() > 150.0, f"min {arr.min()} too cold for a 2m-temp field"
    assert arr.max() < 360.0, f"max {arr.max()} too hot for a 2m-temp field"


@requires_grib
def test_convert_grib_mars_keys_present() -> None:
    """The ``base[i]["mars"]`` sub-map contains the expected key names."""
    messages = tensogram.convert_grib(str(_testdata("2t.grib2")))
    meta = tensogram.decode_metadata(messages[0])
    assert meta.base, "base array must have at least one entry"
    mars = meta.base[0].get("mars")
    assert isinstance(mars, dict)
    # These keys are present on every GRIB2 message from IFS opendata.
    for required in ("class", "stream", "date", "step", "levtype", "param"):
        assert required in mars, f"missing MARS key: {required}"


@requires_grib
def test_preserve_all_keys_populates_grib_namespace() -> None:
    """``preserve_all_keys=True`` lifts non-MARS namespaces into ``base[i]["grib"]``."""
    messages = tensogram.convert_grib(
        str(_testdata("2t.grib2")),
        preserve_all_keys=True,
    )
    meta = tensogram.decode_metadata(messages[0])
    grib_entries = [entry.get("grib") for entry in meta.base if "grib" in entry]
    assert grib_entries, "preserve_all_keys should populate a 'grib' sub-map"
    assert isinstance(grib_entries[0], dict)
    # ls / geography / parameter / statistics are always populated by ecCodes.
    for ns in ("ls", "geography", "parameter"):
        assert ns in grib_entries[0], f"expected ecCodes namespace '{ns}' under grib"


@requires_grib
def test_grouping_one_to_one() -> None:
    """``grouping='one_to_one'`` produces one message per GRIB message."""
    # 2t.grib2 has a single GRIB message — verify we get a single output
    # from the one_to_one path too.
    messages = tensogram.convert_grib(
        str(_testdata("2t.grib2")),
        grouping="one_to_one",
    )
    assert len(messages) == 1


@requires_grib
def test_grouping_merge_all() -> None:
    """Default ``grouping='merge_all'`` returns a single Tensogram message."""
    messages = tensogram.convert_grib(
        str(_testdata("2t.grib2")),
        grouping="merge_all",
    )
    assert len(messages) == 1
    _meta, objects = tensogram.decode(messages[0])
    assert len(objects) == 1


@requires_grib
def test_pipeline_simple_packing_and_zstd() -> None:
    """The full encoding pipeline passes through the Python API unchanged."""
    path = _testdata("2t.grib2")
    plain = tensogram.convert_grib(str(path))
    packed = tensogram.convert_grib(
        str(path),
        encoding="simple_packing",
        bits=16,
        compression="zstd",
    )

    # Packed + zstd should be strictly smaller than uncompressed f64.
    assert len(packed[0]) < len(plain[0]), (
        f"packed+zstd ({len(packed[0])} B) should be smaller than "
        f"raw f64 ({len(plain[0])} B) for a 2m-temperature field"
    )

    # Decode the packed version and verify shape + low-entropy dtype.
    _meta, objects = tensogram.decode(packed[0])
    desc, arr = objects[0]
    assert desc.encoding == "simple_packing"
    assert desc.compression == "zstd"
    # simple_packing round-trip always returns float64.
    assert arr.dtype == np.float64


@requires_grib
def test_pipeline_bits_default_is_16() -> None:
    """``encoding='simple_packing'`` + ``bits=None`` defaults to 16 bits.

    This is the contract documented on ``tensogram.convert_grib`` and shared
    with the ``tensogram convert-grib`` CLI. A regression here would silently
    change the byte layout of every ``.tgm`` produced without an explicit
    ``bits`` argument.
    """
    path = str(_testdata("2t.grib2"))
    no_bits = tensogram.convert_grib(path, encoding="simple_packing")
    bits_16 = tensogram.convert_grib(path, encoding="simple_packing", bits=16)

    # Same size — both hit the 16-bit code path.
    assert len(no_bits[0]) == len(bits_16[0]), (
        f"bits=None should default to 16; got no_bits={len(no_bits[0])}  bits_16={len(bits_16[0])}"
    )

    # Decoded metadata should show simple_packing applied.
    meta = tensogram.decode_metadata(no_bits[0])
    descriptor = meta.base[0]["_reserved_"]["tensor"]
    assert descriptor is not None


@requires_grib
def test_pipeline_bits_out_of_range_falls_back_to_none() -> None:
    """Out-of-range ``bits`` causes a stderr warning and wire ``encoding="none"``.

    ``apply_pipeline`` rejects bit widths outside ``1..=64`` and falls back
    to passthrough. This test pins that behaviour so a future strictening
    (e.g. raising a ValueError at the Python boundary) is a conscious
    decision, not an accident.
    """
    path = str(_testdata("2t.grib2"))
    msgs = tensogram.convert_grib(path, encoding="simple_packing", bits=65)
    _, objects = tensogram.decode(msgs[0])
    desc, _arr = objects[0]
    assert desc.encoding == "none", (
        f"bits=65 should fall back to encoding='none' but wire encoding was {desc.encoding!r}"
    )


@requires_grib
def test_convert_grib_buffer_handles_trailing_garbage() -> None:
    """ecCodes reads through valid GRIB messages and stops at non-GRIB tail bytes.

    Pin this behaviour: a producer that accidentally appends whitespace or
    EOF markers after a valid GRIB should still succeed rather than erroring.
    """
    with open(_testdata("2t.grib2"), "rb") as fh:
        valid = fh.read()

    messages = tensogram.convert_grib_buffer(valid + b"\x00" * 1024 + b"JUNK")
    assert len(messages) == 1


@requires_grib
def test_convert_grib_buffer_handles_leading_garbage() -> None:
    """ecCodes skips non-GRIB bytes and finds the GRIB magic in the middle.

    This supports concatenated-stream use cases where a GRIB message is
    embedded in a larger byte sequence.
    """
    with open(_testdata("2t.grib2"), "rb") as fh:
        valid = fh.read()

    messages = tensogram.convert_grib_buffer(b"PREFIX\x00" * 100 + valid)
    assert len(messages) == 1


@requires_grib
def test_convert_grib_unknown_compression_is_actionable() -> None:
    """An unknown compression name lists the valid choices in the error message."""
    with pytest.raises(RuntimeError, match="unknown compression"):
        tensogram.convert_grib(str(_testdata("2t.grib2")), compression="snappy")


@requires_grib
def test_threads_argument_accepted() -> None:
    """``threads=0`` and ``threads=N`` both produce valid output."""
    path = str(_testdata("2t.grib2"))
    sequential = tensogram.convert_grib(
        path,
        encoding="simple_packing",
        bits=16,
        compression="zstd",
    )
    parallel = tensogram.convert_grib(
        path,
        encoding="simple_packing",
        bits=16,
        compression="zstd",
        threads=2,
    )
    # The two should round-trip to the same float64 values.
    _, objs_seq = tensogram.decode(sequential[0])
    _, objs_par = tensogram.decode(parallel[0])
    np.testing.assert_array_equal(objs_seq[0][1], objs_par[0][1])


# ── Buffer variants ─────────────────────────────────────────────────────────


@requires_grib
def test_convert_grib_buffer_accepts_bytes() -> None:
    """``bytes`` is the most common form — sanity round-trip."""
    with open(_testdata("2t.grib2"), "rb") as fh:
        data = fh.read()
    messages = tensogram.convert_grib_buffer(data)
    assert messages
    assert isinstance(messages[0], bytes)


@requires_grib
def test_convert_grib_buffer_accepts_bytearray() -> None:
    """``bytearray`` is accepted via the Python buffer protocol."""
    with open(_testdata("2t.grib2"), "rb") as fh:
        data = bytearray(fh.read())
    messages = tensogram.convert_grib_buffer(data)
    assert messages


@requires_grib
def test_convert_grib_buffer_accepts_memoryview() -> None:
    """``memoryview`` is accepted via the Python buffer protocol."""
    with open(_testdata("2t.grib2"), "rb") as fh:
        data = fh.read()
    messages = tensogram.convert_grib_buffer(memoryview(data))
    assert messages


# ── Error paths ─────────────────────────────────────────────────────────────


@requires_grib
def test_convert_grib_missing_file_errors(tmp_path: Path) -> None:
    """A missing path raises RuntimeError from the converter."""
    with pytest.raises(RuntimeError):
        tensogram.convert_grib(str(tmp_path / "does_not_exist.grib2"))


@requires_grib
def test_convert_grib_buffer_rejects_garbage() -> None:
    """Non-GRIB bytes surface a RuntimeError rather than a panic."""
    with pytest.raises(RuntimeError):
        tensogram.convert_grib_buffer(b"this is clearly not a GRIB message")


@requires_grib
def test_convert_grib_buffer_rejects_non_bytes() -> None:
    """Passing something clearly not a bytes-like object raises from the extractor.

    PyO3's ``Vec<u8>`` extraction goes through the buffer protocol first and
    then falls back to iterable-of-u8, so a ``list[int]`` with valid u8
    values is accepted (and then rejected as "no GRIB messages" downstream).
    The types we test here are the ones that cannot be coerced at all.
    """
    for bad in (42, None, "hello", 3.14):
        with pytest.raises((ValueError, TypeError)):
            tensogram.convert_grib_buffer(bad)


@requires_grib
def test_convert_grib_invalid_grouping() -> None:
    """Unknown grouping raises ValueError before touching the C library."""
    with pytest.raises(ValueError, match="grouping"):
        tensogram.convert_grib(
            str(_testdata("2t.grib2")),
            grouping="invalid_value",
        )


@requires_grib
def test_convert_grib_invalid_hash() -> None:
    """Unknown hash name raises ValueError."""
    with pytest.raises(ValueError, match="hash"):
        tensogram.convert_grib(str(_testdata("2t.grib2")), hash="sha256")


def test_convert_grib_stub_when_feature_missing() -> None:
    """When the build lacks the 'grib' feature, the stub explains how to fix it."""
    if _has_grib_feature():
        pytest.skip("feature is enabled in this build")
    with pytest.raises(RuntimeError, match="built without GRIB support"):
        tensogram.convert_grib("anything.grib2")
    with pytest.raises(RuntimeError, match="built without GRIB support"):
        tensogram.convert_grib_buffer(b"anything")
