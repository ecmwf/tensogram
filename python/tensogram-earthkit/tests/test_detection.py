# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for :mod:`tensogram_earthkit.detection`.

Covers two responsibilities:

* **Magic-byte detection** — ``TENSOGRM`` (8 bytes) at offset 0 identifies a
  tensogram message.  Both the quick (first-4-bytes) and deeper (search
  whole buffer) passes are tested, matching the two-pass earthkit-data
  reader protocol.
* **MARS-flavour detection** — scans ``meta.base[i]`` for at least one
  entry with a non-empty ``"mars"`` sub-map.  Drives the branch between
  the ``FieldList`` and xarray-only data paths.
"""

from __future__ import annotations

import pytest

from tensogram_earthkit import detection

TENSOGRM = b"TENSOGRM"


# ---------------------------------------------------------------------------
# Magic-byte matcher
# ---------------------------------------------------------------------------


class TestMatchMagic:
    def test_exact_prefix_quick_pass(self) -> None:
        """TENSOGRM at offset 0 matches on the quick pass."""
        assert detection._match_magic(TENSOGRM + b"\x00\x03", deeper_check=False) is True

    def test_full_preamble_quick_pass(self) -> None:
        """A full 24-byte preamble passes the quick check."""
        preamble = TENSOGRM + b"\x00\x03" + b"\x00" * 14  # magic + version 3 + pad
        assert detection._match_magic(preamble, deeper_check=False) is True

    def test_non_tensogram_quick_pass(self) -> None:
        """Other format magics are rejected on the quick pass."""
        assert detection._match_magic(b"GRIB\x00\x00\x00\x02", deeper_check=False) is False
        assert detection._match_magic(b"\x89HDF\r\n\x1a\n", deeper_check=False) is False
        assert detection._match_magic(b"CDF\x01\x00\x00\x00\x00", deeper_check=False) is False

    def test_offset_magic_rejected_quick_pass(self) -> None:
        """TENSOGRM not at offset 0 must NOT match on the quick pass."""
        buf = b"\x00\x00\x00\x00" + TENSOGRM
        assert detection._match_magic(buf, deeper_check=False) is False

    def test_offset_magic_accepted_deeper_pass(self) -> None:
        """Deeper pass searches the whole buffer for TENSOGRM."""
        buf = b"\x00\x00\x00\x00" + TENSOGRM + b"\x00\x03"
        assert detection._match_magic(buf, deeper_check=True) is True

    def test_empty_buffer(self) -> None:
        assert detection._match_magic(b"", deeper_check=False) is False
        assert detection._match_magic(b"", deeper_check=True) is False

    def test_short_buffer_under_four_bytes(self) -> None:
        assert detection._match_magic(b"TEN", deeper_check=False) is False
        assert detection._match_magic(b"TEN", deeper_check=True) is False

    def test_short_buffer_four_to_seven_bytes(self) -> None:
        """Four-to-seven bytes is not enough to confirm the 8-byte magic."""
        assert detection._match_magic(b"TENSOGR", deeper_check=False) is False
        assert detection._match_magic(b"TENSOGR", deeper_check=True) is False

    def test_none_buffer(self) -> None:
        """A ``None`` buffer (magic unavailable) must return False."""
        assert detection._match_magic(None, deeper_check=False) is False
        assert detection._match_magic(None, deeper_check=True) is False

    def test_close_but_wrong_magic(self) -> None:
        """``TENSOGRN`` (one-byte off) must not match."""
        assert detection._match_magic(b"TENSOGRN" + b"\x00" * 16, deeper_check=False) is False
        assert detection._match_magic(b"TENSOGRN" + b"\x00" * 16, deeper_check=True) is False


# ---------------------------------------------------------------------------
# MARS-tensogram detection
# ---------------------------------------------------------------------------


class _FakeMeta:
    """Duck-typed stand-in for ``tensogram.Metadata``."""

    def __init__(self, base: list[dict] | None) -> None:
        self.base = base
        self.extra: dict = {}


class TestIsMarsTensogram:
    def test_no_base_entries(self) -> None:
        meta = _FakeMeta(base=[])
        assert detection.is_mars_tensogram(meta) is False

    def test_base_is_none(self) -> None:
        meta = _FakeMeta(base=None)
        assert detection.is_mars_tensogram(meta) is False

    def test_single_mars_entry(self) -> None:
        meta = _FakeMeta(base=[{"mars": {"param": "2t", "step": 0}}])
        assert detection.is_mars_tensogram(meta) is True

    def test_mixed_entries_any_mars_wins(self) -> None:
        """Any MARS entry anywhere in base[] flips the flag to True."""
        meta = _FakeMeta(base=[{}, {"mars": {"param": "t"}}, {"other": "x"}])
        assert detection.is_mars_tensogram(meta) is True

    def test_empty_mars_sub_map_counts_as_non_mars(self) -> None:
        """An empty ``"mars": {}`` must NOT count as a MARS tensogram.

        Empty MARS namespaces add no semantic value; the FieldList path
        would have nothing to work with so we fall back to xarray-only.
        """
        meta = _FakeMeta(base=[{"mars": {}}])
        assert detection.is_mars_tensogram(meta) is False

    def test_non_dict_mars_value_rejected(self) -> None:
        """``"mars"`` must be a dict — lists / scalars / None don't count."""
        for bad in (None, [], "2t", 42):
            meta = _FakeMeta(base=[{"mars": bad}])
            assert detection.is_mars_tensogram(meta) is False, f"rejected {bad!r}"

    def test_non_dict_base_entry_ignored(self) -> None:
        """If a base entry is not a dict it is silently skipped."""
        meta = _FakeMeta(base=["not-a-dict", {"mars": {"param": "2t"}}])
        assert detection.is_mars_tensogram(meta) is True

    def test_mars_key_at_wrong_level_rejected(self) -> None:
        """``mars`` at message-level (extra) doesn't make it a MARS tensogram."""
        meta = _FakeMeta(base=[{}])
        meta.extra = {"mars": {"param": "2t"}}
        assert detection.is_mars_tensogram(meta) is False


# ---------------------------------------------------------------------------
# Integration: detect MARS in a real encoded tensogram message
# ---------------------------------------------------------------------------


class TestIsMarsTensogramIntegration:
    """Exercise MARS detection against actually encoded ``.tgm`` bytes."""

    def test_real_mars_message(self) -> None:
        tensogram = pytest.importorskip("tensogram")
        np = pytest.importorskip("numpy")

        descriptor = {
            "type": "ntensor",
            "dtype": "float32",
            "ndim": 2,
            "shape": [2, 3],
            "strides": [3, 1],
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        meta_in = {
            "base": [{"mars": {"param": "2t", "step": 0}}],
            "_extra_": {},
        }
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        buf = tensogram.encode(meta_in, [(descriptor, data)])

        meta_out = tensogram.decode_metadata(buf)
        assert detection.is_mars_tensogram(meta_out) is True

    def test_real_non_mars_message(self) -> None:
        tensogram = pytest.importorskip("tensogram")
        np = pytest.importorskip("numpy")

        descriptor = {
            "type": "ntensor",
            "dtype": "float64",
            "ndim": 1,
            "shape": [4],
            "strides": [1],
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        meta_in = {"base": [{}], "_extra_": {"label": "generic"}}
        data = np.arange(4, dtype=np.float64)
        buf = tensogram.encode(meta_in, [(descriptor, data)])

        meta_out = tensogram.decode_metadata(buf)
        assert detection.is_mars_tensogram(meta_out) is False
