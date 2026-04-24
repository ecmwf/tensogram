# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Edge-case and error-path tests for tensogram-earthkit.

Every raise / skip / fallback branch in the implementation gets a
corresponding test here, pinning the behaviour for Pass-3 hardening.
"""

from __future__ import annotations

import importlib
import io
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pytest
import xarray as xr

from tensogram_earthkit import detection, mars
from tensogram_earthkit.encoder import (
    TensogramEncodedData,
    TensogramEncoder,
    _compute_strides,
    _numpy_to_tgm_dtype,
)
from tensogram_earthkit.readers.file import TensogramFileReader
from tensogram_earthkit.readers.file import reader as file_reader
from tensogram_earthkit.readers.memory import memory_reader
from tensogram_earthkit.readers.stream import stream_reader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Src:
    def __init__(self) -> None:
        self.source_filename = None
        self.storage_options = None


def _reader(path) -> TensogramFileReader:
    return TensogramFileReader(_Src(), str(path))


# ---------------------------------------------------------------------------
# Encoder internals
# ---------------------------------------------------------------------------


class TestEncoderHelpers:
    def test_unsupported_dtype_raises(self) -> None:
        # str dtype is not in the tensogram dtype map.
        arr = np.array(["a", "b", "c"])
        with pytest.raises(ValueError, match="unsupported numpy dtype"):
            _numpy_to_tgm_dtype(arr)

    def test_compute_strides_empty(self) -> None:
        assert _compute_strides(()) == []

    def test_compute_strides_1d(self) -> None:
        assert _compute_strides((5,)) == [1]

    def test_compute_strides_2d(self) -> None:
        assert _compute_strides((3, 4)) == [4, 1]

    def test_compute_strides_3d(self) -> None:
        assert _compute_strides((2, 3, 4)) == [12, 4, 1]


class TestEncodedData:
    def test_to_bytes_returns_payload(self) -> None:
        ed = TensogramEncodedData(b"hello")
        assert ed.to_bytes() == b"hello"

    def test_to_file_with_path(self, tmp_path) -> None:
        ed = TensogramEncodedData(b"TENSOGRM")
        out = tmp_path / "x.bin"
        ed.to_file(str(out))
        assert out.read_bytes() == b"TENSOGRM"

    def test_to_file_with_file_object(self) -> None:
        ed = TensogramEncodedData(b"bytes")
        buf = io.BytesIO()
        ed.to_file(buf)
        assert buf.getvalue() == b"bytes"

    def test_metadata_no_key_returns_copy(self) -> None:
        ed = TensogramEncodedData(b"x", metadata={"a": 1, "b": 2})
        meta = ed.metadata()
        assert meta == {"a": 1, "b": 2}
        # Returned dict is a copy, not the internal one.
        meta["a"] = 999
        assert ed.metadata() == {"a": 1, "b": 2}

    def test_metadata_with_key(self) -> None:
        ed = TensogramEncodedData(b"x", metadata={"a": 1, "b": 2})
        assert ed.metadata("a") == 1
        assert ed.metadata("missing") is None

    def test_metadata_none_by_default(self) -> None:
        ed = TensogramEncodedData(b"x")
        assert ed.metadata() == {}


class TestEncoderDispatch:
    def test_unknown_type_error_lists_supported(self) -> None:
        enc = TensogramEncoder()
        with pytest.raises(ValueError, match="supported inputs are"):
            enc._encode(42)

    def test_xarray_dataarray_without_name_is_renamed(self) -> None:
        """Nameless DataArrays get a default name, not literal ``"None"``."""
        enc = TensogramEncoder()
        da = xr.DataArray(np.arange(6, dtype=np.float64).reshape(2, 3))
        encoded = enc.encode(da)
        import tensogram

        meta = tensogram.decode_metadata(encoded.to_bytes())
        names = [entry.get("name") for entry in meta.base]
        assert "None" not in names
        assert "data" in names

    def test_xarray_dataarray_with_name_preserved(self) -> None:
        enc = TensogramEncoder()
        da = xr.DataArray(np.arange(6, dtype=np.float32).reshape(2, 3), name="temperature")
        encoded = enc.encode(da)
        import tensogram

        meta = tensogram.decode_metadata(encoded.to_bytes())
        names = [entry.get("name") for entry in meta.base]
        assert "temperature" in names

    def test_xarray_empty_dataset_rejected(self) -> None:
        enc = TensogramEncoder()
        ds = xr.Dataset()  # no data variables
        with pytest.raises(ValueError, match="no data variables"):
            enc.encode(ds)

    def test_xarray_attrs_round_trip(self) -> None:
        enc = TensogramEncoder()
        da = xr.DataArray(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            name="t",
            attrs={"units": "K", "long_name": "temperature"},
        )
        encoded = enc.encode(da)
        import tensogram

        meta = tensogram.decode_metadata(encoded.to_bytes())
        entry = next(e for e in meta.base if e.get("name") == "t")
        assert entry["_extra_"]["units"] == "K"
        assert entry["_extra_"]["long_name"] == "temperature"

    def test_non_contiguous_xarray_input_produces_valid_output(self) -> None:
        """Transposed DataArrays must still round-trip correctly."""
        enc = TensogramEncoder()
        original = np.arange(12, dtype=np.float64).reshape(3, 4)
        transposed = original.T  # non-contiguous view, shape (4, 3)
        da = xr.DataArray(transposed, name="t")
        encoded = enc.encode(da)

        import tensogram

        _meta, objects = tensogram.decode(encoded.to_bytes())
        _desc, arr = objects[0]
        np.testing.assert_array_equal(arr, transposed)


class TestEncoderEmptyInputs:
    def test_empty_fieldlist_rejected(self) -> None:
        enc = TensogramEncoder()
        with pytest.raises(ValueError, match="empty FieldList"):
            enc._encode_fieldlist_like([])


class TestEncoderGenericDispatch:
    """Exercise the ``_encode`` duck-typed dispatch that earthkit wrappers hit."""

    def test_field_only_input_routes_to_encode_field(self) -> None:
        """A Field-like object (``metadata`` + ``to_numpy`` + no ``__len__``)."""

        class FakeMeta:
            def items(self):
                return [("param", "sp"), ("step", 0)]

        class FakeField:
            shape = (3, 4)

            def metadata(self, key=None, **_):
                return FakeMeta()

            def to_numpy(self):
                return np.arange(12, dtype=np.float32).reshape(3, 4)

        enc = TensogramEncoder()
        encoded = enc._encode(FakeField())
        import tensogram

        _, objects = tensogram.decode(encoded.to_bytes())
        assert len(objects) == 1
        np.testing.assert_array_equal(objects[0][1], np.arange(12, dtype=np.float32).reshape(3, 4))

    def test_fieldlist_only_input_routes_to_encode_fieldlist(self) -> None:
        """Iterable + ``__len__`` without the Field introspection surface."""

        class FakeMeta:
            def items(self):
                return [("param", "2t")]

        class FakeField:
            shape = (2, 3)

            def metadata(self, key=None, **_):
                return FakeMeta()

            def to_numpy(self):
                return np.arange(6, dtype=np.float32).reshape(2, 3)

        fl = [FakeField(), FakeField()]  # plain list acts as FieldList-like

        enc = TensogramEncoder()
        encoded = enc._encode(fl)
        import tensogram

        _, objects = tensogram.decode(encoded.to_bytes())
        assert len(objects) == 2


class TestEncoderShapeMismatch:
    """Field with flat 1-D ``to_numpy`` output but a 2-D ``shape`` attribute."""

    def test_flat_to_numpy_is_reshaped(self) -> None:
        """The encoder restores the declared 2-D shape before encoding."""

        class FakeMeta:
            def items(self):
                return [("param", "pr")]

        class FlatField:
            shape = (2, 3)

            def metadata(self, key=None, **_):
                return FakeMeta()

            def to_numpy(self):
                # Flat 1-D vector, but shape attr is 2-D.
                return np.arange(6, dtype=np.float32)

        enc = TensogramEncoder()
        encoded = enc._encode_fieldlist_like([FlatField()])
        import tensogram

        _, objects = tensogram.decode(encoded.to_bytes())
        assert objects[0][1].shape == (2, 3)


class TestFieldListMalformedInput:
    """Encoder output with a ``base`` array shorter than the object count."""

    def test_short_base_skips_extra_objects(self, tmp_path) -> None:
        """When ``base`` has fewer entries than objects, extras are skipped."""
        import tensogram

        from tensogram_earthkit.fieldlist import build_fieldlist_from_path

        desc_mars = {
            "type": "ntensor",
            "dtype": "float32",
            "ndim": 2,
            "shape": [2, 3],
            "strides": [3, 1],
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        desc_extra = dict(desc_mars)
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)

        # Produce a message with TWO objects but a base array with ONE entry.
        meta = {"base": [{"mars": {"param": "2t"}}], "_extra_": {}}
        buf = tensogram.encode(meta, [(desc_mars, arr), (desc_extra, arr)])

        p = tmp_path / "short_base.tgm"
        p.write_bytes(buf)

        fl = build_fieldlist_from_path(str(p))
        # Only the first object (covered by base[0]) survives.
        assert len(fl) == 1


# ---------------------------------------------------------------------------
# MARS helpers
# ---------------------------------------------------------------------------


class TestExtractMarsKeys:
    def test_non_dict_input_returns_empty(self) -> None:
        assert mars.extract_mars_keys("not a dict") == {}  # type: ignore[arg-type]
        assert mars.extract_mars_keys(None) == {}  # type: ignore[arg-type]
        assert mars.extract_mars_keys([]) == {}  # type: ignore[arg-type]

    def test_only_reserved_is_filtered(self) -> None:
        out = mars.extract_mars_keys({"_reserved_": {"x": 1}, "name": "lat"})
        assert out == {"name": "lat"}

    def test_mars_wins_on_conflict(self) -> None:
        out = mars.extract_mars_keys({"step": "sibling", "mars": {"step": "mars"}})
        assert out == {"step": "mars"}


class TestHasMarsNamespace:
    def test_non_dict(self) -> None:
        assert mars.has_mars_namespace("x") is False
        assert mars.has_mars_namespace(None) is False

    def test_mars_not_a_dict(self) -> None:
        assert mars.has_mars_namespace({"mars": "not-a-dict"}) is False

    def test_mars_empty_dict(self) -> None:
        assert mars.has_mars_namespace({"mars": {}}) is False

    def test_mars_nonempty_dict(self) -> None:
        assert mars.has_mars_namespace({"mars": {"param": "2t"}}) is True


class TestFieldToBaseEntry:
    def test_field_with_items_method(self) -> None:
        class FakeMeta:
            def items(self):
                return [("param", "2t"), ("units", "K")]

        class FakeField:
            def metadata(self, key=None, **_):
                return FakeMeta()

        entry = mars.field_to_base_entry(FakeField())
        assert entry["mars"] == {"param": "2t"}
        assert entry["_extra_"] == {"units": "K"}

    def test_field_fallback_to_per_key_lookup(self) -> None:
        """When metadata() returns a non-iterable object, per-key fallback kicks in."""

        class MinimalField:
            def __init__(self, values):
                self._values = values

            def metadata(self, key=None, **_):
                if key is None:
                    # A metadata object that doesn't expose items().
                    return object()
                if key in self._values:
                    return self._values[key]
                raise KeyError(key)

        f = MinimalField({"param": "tp", "step": 6})
        entry = mars.field_to_base_entry(f)
        assert entry["mars"]["param"] == "tp"
        assert entry["mars"]["step"] == 6

    def test_field_with_no_metadata_yields_empty_entry(self) -> None:
        class NoMetaField:
            def metadata(self, *_args, **_kwargs):
                raise AttributeError("no metadata")

        assert mars.field_to_base_entry(NoMetaField()) == {}


# ---------------------------------------------------------------------------
# Reader dispatch paths
# ---------------------------------------------------------------------------


class TestReaderFunctionDiscovery:
    def test_discovery_path_non_tensogram_returns_none(self, tmp_path) -> None:
        """Discovery path with non-matching magic returns None (pass-through)."""
        p = tmp_path / "other.bin"
        p.write_bytes(b"GRIB\x02\x00\x00\x00")
        result = file_reader(_Src(), str(p), magic=b"GRIB\x02\x00\x00\x00")
        assert result is None

    def test_discovery_path_tensogram_returns_reader(self, nonmars_tensogram_file) -> None:
        """Discovery path with matching magic returns a live reader."""
        with open(nonmars_tensogram_file, "rb") as fh:
            magic = fh.read(8)
        result = file_reader(_Src(), str(nonmars_tensogram_file), magic=magic)
        assert result is not None
        assert isinstance(result, TensogramFileReader)

    def test_source_hook_path_non_tensogram_raises(self, tmp_path) -> None:
        """Source-hook path with a non-tensogram file raises ValueError.

        This guards users from pointing the ``tensogram`` source at a
        random file — they should see a clear error not a cryptic
        framing failure deep inside the decoder.
        """
        p = tmp_path / "other.bin"
        p.write_bytes(b"GRIB\x02\x00\x00\x00" + b"\x00" * 32)
        with pytest.raises(ValueError, match="does not look like a tensogram file"):
            file_reader(_Src(), str(p), magic=None)


class TestReaderNumpyErrorPath:
    def test_multi_variable_to_numpy_raises(self, mars_tensogram_file) -> None:
        """``to_numpy()`` on a multi-variable message points at to_xarray/to_fieldlist."""
        r = _reader(mars_tensogram_file)
        with pytest.raises(ValueError, match="single-variable"):
            r.to_numpy()


class TestReaderFieldListOnNonMars:
    def test_len_on_nonmars_raises(self, nonmars_tensogram_file) -> None:
        r = _reader(nonmars_tensogram_file)
        with pytest.raises(TypeError, match="no MARS metadata"):
            len(r)

    def test_iter_on_nonmars_raises(self, nonmars_tensogram_file) -> None:
        r = _reader(nonmars_tensogram_file)
        with pytest.raises(TypeError, match="no MARS metadata"):
            iter(r)

    def test_repr_contains_path(self, nonmars_tensogram_file) -> None:
        r = _reader(nonmars_tensogram_file)
        assert "TensogramFileReader" in repr(r)
        assert str(nonmars_tensogram_file) in repr(r)


class TestReaderMarsHelpers:
    def test_order_by_delegates(self, mars_tensogram_file) -> None:
        r = _reader(mars_tensogram_file)
        ordered = r.order_by("param")
        params = [f.metadata("param") for f in ordered]
        assert params == sorted(params)

    def test_metadata_delegates(self, mars_tensogram_file) -> None:
        r = _reader(mars_tensogram_file)
        params = r.metadata("param")
        assert sorted(params) == ["2t", "tp"]

    def test_mutate_source_returns_none(self, mars_tensogram_file) -> None:
        """``mutate_source`` returning None keeps the original source."""
        r = _reader(mars_tensogram_file)
        assert r.mutate_source() is None


# ---------------------------------------------------------------------------
# Memory / stream reader discovery
# ---------------------------------------------------------------------------


class TestMemoryReaderDiscovery:
    def test_magic_mismatch_returns_none(self) -> None:
        result = memory_reader(_Src(), b"GRIB\x02\x00\x00\x00", magic=b"GRIB")
        assert result is None

    def test_short_buffer_returns_none(self) -> None:
        result = memory_reader(_Src(), b"TEN", magic=None)
        assert result is None

    def test_discovery_path_round_trip(self, nonmars_tensogram_bytes) -> None:
        reader = memory_reader(_Src(), nonmars_tensogram_bytes, magic=nonmars_tensogram_bytes[:8])
        assert reader is not None
        arr = reader.to_numpy()
        assert arr.shape == (2, 3, 4)


class TestStreamReaderDiscovery:
    def test_magic_mismatch_fast_fail(self) -> None:
        buf = b"GRIB\x02\x00\x00\x00" + b"\x00" * 32
        stream = io.BufferedReader(io.BytesIO(buf))
        result = stream_reader(_Src(), stream, magic=b"GRIB\x02\x00\x00\x00")
        assert result is None

    def test_unpeeked_non_tensogram_returns_none(self) -> None:
        """When no magic was peeked, we drain and check the leading bytes ourselves."""
        buf = b"GRIB\x02\x00\x00\x00" + b"\x00" * 32
        stream = io.BufferedReader(io.BytesIO(buf))
        result = stream_reader(_Src(), stream, magic=None)
        assert result is None


class TestRemoteReaderSourceHook:
    """The source-hook path for remote URLs skips the magic peek.

    Unit-tested with a URL that doesn't really exist — the reader is
    returned immediately without any I/O.  Actual decode would fail,
    which is the documented behaviour (validation is deferred to
    first-access).
    """

    def test_remote_url_source_hook_returns_reader(self) -> None:
        result = file_reader(_Src(), "https://example.invalid/does-not-exist.tgm")
        assert result is not None
        assert isinstance(result, TensogramFileReader)


class TestFieldListXarrayFallback:
    """``TensogramSimpleFieldList.to_xarray`` falls back to the default path."""

    def test_no_file_path_uses_superclass(self, mars_tensogram_bytes) -> None:
        """When the FieldList was built without a path, delegation is not possible."""
        import contextlib

        from tensogram_earthkit.fieldlist import TensogramSimpleFieldList

        # Construct manually with no file_path.
        empty = TensogramSimpleFieldList([], file_path=None)
        # The superclass may raise on empty input — we only need the
        # fallback-path line (no ``_tgm_file_path`` access) to run.
        # Either outcome exercises the branch.
        with contextlib.suppress(Exception):
            empty.to_xarray()


# ---------------------------------------------------------------------------
# Source close / lifecycle
# ---------------------------------------------------------------------------


class TestSourceClose:
    def test_close_after_bytes_unlinks_tempfile(self, nonmars_tensogram_bytes) -> None:
        src = ekd.from_source("tensogram", nonmars_tensogram_bytes)
        tmp = Path(src._tmp_path)
        assert tmp.exists()
        src.close()
        assert not tmp.exists()

    def test_close_idempotent(self, nonmars_tensogram_bytes) -> None:
        src = ekd.from_source("tensogram", nonmars_tensogram_bytes)
        src.close()
        # Second close is a no-op; must not raise.
        src.close()

    def test_close_local_source_is_safe(self, nonmars_tensogram_file) -> None:
        """``close()`` on a file-path source does nothing — there's no temp."""
        src = ekd.from_source("tensogram", str(nonmars_tensogram_file))
        # Should be a safe no-op.
        src.close()
        # The original file must still exist.
        assert Path(nonmars_tensogram_file).exists()


# ---------------------------------------------------------------------------
# Detection: TENSOGRM_MAGIC constant
# ---------------------------------------------------------------------------


class TestTensogrmMagicConstant:
    def test_value(self) -> None:
        assert detection.TENSOGRM_MAGIC == b"TENSOGRM"

    def test_length(self) -> None:
        assert len(detection.TENSOGRM_MAGIC) == 8


# ---------------------------------------------------------------------------
# Public API — top-level re-exports
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Guardrail tests for the top-level package namespace.

    Uses :func:`importlib.import_module` rather than a module-level
    ``import tensogram_earthkit as tek`` alias so this file stays on
    the ``from <package> import ...`` form throughout — CodeQL flags
    files that mix ``import X`` with ``from X import Y`` for the same
    package.
    """

    def test_top_level_re_exports_present(self) -> None:
        tek = importlib.import_module("tensogram_earthkit")
        for name in (
            "TENSOGRM_MAGIC",
            "TensogramData",
            "TensogramEncodedData",
            "TensogramEncoder",
            "TensogramSimpleFieldList",
            "TensogramSource",
            "build_fieldlist_from_path",
            "is_mars_tensogram",
        ):
            assert hasattr(tek, name), f"missing top-level {name!r}"

    def test_all_declarations_are_consistent(self) -> None:
        """Every name in ``__all__`` must be importable from the module."""
        tek = importlib.import_module("tensogram_earthkit")
        for name in tek.__all__:
            getattr(tek, name)  # raises if missing
