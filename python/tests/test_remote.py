# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for remote URL support in Python bindings, xarray, and zarr."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import tensogram

# serve_tgm_bytes fixture is auto-discovered from conftest.py


def make_global_meta(version: int = 3, **extra: Any) -> dict[str, Any]:
    return {"version": version, **extra}


def make_descriptor(shape: list[int], dtype: str = "float32") -> dict[str, Any]:
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": dtype,
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


def encode_test_message(shape: list[int], fill: float = 42.0) -> bytes:
    meta = make_global_meta()
    desc = make_descriptor(shape)
    data = np.full(shape, fill, dtype=np.float32)
    return tensogram.encode(meta, [(desc, data)])


# ---------------------------------------------------------------------------
# Python binding remote tests
# ---------------------------------------------------------------------------


class TestPythonRemoteOpen:
    def test_open_auto_detects_remote(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open(url) as f:
            assert f.is_remote()
            assert f.source() == url
            assert f.message_count() == 1

    def test_open_remote_explicit(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open_remote(url) as f:
            assert f.is_remote()
            assert f.message_count() == 1

    def test_open_remote_with_storage_options(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open_remote(url, {"allow_http": "true"}) as f:
            assert f.is_remote()

    def test_open_local_still_works(self, tmp_path):
        path = str(tmp_path / "local.tgm")
        with tensogram.TensogramFile.create(path) as f:
            meta = make_global_meta()
            desc = make_descriptor([4])
            data = np.full([4], 1.0, dtype=np.float32)
            f.append(meta, [(desc, data)])

        with tensogram.TensogramFile.open(path) as f:
            assert not f.is_remote()
            assert f.message_count() == 1


class TestPythonRemoteDecode:
    def test_file_decode_metadata(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open(url) as f:
            meta = f.file_decode_metadata(0)
            assert meta.version == 3

    def test_file_decode_descriptors(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open(url) as f:
            result = f.file_decode_descriptors(0)
            assert "metadata" in result
            assert "descriptors" in result
            descs = result["descriptors"]
            assert len(descs) == 1
            assert list(descs[0].shape) == [4]

    def test_file_decode_object(self, serve_tgm_bytes):
        msg = encode_test_message([4], fill=99.0)
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open(url) as f:
            result = f.file_decode_object(0, 0)
            assert "metadata" in result
            assert "descriptor" in result
            assert "data" in result
            arr = result["data"]
            assert arr.shape == (4,)
            np.testing.assert_allclose(arr, np.full(4, 99.0, dtype=np.float32))

    def test_decode_message_still_works_remote(self, serve_tgm_bytes):
        msg = encode_test_message([4], fill=7.0)
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open(url) as f:
            meta, objects = f.decode_message(0)
            assert meta.version == 3
            assert len(objects) == 1
            np.testing.assert_allclose(objects[0][1], np.full(4, 7.0, dtype=np.float32))

    def test_remote_matches_local_decode(self, serve_tgm_bytes, tmp_path):
        msg = encode_test_message([10], fill=3.14)

        url = serve_tgm_bytes(msg)
        local_path = str(tmp_path / "local.tgm")
        with open(local_path, "wb") as fh:
            fh.write(msg)

        with tensogram.TensogramFile.open(url) as remote:
            remote_result = remote.file_decode_object(0, 0)

        with tensogram.TensogramFile.open(local_path) as local:
            _local_meta, local_objects = local.decode_message(0)

        np.testing.assert_array_equal(remote_result["data"], local_objects[0][1])


class TestPythonRemoteErrors:
    def test_invalid_url(self):
        with pytest.raises(OSError, match="invalid"):
            tensogram.TensogramFile.open("http://[invalid]/file.tgm")

    def test_open_remote_bad_storage_option_value(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        class Unconvertible:
            def __str__(self):
                raise RuntimeError("cannot convert")

        with pytest.raises(ValueError, match="convertible to string"):
            tensogram.TensogramFile.open_remote(url, {"key": Unconvertible()})

    def test_iteration_not_supported_on_remote(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with (
            tensogram.TensogramFile.open(url) as f,
            pytest.raises(RuntimeError, match="iteration not supported"),
        ):
            iter(f)


# ---------------------------------------------------------------------------
# Bidirectional scan opt-in
# ---------------------------------------------------------------------------


class TestBidirectionalKwarg:
    def test_default_is_forward_only(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open_remote(url) as f:
            assert f.message_count() == 1

    def test_open_remote_bidirectional_layouts_match_forward_only(
        self, serve_tgm_bytes
    ):
        msg1 = encode_test_message([4], fill=10.0)
        msg2 = encode_test_message([8], fill=20.0)
        msg3 = encode_test_message([16], fill=30.0)
        url = serve_tgm_bytes(msg1 + msg2 + msg3)

        with tensogram.TensogramFile.open_remote(url) as fwd:
            fwd_count = fwd.message_count()

        with tensogram.TensogramFile.open_remote(url, bidirectional=True) as bidir:
            bidir_count = bidir.message_count()

        assert fwd_count == bidir_count == 3

    def test_open_with_bidirectional_kwarg(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open(url, bidirectional=True) as f:
            assert f.is_remote()
            assert f.message_count() == 1

    def test_open_local_path_with_bidirectional_kwarg_is_no_op(self, tmp_path):
        path = str(tmp_path / "local.tgm")
        with tensogram.TensogramFile.create(path) as f:
            meta = make_global_meta()
            desc = make_descriptor([4])
            data = np.full([4], 1.0, dtype=np.float32)
            f.append(meta, [(desc, data)])

        with tensogram.TensogramFile.open(path, bidirectional=True) as f:
            assert not f.is_remote()
            assert f.message_count() == 1

    def test_bidirectional_must_be_bool_int_rejected(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with pytest.raises(TypeError):
            tensogram.TensogramFile.open_remote(url, bidirectional=1)

    def test_bidirectional_must_be_bool_str_rejected(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with pytest.raises(TypeError):
            tensogram.TensogramFile.open_remote(url, bidirectional="yes")

    def test_bidirectional_is_keyword_only(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with pytest.raises(TypeError):
            tensogram.TensogramFile.open_remote(url, None, True)
