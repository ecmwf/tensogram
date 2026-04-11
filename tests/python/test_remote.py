"""Tests for remote URL support in Python bindings, xarray, and zarr."""

from __future__ import annotations

import http.server
import threading
from typing import Any

import numpy as np
import pytest
import tensogram


# ---------------------------------------------------------------------------
# Mock HTTP server with Range support
# ---------------------------------------------------------------------------


def _make_handler(file_data: bytes):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(file_data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            data = file_data
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                range_spec = range_header[6:]
                if range_spec.startswith("-"):
                    suffix = int(range_spec[1:])
                    start = max(0, len(data) - suffix)
                    end = len(data)
                else:
                    parts = range_spec.split("-")
                    start = int(parts[0])
                    end = int(parts[1]) + 1 if parts[1] else len(data)
                    end = min(end, len(data))
                if start >= len(data):
                    self.send_response(416)
                    self.end_headers()
                    return
                chunk = data[start:end]
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{end - 1}/{len(data)}"
                )
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    return Handler


@pytest.fixture
def serve_tgm_bytes():
    """Fixture that starts a mock HTTP server for given bytes, isolated per call."""
    servers = []

    def _serve(data: bytes) -> str:
        handler = _make_handler(data)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        return f"http://127.0.0.1:{port}/test.tgm"

    yield _serve

    for s in servers:
        s.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_global_meta(version: int = 2, **extra: Any) -> dict[str, Any]:
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
            assert meta.version == 2

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
            assert meta.version == 2
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
            local_meta, local_objects = local.decode_message(0)

        np.testing.assert_array_equal(remote_result["data"], local_objects[0][1])


class TestPythonRemoteErrors:
    def test_invalid_url(self):
        with pytest.raises(Exception):
            tensogram.TensogramFile.open("http://[invalid]/file.tgm")

    def test_open_remote_bad_storage_option_value(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        class Unconvertible:
            def __str__(self):
                raise RuntimeError("cannot convert")

        with pytest.raises(Exception, match="convertible to string"):
            tensogram.TensogramFile.open_remote(url, {"key": Unconvertible()})

    def test_iteration_not_supported_on_remote(self, serve_tgm_bytes):
        msg = encode_test_message([4])
        url = serve_tgm_bytes(msg)

        with tensogram.TensogramFile.open(url) as f:
            with pytest.raises(RuntimeError, match="iteration not supported"):
                iter(f)
