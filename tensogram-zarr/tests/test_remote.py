"""Remote URL support tests for tensogram-zarr."""

from __future__ import annotations

import http.server
import threading

import numpy as np
import pytest
import tensogram
from tensogram_zarr.store import TensogramStore


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
                spec = range_header[6:]
                if spec.startswith("-"):
                    n = int(spec[1:])
                    s, e = max(0, len(data) - n), len(data)
                else:
                    parts = spec.split("-")
                    s = int(parts[0])
                    e = int(parts[1]) + 1 if parts[1] else len(data)
                    e = min(e, len(data))
                if s >= len(data):
                    self.send_response(416)
                    self.end_headers()
                    return
                chunk = data[s:e]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {s}-{e - 1}/{len(data)}")
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
    """Start a mock HTTP server for given bytes, isolated per call."""
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
        s.server_close()


def _encode_simple_message() -> bytes:
    meta = {"version": 2}
    desc = {
        "type": "ntensor",
        "shape": [4],
        "dtype": "float32",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }
    data = np.arange(4, dtype=np.float32)
    return tensogram.encode(meta, [(desc, data)])


class TestZarrRemoteRead:
    def test_open_tgm_remote(self, serve_tgm_bytes):
        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        assert store._path == url

    def test_open_tgm_remote_with_storage_options(self, serve_tgm_bytes):
        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url, storage_options={"allow_http": "true"})
        assert store._path == url


class TestZarrRemoteWriteRejection:
    def test_remote_write_rejected(self):
        with pytest.raises(ValueError, match="remote URLs do not support"):
            TensogramStore("s3://bucket/file.tgm", mode="w")

    def test_remote_append_rejected(self):
        with pytest.raises(ValueError, match="remote URLs do not support"):
            TensogramStore("s3://bucket/file.tgm", mode="a")

    def test_local_write_still_works(self, tmp_path):
        path = str(tmp_path / "write.tgm")
        store = TensogramStore(path, mode="w")
        assert store.supports_writes
