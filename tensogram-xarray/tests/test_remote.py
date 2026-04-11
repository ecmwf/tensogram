"""Remote URL support tests for tensogram-xarray."""

from __future__ import annotations

import http.server
import threading

import numpy as np
import pytest
import tensogram
import xarray as xr


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
def serve_tgm(tmp_path):
    """Create a .tgm file, serve it over HTTP, return (url, local_path)."""
    servers = []

    def _serve(data: np.ndarray, shape: list[int], dtype: str = "float32") -> tuple[str, str]:
        meta = {"version": 2}
        desc = {
            "type": "ntensor",
            "shape": shape,
            "dtype": dtype,
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        msg_bytes = tensogram.encode(meta, [(desc, data)])

        local_path = str(tmp_path / "test.tgm")
        with open(local_path, "wb") as f:
            f.write(msg_bytes)

        handler = _make_handler(msg_bytes)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        return f"http://127.0.0.1:{port}/test.tgm", local_path

    yield _serve

    for s in servers:
        s.shutdown()
        s.server_close()


class TestXarrayRemoteOpen:
    def test_open_dataset_remote(self, serve_tgm):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        url, _ = serve_tgm(data, [3, 4])

        ds = xr.open_dataset(url, engine="tensogram")
        assert len(ds.data_vars) == 1
        var = next(iter(ds.data_vars.values()))
        np.testing.assert_array_equal(var.values, data)

    def test_remote_matches_local(self, serve_tgm):
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        url, local_path = serve_tgm(data, [4, 5])

        ds_remote = xr.open_dataset(url, engine="tensogram")
        ds_local = xr.open_dataset(local_path, engine="tensogram")

        for var_name in ds_local.data_vars:
            np.testing.assert_array_equal(
                ds_remote[var_name].values,
                ds_local[var_name].values,
            )

    def test_open_dataset_with_storage_options(self, serve_tgm):
        data = np.full((2, 3), 7.0, dtype=np.float32)
        url, _ = serve_tgm(data, [2, 3])

        ds = xr.open_dataset(
            url,
            engine="tensogram",
            storage_options={"allow_http": "true"},
        )
        var = next(iter(ds.data_vars.values()))
        np.testing.assert_allclose(var.values, 7.0)
