# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Shared pytest fixtures for tensogram Python tests."""

from __future__ import annotations

import http.server
import threading

import numpy as np
import pytest
import tensogram


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
    """Start a mock HTTP server with Range support, isolated per call."""
    entries: list[tuple] = []

    def _serve(data: bytes) -> str:
        handler = _make_handler(data)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        entries.append((server, thread))
        return f"http://127.0.0.1:{port}/test.tgm"

    yield _serve

    for server, thread in entries:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def tgm_path(tmp_path):
    """Create a local .tgm with 3 messages (float32 [10], fill=0.0/1.0/2.0)."""
    path = str(tmp_path / "test.tgm")
    with tensogram.TensogramFile.create(path) as f:
        for i in range(3):
            meta = {"version": 2, "base": [{"index": i}]}
            desc = {
                "type": "ntensor",
                "shape": [10],
                "dtype": "float32",
                "byte_order": "little",
                "encoding": "none",
                "filter": "none",
                "compression": "none",
            }
            data = np.full(10, float(i), dtype=np.float32)
            f.append(meta, [(desc, data)])
    return path
