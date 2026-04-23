# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Remote URL handling for the tensogram earthkit source.

Uses an in-process HTTP server with HTTP Range support so we can test
the remote path without any external network dependency.  The pattern
mirrors ``python/tensogram-xarray/tests/test_remote.py`` so behaviour
stays consistent across the tensogram Python stack.

Contract:

* ``ekd.from_source("tensogram", "http://…/f.tgm")`` opens the file
  remotely without downloading it up-front.
* ``.to_xarray()`` and ``.to_numpy()`` decode correctly.
* For MARS content, ``.to_fieldlist()`` decodes correctly.
* ``storage_options`` are forwarded and produce no error for HTTP
  (they are meaningful for S3/GCS/Azure).
* Parity: remote vs local reads match byte-for-byte.
"""

from __future__ import annotations

import http.server
import threading

import earthkit.data as ekd
import numpy as np
import pytest
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
def http_server_nonmars(nonmars_tensogram_bytes):
    """Spin up an HTTP server for the non-MARS tensogram bytes."""
    handler = _make_handler(nonmars_tensogram_bytes)
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/generic.tgm"
    try:
        yield url
    finally:
        server.shutdown()
        thread.join(timeout=2)


@pytest.fixture
def http_server_mars(mars_tensogram_bytes):
    """Spin up an HTTP server for the MARS tensogram bytes."""
    handler = _make_handler(mars_tensogram_bytes)
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/mars.tgm"
    try:
        yield url
    finally:
        server.shutdown()
        thread.join(timeout=2)


class TestRemoteNonMars:
    def test_open_via_http_returns_source(self, http_server_nonmars) -> None:
        data = ekd.from_source("tensogram", http_server_nonmars)
        assert data is not None
        assert hasattr(data, "to_xarray")

    def test_http_to_xarray(self, http_server_nonmars) -> None:
        data = ekd.from_source("tensogram", http_server_nonmars)
        ds = data.to_xarray()
        assert isinstance(ds, xr.Dataset)
        var = next(iter(ds.data_vars.values()))
        assert var.shape == (2, 3, 4)

    def test_http_to_numpy_values_match_local(
        self, http_server_nonmars, nonmars_tensogram_file
    ) -> None:
        via_http = ekd.from_source("tensogram", http_server_nonmars).to_numpy()
        via_file = ekd.from_source("tensogram", str(nonmars_tensogram_file)).to_numpy()
        np.testing.assert_array_equal(via_http, via_file)

    def test_storage_options_accepted(self, http_server_nonmars) -> None:
        """``storage_options`` threads through to the xarray backend."""
        # Empty dict is always safe — exercises the forwarding path
        # without requiring any specific object_store key shape.
        data = ekd.from_source("tensogram", http_server_nonmars, storage_options={})
        ds = data.to_xarray()
        assert isinstance(ds, xr.Dataset)

    def test_storage_options_stashed_on_source(self, http_server_nonmars) -> None:
        """The storage_options dict is accessible to the reader path."""
        data = ekd.from_source(
            "tensogram",
            http_server_nonmars,
            storage_options={"example": "sentinel"},
        )
        assert data.storage_options == {"example": "sentinel"}


class TestRemoteMars:
    def test_http_to_fieldlist(self, http_server_mars) -> None:
        data = ekd.from_source("tensogram", http_server_mars)
        fl = data.to_fieldlist()
        assert len(fl) == 2
        params = sorted(f.metadata("param") for f in fl)
        assert params == ["2t", "tp"]

    def test_remote_mars_matches_local(self, http_server_mars, mars_tensogram_file) -> None:
        fl_remote = ekd.from_source("tensogram", http_server_mars).to_fieldlist()
        fl_local = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        assert len(fl_remote) == len(fl_local)
        for remote, local in zip(fl_remote, fl_local, strict=True):
            assert remote.metadata("param") == local.metadata("param")
            np.testing.assert_array_equal(remote.to_numpy(), local.to_numpy())


class TestRemoteDetection:
    def test_is_remote_url_matches_tensogram_core(self) -> None:
        """Our source detects remote URLs using tensogram's own helper."""
        import tensogram

        assert tensogram.is_remote_url("https://example.com/a.tgm") is True
        assert tensogram.is_remote_url("s3://bucket/a.tgm") is True
        assert tensogram.is_remote_url("/tmp/a.tgm") is False
