# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Remote URL support tests for tensogram-zarr."""

from __future__ import annotations

import http.server
import threading
import warnings

import numpy as np
import pytest
import tensogram
import zarr
from tensogram_zarr.store import TensogramStore


async def _collect_async_iter(ait):
    return [item async for item in ait]


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


class TestZarrRemoteLazyReads:
    def test_remote_open_does_not_fetch_chunk_data(self, serve_tgm_bytes):
        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        chunk_keys_in_keys = [k for k in store._keys if "/c/" in k]
        assert len(chunk_keys_in_keys) == 0, "chunk data should not be in _keys for remote"
        assert len(store._chunk_index) == 1, "chunk index should have one entry"
        assert store._file is not None, "file handle should be kept open for lazy reads"

    def test_remote_chunk_decoded_on_access(self, serve_tgm_bytes):
        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        chunk_key = next(iter(store._chunk_index))
        data = store._decode_chunk(chunk_key)
        assert data is not None
        arr = np.frombuffer(data, dtype=np.float32)
        np.testing.assert_array_equal(arr, np.arange(4, dtype=np.float32))

    def test_remote_close_releases_handle(self, serve_tgm_bytes):
        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        assert store._file is not None
        store.close()
        assert store._file is None
        assert len(store._chunk_index) == 0

    def test_remote_chunk_cached_on_repeat_access(self, serve_tgm_bytes):
        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        chunk_key = next(iter(store._chunk_index))
        assert chunk_key not in store._keys

        store._decode_chunk(chunk_key)
        assert chunk_key in store._keys, "decoded chunk should be cached in _keys"
        assert chunk_key not in store._chunk_index, "should be removed from index after caching"

    def test_remote_exists_sees_lazy_chunk(self, serve_tgm_bytes):
        import asyncio

        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        chunk_key = next(iter(store._chunk_index))
        assert asyncio.run(store.exists(chunk_key))

    def test_remote_list_includes_lazy_chunks(self, serve_tgm_bytes):
        import asyncio

        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        keys = asyncio.run(_collect_async_iter(store.list()))
        chunk_keys = [k for k in keys if "/c/" in k]
        assert len(chunk_keys) == 1

    def test_remote_list_no_duplicates_after_cache(self, serve_tgm_bytes):
        import asyncio

        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        chunk_key = next(iter(store._chunk_index))
        store._decode_chunk(chunk_key)

        keys = asyncio.run(_collect_async_iter(store.list()))
        chunk_keys = [k for k in keys if "/c/" in k]
        assert len(chunk_keys) == 1, f"expected 1 chunk key, got {chunk_keys}"

    def test_remote_exit_cleans_up_on_exception(self, serve_tgm_bytes):
        msg = _encode_simple_message()
        url = serve_tgm_bytes(msg)

        store = TensogramStore.open_tgm(url)
        try:
            with store:
                raise RuntimeError("test exception")
        except RuntimeError:
            pass
        assert store._file is None
        assert len(store._chunk_index) == 0
        assert not store._is_open

    def test_local_still_eager(self, tmp_path):
        msg = _encode_simple_message()
        path = str(tmp_path / "test.tgm")
        with open(path, "wb") as fh:
            fh.write(msg)

        store = TensogramStore.open_tgm(path)
        chunk_keys_in_keys = [k for k in store._keys if "/c/" in k]
        assert len(chunk_keys_in_keys) == 1, "local files should decode chunks eagerly"
        assert len(store._chunk_index) == 0, "no lazy index for local files"
        assert store._file is None, "no persistent handle for local files"


class TestRemoteIssue67DescriptorNameFallback:
    def test_remote_scan_surfaces_descriptor_level_name(self, serve_tgm_bytes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            meta = {"version": 2}
            desc = {
                "type": "ntensor",
                "shape": [4],
                "dtype": "float32",
                "byte_order": "little",
                "encoding": "none",
                "filter": "none",
                "compression": "none",
                "name": "temperature",
            }
            msg = tensogram.encode(meta, [(desc, np.arange(4, dtype=np.float32))])

        url = serve_tgm_bytes(msg)
        with TensogramStore.open_tgm(url) as store:
            root = zarr.open_group(store=store, mode="r")
            assert list(root.keys()) == ["temperature"]


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
