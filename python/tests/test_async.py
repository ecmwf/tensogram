# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for AsyncTensogramFile — Python asyncio bindings."""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np
import pytest
import tensogram


def _encode_test_message(shape: list[int], fill: float = 42.0) -> bytes:
    meta = {"version": 3, "base": [{}]}
    desc = {
        "type": "ntensor",
        "shape": shape,
        "dtype": "float32",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }
    data = np.full(shape, fill, dtype=np.float32)
    return tensogram.encode(meta, [(desc, data)])


class TestAsyncOpen:
    @pytest.mark.asyncio
    async def test_open_local(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        assert not f.is_remote()
        assert f.source() == tgm_path

    @pytest.mark.asyncio
    async def test_open_remote(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        assert f.is_remote()
        assert f.source() == url

    @pytest.mark.asyncio
    async def test_open_remote_explicit(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open_remote(url)
        assert f.is_remote()

    @pytest.mark.asyncio
    async def test_open_remote_with_storage_options(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open_remote(url, {"allow_http": "true"})
        assert f.is_remote()

    @pytest.mark.asyncio
    async def test_open_nonexistent(self):
        with pytest.raises(OSError, match=r"[Nn]o such|not found"):
            _ = await tensogram.AsyncTensogramFile.open("/nonexistent/path.tgm")

    @pytest.mark.asyncio
    async def test_repr(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        r = repr(f)
        assert "AsyncTensogramFile" in r
        assert tgm_path in r

    @pytest.mark.asyncio
    async def test_open_remote_bidirectional_kwarg(self, serve_tgm_bytes):
        msg1 = _encode_test_message([4], fill=10.0)
        msg2 = _encode_test_message([8], fill=20.0)
        msg3 = _encode_test_message([16], fill=30.0)
        url = serve_tgm_bytes(msg1 + msg2 + msg3)

        fwd = await tensogram.AsyncTensogramFile.open_remote(url)
        fwd_count = await fwd.message_count()

        bidir = await tensogram.AsyncTensogramFile.open_remote(url, bidirectional=True)
        bidir_count = await bidir.message_count()

        assert fwd_count == bidir_count == 3

    @pytest.mark.asyncio
    async def test_open_with_bidirectional_kwarg(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url, bidirectional=True)
        assert f.is_remote()
        assert await f.message_count() == 1

    @pytest.mark.asyncio
    async def test_open_local_path_with_bidirectional_kwarg_is_no_op(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path, bidirectional=True)
        assert not f.is_remote()

    @pytest.mark.asyncio
    async def test_bidirectional_int_rejected_eagerly(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        with pytest.raises(TypeError):
            tensogram.AsyncTensogramFile.open_remote(url, bidirectional=1)

    @pytest.mark.asyncio
    async def test_bidirectional_str_rejected_eagerly(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        with pytest.raises(TypeError):
            tensogram.AsyncTensogramFile.open_remote(url, bidirectional="yes")


class TestAsyncMessageCount:
    @pytest.mark.asyncio
    async def test_message_count(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        count = await f.message_count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_message_count_single(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        count = await f.message_count()
        assert count == 1


class TestAsyncDecode:
    @pytest.mark.asyncio
    async def test_decode_message(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        meta, objects = await f.decode_message(0)
        assert meta.version == 3
        assert len(objects) == 1
        _desc, arr = objects[0]
        assert arr.shape == (10,)
        np.testing.assert_allclose(arr, np.zeros(10, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_decode_message_verify_hash_succeeds_on_hashed(self, tgm_path):
        """Async ``decode_message`` accepts ``verify_hash=True`` and
        returns the data when the per-frame hash flag is set + slot
        matches.  See ``test_tensogram.py`` for the synchronous matrix."""
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        meta, objects = await f.decode_message(0, verify_hash=True)
        assert meta.version == 3
        assert len(objects) == 1

    @pytest.mark.asyncio
    async def test_file_decode_metadata(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        meta = await f.file_decode_metadata(0)
        assert meta.version == 3

    @pytest.mark.asyncio
    async def test_file_decode_descriptors(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        result = await f.file_decode_descriptors(0)
        assert "metadata" in result
        assert "descriptors" in result
        descs = result["descriptors"]
        assert len(descs) == 1
        assert list(descs[0].shape) == [10]
        assert descs[0].dtype == "float32"

    @pytest.mark.asyncio
    async def test_file_decode_object(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        result = await f.file_decode_object(0, 0)
        assert "metadata" in result
        assert "descriptor" in result
        assert "data" in result
        np.testing.assert_allclose(result["data"], np.zeros(10, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_file_decode_object_second_message(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        result = await f.file_decode_object(1, 0)
        np.testing.assert_allclose(result["data"], np.ones(10, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_file_decode_object_third_message(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        result = await f.file_decode_object(2, 0)
        np.testing.assert_allclose(result["data"], np.full(10, 2.0, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_read_message(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        raw = await f.read_message(0)
        assert isinstance(raw, bytes)
        assert len(raw) > 0
        meta, _ = tensogram.decode(raw)
        assert meta.version == 3


class TestAsyncGather:
    @pytest.mark.asyncio
    async def test_gather_decode_objects(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        results = await asyncio.gather(
            f.file_decode_object(0, 0),
            f.file_decode_object(1, 0),
            f.file_decode_object(2, 0),
        )
        for i, r in enumerate(results):
            np.testing.assert_allclose(
                r["data"], np.full(10, float(i), dtype=np.float32)
            )

    @pytest.mark.asyncio
    async def test_gather_mixed_operations(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        meta, obj = await asyncio.gather(
            f.file_decode_metadata(0),
            f.file_decode_object(1, 0),
        )
        assert meta.version == 3
        assert obj["data"].shape == (10,)

    @pytest.mark.asyncio
    async def test_gather_all_messages(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        messages = await asyncio.gather(
            f.decode_message(0),
            f.decode_message(1),
            f.decode_message(2),
        )
        for i, (meta, objects) in enumerate(messages):
            assert meta.version == 3
            np.testing.assert_allclose(
                objects[0][1], np.full(10, float(i), dtype=np.float32)
            )


class TestAsyncParity:
    @pytest.mark.asyncio
    async def test_decode_message_matches_sync(self, tgm_path):
        sync_file = tensogram.TensogramFile.open(tgm_path)
        sync_meta, sync_objects = sync_file.decode_message(0)

        async_file = await tensogram.AsyncTensogramFile.open(tgm_path)
        async_meta, async_objects = await async_file.decode_message(0)

        assert sync_meta.version == async_meta.version
        assert len(sync_objects) == len(async_objects)
        np.testing.assert_array_equal(sync_objects[0][1], async_objects[0][1])

    @pytest.mark.asyncio
    async def test_file_decode_object_matches_sync(self, tgm_path):
        sync_file = tensogram.TensogramFile.open(tgm_path)
        sync_result = sync_file.file_decode_object(0, 0)

        async_file = await tensogram.AsyncTensogramFile.open(tgm_path)
        async_result = await async_file.file_decode_object(0, 0)

        np.testing.assert_array_equal(sync_result["data"], async_result["data"])

    @pytest.mark.asyncio
    async def test_read_message_matches_sync(self, tgm_path):
        sync_file = tensogram.TensogramFile.open(tgm_path)
        sync_raw = sync_file.read_message(0)

        async_file = await tensogram.AsyncTensogramFile.open(tgm_path)
        async_raw = await async_file.read_message(0)

        assert sync_raw == async_raw


class TestAsyncErrors:
    @pytest.mark.asyncio
    async def test_decode_message_out_of_range(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.decode_message(999)

    @pytest.mark.asyncio
    async def test_file_decode_object_out_of_range(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.file_decode_object(999, 0)

    @pytest.mark.asyncio
    async def test_file_decode_metadata_out_of_range(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.file_decode_metadata(999)

    @pytest.mark.asyncio
    async def test_file_decode_descriptors_out_of_range(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.file_decode_descriptors(999)

    @pytest.mark.asyncio
    async def test_read_message_out_of_range(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.read_message(999)

    @pytest.mark.asyncio
    async def test_decode_message_verify_hash_raises_missing_hash_on_unhashed_async(
        self, tmp_path
    ):
        """Async ``decode_message(verify_hash=True)`` on an unhashed
        message raises ``MissingHashError`` with the offending object
        index.  Pins the strict-input contract from PR #110 on the
        async surface."""
        path = str(tmp_path / "unhashed.tgm")
        # Build a hashing=false message via the tensogram.encode entry
        # point with hash=None.
        meta = {"version": 3}
        desc = {
            "type": "ntensor",
            "ndim": 1,
            "shape": [4],
            "strides": [1],
            "dtype": "float32",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = np.zeros(4, dtype=np.float32)
        msg = bytes(tensogram.encode(meta, [(desc, data)], hash=None))
        with open(path, "wb") as fh:
            fh.write(msg)
        f = await tensogram.AsyncTensogramFile.open(path)
        with pytest.raises(tensogram.MissingHashError) as excinfo:
            _ = await f.decode_message(0, verify_hash=True)
        assert excinfo.value.object_index == 0

    @pytest.mark.asyncio
    async def test_open_remote_bad_storage_options(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)

        class Unconvertible:
            def __str__(self):
                raise RuntimeError("cannot convert")

        with pytest.raises(ValueError, match="convertible to string"):
            tensogram.AsyncTensogramFile.open_remote(url, {"key": Unconvertible()})

    @pytest.mark.asyncio
    async def test_cancel_then_reuse_handle(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)

        fut = f.file_decode_object(0, 0)
        task = asyncio.ensure_future(fut)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            _ = await task

        result = await f.file_decode_object(1, 0)
        np.testing.assert_allclose(result["data"], np.ones(10, dtype=np.float32))


class TestAsyncRemote:
    @pytest.mark.asyncio
    async def test_remote_decode_object(self, serve_tgm_bytes):
        msg = _encode_test_message([4], fill=42.0)
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        assert f.is_remote()
        result = await f.file_decode_object(0, 0)
        np.testing.assert_allclose(result["data"], np.full(4, 42.0, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_remote_decode_message(self, serve_tgm_bytes):
        msg = _encode_test_message([6], fill=7.0)
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        meta, objects = await f.decode_message(0)
        assert meta.version == 3
        np.testing.assert_allclose(objects[0][1], np.full(6, 7.0, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_remote_matches_sync(self, serve_tgm_bytes):
        msg = _encode_test_message([10], fill=3.14)
        url = serve_tgm_bytes(msg)

        sync_file = tensogram.TensogramFile.open(url)
        sync_result = sync_file.file_decode_object(0, 0)

        async_file = await tensogram.AsyncTensogramFile.open(url)
        async_result = await async_file.file_decode_object(0, 0)

        np.testing.assert_array_equal(sync_result["data"], async_result["data"])

    @pytest.mark.asyncio
    async def test_remote_read_message(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        raw = await f.read_message(0)
        assert isinstance(raw, bytes)
        assert len(raw) > 0


class TestAsyncDecodeRange:
    @pytest.mark.asyncio
    async def test_file_decode_range_basic(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        result = await f.file_decode_range(0, 0, [(0, 5)])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].shape == (5,)

    @pytest.mark.asyncio
    async def test_file_decode_range_join(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        result = await f.file_decode_range(0, 0, [(0, 3), (7, 3)], join=True)
        assert not isinstance(result, list)
        assert result.shape == (6,)

    @pytest.mark.asyncio
    async def test_file_decode_range_out_of_range(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.file_decode_range(999, 0, [(0, 5)])

    @pytest.mark.asyncio
    async def test_file_decode_range_matches_sync(self, tgm_path):
        sync_file = tensogram.TensogramFile.open(tgm_path)
        sync_result = sync_file.file_decode_range(0, 0, [(0, 5), (5, 5)])

        async_file = await tensogram.AsyncTensogramFile.open(tgm_path)
        async_result = await async_file.file_decode_range(0, 0, [(0, 5), (5, 5)])

        for s, a in zip(sync_result, async_result):
            np.testing.assert_array_equal(s, a)


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_async_with(self, tgm_path):
        async with await tensogram.AsyncTensogramFile.open(tgm_path) as f:
            meta, _objects = await f.decode_message(0)
            assert meta.version == 3

    @pytest.mark.asyncio
    async def test_async_with_exception_propagates(self, tgm_path):
        with pytest.raises(ValueError, match="test error"):  # noqa: PT012
            async with await tensogram.AsyncTensogramFile.open(tgm_path) as f:
                await f.decode_message(0)
                raise ValueError("test error")


class TestAsyncLen:
    @pytest.mark.asyncio
    async def test_len_after_message_count(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        count = await f.message_count()
        assert count == 3
        assert len(f) == 3

    @pytest.mark.asyncio
    async def test_len_without_message_count_raises(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises(RuntimeError, match="message_count"):
            len(f)

    @pytest.mark.asyncio
    async def test_len_matches_sync(self, tgm_path):
        sync_file = tensogram.TensogramFile.open(tgm_path)
        async_file = await tensogram.AsyncTensogramFile.open(tgm_path)
        await async_file.message_count()
        assert len(sync_file) == len(async_file)


class TestAsyncIteration:
    @pytest.mark.asyncio
    async def test_aiter_all_messages(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        await f.message_count()
        messages = []
        async for msg in f:
            messages.append(msg)
        assert len(messages) == 3
        for i, (meta, objects) in enumerate(messages):
            assert meta.version == 3
            np.testing.assert_allclose(
                objects[0][1], np.full(10, float(i), dtype=np.float32)
            )

    @pytest.mark.asyncio
    async def test_aiter_early_break(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        await f.message_count()
        count = 0
        async for _msg in f:
            count += 1
            if count == 1:
                break
        assert count == 1

    @pytest.mark.asyncio
    async def test_aiter_matches_sync(self, tgm_path):
        sync_file = tensogram.TensogramFile.open(tgm_path)
        sync_messages = list(sync_file)

        async_file = await tensogram.AsyncTensogramFile.open(tgm_path)
        await async_file.message_count()
        async_messages = []
        async for msg in async_file:
            async_messages.append(msg)

        assert len(sync_messages) == len(async_messages)
        for (s_meta, s_objs), (a_meta, a_objs) in zip(sync_messages, async_messages):
            assert s_meta.version == a_meta.version
            np.testing.assert_array_equal(s_objs[0][1], a_objs[0][1])

    @pytest.mark.asyncio
    async def test_aiter_remote(self, serve_tgm_bytes):
        msg = _encode_test_message([4], fill=99.0)
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        await f.message_count()
        messages = []
        async for m in f:
            messages.append(m)
        assert len(messages) == 1
        np.testing.assert_allclose(
            messages[0][1][0][1], np.full(4, 99.0, dtype=np.float32)
        )

    @pytest.mark.asyncio
    async def test_aiter_repr(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        await f.message_count()
        it = f.__aiter__()
        r = repr(it)
        assert "AsyncTensogramFileIter" in r
        assert "position=0" in r
        assert "remaining=3" in r

    @pytest.mark.asyncio
    async def test_aiter_len(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        await f.message_count()
        it = f.__aiter__()
        assert len(it) == 3
        await it.__anext__()
        assert len(it) == 2

    @pytest.mark.asyncio
    async def test_aiter_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.tgm")
        with tensogram.TensogramFile.create(path):
            pass
        f = await tensogram.AsyncTensogramFile.open(path)
        await f.message_count()
        messages = []
        async for msg in f:
            messages.append(msg)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_aiter_exhaustion(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        await f.message_count()
        it = f.__aiter__()
        for _ in range(3):
            await it.__anext__()
        with pytest.raises(StopAsyncIteration):
            _ = await it.__anext__()

    @pytest.mark.asyncio
    async def test_aiter_concurrent_iterators(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        await f.message_count()
        it1 = f.__aiter__()
        it2 = f.__aiter__()
        msg1_first = await it1.__anext__()
        msg2_first = await it2.__anext__()
        assert msg1_first[0].version == msg2_first[0].version
        assert len(it1) == 2
        assert len(it2) == 2


class TestAsyncDecodeRangeRemote:
    @pytest.mark.asyncio
    async def test_file_decode_range_remote(self, serve_tgm_bytes):
        msg = _encode_test_message([10], fill=5.0)
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        result = await f.file_decode_range(0, 0, [(0, 5)])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].shape == (5,)

    @pytest.mark.asyncio
    async def test_file_decode_range_remote_join(self, serve_tgm_bytes):
        msg = _encode_test_message([10], fill=7.0)
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        result = await f.file_decode_range(0, 0, [(0, 3), (7, 3)], join=True)
        assert result.shape == (6,)


class TestAsyncLenRemote:
    @pytest.mark.asyncio
    async def test_len_remote(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        await f.message_count()
        assert len(f) == 1


class TestAsyncMessages:
    @pytest.mark.asyncio
    async def test_messages_returns_list_of_bytes(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        msgs = await f.messages()
        assert isinstance(msgs, list)
        assert len(msgs) == 3
        for m in msgs:
            assert isinstance(m, bytes)

    @pytest.mark.asyncio
    async def test_messages_matches_sync(self, tgm_path):
        sync_file = tensogram.TensogramFile.open(tgm_path)
        sync_msgs = sync_file.messages()

        async_file = await tensogram.AsyncTensogramFile.open(tgm_path)
        async_msgs = await async_file.messages()

        assert len(sync_msgs) == len(async_msgs)
        for s, a in zip(sync_msgs, async_msgs):
            assert s == a

    @pytest.mark.asyncio
    async def test_messages_remote(self, serve_tgm_bytes):
        msg = _encode_test_message([4])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        msgs = await f.messages()
        assert len(msgs) == 1
        assert isinstance(msgs[0], bytes)


class TestAsyncBatchRange:
    @pytest.mark.asyncio
    async def test_batch_basic(self, serve_tgm_bytes):
        meta = {"version": 3, "base": [{}]}
        desc = {
            "type": "ntensor",
            "shape": [10],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = b""
        for i in range(3):
            arr = np.full(10, float(i), dtype=np.float32)
            data += tensogram.encode(meta, [(desc, arr)])
        url = serve_tgm_bytes(data)
        f = await tensogram.AsyncTensogramFile.open(url)
        results = await f.file_decode_range_batch([0, 1, 2], 0, [(0, 5)], join=True)
        assert isinstance(results, list)
        assert len(results) == 3
        for i, arr in enumerate(results):
            np.testing.assert_allclose(arr[:5], np.full(5, float(i), dtype=np.float32))

    @pytest.mark.asyncio
    async def test_batch_matches_individual(self, serve_tgm_bytes):
        meta = {"version": 3, "base": [{}]}
        desc = {
            "type": "ntensor",
            "shape": [10],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = b""
        for i in range(3):
            arr = np.full(10, float(i), dtype=np.float32)
            data += tensogram.encode(meta, [(desc, arr)])
        url = serve_tgm_bytes(data)
        f = await tensogram.AsyncTensogramFile.open(url)

        batch = await f.file_decode_range_batch([0, 1, 2], 0, [(2, 3)], join=True)
        individual = []
        for idx in range(3):
            r = await f.file_decode_range(idx, 0, [(2, 3)], join=True)
            individual.append(r)

        for b, i in zip(batch, individual):
            np.testing.assert_array_equal(b, i)

    @pytest.mark.asyncio
    async def test_batch_empty_indices(self, serve_tgm_bytes):
        msg = _encode_test_message([10])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        results = await f.file_decode_range_batch([], 0, [(0, 5)], join=True)
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_out_of_range(self, serve_tgm_bytes):
        msg = _encode_test_message([10])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.file_decode_range_batch([999], 0, [(0, 5)], join=True)


class TestAsyncBatchObject:
    @pytest.mark.asyncio
    async def test_batch_basic(self, serve_tgm_bytes):
        meta = {"version": 3, "base": [{}]}
        desc = {
            "type": "ntensor",
            "shape": [10],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = b""
        for i in range(3):
            arr = np.full(10, float(i), dtype=np.float32)
            data += tensogram.encode(meta, [(desc, arr)])
        url = serve_tgm_bytes(data)
        f = await tensogram.AsyncTensogramFile.open(url)
        results = await f.file_decode_object_batch([0, 1, 2], 0)
        assert isinstance(results, list)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert "data" in r
            assert "metadata" in r
            assert "descriptor" in r
            np.testing.assert_allclose(
                r["data"], np.full(10, float(i), dtype=np.float32)
            )

    @pytest.mark.asyncio
    async def test_batch_matches_individual(self, serve_tgm_bytes):
        meta = {"version": 3, "base": [{}]}
        desc = {
            "type": "ntensor",
            "shape": [10],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = b""
        for i in range(3):
            arr = np.full(10, float(i), dtype=np.float32)
            data += tensogram.encode(meta, [(desc, arr)])
        url = serve_tgm_bytes(data)
        f = await tensogram.AsyncTensogramFile.open(url)
        batch = await f.file_decode_object_batch([0, 1, 2], 0)
        for i in range(3):
            individual = await f.file_decode_object(i, 0)
            np.testing.assert_array_equal(batch[i]["data"], individual["data"])

    @pytest.mark.asyncio
    async def test_batch_empty(self, serve_tgm_bytes):
        msg = _encode_test_message([10])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        results = await f.file_decode_object_batch([], 0)
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_out_of_range(self, serve_tgm_bytes):
        msg = _encode_test_message([10])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.file_decode_object_batch([999], 0)


class TestSyncBatchObject:
    def test_batch_basic(self, serve_tgm_bytes):
        meta = {"version": 3, "base": [{}]}
        desc = {
            "type": "ntensor",
            "shape": [10],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = b""
        for i in range(3):
            arr = np.full(10, float(i), dtype=np.float32)
            data += tensogram.encode(meta, [(desc, arr)])
        url = serve_tgm_bytes(data)
        f = tensogram.TensogramFile.open(url)
        results = f.file_decode_object_batch([0, 1, 2], 0)
        assert len(results) == 3
        for i, r in enumerate(results):
            np.testing.assert_allclose(
                r["data"], np.full(10, float(i), dtype=np.float32)
            )

    def test_batch_matches_individual(self, serve_tgm_bytes):
        meta = {"version": 3, "base": [{}]}
        desc = {
            "type": "ntensor",
            "shape": [10],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = b""
        for i in range(3):
            arr = np.full(10, float(i), dtype=np.float32)
            data += tensogram.encode(meta, [(desc, arr)])
        url = serve_tgm_bytes(data)
        f = tensogram.TensogramFile.open(url)
        batch = f.file_decode_object_batch([0, 1, 2], 0)
        for i in range(3):
            individual = f.file_decode_object(i, 0)
            np.testing.assert_array_equal(batch[i]["data"], individual["data"])


class TestSyncBatchRange:
    def test_batch_basic(self, serve_tgm_bytes):
        meta = {"version": 3, "base": [{}]}
        desc = {
            "type": "ntensor",
            "shape": [10],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = b""
        for i in range(3):
            arr = np.full(10, float(i), dtype=np.float32)
            data += tensogram.encode(meta, [(desc, arr)])
        url = serve_tgm_bytes(data)
        f = tensogram.TensogramFile.open(url)
        results = f.file_decode_range_batch([0, 1, 2], 0, [(0, 5)], join=True)
        assert isinstance(results, list)
        assert len(results) == 3
        for i, arr in enumerate(results):
            np.testing.assert_allclose(arr[:5], np.full(5, float(i), dtype=np.float32))

    def test_batch_matches_individual(self, serve_tgm_bytes):
        meta = {"version": 3, "base": [{}]}
        desc = {
            "type": "ntensor",
            "shape": [10],
            "dtype": "float32",
            "byte_order": "little",
            "encoding": "none",
            "filter": "none",
            "compression": "none",
        }
        data = b""
        for i in range(3):
            arr = np.full(10, float(i), dtype=np.float32)
            data += tensogram.encode(meta, [(desc, arr)])
        url = serve_tgm_bytes(data)
        f = tensogram.TensogramFile.open(url)

        batch = f.file_decode_range_batch([0, 1, 2], 0, [(2, 3)], join=True)
        individual = []
        for idx in range(3):
            r = f.file_decode_range(idx, 0, [(2, 3)], join=True)
            individual.append(r)

        for b, i in zip(batch, individual):
            np.testing.assert_array_equal(b, i)


class TestBatchLocalFileError:
    """Batch methods require remote backend; calling on local files must raise."""

    def test_sync_object_batch_raises_on_local(self, tgm_path):
        f = tensogram.TensogramFile.open(tgm_path)
        with pytest.raises(OSError, match="remote backend"):
            f.file_decode_object_batch([0, 1], 0)

    def test_sync_range_batch_raises_on_local(self, tgm_path):
        f = tensogram.TensogramFile.open(tgm_path)
        with pytest.raises(OSError, match="remote backend"):
            f.file_decode_range_batch([0, 1], 0, [(0, 5)])

    @pytest.mark.asyncio
    async def test_async_object_batch_raises_on_local(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises(OSError, match="remote backend"):
            _ = await f.file_decode_object_batch([0, 1], 0)

    @pytest.mark.asyncio
    async def test_async_range_batch_raises_on_local(self, tgm_path):
        f = await tensogram.AsyncTensogramFile.open(tgm_path)
        with pytest.raises(OSError, match="remote backend"):
            _ = await f.file_decode_range_batch([0, 1], 0, [(0, 5)])


class TestAsyncPrefetchLayouts:
    @pytest.mark.asyncio
    async def test_prefetch_then_decode(self, serve_tgm_bytes):
        msg = _encode_test_message([10], fill=7.0)
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        await f.prefetch_layouts([0])
        result = await f.file_decode_object(0, 0)
        np.testing.assert_allclose(result["data"], np.full(10, 7.0, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_prefetch_empty(self, serve_tgm_bytes):
        msg = _encode_test_message([10])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        await f.prefetch_layouts([])

    @pytest.mark.asyncio
    async def test_prefetch_idempotent(self, serve_tgm_bytes):
        msg = _encode_test_message([10])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        await f.prefetch_layouts([0])
        await f.prefetch_layouts([0])
        result = await f.file_decode_object(0, 0)
        assert result["data"].shape == (10,)

    @pytest.mark.asyncio
    async def test_prefetch_out_of_range(self, serve_tgm_bytes):
        msg = _encode_test_message([10])
        url = serve_tgm_bytes(msg)
        f = await tensogram.AsyncTensogramFile.open(url)
        with pytest.raises((ValueError, RuntimeError)):
            _ = await f.prefetch_layouts([999])
