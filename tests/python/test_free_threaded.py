# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Thread-safety tests for free-threaded Python support.

Verifies that tensogram's Python bindings can be used safely from
multiple threads simultaneously. Each test launches 8 threads that
perform 50 iterations of a specific operation, checking for
correctness and absence of crashes/deadlocks.
"""

import os
import tempfile
import threading

import numpy as np
import pytest
import tensogram

NUM_THREADS = 8
ITERATIONS = 50


def _make_message(size=1000, encoding="none", compression="none", dtype="float32", seed=42):
    """Create a test message with the given parameters."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(size).astype(getattr(np, dtype))
    meta = {"version": 2, "base": [{}]}
    desc = {
        "type": "ntensor",
        "shape": [size],
        "dtype": dtype,
        "encoding": encoding,
        "compression": compression,
    }
    if encoding == "simple_packing":
        params = tensogram.compute_packing_params(data.astype(np.float64).ravel(), 16, 0)
        desc.update(params)
    msg = tensogram.encode(meta, [(desc, data)])
    return data, meta, desc, msg


def _run_threaded(fn, num_threads=NUM_THREADS):
    """Run fn(thread_id) on num_threads threads; raise on any error."""
    errors = []
    errors_lock = threading.Lock()
    barrier = threading.Barrier(num_threads)

    def wrapper(tid):
        try:
            barrier.wait(timeout=10)
            fn(tid)
        except Exception as e:
            with errors_lock:
                errors.append((tid, e))

    threads = [threading.Thread(target=wrapper, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for t in threads:
        assert not t.is_alive(), "thread did not finish (possible deadlock)"
    assert not errors, f"Thread errors: {errors}"


# ── Concurrent encode/decode with independent data ──


class TestConcurrentEncodeDecode:
    def test_concurrent_encode_none(self):
        """Each thread encodes its own data with no encoding/compression."""

        def work(tid):
            data = np.arange(500, dtype=np.float32) + tid
            meta = {"version": 2, "base": [{}]}
            desc = {"type": "ntensor", "shape": [500], "dtype": "float32"}
            for _ in range(ITERATIONS):
                msg = tensogram.encode(meta, [(desc, data)])
                result = tensogram.decode(msg)
                arr = result.objects[0][1]
                np.testing.assert_array_equal(arr, data)

        _run_threaded(work)


# ── Additional coverage for all detached paths ──


class TestConcurrentDetachedPaths:
    def test_encode_pre_encoded_concurrent(self):
        data = np.arange(100, dtype=np.float32)
        meta = {"version": 2, "base": [{}]}
        desc = {"type": "ntensor", "shape": [100], "dtype": "float32"}
        raw_bytes = data.tobytes()

        def work(tid):
            for _ in range(ITERATIONS):
                encoded = tensogram.encode_pre_encoded(meta, [(desc, raw_bytes)])
                assert len(encoded) > 0

        _run_threaded(work)

    def test_decode_range_concurrent(self):
        data = np.arange(1000, dtype=np.float32)
        meta = {"version": 2, "base": [{}]}
        desc = {"type": "ntensor", "shape": [1000], "dtype": "float32"}
        msg = tensogram.encode(meta, [(desc, data)])

        def work(tid):
            for _ in range(ITERATIONS):
                parts = tensogram.decode_range(msg, 0, [(0, 10), (500, 10)])
                assert len(parts) == 2
                np.testing.assert_array_equal(parts[0], data[:10])
                np.testing.assert_array_equal(parts[1], data[500:510])

        _run_threaded(work)

    def test_validate_file_concurrent(self):
        with tempfile.NamedTemporaryFile(suffix=".tgm", delete=False) as tmp:
            path = tmp.name
        try:
            data = np.arange(100, dtype=np.float32)
            meta = {"version": 2, "base": [{}]}
            desc = {"type": "ntensor", "shape": [100], "dtype": "float32"}
            with tensogram.TensogramFile.create(path) as f:
                f.append(meta, [(desc, data)])

            def work(tid):
                for _ in range(ITERATIONS):
                    report = tensogram.validate_file(path)
                    assert len(report["messages"]) == 1
                    assert report["messages"][0]["object_count"] == 1

            _run_threaded(work)
        finally:
            os.unlink(path)

    def test_iter_messages_concurrent(self):
        data = np.arange(50, dtype=np.float32)
        meta = {"version": 2, "base": [{}]}
        desc = {"type": "ntensor", "shape": [50], "dtype": "float32"}
        msg1 = tensogram.encode(meta, [(desc, data)])
        msg2 = tensogram.encode(meta, [(desc, data + 1)])
        combined = msg1 + msg2

        def work(tid):
            for _ in range(ITERATIONS):
                msgs = list(tensogram.iter_messages(combined))
                assert len(msgs) == 2
                np.testing.assert_array_equal(msgs[0].objects[0][1], data)

        _run_threaded(work)

    def test_bytes_input_decode(self):
        """Verify decode works with plain bytes (zero-copy via PyBackedBytes)."""
        data = np.arange(100, dtype=np.float32)
        meta = {"version": 2, "base": [{}]}
        desc = {"type": "ntensor", "shape": [100], "dtype": "float32"}
        msg = bytes(tensogram.encode(meta, [(desc, data)]))

        def work(tid):
            for _ in range(ITERATIONS):
                result = tensogram.decode(msg)
                np.testing.assert_array_equal(result.objects[0][1], data)

        _run_threaded(work)


class TestSharedFileHandle:
    def test_shared_file_concurrent_decode_message(self):
        """Concurrent reads on the same TensogramFile handle succeed."""
        with tempfile.NamedTemporaryFile(suffix=".tgm", delete=False) as tmp:
            path = tmp.name
        try:
            data = np.arange(100, dtype=np.float32)
            meta = {"version": 2, "base": [{}]}
            desc = {"type": "ntensor", "shape": [100], "dtype": "float32"}
            with tensogram.TensogramFile.create(path) as f:
                for _ in range(10):
                    f.append(meta, [(desc, data)])

            with tensogram.TensogramFile.open(path) as shared_file:

                def work(tid):
                    for _ in range(ITERATIONS):
                        msg = shared_file.decode_message(tid % 10)
                        arr = msg.objects[0][1]
                        np.testing.assert_array_equal(arr, data)

                _run_threaded(work)
        finally:
            os.unlink(path)

    def test_shared_file_concurrent_mixed_reads(self):
        with tempfile.NamedTemporaryFile(suffix=".tgm", delete=False) as tmp:
            path = tmp.name
        try:
            data = np.arange(200, dtype=np.float32)
            meta = {"version": 2, "base": [{}]}
            desc = {"type": "ntensor", "shape": [200], "dtype": "float32"}
            with tensogram.TensogramFile.create(path) as f:
                for _ in range(5):
                    f.append(meta, [(desc, data)])

            with tensogram.TensogramFile.open(path) as shared:

                def work(tid):
                    for _ in range(ITERATIONS):
                        if tid % 4 == 0:
                            assert shared.message_count() == 5
                        elif tid % 4 == 1:
                            shared.file_decode_metadata(0)
                        elif tid % 4 == 2:
                            shared.file_decode_descriptors(0)
                        else:
                            msg = shared.decode_message(tid % 5)
                            np.testing.assert_array_equal(msg.objects[0][1], data)

                _run_threaded(work)
        finally:
            os.unlink(path)

    def test_shared_file_concurrent_read_and_append(self):
        with tempfile.NamedTemporaryFile(suffix=".tgm", delete=False) as tmp:
            path = tmp.name
        try:
            data = np.arange(100, dtype=np.float32)
            meta = {"version": 2, "base": [{}]}
            desc = {"type": "ntensor", "shape": [100], "dtype": "float32"}
            with tensogram.TensogramFile.create(path) as f:
                f.append(meta, [(desc, data)])

            with tensogram.TensogramFile.open(path) as shared:
                errors_lock = threading.Lock()
                runtime_errors = [0]
                other_errors = []
                barrier = threading.Barrier(4)

                def reader(tid):
                    try:
                        barrier.wait(timeout=10)
                        for _ in range(50):
                            shared.decode_message(0)
                    except RuntimeError:
                        with errors_lock:
                            runtime_errors[0] += 1
                    except Exception as e:
                        with errors_lock:
                            other_errors.append((tid, type(e).__name__, str(e)))

                def writer(tid):
                    try:
                        barrier.wait(timeout=10)
                        for _ in range(50):
                            shared.append(meta, [(desc, data)])
                    except RuntimeError:
                        with errors_lock:
                            runtime_errors[0] += 1
                    except Exception as e:
                        with errors_lock:
                            other_errors.append((tid, type(e).__name__, str(e)))

                threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)] + [
                    threading.Thread(target=writer, args=(99,))
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=30)
                for t in threads:
                    assert not t.is_alive(), "thread did not finish (possible deadlock)"

                # RuntimeError from PyO3 borrow check is expected when
                # read (&self) and write (&mut self) overlap. Whether it
                # actually triggers depends on timing, so we only assert
                # no unexpected exception types occurred.
                assert not other_errors, f"Unexpected errors: {other_errors}"
        finally:
            os.unlink(path)


# ── Concurrent decode of the same immutable buffer ──


class TestConcurrentDecode:
    def test_shared_buffer_decode(self):
        """Multiple threads decode the same message bytes concurrently."""
        data, _meta, _desc, msg = _make_message(2000)

        def work(tid):
            for _ in range(ITERATIONS):
                result = tensogram.decode(msg)
                arr = result.objects[0][1]
                np.testing.assert_array_equal(arr, data)

        _run_threaded(work)

    def test_shared_buffer_decode_object(self):
        """Multiple threads use decode_object on the same buffer."""
        data, _meta, _desc, msg = _make_message(2000)

        def work(tid):
            for _ in range(ITERATIONS):
                _, _, arr = tensogram.decode_object(msg, 0)
                np.testing.assert_array_equal(arr, data)

        _run_threaded(work)

    def test_shared_buffer_decode_packing(self):
        """Multiple threads decode same simple_packing+zstd message."""
        data, _meta, _desc, msg = _make_message(2000, "simple_packing", "zstd", "float64")

        def work(tid):
            for _ in range(ITERATIONS):
                result = tensogram.decode(msg)
                arr = result.objects[0][1]
                np.testing.assert_allclose(arr, data, atol=1e-4)

        _run_threaded(work)


# ── Concurrent scan ──


class TestConcurrentScan:
    def test_shared_buffer_scan(self):
        """Multiple threads scan the same multi-message buffer."""
        _, _, _, msg1 = _make_message(500)
        _, _, _, msg2 = _make_message(800)
        combined = msg1 + msg2

        def work(tid):
            for _ in range(ITERATIONS):
                offsets = tensogram.scan(combined)
                assert len(offsets) == 2

        _run_threaded(work)


# ── Concurrent validate ──


class TestConcurrentValidate:
    def test_validate_concurrent(self):
        """Multiple threads validate different messages."""

        def work(tid):
            _, _, _, msg = _make_message(500 + tid * 100)
            for _ in range(ITERATIONS):
                report = tensogram.validate(msg)
                assert report["object_count"] == 1

        _run_threaded(work)


# ── Concurrent file operations (separate handles) ──


class TestConcurrentFile:
    def test_separate_file_handles(self):
        """Each thread opens its own TensogramFile handle for reading."""
        with tempfile.NamedTemporaryFile(suffix=".tgm", delete=False) as tmp:
            path = tmp.name

        try:
            data = np.arange(100, dtype=np.float32)
            meta = {"version": 2, "base": [{}]}
            desc = {"type": "ntensor", "shape": [100], "dtype": "float32"}
            with tensogram.TensogramFile.create(path) as f:
                for _ in range(5):
                    f.append(meta, [(desc, data)])

            def work(tid):
                with tensogram.TensogramFile.open(path) as f:
                    for _ in range(ITERATIONS):
                        msg = f.decode_message(tid % 5)
                        arr = msg.objects[0][1]
                        np.testing.assert_array_equal(arr, data)

            _run_threaded(work)
        finally:
            os.unlink(path)


# ── Concurrent StreamingEncoder (independent instances) ──


class TestConcurrentStreamingEncoder:
    def test_independent_streaming_encoders(self):
        """Each thread creates and finishes its own StreamingEncoder."""

        def work(tid):
            for _ in range(ITERATIONS):
                enc = tensogram.StreamingEncoder({"version": 2})
                data = np.arange(100, dtype=np.float32) + tid
                enc.write_object({"type": "ntensor", "shape": [100], "dtype": "float32"}, data)
                msg = enc.finish()
                result = tensogram.decode(msg)
                arr = result.objects[0][1]
                np.testing.assert_array_equal(arr, data)

        _run_threaded(work)


# ── Concurrent metadata access ──


class TestConcurrentMetadata:
    def test_metadata_read_concurrent(self):
        """Multiple threads read metadata from the same decoded message."""
        _, _, _, msg = _make_message(100)
        result = tensogram.decode(msg)
        meta = result.metadata

        def work(tid):
            for _ in range(ITERATIONS):
                assert meta.version == 2
                _ = meta.base
                _ = meta.reserved
                _ = meta.extra

        _run_threaded(work)

    def test_decode_metadata_concurrent(self):
        """Multiple threads call decode_metadata on the same buffer."""
        _, _, _, msg = _make_message(100)

        def work(tid):
            for _ in range(ITERATIONS):
                meta = tensogram.decode_metadata(msg)
                assert meta.version == 2

        _run_threaded(work)


# ── Codec backend smoke tests ──


class TestConcurrentCodecBackends:
    """Smoke-test all codec combinations under concurrent load.

    Native libraries (szip/libaec, zstd, lz4) have internal state
    that needs coverage under concurrency.
    """

    @pytest.mark.parametrize(
        ("encoding", "compression"),
        [
            ("none", "none"),
            ("none", "zstd"),
            ("none", "lz4"),
            ("simple_packing", "zstd"),
            ("simple_packing", "lz4"),
        ],
    )
    def test_codec_concurrent(self, encoding, compression):
        def work(tid):
            rng = np.random.default_rng(tid)
            data = rng.standard_normal(500).astype(np.float64) + tid
            meta = {"version": 2, "base": [{}]}
            desc = {
                "type": "ntensor",
                "shape": [500],
                "dtype": "float64",
                "encoding": encoding,
                "compression": compression,
            }
            if encoding == "simple_packing":
                params = tensogram.compute_packing_params(data.ravel(), 24, 0)
                desc.update(params)
            for _ in range(20):
                msg = tensogram.encode(meta, [(desc, data)])
                result = tensogram.decode(msg)
                arr = result.objects[0][1]
                if encoding == "none":
                    np.testing.assert_array_equal(arr, data)
                else:
                    np.testing.assert_allclose(arr, data, atol=1e-4)

        _run_threaded(work)
