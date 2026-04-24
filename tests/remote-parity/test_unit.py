# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Unit tests for the remote-parity harness internals.

Every test here runs with no external build artefacts (no Rust driver
binary, no `npm install`, no fixtures). It exercises:

- `classifier.classify` — status-aware categorisation, range parsing
- `classifier.RoundBuilder` — round assignment invariants
- `mock_server` — HEAD/Range/404/416/path-traversal behaviour
- `run_parity.collect_events` — duplicate-run_id guard
- Schema contract — keys + enums match the ScanEvent schema
"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from classifier import (
    END_MAGIC_BYTES,
    NON_SCAN_ROUND,
    POSTAMBLE_BYTES,
    PREAMBLE_BYTES,
    Observation,
    RoundBuilder,
    classify,
)
from mock_server import MockServer
from run_parity import DriverCase, _check_no_duplicate_run_ids

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_FIXTURES_DIR = _THIS_DIR / "fixtures"
_SCHEMA_PATH = _THIS_DIR / "schema.json"


class TestClassify:
    def test_head_is_probe(self) -> None:
        r = classify(Observation("HEAD", None, 200, 0))
        assert r.category == "probe"
        assert r.direction == "none"
        assert r.logical_range == (0, 0)

    def test_get_200_is_fallback(self) -> None:
        r = classify(Observation("GET", None, 200, 1000))
        assert r.category == "fallback"
        assert r.direction == "none"
        assert r.logical_range == (0, 1000)

    def test_get_206_preamble_is_scan_forward(self) -> None:
        r = classify(Observation("GET", "bytes=0-23", 206, PREAMBLE_BYTES))
        assert r.category == "scan"
        assert r.direction == "forward"
        assert r.logical_range == (0, PREAMBLE_BYTES)

    def test_get_206_postamble_is_scan(self) -> None:
        r = classify(Observation("GET", "bytes=500-523", 206, POSTAMBLE_BYTES))
        assert r.category == "scan"
        assert r.logical_range == (500, 524)

    def test_get_206_end_magic_is_scan(self) -> None:
        r = classify(Observation("GET", "bytes=1000-1007", 206, END_MAGIC_BYTES))
        assert r.category == "scan"
        assert r.logical_range == (1000, 1008)

    def test_get_206_large_is_payload(self) -> None:
        r = classify(Observation("GET", "bytes=100-10099", 206, 10000))
        assert r.category == "payload"
        assert r.direction == "none"

    @pytest.mark.parametrize("status", [404, 416, 500, 503])
    def test_non_success_status_is_error(self, status: int) -> None:
        r = classify(Observation("GET", "bytes=0-23", status, 0))
        assert r.category == "error"
        assert r.direction == "none"

    def test_unknown_method_is_error(self) -> None:
        r = classify(Observation("PATCH", None, 200, 0))
        assert r.category == "error"

    def test_missing_range_uses_response_bytes(self) -> None:
        r = classify(Observation("GET", None, 200, 512))
        assert r.logical_range == (0, 512)

    def test_open_ended_range_uses_response_bytes(self) -> None:
        r = classify(Observation("GET", "bytes=10-", 206, 100))
        assert r.logical_range == (10, 110)

    def test_malformed_range_falls_back(self) -> None:
        r = classify(Observation("GET", "bytes=not-valid", 206, PREAMBLE_BYTES))
        assert r.logical_range == (0, PREAMBLE_BYTES)


class TestRoundBuilder:
    def test_each_scan_gets_new_round(self) -> None:
        builder = RoundBuilder()
        a = classify(Observation("GET", "bytes=0-23", 206, PREAMBLE_BYTES))
        b = classify(Observation("GET", "bytes=100-123", 206, PREAMBLE_BYTES))
        assert builder.assign(a) == 0
        assert builder.assign(b) == 1

    def test_non_scan_events_get_non_scan_round(self) -> None:
        builder = RoundBuilder()
        probe = classify(Observation("HEAD", None, 200, 0))
        assert builder.assign(probe) == NON_SCAN_ROUND

    def test_round_counter_ignores_non_scan(self) -> None:
        builder = RoundBuilder()
        scan0 = classify(Observation("GET", "bytes=0-23", 206, PREAMBLE_BYTES))
        probe = classify(Observation("HEAD", None, 200, 0))
        scan1 = classify(Observation("GET", "bytes=100-123", 206, PREAMBLE_BYTES))
        assert builder.assign(scan0) == 0
        assert builder.assign(probe) == NON_SCAN_ROUND
        assert builder.assign(scan1) == 1


class TestMockServer:
    @pytest.fixture(autouse=True)
    def _fixtures_ready(self) -> None:
        if not _FIXTURES_DIR.exists() or not any(_FIXTURES_DIR.glob("*.tgm")):
            pytest.skip("fixtures/ missing; run tools/gen_fixtures.py")

    def test_head_returns_accept_ranges(self) -> None:
        with MockServer(_FIXTURES_DIR) as server:
            url = server.url_for("unit-test-head", "single-msg.tgm")
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
                assert resp.status == 200
                assert resp.headers["Accept-Ranges"] == "bytes"
                assert int(resp.headers["Content-Length"]) > 0

    def test_range_request_returns_206(self) -> None:
        with MockServer(_FIXTURES_DIR) as server:
            url = server.url_for("unit-test-range", "single-msg.tgm")
            req = urllib.request.Request(url, headers={"Range": "bytes=0-23"})
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 206
                body = resp.read()
                assert len(body) == 24

    def test_suffix_range_supported(self) -> None:
        with MockServer(_FIXTURES_DIR) as server:
            url = server.url_for("unit-test-suffix", "single-msg.tgm")
            req = urllib.request.Request(url, headers={"Range": "bytes=-8"})
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 206
                body = resp.read()
                assert len(body) == 8

    def test_out_of_range_returns_416(self) -> None:
        with MockServer(_FIXTURES_DIR) as server:
            url = server.url_for("unit-test-416", "single-msg.tgm")
            req = urllib.request.Request(url, headers={"Range": "bytes=999999-9999999"})
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(req).read()
            assert excinfo.value.code == 416

    def test_unknown_fixture_returns_404(self) -> None:
        with MockServer(_FIXTURES_DIR) as server:
            url = server.url_for("unit-test-404", "does-not-exist.tgm")
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(url).read()
            assert excinfo.value.code == 404

    def test_path_traversal_rejected(self) -> None:
        with MockServer(_FIXTURES_DIR) as server:
            url = f"{server.base_url}/unit-test-traversal/../../../etc/passwd"
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(url).read()
            assert excinfo.value.code == 404

    def test_log_endpoint_returns_per_run(self) -> None:
        with MockServer(_FIXTURES_DIR) as server:
            url = server.url_for("unit-test-log-a", "single-msg.tgm")
            req = urllib.request.Request(url, headers={"Range": "bytes=0-23"})
            with urllib.request.urlopen(req) as resp:
                resp.read()
            log_url = f"{server.base_url}/_log/unit-test-log-a"
            with urllib.request.urlopen(log_url) as log_resp:
                log_body = log_resp.read().decode("utf-8")
            entries = [json.loads(line) for line in log_body.splitlines() if line.strip()]
            assert len(entries) == 1
            assert entries[0]["method"] == "GET"
            assert entries[0]["status"] == 206

    def test_reset_clears_log(self) -> None:
        with MockServer(_FIXTURES_DIR) as server:
            url = server.url_for("unit-test-reset", "single-msg.tgm")
            req = urllib.request.Request(url, headers={"Range": "bytes=0-23"})
            with urllib.request.urlopen(req) as resp:
                resp.read()

            reset = urllib.request.Request(f"{server.base_url}/_reset", method="POST")
            with urllib.request.urlopen(reset) as resp:
                assert resp.status == 204

            log_url = f"{server.base_url}/_log/unit-test-reset"
            with urllib.request.urlopen(log_url) as log_resp:
                log_body = log_resp.read().decode("utf-8")
            entries = [line for line in log_body.splitlines() if line.strip()]
            assert entries == []


class TestDuplicateRunIdGuard:
    def test_rejects_duplicates(self) -> None:
        case = DriverCase(fixture="single-msg", language="rust", op="open")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="duplicate run_id"):
            _check_no_duplicate_run_ids([case, case])

    def test_accepts_unique_cases(self) -> None:
        cases = [
            DriverCase(fixture="single-msg", language="rust", op="open"),  # type: ignore[arg-type]
            DriverCase(fixture="single-msg", language="ts", op="open"),  # type: ignore[arg-type]
        ]
        _check_no_duplicate_run_ids(cases)


class TestSchemaContract:
    @pytest.fixture(scope="class")
    def schema(self) -> dict:
        return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_schema_is_valid_json(self, schema: dict) -> None:
        assert schema["title"] == "Remote parity scan event"
        required = set(schema["required"])
        assert required == {
            "run_id",
            "scan_round",
            "direction",
            "category",
            "logical_range",
            "physical_requests",
        }

    def test_classified_event_serialises_to_required_shape(self, schema: dict) -> None:
        from run_parity import ScanEvent

        event = ScanEvent(
            run_id="unit-test-shape",
            scan_round=0,
            direction="forward",
            category="scan",
            logical_range=(0, PREAMBLE_BYTES),
            physical_requests=({"method": "GET", "headers": {}, "status": 206},),
        )
        rendered = event.to_dict()
        assert set(rendered.keys()) == set(schema["required"])
        assert rendered["category"] in schema["properties"]["category"]["enum"]
        assert rendered["direction"] in schema["properties"]["direction"]["enum"]
