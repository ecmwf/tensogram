#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Remote-scan walker microbench (Python).

Mirrors the cell matrix and ``cold_open_plus_operation_plus_close``
sample semantics of ``rust/benchmarks/benches/remote_scan.rs`` and
``typescript/tests/remote_scan.bench.ts`` so the resulting NDJSON
sidecars stack up cell-for-cell against each other.

Reuses ``tests/remote-parity/mock_server.py`` directly: the
``MockServer.log_for(run_id)`` accessor surfaces enough request
metadata to discriminate ``total_requests``, ``range_get_requests``,
``head_requests``, and ``response_body_bytes`` per cell.

Usage:
    python bench_remote_scan.py                      # full matrix, sync
    python bench_remote_scan.py --quick              # CI smoke (N=10 only)
    python bench_remote_scan.py --headline           # N=100 iter, both walkers
    python bench_remote_scan.py --mode async         # AsyncTensogramFile
    python bench_remote_scan.py --json out.ndjson    # custom output path
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
from dataclasses import dataclass

import tensogram

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent
_PARITY_DIR = _REPO_ROOT / "tests" / "remote-parity"
_FIXTURES_DIR = _PARITY_DIR / "fixtures"
_DEFAULT_OUT = _REPO_ROOT / "target" / "remote-scan-bench" / "python.ndjson"

sys.path.insert(0, str(_PARITY_DIR))
from mock_server import MockServer, RequestRecord  # noqa: E402

_HEADER_FIXTURES = {
    1: "single-msg.tgm",
    10: "ten-msg.tgm",
    100: "hundred-msg.tgm",
    1000: "thousand-msg.tgm",
}
_FOOTER_FIXTURES = {
    1: "single-msg-footer.tgm",
    10: "ten-msg-footer.tgm",
    100: "hundred-msg-footer.tgm",
    1000: "thousand-msg-footer.tgm",
}
_STREAMING_TAIL = ("streaming-tail.tgm", 10)

_SCENARIOS = (
    "message_count",
    "read_message(0)",
    "read_message(N-1)",
    "read_message(N/2)",
    "iter",
)


@dataclass(frozen=True)
class CellSpec:
    fixture_kind: str
    fixture_name: str
    tier: int


def _matrix() -> list[CellSpec]:
    out: list[CellSpec] = []
    for tier, name in _HEADER_FIXTURES.items():
        out.append(CellSpec("header-indexed", name, tier))
    for tier, name in _FOOTER_FIXTURES.items():
        out.append(CellSpec("footer-indexed", name, tier))
    out.append(CellSpec("streaming-tail", _STREAMING_TAIL[0], _STREAMING_TAIL[1]))
    return out


def _quick_matrix() -> list[CellSpec]:
    return [s for s in _matrix() if s.tier == 10]


def _headline_matrix() -> list[CellSpec]:
    return [
        CellSpec("header-indexed", "hundred-msg.tgm", 100),
        CellSpec("footer-indexed", "hundred-msg-footer.tgm", 100),
    ]


def _classify_log(records: list[RequestRecord]) -> dict[str, int]:
    return {
        "total_requests": len(records),
        "range_get_requests": sum(
            1 for r in records if r.method == "GET" and r.range_header is not None
        ),
        "head_requests": sum(1 for r in records if r.method == "HEAD"),
        "response_body_bytes": sum(r.response_bytes for r in records),
    }


def _run_scenario_sync(url: str, n: int, scenario: str, bidirectional: bool) -> None:
    with tensogram.TensogramFile.open(url, bidirectional=bidirectional) as file:
        if scenario == "message_count":
            _ = file.message_count()
        elif scenario == "read_message(0)":
            _ = file.read_message(0)
        elif scenario == "read_message(N-1)":
            _ = file.read_message(n - 1)
        elif scenario == "read_message(N/2)":
            _ = file.read_message(n // 2)
        elif scenario == "iter":
            count = file.message_count()
            for i in range(count):
                _ = file.read_message(i)
        else:
            raise ValueError(f"unknown scenario: {scenario}")


async def _run_scenario_async(
    url: str, n: int, scenario: str, bidirectional: bool
) -> None:
    file = await tensogram.AsyncTensogramFile.open(url, bidirectional=bidirectional)
    async with file:
        if scenario == "message_count":
            _ = await file.message_count()
        elif scenario == "read_message(0)":
            _ = await file.read_message(0)
        elif scenario == "read_message(N-1)":
            _ = await file.read_message(n - 1)
        elif scenario == "read_message(N/2)":
            _ = await file.read_message(n // 2)
        elif scenario == "iter":
            count = await file.message_count()
            for i in range(count):
                _ = await file.read_message(i)
        else:
            raise ValueError(f"unknown scenario: {scenario}")


def _bench_one_cell(
    server: MockServer,
    spec: CellSpec,
    scenario: str,
    bidirectional: bool,
    mode: str,
    cell_id: int,
) -> dict[str, object]:
    walker_label = "bidirectional" if bidirectional else "forward-only"
    run_id = f"py-{cell_id}"
    url = server.url_for(run_id, spec.fixture_name)

    started = time.perf_counter_ns()
    if mode == "sync":
        _run_scenario_sync(url, spec.tier, scenario, bidirectional)
    else:
        asyncio.run(_run_scenario_async(url, spec.tier, scenario, bidirectional))
    wall_ms = (time.perf_counter_ns() - started) / 1e6

    counters = _classify_log(server.log_for(run_id))
    return {
        "language": "python",
        "mode": mode,
        "fixture_kind": spec.fixture_kind,
        "fixture_name": spec.fixture_name,
        "tier": spec.tier,
        "scenario": scenario,
        "walker": walker_label,
        "total_requests": counters["total_requests"],
        "range_get_requests": counters["range_get_requests"],
        "head_requests": counters["head_requests"],
        "response_body_bytes": counters["response_body_bytes"],
        "wall_ms": wall_ms,
        "semantics": "cold_open_plus_operation_plus_close",
    }


def _run_matrix(
    matrix: list[CellSpec],
    scenarios: tuple[str, ...],
    mode: str,
) -> list[dict[str, object]]:
    server = MockServer(_FIXTURES_DIR)
    server.start()
    records: list[dict[str, object]] = []
    cell_id = 0
    try:
        for spec in matrix:
            for scenario in scenarios:
                for bidirectional in (False, True):
                    cell_id += 1
                    record = _bench_one_cell(
                        server, spec, scenario, bidirectional, mode, cell_id
                    )
                    records.append(record)
    finally:
        server.stop()
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick", action="store_true", help="N=10 cells only (CI smoke)"
    )
    mode_group.add_argument(
        "--headline",
        action="store_true",
        help="N=100 iter both walkers, header + footer",
    )
    parser.add_argument("--mode", choices=("sync", "async"), default="sync")
    parser.add_argument(
        "--json",
        type=pathlib.Path,
        default=_DEFAULT_OUT,
        help=f"NDJSON output path (default: {_DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)

    if args.quick:
        matrix = _quick_matrix()
        scenarios = _SCENARIOS
    elif args.headline:
        matrix = _headline_matrix()
        scenarios = ("iter",)
    else:
        matrix = _matrix()
        scenarios = _SCENARIOS

    args.json.parent.mkdir(parents=True, exist_ok=True)
    records = _run_matrix(matrix, scenarios, args.mode)
    with args.json.open("w") as f:
        for record in records:
            json.dump(record, f, separators=(",", ":"))
            f.write("\n")
    print(f"wrote {len(records)} cells to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
