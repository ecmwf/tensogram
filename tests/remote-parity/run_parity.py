# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Cross-language HTTP parity orchestrator.

Starts the mock server on an auto-chosen port, runs each configured
driver one at a time, fetches the server-side request log, and
normalises it to the ScanEvent schema. The pytest suite consumes the
result and asserts parametric invariants (cross-language equivalence,
scan-event shape, scaling against the fixture's actual layout).
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from classifier import NON_SCAN_ROUND, Category, Direction, Observation, RoundBuilder, classify
from mock_server import MockServer, RequestRecord

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_FIXTURES_DIR = _THIS_DIR / "fixtures"
_DRIVERS_DIR = _THIS_DIR / "drivers"

_RUST_DRIVER_BIN = (
    _DRIVERS_DIR / "rust_driver" / "target" / "release" / "remote-parity-rust-driver"
)
_TS_DRIVER_SCRIPT = _DRIVERS_DIR / "ts_driver.ts"

Language = Literal["rust", "ts"]
Op = Literal["open", "message-count", "read-first", "read-last"]

_FIXTURES: tuple[str, ...] = (
    "single-msg",
    "two-msg",
    "ten-msg",
    "hundred-msg",
)

_OPS: tuple[Op, ...] = (
    "open",
    "message-count",
    "read-first",
    "read-last",
)


@dataclass(frozen=True)
class ScanEvent:
    run_id: str
    scan_round: int
    direction: Direction
    category: Category
    logical_range: tuple[int, int]
    physical_requests: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "scan_round": self.scan_round,
            "direction": self.direction,
            "category": self.category,
            "logical_range": list(self.logical_range),
            "physical_requests": list(self.physical_requests),
        }


@dataclass(frozen=True)
class DriverCase:
    fixture: str
    language: Language
    op: Op
    mode: Literal["forward"] = "forward"

    @property
    def run_id(self) -> str:
        return f"{self.fixture}-{self.language}-{self.op}-{self.mode}"


def normalise_log(run_id: str, records: list[RequestRecord]) -> list[ScanEvent]:
    builder = RoundBuilder()
    events: list[ScanEvent] = []
    for record in records:
        classified = classify(
            Observation(
                method=record.method,
                range_header=record.range_header,
                status=record.status,
                response_bytes=record.response_bytes,
            )
        )
        scan_round = builder.assign(classified)
        events.append(
            ScanEvent(
                run_id=run_id,
                scan_round=scan_round if classified.category == "scan" else NON_SCAN_ROUND,
                direction=classified.direction,
                category=classified.category,
                logical_range=classified.logical_range,
                physical_requests=(classified.physical,),
            )
        )
    return events


def filter_scan_events(events: list[ScanEvent]) -> list[ScanEvent]:
    return [e for e in events if e.category == "scan"]


def _all_cases() -> list[DriverCase]:
    cases = []
    for fixture in _FIXTURES:
        for language in ("rust", "ts"):
            for op in _OPS:
                cases.append(DriverCase(fixture=fixture, language=language, op=op))
    return cases


def missing_fixtures() -> list[str]:
    expected = [_FIXTURES_DIR / f"{name}.tgm" for name in _FIXTURES]
    absent = [p for p in expected if not p.exists()]
    if not absent:
        return []
    names = ", ".join(p.name for p in absent)
    return [f"fixtures ({names}) — run `python {_THIS_DIR / 'tools/gen_fixtures.py'}`"]


def missing_prereqs() -> list[str]:
    missing: list[str] = []
    missing.extend(missing_fixtures())
    if not _RUST_DRIVER_BIN.exists():
        missing.append(
            "rust driver (run `cargo build --release --manifest-path "
            f"{_DRIVERS_DIR / 'rust_driver/Cargo.toml'}`)"
        )
    node_modules = _DRIVERS_DIR / "node_modules"
    if not node_modules.exists():
        missing.append(
            f"ts driver deps (run `cd {_DRIVERS_DIR} && npm install --no-audit --no-fund`)"
        )
    if not shutil.which("npx"):
        missing.append("npx (install Node.js ≥ 20)")
    return missing


def _ensure_prereqs() -> None:
    missing = missing_prereqs()
    if missing:
        raise SystemExit("remote-parity prerequisites missing:\n  - " + "\n  - ".join(missing))


_DRIVER_TIMEOUT_S = 60


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Reap a timed-out driver and its descendants.

    `killpg` on the new session reaps the whole tree, but it can fail
    with `ProcessLookupError` when the group is already gone, or with
    a broader `OSError` on platforms without process-group support.
    Either way we fall back to killing the direct child if it's still
    around so the timeout never turns into a secondary harness error.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        return
    except ProcessLookupError:
        return
    except OSError:
        pass
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


def _run_driver(case: DriverCase, url: str) -> None:
    if case.language == "rust":
        cmd = [str(_RUST_DRIVER_BIN), "--url", url, "--op", case.op]
    else:
        cmd = [
            "npx",
            "tsx",
            str(_TS_DRIVER_SCRIPT),
            "--url",
            url,
            "--op",
            case.op,
        ]
    env_cwd = _DRIVERS_DIR if case.language == "ts" else _THIS_DIR

    # `start_new_session=True` puts the driver in its own process
    # group so `os.killpg` on timeout reaps the whole tree, including
    # any descendants spawned by `npx -> tsx -> node`. Without this,
    # `subprocess.run(timeout=...)` would only kill the direct child
    # and leave grandchildren attached to the harness server.
    with subprocess.Popen(
        cmd,
        cwd=env_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=_DRIVER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"driver {case.language}/{case.op}@{case.fixture} "
                f"timed out after {_DRIVER_TIMEOUT_S}s\n"
                f"  cmd: {cmd}\n"
                f"  stdout: {stdout.strip()}\n"
                f"  stderr: {stderr.strip()}"
            ) from None

    if proc.returncode != 0:
        raise RuntimeError(
            f"driver {case.language}/{case.op}@{case.fixture} failed "
            f"(rc={proc.returncode})\n"
            f"  cmd: {cmd}\n"
            f"  stdout: {stdout.strip()}\n"
            f"  stderr: {stderr.strip()}"
        )


def _check_no_duplicate_run_ids(cases: list[DriverCase]) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.run_id in seen:
            raise ValueError(
                f"duplicate run_id '{case.run_id}' in cases list — "
                "each DriverCase must map to a unique run_id to avoid "
                "mock-server log conflation."
            )
        seen.add(case.run_id)


def collect_events(cases: list[DriverCase]) -> dict[str, list[ScanEvent]]:
    if not cases:
        raise ValueError("collect_events: no driver cases provided")
    _check_no_duplicate_run_ids(cases)
    _ensure_prereqs()

    result: dict[str, list[ScanEvent]] = {}
    with MockServer(_FIXTURES_DIR) as server:
        for case in cases:
            url = server.url_for(case.run_id, f"{case.fixture}.tgm")
            _run_driver(case, url)
            records = server.log_for(case.run_id)
            if not records:
                raise RuntimeError(
                    f"driver {case.language}/{case.op}@{case.fixture} produced "
                    "no server-side requests — likely a wrong URL or run_id; "
                    "check the driver's stderr."
                )
            result[case.run_id] = normalise_log(case.run_id, records)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        action="append",
        help="Limit to these fixtures (default: all)",
    )
    args = parser.parse_args(argv)

    cases = _all_cases()
    if args.fixture:
        cases = [c for c in cases if c.fixture in args.fixture]
    if not cases:
        print("no cases match", file=sys.stderr)
        return 2

    events = collect_events(cases)
    print(f"{'case':<55} {'scan events':>12}")
    print("-" * 70)
    for case in cases:
        scan_count = len(filter_scan_events(events[case.run_id]))
        print(f"{case.run_id:<55} {scan_count:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
