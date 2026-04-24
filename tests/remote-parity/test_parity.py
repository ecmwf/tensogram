# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""pytest entry for the cross-language remote-parity harness.

Each (fixture, language, op) is its own test case so failures name the
exact case that diverged. Tests skip (not fail) when prerequisites are
missing — they require external build artifacts (Rust driver binary,
TS driver node_modules).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_parity import (
    _DRIVERS_DIR,
    _FIXTURES,
    _FIXTURES_DIR,
    _OPS,
    _RUST_DRIVER_BIN,
    DriverCase,
    collect_events,
    filter_scan_events,
    load_golden,
)

_CASES = [
    DriverCase(fixture=fixture, language=language, op=op)
    for fixture in _FIXTURES
    for language in ("rust", "ts")
    for op in _OPS
]


def _missing_prereqs() -> list[str]:
    missing: list[str] = []
    if not _FIXTURES_DIR.exists() or not any(_FIXTURES_DIR.glob("*.tgm")):
        missing.append("fixtures/ (run tools/gen_fixtures.py)")
    if not _RUST_DRIVER_BIN.exists():
        missing.append("rust driver release binary (run cargo build --release)")
    if not (_DRIVERS_DIR / "node_modules").exists():
        missing.append("drivers/node_modules (run npm install in drivers/)")
    return missing


@pytest.fixture(scope="module")
def events():
    missing = _missing_prereqs()
    if missing:
        pytest.skip("remote-parity prereqs missing: " + ", ".join(missing))
    return collect_events(list(_CASES))


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.run_id)
def test_parity_against_golden(case: DriverCase, events) -> None:
    actual = filter_scan_events(events[case.run_id])
    expected = load_golden(case)
    if not expected:
        pytest.skip(f"no golden for {case.run_id}. Run tools/regen_goldens.py --regen to create.")
    assert actual == expected, f"parity mismatch for {case.run_id}"


_CROSS_LANGUAGE_PARITY_OPS: tuple[str, ...] = ("message-count", "read-last")


@pytest.mark.parametrize("fixture", _FIXTURES, ids=list(_FIXTURES))
@pytest.mark.parametrize("op", _CROSS_LANGUAGE_PARITY_OPS, ids=list(_CROSS_LANGUAGE_PARITY_OPS))
def test_rust_ts_parity_scan_only(fixture: str, op: str, events) -> None:
    """Rust and TS must issue the same `scan`-category request sequence.

    Restricted to ops where both backends do a *full* forward scan: current
    Rust `open_remote` is lazy (reads only the first preamble at open), but
    TS `fromUrl` eagerly walks every preamble at open time. `message-count`
    and `read-last` force Rust to catch up to the same coverage, so the
    request multisets match there. See `test_open_divergence` for the
    documented laziness difference on `open` / `read-first`.
    """
    rust_case = DriverCase(fixture=fixture, language="rust", op=op)  # type: ignore[arg-type]
    ts_case = DriverCase(fixture=fixture, language="ts", op=op)  # type: ignore[arg-type]
    rust_scan = filter_scan_events(events[rust_case.run_id])
    ts_scan = filter_scan_events(events[ts_case.run_id])
    assert [(e.scan_round, e.direction, e.logical_range) for e in rust_scan] == [
        (e.scan_round, e.direction, e.logical_range) for e in ts_scan
    ], f"Rust vs TS scan-event divergence for {fixture}/{op}"


_EXPECTED_MESSAGE_COUNTS: dict[str, int] = {
    "single-msg": 1,
    "two-msg": 2,
    "ten-msg": 10,
    "hundred-msg": 100,
}


@pytest.mark.parametrize("fixture", _FIXTURES, ids=list(_FIXTURES))
def test_open_divergence_rust_lazy_ts_eager(fixture: str, events) -> None:
    """Documents the known asymmetry: Rust opens lazily, TS opens eagerly.

    If this test ever fails, one of the backends has changed its open-time
    behaviour. Update the assertion deliberately and re-baseline goldens.
    """
    expected_n = _EXPECTED_MESSAGE_COUNTS[fixture]
    rust_open = filter_scan_events(events[f"{fixture}-rust-open-forward"])
    ts_open = filter_scan_events(events[f"{fixture}-ts-open-forward"])

    assert len(rust_open) == 1, (
        f"Rust open is expected to issue exactly one preamble GET; "
        f"got {len(rust_open)} for {fixture}"
    )
    assert len(ts_open) == expected_n, (
        f"TS open is expected to walk all {expected_n} preambles; got {len(ts_open)} for {fixture}"
    )
