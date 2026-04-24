# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Parametric parity assertions for the remote-parity harness.

Each (fixture, language, op) is its own test case so failures name the
exact case that diverged. Tests skip (not fail) when prerequisites are
missing — they require external build artefacts (Rust driver binary,
TS driver node_modules).

Three layers of invariants:

1. Cross-language equivalence on ops where both backends do a full
   forward scan (`message-count`, `read-last`).
2. Per-language scan-pattern shape: every scan event is a 24-byte
   forward read whose offset matches the fixture's actual layout
   (computed live via `tensogram.scan`).
3. The Rust-lazy / TS-eager open-time divergence is documented and
   pinned: changing it must be a deliberate test update.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import tensogram

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from classifier import PREAMBLE_BYTES
from run_parity import (
    _DRIVERS_DIR,
    _FIXTURES,
    _FIXTURES_DIR,
    _OPS,
    _RUST_DRIVER_BIN,
    DriverCase,
    Language,
    Op,
    collect_events,
    filter_scan_events,
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


@pytest.fixture(scope="module")
def fixture_layouts() -> dict[str, list[tuple[int, int]]]:
    return {
        name: [
            (offset, length)
            for offset, length in tensogram.scan((_FIXTURES_DIR / f"{name}.tgm").read_bytes())
        ]
        for name in _FIXTURES
    }


_FULL_SCAN_OPS: tuple[str, ...] = ("message-count", "read-last")


@pytest.mark.parametrize("fixture", _FIXTURES, ids=list(_FIXTURES))
@pytest.mark.parametrize("op", _FULL_SCAN_OPS, ids=list(_FULL_SCAN_OPS))
def test_rust_ts_parity_on_full_scan_ops(fixture: str, op: Op, events) -> None:
    rust_case = DriverCase(fixture=fixture, language="rust", op=op)
    ts_case = DriverCase(fixture=fixture, language="ts", op=op)
    rust_scan = filter_scan_events(events[rust_case.run_id])
    ts_scan = filter_scan_events(events[ts_case.run_id])
    assert [(e.scan_round, e.direction, e.logical_range) for e in rust_scan] == [
        (e.scan_round, e.direction, e.logical_range) for e in ts_scan
    ], f"Rust vs TS scan-event divergence for {fixture}/{op}"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.run_id)
def test_scan_event_shape_invariants(case: DriverCase, events) -> None:
    scan_events = filter_scan_events(events[case.run_id])
    for i, event in enumerate(scan_events):
        where = f"{case.run_id}[{i}]"
        assert event.direction == "forward", f"{where}: forward-only walk expected"
        assert event.scan_round == i, f"{where}: rounds must be contiguous from 0"
        start, end = event.logical_range
        assert end - start == PREAMBLE_BYTES, (
            f"{where}: scan range must be PREAMBLE_BYTES ({PREAMBLE_BYTES}); got {end - start}"
        )

    offsets = [e.logical_range[0] for e in scan_events]
    assert offsets == sorted(offsets), (
        f"{case.run_id}: scan offsets must be monotonically increasing"
    )
    assert len(set(offsets)) == len(offsets), f"{case.run_id}: scan offsets must be unique"


_FULL_SCAN_OPS_LAYOUT: tuple[str, ...] = ("message-count", "read-last")


@pytest.mark.parametrize("language", ["rust", "ts"], ids=["rust", "ts"])
@pytest.mark.parametrize("op", _FULL_SCAN_OPS_LAYOUT, ids=list(_FULL_SCAN_OPS_LAYOUT))
@pytest.mark.parametrize("fixture", _FIXTURES, ids=list(_FIXTURES))
def test_full_scan_offsets_match_fixture_layout(
    fixture: str, language: Language, op: Op, events, fixture_layouts
) -> None:
    case = DriverCase(fixture=fixture, language=language, op=op)
    scan_events = filter_scan_events(events[case.run_id])
    expected_starts = [offset for offset, _length in fixture_layouts[fixture]]
    actual_starts = [e.logical_range[0] for e in scan_events]
    assert actual_starts == expected_starts, (
        f"{case.run_id}: scan offsets must match the fixture's message starts; "
        f"expected {expected_starts}, got {actual_starts}"
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.run_id)
def test_no_fallback_or_error_events(case: DriverCase, events) -> None:
    """Header-indexed non-streaming fixtures must never trigger eager fallback.

    If TS bails to a full GET (`category="fallback"`) or any backend
    surfaces an HTTP error on these fixtures, that's a regression — the
    lazy Range path should handle every committed fixture cleanly.
    """
    bad = [e for e in events[case.run_id] if e.category in ("fallback", "error")]
    assert not bad, f"{case.run_id}: unexpected non-scan events: " + ", ".join(
        f"{e.category}@{e.logical_range}" for e in bad
    )


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
    behaviour. Update the assertion deliberately.
    """
    expected_n = _EXPECTED_MESSAGE_COUNTS[fixture]
    rust_case = DriverCase(fixture=fixture, language="rust", op="open")
    ts_case = DriverCase(fixture=fixture, language="ts", op="open")
    rust_open = filter_scan_events(events[rust_case.run_id])
    ts_open = filter_scan_events(events[ts_case.run_id])

    assert len(rust_open) == 1, (
        f"Rust open is expected to issue exactly one preamble GET; "
        f"got {len(rust_open)} for {fixture}"
    )
    assert len(ts_open) == expected_n, (
        f"TS open is expected to walk all {expected_n} preambles; got {len(ts_open)} for {fixture}"
    )
