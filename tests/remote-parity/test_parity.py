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

import json
import pathlib
import sys

import pytest
import tensogram

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from classifier import PREAMBLE_BYTES
from run_parity import (
    _FIXTURES,
    _FIXTURES_DIR,
    _OPS,
    DriverCase,
    Language,
    Op,
    collect_events,
    filter_scan_events,
    missing_fixtures,
    missing_prereqs,
)

_CASES = [
    DriverCase(fixture=fixture, language=language, op=op)
    for fixture in _FIXTURES
    for language in ("rust", "ts")
    for op in _OPS
]

_LAYOUT_CASES = [
    DriverCase(fixture=fixture, language=language, op="dump-layout", mode=mode)
    for fixture in _FIXTURES
    for language in ("rust", "ts")
    for mode in ("forward", "bidirectional")
]


@pytest.fixture(scope="module")
def events():
    missing = missing_prereqs()
    if missing:
        pytest.skip("remote-parity prereqs missing: " + ", ".join(missing))
    return collect_events([*_CASES, *_LAYOUT_CASES])


@pytest.fixture(scope="module")
def fixture_layouts() -> dict[str, list[tuple[int, int]]]:
    # Only needs the fixture files; the Rust/TS drivers don't matter here.
    fix_missing = missing_fixtures()
    if fix_missing:
        pytest.skip("remote-parity fixtures missing: " + ", ".join(fix_missing))
    return {
        name: list(tensogram.scan((_FIXTURES_DIR / f"{name}.tgm").read_bytes()))
        for name in _FIXTURES
    }


_FULL_SCAN_OPS: tuple[str, ...] = ("message-count", "read-last")


@pytest.mark.parametrize("fixture", _FIXTURES, ids=list(_FIXTURES))
@pytest.mark.parametrize("op", _FULL_SCAN_OPS, ids=list(_FULL_SCAN_OPS))
def test_rust_ts_parity_on_full_scan_ops(fixture: str, op: Op, events) -> None:
    rust_case = DriverCase(fixture=fixture, language="rust", op=op)
    ts_case = DriverCase(fixture=fixture, language="ts", op=op)
    rust_scan = filter_scan_events(events[rust_case.run_id].events)
    ts_scan = filter_scan_events(events[ts_case.run_id].events)
    assert [(e.scan_round, e.direction, e.logical_range) for e in rust_scan] == [
        (e.scan_round, e.direction, e.logical_range) for e in ts_scan
    ], f"Rust vs TS scan-event divergence for {fixture}/{op}"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.run_id)
def test_scan_event_shape_invariants(case: DriverCase, events) -> None:
    scan_events = filter_scan_events(events[case.run_id].events)
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
    scan_events = filter_scan_events(events[case.run_id].events)
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
    bad = [e for e in events[case.run_id].events if e.category in ("fallback", "error")]
    assert not bad, f"{case.run_id}: unexpected non-scan events: " + ", ".join(
        f"{e.category}@{e.logical_range}" for e in bad
    )


_EXPECTED_MESSAGE_COUNTS: dict[str, int] = {
    "single-msg": 1,
    "two-msg": 2,
    "ten-msg": 10,
    "hundred-msg": 100,
    "single-msg-footer": 1,
    "ten-msg-footer": 10,
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
    rust_open = filter_scan_events(events[rust_case.run_id].events)
    ts_open = filter_scan_events(events[ts_case.run_id].events)

    assert len(rust_open) == 1, (
        f"Rust open is expected to issue exactly one preamble GET; "
        f"got {len(rust_open)} for {fixture}"
    )
    assert len(ts_open) == expected_n, (
        f"TS open is expected to walk all {expected_n} preambles; got {len(ts_open)} for {fixture}"
    )


@pytest.mark.parametrize("fixture", _FIXTURES, ids=list(_FIXTURES))
def test_read_first_divergence_rust_lazy_ts_eager(fixture: str, events) -> None:
    """Same Rust-lazy / TS-eager asymmetry on `read-first`.

    Rust only needs the first preamble to satisfy `read_message(0)`; TS
    has already walked every preamble at `fromUrl` time, so its
    `read-first` issues no further scan events on top of the open-time
    walk.
    """
    expected_n = _EXPECTED_MESSAGE_COUNTS[fixture]
    rust_case = DriverCase(fixture=fixture, language="rust", op="read-first")
    ts_case = DriverCase(fixture=fixture, language="ts", op="read-first")
    rust_scan = filter_scan_events(events[rust_case.run_id].events)
    ts_scan = filter_scan_events(events[ts_case.run_id].events)

    assert len(rust_scan) == 1, (
        f"Rust read-first is expected to issue exactly one preamble GET; "
        f"got {len(rust_scan)} for {fixture}"
    )
    assert len(ts_scan) == expected_n, (
        f"TS read-first is expected to have already walked all {expected_n} "
        f"preambles at open time; got {len(ts_scan)} for {fixture}"
    )


@pytest.mark.parametrize("language", ["rust", "ts"], ids=["rust", "ts"])
@pytest.mark.parametrize("fixture", _FIXTURES, ids=list(_FIXTURES))
def test_forward_vs_bidirectional_layouts_equal(fixture: str, language: Language, events) -> None:
    """Forward-only and bidirectional walkers must agree on the final layout set.

    The walkers may issue different HTTP request patterns — that is the
    whole point of the bidirectional optimisation — but the discovered
    `(offset, length)` layout per message must match exactly.  Any
    drift indicates the bidirectional walker has gained or lost a
    message somewhere along its inward sweep.

    Runs once per language so a divergence in either Rust or TS gets
    its own named failure.
    """
    fwd_case = DriverCase(fixture=fixture, language=language, op="dump-layout", mode="forward")
    bidir_case = DriverCase(
        fixture=fixture, language=language, op="dump-layout", mode="bidirectional"
    )
    fwd_layouts = json.loads(events[fwd_case.run_id].stdout)
    bidir_layouts = json.loads(events[bidir_case.run_id].stdout)
    assert fwd_layouts == bidir_layouts, (
        f"forward vs bidirectional layout divergence for {fixture}/{language}: "
        f"forward={fwd_layouts}, bidirectional={bidir_layouts}"
    )
