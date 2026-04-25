# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Request classification and scan-round assignment.

Three responsibilities, kept deliberately separate so future scan-walker
extensions can extend them without touching the rest of the harness:

- ``Classifier`` maps one raw HTTP observation to ``(category, role,
  direction, logical_range)``.  Status-aware: non-2xx responses become
  ``error``.  When a ``FixtureLayout`` is supplied, 24-byte explicit
  Range fetches are role-tagged by matching the range's start/end
  against the fixture's known message starts and ends:
    - ``[msg_start, msg_start + 24)`` → ``fwd_preamble`` (forward
      preamble OR backward preamble validation; the round builder
      decides which).
    - ``[msg_end - 24, msg_end)`` → ``bwd_postamble`` (backward).
  Without a layout, the fallback labels every 24-byte fetch
  ``fwd_preamble`` (forward-only contract).

- ``RoundBuilder`` walks classified events in observed order and
  assigns ``scan_round``.  Forward-only mode: every scan event starts
  a new round.  Bidirectional mode: a round contains at most one
  ``fwd_preamble``, one ``bwd_postamble``, and one
  ``bwd_preamble_validation`` (a second ``fwd_preamble``-shaped fetch
  AFTER a ``bwd_postamble``).  Either of the paired requests can
  start a round (``Promise.allSettled`` / ``store.get_ranges`` may
  reorder the pair).

Schema contract is defined by ``schema.json``.  Non-scan events use
``scan_round = -1`` and ``direction = "none"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Category = Literal["probe", "scan", "payload", "fallback", "error"]
Direction = Literal["forward", "backward", "none"]
ScanRole = Literal["fwd_preamble", "bwd_postamble", "bwd_preamble_validation", "none"]

PREAMBLE_BYTES = 24
POSTAMBLE_BYTES = 24
NON_SCAN_ROUND = -1

_EXPLICIT_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d+)$")


@dataclass(frozen=True)
class FixtureLayout:
    """Pre-computed message starts and ends for one fixture.

    Used by the classifier to role-tag 24-byte explicit Range fetches
    in bidirectional mode without having to observe the walker's state.
    Computed once per fixture from ``tensogram.scan(fixture_bytes)``.
    """

    message_starts: frozenset[int]
    message_ends: frozenset[int]
    total_size: int


@dataclass(frozen=True)
class Observation:
    """One raw observation from the mock server, pre-classification."""

    method: str
    range_header: str | None
    status: int
    response_bytes: int


@dataclass(frozen=True)
class Classified:
    """Result of classifying one ``Observation``."""

    category: Category
    direction: Direction
    logical_range: tuple[int, int]
    physical: dict
    role: ScanRole = "none"


def classify(observation: Observation, layout: FixtureLayout | None = None) -> Classified:
    physical = _physical_dict(observation)
    logical_range = _logical_from_range(observation.range_header, observation.response_bytes)

    if observation.method == "HEAD":
        if observation.status == 200:
            return Classified("probe", "none", (0, 0), physical)
        return Classified("error", "none", logical_range, physical)

    if observation.method != "GET":
        return Classified("error", "none", logical_range, physical)

    if observation.status == 200:
        return Classified("fallback", "none", logical_range, physical)

    if observation.status != 206:
        return Classified("error", "none", logical_range, physical)

    explicit = _parse_explicit_single_range(observation.range_header)
    if explicit is None:
        return Classified("payload", "none", logical_range, physical)

    start, end_inclusive = explicit
    actual_end = min(end_inclusive + 1, start + observation.response_bytes)
    span = actual_end - start
    if span != PREAMBLE_BYTES:
        return Classified("payload", "none", (start, actual_end), physical)

    if layout is None:
        return Classified("scan", "forward", (start, actual_end), physical, "fwd_preamble")

    if start in layout.message_starts:
        return Classified("scan", "forward", (start, actual_end), physical, "fwd_preamble")
    if actual_end in layout.message_ends:
        return Classified("scan", "backward", (start, actual_end), physical, "bwd_postamble")
    return Classified("payload", "none", (start, actual_end), physical)


def _parse_explicit_single_range(range_header: str | None) -> tuple[int, int] | None:
    if range_header is None:
        return None
    m = _EXPLICIT_RANGE_RE.match(range_header.strip())
    if not m:
        return None
    start = int(m.group(1))
    end_inclusive = int(m.group(2))
    if end_inclusive < start:
        return None
    return start, end_inclusive


def _logical_from_range(range_header: str | None, response_bytes: int) -> tuple[int, int]:
    if range_header is None:
        return (0, response_bytes)
    if not range_header.startswith("bytes="):
        return (0, response_bytes)
    spec = range_header[len("bytes=") :].strip()

    if spec.startswith("-"):
        try:
            int(spec[1:])
        except ValueError:
            return (0, response_bytes)
        return (0, response_bytes)

    if "-" not in spec:
        return (0, response_bytes)
    start_s, end_s = spec.split("-", 1)
    try:
        start = int(start_s)
    except ValueError:
        return (0, response_bytes)
    if start < 0:
        return (0, response_bytes)

    served_end = start + response_bytes
    if not end_s:
        return (start, served_end)
    try:
        end_inclusive = int(end_s)
    except ValueError:
        return (start, served_end)
    if end_inclusive < start:
        return (start, served_end)
    return (start, min(end_inclusive + 1, served_end))


def _physical_dict(observation: Observation) -> dict:
    headers: dict[str, str] = {}
    if observation.range_header is not None:
        headers["Range"] = observation.range_header
    return {"method": observation.method, "headers": headers, "status": observation.status}


@dataclass
class RoundBuilder:
    """Assigns ``scan_round`` to classified events in observation order.

    Forward-only contract: every ``scan`` event starts a new round.
    Non-scan events get ``scan_round = NON_SCAN_ROUND``.

    Bidirectional contract: a round contains at most one
    ``fwd_preamble``, one ``bwd_postamble``, and one
    ``bwd_preamble_validation`` (which is a ``fwd_preamble``-shaped
    fetch following a ``bwd_postamble`` in the same round).  Either
    of the paired requests can start a round: ``Promise.allSettled``
    and ``store.get_ranges`` parallelise the pair, so the mock-server
    log can observe them in either order.
    """

    forward_only: bool = True
    _next_round: int = 0
    _round_has: set[ScanRole] = field(default_factory=set)

    def assign(self, c: Classified) -> int:
        if c.category != "scan":
            return NON_SCAN_ROUND
        if self.forward_only:
            r = self._next_round
            self._next_round += 1
            return r
        return self._assign_bidirectional(c.role)

    def _assign_bidirectional(self, role: ScanRole) -> int:
        # The 8 disambiguation rules below cover every possible
        # arrival order of a paired round + its preamble validation.
        relabel: ScanRole = role
        if role == "fwd_preamble" and "bwd_postamble" in self._round_has:
            # A `fwd_preamble`-shaped fetch following a `bwd_postamble`
            # within the same round is the backward preamble validation,
            # not a fresh forward preamble.
            relabel = "bwd_preamble_validation"

        # Open a new round when this role would conflict with the
        # current round's already-seen roles.
        opens_new = (
            (relabel == "fwd_preamble" and "fwd_preamble" in self._round_has)
            or (relabel == "bwd_postamble" and "bwd_postamble" in self._round_has)
            or (
                relabel == "bwd_preamble_validation"
                and "bwd_preamble_validation" in self._round_has
            )
        )
        if opens_new:
            self._open_new_round()
        self._round_has.add(relabel)
        return self._next_round

    def _open_new_round(self) -> None:
        self._next_round += 1
        self._round_has = set()
