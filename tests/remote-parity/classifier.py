# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Request classification and scan-round assignment.

Two responsibilities, kept deliberately separate so future backward-scan
and eager footer-discovery work can extend them without touching the
rest of the harness:

- `Classifier` maps one raw HTTP observation to `(category, direction,
  logical_range)` based on status + method + range length + response
  bytes. Status-aware: non-2xx responses become ``error``.
- `RoundBuilder` walks classified events in observed order and assigns
  `scan_round`. Forward-only today (one scan event per round);
  bidirectional will pair forward + backward events per round.

Schema contract is defined by `schema.json`. Non-scan events use
``scan_round = -1`` and ``direction = "none"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Category = Literal["probe", "scan", "payload", "fallback", "error"]
Direction = Literal["forward", "backward", "none"]

PREAMBLE_BYTES = 24
NON_SCAN_ROUND = -1

_EXPLICIT_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d+)$")


@dataclass(frozen=True)
class Observation:
    """One raw observation from the mock server, pre-classification."""

    method: str
    range_header: str | None
    status: int
    response_bytes: int


@dataclass(frozen=True)
class Classified:
    """Result of classifying one `Observation`."""

    category: Category
    direction: Direction
    logical_range: tuple[int, int]
    physical: dict


def classify(observation: Observation) -> Classified:
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

    # Scan walks issue strict explicit ranges (`bytes=N-M`, decimals
    # only). Suffix (`bytes=-N`), open-ended (`bytes=N-`), multi-range,
    # missing, or otherwise malformed Range headers all fall to
    # `payload` — they are not part of the current forward-only scan
    # contract.
    explicit = _parse_explicit_single_range(observation.range_header)
    if explicit is None:
        return Classified("payload", "none", logical_range, physical)

    start, end_inclusive = explicit
    actual_end = min(end_inclusive + 1, start + observation.response_bytes)
    if actual_end - start == PREAMBLE_BYTES:
        return Classified("scan", "forward", (start, actual_end), physical)
    return Classified("payload", "none", (start, actual_end), physical)


def _parse_explicit_single_range(range_header: str | None) -> tuple[int, int] | None:
    """Parse a strict ``bytes=N-M`` header into ``(start, end_inclusive)``.

    Returns ``None`` for missing, suffix (``bytes=-N``), open-ended
    (``bytes=N-``), multi-range (``bytes=a-b,c-d``), signed, or any
    other malformed input. Matching is intentionally strict because
    only this form is part of the scan-walk contract today.
    """
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
        # Suffix ranges (`bytes=-n`) identify the last n bytes of the
        # object, but without the total object size or a parsed
        # `Content-Range` header we can't recover the true logical
        # start offset here. Fall back to the observed response span;
        # capturing total size is tracked for when bidirectional scan
        # adds suffix-range usage.
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

    # Always clamp the upper bound to the bytes actually served: if the
    # client asked beyond EOF, the mock server (and most HTTP servers)
    # return only the available suffix. Trusting the request header here
    # would inflate logical_range past what was observed and cause
    # spurious parity mismatches.
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


class RoundBuilder:
    """Assigns `scan_round` to classified events in observation order.

    Forward-only contract: every ``scan`` event starts a new round.
    Non-scan events get `scan_round = NON_SCAN_ROUND`.

    Bidirectional (future): forward + backward events belonging to the
    same logical round share a `scan_round`. This class is the
    extension point — override or subclass without touching callers.
    """

    def __init__(self) -> None:
        self._next_round = 0

    def assign(self, classified: Classified) -> int:
        if classified.category != "scan":
            return NON_SCAN_ROUND
        round_id = self._next_round
        self._next_round += 1
        return round_id
