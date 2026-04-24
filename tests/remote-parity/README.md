# Remote Parity Harness

Cross-language HTTP Range-request parity tests for the Tensogram remote
backend.

## Purpose

Ensure that the Rust (`TensogramFile::open_remote`) and TypeScript
(`TensogramFile.fromUrl`) remote backends issue **equivalent HTTP Range
request sequences** when reading the same `.tgm` file. This is the
foundation substrate against which any change to the remote read path
is validated — in particular the upcoming bidirectional-scan work
tracked in `plans/TODO.md` → *remote 8*.

The harness currently covers **forward-only** behaviour; any change
must keep the parametric invariants in `test_parity.py` green.

## Design

A Python mock HTTP server with Range + HEAD support logs every incoming
request, tagged with a `run_id` taken from the URL path. Per-language
drivers drive `TensogramFile` operations against that server; the
orchestrator then normalises the captured request log into a stream
of `ScanEvent`s (see `schema.json`), which the pytest suite asserts
against parametric invariants — no committed snapshot files.

**No production code changes are required.** Instrumentation lives
entirely on the mock-server side.

## What we assert

The harness deliberately avoids snapshot/golden files: this code is
still evolving, and pinning the exact request sequence would mean
regenerating snapshots on every legitimate change. Instead it asserts:

- **Cross-language equivalence** for ops where both backends do a
  full forward scan (`message-count`, `read-last`): Rust and TS must
  emit the same `(scan_round, direction, logical_range)` sequence.
- **Per-event shape invariants**: every scan event is a 24-byte
  forward read; `scan_round` increments contiguously from 0; offsets
  are strictly increasing and unique.
- **Layout match**: for `message-count`, the offsets emitted match
  the fixture's actual message starts (computed live via
  `tensogram.scan(fixture_bytes)` — no offset values hardcoded).
- **Documented divergence**: Rust `open` issues exactly 1 scan event;
  TS `open` walks all N preambles. If either backend changes its
  open-time laziness, the test fails loudly and the assertion is
  updated deliberately.

### Event schema

See `schema.json`. Each logged request is normalised to:

```
{
  "run_id":     "single-msg-rust-forward",
  "scan_round": 0,
  "direction":  "forward",
  "category":   "probe" | "scan" | "payload" | "fallback",
  "logical_range": [start, end_exclusive],
  "physical_requests": [
    { "method": "GET", "headers": { "Range": "bytes=0-23" }, "status": 206 }
  ]
}
```

Only `"scan"` events are compared for parity. `"probe"` (HEAD / Range
support probes) and `"payload"` (bulk payload fetches) are logged but
excluded from parity assertions; they are language- and implementation-
specific by design.

### Parity comparison rules

- **Ordered** across `scan_round`s (round N must match across languages).
- **Multiset-per-round** within a round (`Promise.all` and `get_ranges`
  may reorder; duplicate requests are detected by count).
- Inclusive HTTP `Range: bytes=a-b` normalised to `[a, b+1)`.
- HEAD requests categorised as `"probe"`, filtered out of parity.

### Known laziness asymmetry

Current `TensogramFile.fromUrl` in TypeScript walks **every** message
preamble at open time (see `typescript/src/file.ts:1178`
`lazyScanMessages`). The Rust `TensogramFile::open_remote` is
lazier: it reads only the first preamble at open, then walks the
rest on demand through `ensure_message_locked`.

Cross-language parity therefore only holds for operations that force
both backends to cover the same range — `message-count` and
`read-last`. For `open` and `read-first` the request sequences
diverge by design. This is captured by
`test_open_divergence_rust_lazy_ts_eager`: if either backend changes
its open-time laziness, the test fails loudly and the assertion is
updated deliberately.

This asymmetry is an input to the bidirectional-scan design — later
work must decide whether to keep it, reconcile it, or document it as
intentional.

### Run isolation

Runs are **serialised** — `run_parity.py` runs one driver at a time.
URLs include the run_id as the first path segment so even parallel
future runs can be disambiguated.

## Layout

```
tests/remote-parity/
├── README.md                  # this file
├── schema.json                # JSON Schema for ScanEvent
├── mock_server.py             # Python http.server with Range + request log
├── classifier.py              # request → ScanEvent classification + round builder
├── run_parity.py              # orchestrator: starts server, runs each driver
├── test_parity.py             # pytest: parametric parity invariants
├── test_unit.py               # pytest: classifier, mock server, schema contract
├── tools/
│   └── gen_fixtures.py        # .tgm fixture generator
├── fixtures/                  # generated .tgm files (checked in)
│   ├── single-msg.tgm
│   ├── two-msg.tgm
│   ├── ten-msg.tgm
│   └── hundred-msg.tgm
└── drivers/
    ├── rust_driver/           # standalone cargo bin
    │   ├── Cargo.toml
    │   └── src/main.rs
    └── ts_driver.ts           # Node script using @ecmwf.int/tensogram
```

## Running

```bash
# Generate fixtures (once, or after format changes):
python tests/remote-parity/tools/gen_fixtures.py

# Run all driver/op combinations and print scan-event counts:
python tests/remote-parity/run_parity.py

# As pytest:
make remote-parity            # builds drivers and runs the full suite
pytest tests/remote-parity/   # runs the suite assuming drivers are built
```

## Current scope

**Covered:**
- Header-indexed, non-streaming fixtures only (single/two/ten/hundred messages).
- Forward-only parity between Rust and TypeScript.
- Server-side request logging (no production code changes).

**Deferred to later work:**
- Bidirectional scan assertions — added when the Rust and TypeScript backends gain a backward walker.
- Streaming-tail / streaming-mid fixtures — TypeScript currently bails to eager download when it encounters a streaming (`total_length == 0`) message, which makes `scan`-category request logs non-comparable. These fixtures land once that divergence is fixed or explicitly documented.
- Footer-indexed fixtures — added alongside eager footer-indexed backward discovery.
- Mixed header+footer fixtures — same.
