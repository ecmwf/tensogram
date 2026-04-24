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
must keep parity or explicitly regenerate goldens with review.

## Design

A Python mock HTTP server with Range + HEAD support logs every incoming
request, tagged with a `run_id` taken from the URL path. Per-language
drivers drive `TensogramFile` operations against that server; the
orchestrator then compares the server-side request log between languages
and against checked-in golden logs.

**No production code changes are required.** Instrumentation lives
entirely on the mock-server side.

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
its open-time laziness, the test fails loudly and the assertion (and
any affected goldens) must be updated deliberately.

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
├── run_parity.py              # orchestrator: starts server, drives each language, diffs
├── test_parity.py             # pytest entry
├── tools/
│   ├── gen_fixtures.py        # deterministic .tgm generator
│   └── regen_goldens.py       # fail-closed regeneration (explicit --regen)
├── fixtures/                  # generated .tgm files (checked in)
│   ├── single-msg.tgm
│   ├── two-msg.tgm
│   ├── ten-msg.tgm
│   └── hundred-msg.tgm
├── goldens/                   # expected event logs per (fixture × language)
│   └── <fixture>-<lang>-forward.json
└── drivers/
    ├── rust_driver/           # cargo bin (workspace crate)
    │   ├── Cargo.toml
    │   └── src/main.rs
    └── ts_driver.ts           # Node script using @ecmwf.int/tensogram
```

## Running

```bash
# Generate fixtures (once, or after format changes):
python tests/remote-parity/tools/gen_fixtures.py

# Full parity check (starts server, runs all drivers, diffs against goldens):
python tests/remote-parity/run_parity.py

# As pytest (CI):
pytest tests/remote-parity/test_parity.py -v

# Regenerate goldens after a known-intentional change:
python tests/remote-parity/tools/regen_goldens.py --regen --fixture single-msg
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

## Golden update policy

Goldens are **fail-closed**: CI never auto-regenerates them. When a change
legitimately alters the scan request sequence, re-run
`tools/regen_goldens.py --regen --fixture <name>` locally, review the
diff, and commit.
