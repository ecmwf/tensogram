# Features Decided to Implement

Accepted features that are planned but not yet implemented.
For speculative ideas, see `IDEAS.md`.

## API

- [ ] Populate `reserved` metadata field with provenance information:
  - `reserved.encoder.name` — library name (`"tensogram"`)
  - `reserved.encoder.version` — library version at encode time
  - `reserved.time` — UTC timestamp of encoding
  - `reserved.uuid` — RFC 4122 UUID for provenance tracking
  (Specified in WIRE_FORMAT.md, not yet implemented in code)

## CLI

- [ ] `tensogram merge` — merge common metadata from multiple files (currently first-takes-precedence; should support configurable merge strategies)

## Metadata

- (none currently)

## Documentation

- [ ] Document all error paths in docs/ (error handling reference page)

## Builds

- [ ] CI matrix for all three language test suites on every commit (partially done — Rust, Python, C++ run but GRIB tests gated on ecCodes availability)

## Code Quality

- [ ] Reach 95% line coverage (currently 90.5%)
