# Features Decided to Implement

Accepted features that are planned but not yet implemented.
For speculative ideas, see `IDEAS.md`.

## API

- [x] ~~Populate `reserved` metadata field with provenance information~~ → `encode.rs:populate_reserved_provenance()`

## CLI

- [ ] `tensogram merge` — merge common metadata from multiple files (currently first-takes-precedence; should support configurable merge strategies)

## Metadata

- [ ] *metadata-major-refactor*:
  - currently 'extra' is the serde catch-all for forward/backward compatibility. It ensures any CBOR key in a message survives round-trip even if the decoder doesn't know about it. In practice, most structured metadata currently should go into common (shared) or payload (per-object), with extra used only for ad-hoc annotations or unknown keys.
  - refactor to remove 'common' and 'payload' from metadata objects. all structured will go into a single 'metadata' map.
  - 'metadata' becomes an array per data object. all entries are independent.
  - commonalities are computed in software after decoding cbor and only if requested or needed for algorithmic purposes (like merges and splits).
  - namespaces still exist but not hardcoded and used for semantics.
  - only special maps that remain are:
    - '_reserved_' formerly 'reserved', used for internals of the library
    - '_extra_' formerly 'extra', continues to serve as a catch-all
  - make all this explicity and updated in the docs/ and wire format.

## Documentation

- [x] ~~Document all error paths in docs/ (error handling reference page)~~ → `docs/src/guide/error-handling.md`

## Builds

- [ ] CI matrix for all three language test suites on every commit (partially done — Rust, Python, C++ run but GRIB tests gated on ecCodes availability)

## Tests and Examples

- [ ] *consumer-side-streaming*:
  - please ensure there is python example in examples/  showing how to decode a stream, from downloading from a https file (you may need to moc the server) and decode the tensogram stream as it arrives without any large memory buffer allocation containing the whole tensogram message. At the recieving end, it should create xarray datasets locally in memory as the data object frames arrive. 
  - Check that internally this indeed to parses the data objects as they arrive (we assume we can cope with a single data object at a time in memory) and transform them to xarray. This is possible because in streaming we insert the PRECEDER METADATA FRAME.

## Optimisation
 
- [ ] *minimise-mem-alloc*: 
  - Tensogram should minimise large mem allocations or decoding of data where possible. 
  - Decoding of actual data into tensors should be delayed until absolutely necessary (when data actually access for caller usage). 
  - Use the metadata for dims sizes and shapes to prepare lazy objects where necessary.
  - ensure this is reflected in the docs and as a strategic design choice in DESIGN.md

## Code Quality

- [ ] improve code coverage until it reachs >95% line coverage
- [ ] add logging trace (not to stdout) of more performance critical information, so that we can analyse what is going on to be able to improve it.
