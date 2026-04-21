// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for the hash-while-encoding optimisation.
//!
//! **v3 note.** The contract these tests guarded — that
//! `descriptor.hash` is populated with `xxh3_64(encoded_payload)` —
//! no longer holds.  v3 moves the per-object hash to the inline
//! slot in the data-object frame's footer and hashes a different
//! scope (`payload + masks + CBOR`, see `plans/WIRE_FORMAT.md`
//! §2.4).  Phase 6 of `plans/WIRE_FORMAT_CHANGES.md` rewrites this
//! suite against the inline slot.
//!
//! Until then the whole file is a stub that ensures the test
//! harness still compiles; there are no active assertions.

#[test]
#[ignore = "v3: hash moved to frame footer — re-enable in phase 6"]
fn hash_while_encoding_v3_placeholder() {
    // Intentionally empty.  See the module doc-comment above.
}
