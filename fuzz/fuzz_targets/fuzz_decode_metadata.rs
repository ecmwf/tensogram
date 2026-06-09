// Fuzz the metadata-only decode path: arbitrary bytes -> framing ->
// CBOR metadata parse.  Targets CBOR-level attacks (deeply nested
// maps, huge claimed sizes, type confusion) without touching payloads.
//
// Security invariant: never panic / hang / UB on any input.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensogram::decode_metadata;

fuzz_target!(|data: &[u8]| {
    let _ = decode_metadata(data);
});
