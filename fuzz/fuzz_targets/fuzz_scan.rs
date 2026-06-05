// Fuzz the multi-message scanner: arbitrary bytes -> scan boundaries.
// Targets the skip-to-next-marker recovery loop and the
// total_length-driven jump (loop-termination / superlinear-scan DoS,
// out-of-bounds slice on a crafted total_length).
//
// Security invariant: never panic / hang / UB on any input; the scan
// must always terminate.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensogram::scan;

fuzz_target!(|data: &[u8]| {
    let offsets = scan(data);
    // Defensive cross-check: every reported (offset, len) must lie
    // within the buffer — a scanner returning an out-of-range slice
    // would be a latent OOB in any consumer.
    for (off, len) in offsets {
        assert!(off <= data.len(), "scan offset {off} > buf {}", data.len());
        let end = off.checked_add(len).expect("scan offset+len overflow");
        assert!(end <= data.len(), "scan end {end} > buf {}", data.len());
    }
});
