// Fuzz the validator across all levels (structure, metadata,
// integrity, fidelity).  `validate_buffer` is designed to never error
// out the process — it returns a report — so the invariant is simply
// "no panic / hang / UB on any input".
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensogram::{ValidateOptions, ValidationLevel, validate_buffer};

fuzz_target!(|data: &[u8]| {
    // Full fidelity validation walks every level, including the
    // decode round-trip and the NaN/Inf scan.
    let full = ValidateOptions {
        max_level: ValidationLevel::Fidelity,
        check_canonical: true,
        checksum_only: false,
    };
    let _ = validate_buffer(data, &full);

    // Checksum-only path is a distinct branch (Level 3 hashing only).
    let checksum = ValidateOptions {
        max_level: ValidationLevel::Integrity,
        check_canonical: false,
        checksum_only: true,
    };
    let _ = validate_buffer(data, &checksum);
});
