// Fuzz the full decode path: arbitrary bytes -> framing -> metadata ->
// per-object descriptor -> codec decode.  This is the primary
// untrusted-input boundary (a `.tgm` downloaded from a remote server).
//
// Security invariant: `decode` must NEVER panic, abort, hang, read out
// of bounds, or otherwise exhibit UB on ANY input.  It must return
// `Ok` or a structured `Err`.  libFuzzer + ASan enforce this.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensogram::{DecodeOptions, decode};

fuzz_target!(|data: &[u8]| {
    // Default options exercise the common path (native byte order,
    // restore_non_finite, no hash verification).
    let _ = decode(data, &DecodeOptions::default());

    // Also exercise the hash-verifying path: a different set of
    // branches (the verify pre-pass) walks frame headers and
    // recomputes digests over attacker-controlled body bytes.
    let verify = DecodeOptions {
        verify_hash: true,
        ..DecodeOptions::default()
    };
    let _ = decode(data, &verify);

    // And the raw-wire-order path (skips the native byteswap branch).
    let raw = DecodeOptions {
        native_byte_order: false,
        ..DecodeOptions::default()
    };
    let _ = decode(data, &raw);
});
