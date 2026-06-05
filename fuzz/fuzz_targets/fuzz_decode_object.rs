// Fuzz single-object decode by index.  The first byte selects an
// object index; the remainder is the message buffer.  Exercises the
// per-object fast path and the hash verify pre-pass with attacker
// control over both the index and the bytes.
//
// Security invariant: never panic / hang / UB on any input.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensogram::{decode_object, DecodeOptions};

fuzz_target!(|data: &[u8]| {
    // Derive an attacker-controlled object index from the first byte
    // (covers in-range, boundary, and wildly-out-of-range indices),
    // then feed the rest as the message.
    let (index, buf): (usize, &[u8]) = match data.split_first() {
        Some((&b, rest)) => (b as usize, rest),
        None => (0, data),
    };

    let _ = decode_object(buf, index, &DecodeOptions::default());

    let verify = DecodeOptions {
        verify_hash: true,
        ..DecodeOptions::default()
    };
    let _ = decode_object(buf, index, &verify);
});
