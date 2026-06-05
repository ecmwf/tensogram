// Fuzz the codec decode path directly: arbitrary "compressed" bytes +
// an arbitrary descriptor fed straight into `decode_pipeline`, for each
// compression codec.  This reaches the decompressors (szip / zstd /
// lz4 / blosc2 / zfp / sz3 / rle / roaring) with attacker-controlled
// payloads and a hostile `num_values` (decompression-bomb / output-
// size-overflow surface) far more directly than going through full
// framing.
//
// Security invariant: never panic / hang / UB / OOM-abort on any input.
// A malformed payload or an impossible descriptor must return a
// structured `Err` (or decode), bounded by the fallible-allocation
// guards.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensogram_encodings::pipeline::{
    self, ByteOrder, CompressionType, EncodingType, FilterType, PipelineConfig,
};

fuzz_target!(|data: &[u8]| {
    // First two bytes pick the codec and a (bounded) num_values
    // exponent; the rest is the "compressed" payload.
    let (codec_sel, num_exp, payload): (u8, u8, &[u8]) = match data {
        [a, b, rest @ ..] => (*a, *b, rest),
        _ => return,
    };

    // num_values is attacker-controlled but we cap the *test* loop's
    // exponent so the harness itself doesn't legitimately try to alloc
    // exabytes for every iteration (the library's own guards are what
    // we're testing; we just want a spread of small..huge claims).
    let num_values: usize = 1usize << (num_exp % 40); // up to ~1 TiB claim

    let compression = match codec_sel % 8 {
        0 => CompressionType::None,
        1 => CompressionType::Lz4,
        2 => CompressionType::Zstd { level: 3 },
        3 => CompressionType::Szip {
            rsi: 128,
            block_size: 16,
            flags: 0,
            bits_per_sample: 32,
        },
        4 => CompressionType::Blosc2 {
            codec: pipeline::Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 4,
        },
        5 => CompressionType::Zfp {
            mode: pipeline::ZfpMode::FixedRate { rate: 16.0 },
        },
        6 => CompressionType::Sz3 {
            error_bound: pipeline::Sz3ErrorBound::Absolute(1e-3),
        },
        _ => CompressionType::None,
    };

    let config = PipelineConfig {
        encoding: EncodingType::None,
        filter: FilterType::None,
        compression,
        num_values,
        byte_order: ByteOrder::Little,
        dtype_byte_width: 4,
        swap_unit_size: 4,
        compression_backend: pipeline::CompressionBackend::default(),
        intra_codec_threads: 0,
        compute_hash: false,
    };

    // native_byte_order=true exercises the byteswap branch too.
    let _ = pipeline::decode_pipeline(payload, &config, true);
});
