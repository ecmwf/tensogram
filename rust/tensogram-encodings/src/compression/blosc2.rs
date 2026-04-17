// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::sync::Once;

use blosc2::chunk::SChunk;
use blosc2::{CParams, CompressAlgo, DParams};

use super::{CompressResult, CompressionError, Compressor};
use crate::pipeline::Blosc2Codec;

/// Ensure the blosc2 C library is initialized.
///
/// Workaround: the `blosc2` crate (v0.2.2) calls `blosc2_init()` inside
/// `SChunk::new()` but not `SChunk::from_buffer()`. Decode-only processes
/// that never compress will hit an uninitialized library and fail.
fn ensure_blosc2_init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: blosc2_init() has no preconditions and is safe to call multiple times.
        unsafe { blosc2_sys::blosc2_init() };
    });
}

fn map_err(e: blosc2::Error) -> CompressionError {
    CompressionError::Blosc2(format!("{e:?}"))
}

fn codec_to_algo(codec: &Blosc2Codec) -> CompressAlgo {
    match codec {
        Blosc2Codec::Blosclz => CompressAlgo::Blosclz,
        Blosc2Codec::Lz4 => CompressAlgo::Lz4,
        Blosc2Codec::Lz4hc => CompressAlgo::Lz4hc,
        Blosc2Codec::Zlib => CompressAlgo::Zlib,
        Blosc2Codec::Zstd => CompressAlgo::Zstd,
    }
}

pub struct Blosc2Compressor {
    pub codec: Blosc2Codec,
    pub clevel: i32,
    pub typesize: usize,
    /// Number of threads blosc2 may use internally (forwarded to
    /// `CParams::nthreads` / `DParams::nthreads`).  `0` means
    /// "sequential" — blosc2 is told to use exactly 1 thread, giving
    /// byte-identical output to previous releases.  Values `>= 1` are
    /// passed through verbatim.
    pub nthreads: u32,
}

/// Normalise the tensogram `nthreads` semantics (0 == sequential) to
/// blosc2's which has `.nthreads(x).max(1)` as its internal clamp but
/// where an explicit 1 is the canonical single-threaded setting.
#[inline]
fn blosc2_nthreads(n: u32) -> usize {
    if n == 0 { 1 } else { n as usize }
}

impl Blosc2Compressor {
    fn build_cparams(&self) -> Result<CParams, CompressionError> {
        let algo = codec_to_algo(&self.codec);
        let mut cparams = CParams::default();
        cparams
            .compressor(algo)
            .clevel(self.clevel as u32)
            .typesize(self.typesize)
            .map_err(map_err)?;
        cparams.nthreads(blosc2_nthreads(self.nthreads));
        Ok(cparams)
    }

    fn build_dparams(&self) -> DParams {
        let mut dparams = DParams::default();
        dparams.nthreads(blosc2_nthreads(self.nthreads));
        dparams
    }
}

impl Compressor for Blosc2Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        ensure_blosc2_init();
        let cparams = self.build_cparams()?;
        let dparams = self.build_dparams();

        let mut schunk = SChunk::new(cparams, dparams).map_err(map_err)?;
        schunk.append(data).map_err(map_err)?;

        let buf = schunk.to_buffer().map_err(map_err)?;
        Ok(CompressResult {
            data: buf.as_slice().to_vec(),
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        ensure_blosc2_init();
        // NOTE: blosc2's safe `SChunk::from_buffer` reads dparams from
        // the buffer itself; there is no cheap runtime override through
        // the high-level API (compare against `Chunk::set_dparams`, which
        // works at the single-chunk level).  We therefore run decompress
        // sequentially and rely on the compress path for axis-B wins.
        // Compress is the expensive direction; blosc2 decompress is
        // largely memory-bound anyway.
        let schunk = SChunk::from_buffer(data.into()).map_err(map_err)?;
        let num_items = schunk.items_num();
        if num_items == 0 {
            return Ok(Vec::new());
        }
        schunk.items(0..num_items).map_err(map_err)
    }

    fn decompress_range(
        &self,
        data: &[u8],
        _block_offsets: &[u64],
        byte_pos: usize,
        byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        ensure_blosc2_init();
        // See note in `decompress()` — decompress path is single-threaded.
        let schunk = SChunk::from_buffer(data.into()).map_err(map_err)?;
        let ts = schunk.typesize();
        if ts == 0 {
            return Err(CompressionError::Blosc2("typesize is 0".to_string()));
        }

        // Convert byte range to item range
        let item_start = byte_pos / ts;
        let item_end = (byte_pos + byte_size).div_ceil(ts);

        let items = schunk.items(item_start..item_end).map_err(map_err)?;

        // Trim to exact byte range within the item-aligned result
        let offset_in_items = byte_pos % ts;
        let end = offset_in_items
            .checked_add(byte_size)
            .ok_or_else(|| CompressionError::Blosc2("range overflow".to_string()))?;
        if end > items.len() {
            return Err(CompressionError::Blosc2(format!(
                "range exceeds decompressed data: need {end} bytes, got {}",
                items.len()
            )));
        }
        Ok(items[offset_in_items..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blosc2_round_trip() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
            nthreads: 0,
        };

        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn blosc2_round_trip_4byte() {
        let data: Vec<u8> = (0..4000).flat_map(|i: u32| i.to_ne_bytes()).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Blosclz,
            clevel: 5,
            typesize: 4,
            nthreads: 0,
        };

        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn blosc2_range_decode() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
            nthreads: 0,
        };

        let result = compressor.compress(&data).unwrap();
        let partial = compressor
            .decompress_range(&result.data, &[], 200, 500)
            .unwrap();
        assert_eq!(partial.len(), 500);
        assert_eq!(&partial[..], &data[200..700]);
    }

    #[test]
    fn blosc2_round_trip_zstd() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Zstd,
            clevel: 3,
            typesize: 1,
            nthreads: 0,
        };
        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn blosc2_round_trip_zlib() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Zlib,
            clevel: 5,
            typesize: 1,
            nthreads: 0,
        };
        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    /// nthreads > 1 must produce losslessly-decodable output that
    /// round-trips to the original bytes.
    ///
    /// Determinism contract for opaque codecs (blosc2, zstd with
    /// `nbWorkers > 0`): compressed bytes MAY differ between parallel
    /// and sequential runs because the codec reorders blocks in the
    /// offset table by worker completion order.  What MUST hold is
    /// that every parallel variant round-trips losslessly and that
    /// cross-thread-count decode works (encode with nthreads=N,
    /// decompress with any M).
    ///
    /// This is the honest contract — callers wanting byte-identical
    /// output across thread counts should use `threads = 0`.  See
    /// docs/src/guide/multi-threaded-pipeline.md for the full policy.
    #[test]
    fn blosc2_nthreads_round_trip_lossless() {
        let data: Vec<u8> = (0..256 * 1024).map(|i| ((i * 31) % 256) as u8).collect();

        let seq_compressor = Blosc2Compressor {
            codec: Blosc2Codec::Zstd,
            clevel: 3,
            typesize: 4,
            nthreads: 0,
        };
        let seq_bytes = seq_compressor.compress(&data).unwrap().data;
        let seq_rt = seq_compressor.decompress(&seq_bytes, data.len()).unwrap();
        assert_eq!(seq_rt, data);

        for n in [1u32, 2, 4, 8] {
            let par_compressor = Blosc2Compressor {
                codec: Blosc2Codec::Zstd,
                clevel: 3,
                typesize: 4,
                nthreads: n,
            };
            let par_bytes = par_compressor.compress(&data).unwrap().data;
            // Decoded values must always round-trip exactly.
            let par_rt = par_compressor.decompress(&par_bytes, data.len()).unwrap();
            assert_eq!(
                par_rt, data,
                "blosc2 nthreads={n} round-trip must match original"
            );
            // Cross-thread-count decode: encode with N threads, decode
            // with 0 threads (single-threaded) — must still work.
            let cross_rt = seq_compressor.decompress(&par_bytes, data.len()).unwrap();
            assert_eq!(cross_rt, data);
        }
    }

    /// At `nthreads=0` (default), blosc2 compress output is stable
    /// across runs and matches the pre-0.13.0 byte layout.  Together
    /// with the cross-language golden tests, this locks in the
    /// "sequential is byte-identical" guarantee.
    #[test]
    fn blosc2_nthreads_zero_is_deterministic_across_runs() {
        let data: Vec<u8> = (0..64 * 1024).map(|i| ((i * 17) % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Zstd,
            clevel: 3,
            typesize: 4,
            nthreads: 0,
        };
        let a = compressor.compress(&data).unwrap().data;
        let b = compressor.compress(&data).unwrap().data;
        assert_eq!(a, b, "blosc2 nthreads=0 must be deterministic");
    }

    /// Regression guard for the decompress path.
    ///
    /// The real bug (missing blosc2_init in decode-only processes) cannot be
    /// fully reproduced in a unit test because blosc2_init is global state —
    /// once called by compress(), it stays initialized. This test still guards
    /// against regressions in the decompress wiring itself.
    #[test]
    fn blosc2_decompress_path() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
            nthreads: 0,
        };
        let compressed = compressor.compress(&data).unwrap().data;

        let decoder = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 0,
            typesize: 1,
            nthreads: 0,
        };
        let decompressed = decoder.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }
}
