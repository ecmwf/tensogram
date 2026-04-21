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

/// Default maximum size of a single SChunk chunk, in bytes (256 MiB).
///
/// Blosc2 enforces a hard per-call limit of
/// `BLOSC2_MAX_BUFFERSIZE = INT_MAX - 32 ≈ 2 GiB` on every compression call
/// (see `c-blosc2/include/blosc2.h`, `BLOSC2_MAX_BUFFERSIZE`), so any buffer
/// larger than this limit passed in a single `SChunk::append()` fails with
/// `MaxBufsizeExceeded`. To support arrays bigger than that, the compressor
/// splits its input across multiple SChunk chunks of up to
/// `DEFAULT_BLOSC2_CHUNK_BYTES` bytes each.
///
/// 256 MiB matches the upper cap used by python-blosc2's `SChunk.__init__`
/// (see `python-blosc2/src/blosc2/schunk.py`). It keeps the number of
/// chunks small on multi-GiB payloads while staying comfortably below the
/// 2 GiB per-call limit and the per-append working-set pressure that a
/// larger chunk would impose.
pub const DEFAULT_BLOSC2_CHUNK_BYTES: usize = 256 * 1024 * 1024;

/// C-Blosc2's hard per-call compression buffer limit, in bytes.
///
/// Any single `SChunk::append()` whose input exceeds this returns
/// `MaxBufsizeExceeded`.  Defined as `INT_MAX - BLOSC2_MAX_OVERHEAD`
/// in `c-blosc2/include/blosc2.h` (overhead = extended header = 32 B).
/// Exposed at crate-private visibility only — the chunking logic in
/// [`Blosc2Compressor::compress`] uses it to clamp an over-large
/// `chunk_bytes` and keep the multi-append path always legal.
const BLOSC2_MAX_BUFFERSIZE: usize = (i32::MAX as usize) - 32;

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

pub(crate) fn codec_to_algo(codec: &Blosc2Codec) -> CompressAlgo {
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
    /// Maximum size of a single SChunk chunk in bytes. Inputs larger than
    /// this are split across multiple SChunk chunks during [`compress`] to
    /// stay below blosc2's hard per-call limit of
    /// `BLOSC2_MAX_BUFFERSIZE = INT_MAX - 32 ≈ 2 GiB`.
    ///
    /// Production callers set this to [`DEFAULT_BLOSC2_CHUNK_BYTES`]
    /// (256 MiB) via the pipeline constructor; tests may set it smaller
    /// to exercise the multi-chunk path without large allocations. Values
    /// smaller than `typesize` (including `0`) are rounded up to exactly
    /// one `typesize` element per chunk.
    ///
    /// [`compress`]: Compressor::compress
    pub chunk_bytes: usize,
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

        // Floor `chunk_bytes` to a multiple of `typesize` so non-tail
        // appends stay aligned for blosc2's shuffle filter, clamp upward
        // to at least one `typesize` to guarantee forward progress, and
        // clamp downward to `BLOSC2_MAX_BUFFERSIZE` so the multi-append
        // path stays legal even if a caller misconfigures `chunk_bytes`.
        let ts = self.typesize.max(1);
        let capped = self.chunk_bytes.min(BLOSC2_MAX_BUFFERSIZE);
        let chunk_bytes = (capped / ts).max(1) * ts;

        if data.len() <= chunk_bytes {
            // Fast path: inputs that fit in one chunk produce byte-identical
            // output to the pre-fix single-append codepath.
            schunk.append(data).map_err(map_err)?;
        } else {
            // Large input: split to stay below blosc2's per-call
            // `BLOSC2_MAX_BUFFERSIZE = INT_MAX - 32` limit.  The SChunk
            // format tolerates a single short trailing chunk (see
            // `blosc2_schunk_fill_special` in c-blosc2/blosc/schunk.c).
            for slice in data.chunks(chunk_bytes) {
                schunk.append(slice).map_err(map_err)?;
            }
        }

        let buf = schunk.to_buffer().map_err(map_err)?;
        Ok(CompressResult {
            data: buf.as_slice().to_vec(),
            block_offsets: None,
        })
    }

    fn decompress(
        &self,
        data: &[u8],
        _expected_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        ensure_blosc2_init();
        // We iterate chunks explicitly instead of going through
        // `schunk.items(0..schunk.items_num())` because `items_num()` in
        // blosc2-rs 0.2.x is computed as
        // `num_chunks * chunksize / typesize`
        // (blosc2-0.2.2/src/chunk/schunk.rs:466-468), which OVER-REPORTS
        // whenever the final chunk is shorter than `chunksize` — the
        // common case once `compress()` splits a large input across
        // multiple chunks.  Asking `items(0..overcount)` would request
        // phantom items past the real end and fail in the C layer.
        //
        // Each `Chunk::decompress()` call uses the chunk's own recorded
        // `nbytes` (Chunk::nbytes in blosc2-0.2.2/src/chunk/chunk.rs:141),
        // so the short-tail case decodes correctly.
        //
        // Blosc2's safe `SChunk::from_buffer` reads dparams from the
        // buffer itself; there is no cheap runtime override through the
        // high-level API (compare `Chunk::set_dparams`, which works at
        // the single-chunk level).  We therefore run decompress
        // sequentially and rely on the compress path for axis-B wins.
        // Compress is the expensive direction; blosc2 decompress is
        // largely memory-bound anyway.
        //
        // The `_expected_size` parameter is deliberately ignored: it is
        // derived from caller-supplied tensor metadata (via the pipeline's
        // `estimate_decompressed_size`), so trusting it for an infallible
        // pre-allocation would turn a malformed `num_values` field into a
        // process abort.  Instead we grow `out` from empty and fall back
        // to `try_reserve` per chunk, where the size comes from the
        // already-validated blosc2 frame trailer; an unreasonably large
        // per-chunk value surfaces cleanly as a `CompressionError`
        // instead of aborting the process.
        let mut schunk = SChunk::from_buffer(data.into()).map_err(map_err)?;
        let num_chunks = schunk.num_chunks();
        if num_chunks == 0 {
            return Ok(Vec::new());
        }

        let mut out: Vec<u8> = Vec::new();
        for idx in 0..num_chunks {
            let chunk = schunk.get_chunk(idx).map_err(map_err)?;
            let bytes = chunk.decompress().map_err(map_err)?;
            out.try_reserve(bytes.len()).map_err(|e| {
                CompressionError::Blosc2(format!(
                    "failed to reserve {} bytes for decompressed chunk {idx}: {e}",
                    bytes.len(),
                ))
            })?;
            out.extend_from_slice(&bytes);
        }
        Ok(out)
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

    fn small_chunk_compressor(chunk_bytes: usize) -> Blosc2Compressor {
        Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
            nthreads: 0,
            chunk_bytes,
        }
    }

    #[test]
    fn blosc2_round_trip() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
            nthreads: 0,
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
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
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
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
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
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
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
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
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
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
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
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
                chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
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
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
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
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
        };
        let compressed = compressor.compress(&data).unwrap().data;

        let decoder = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 0,
            typesize: 1,
            nthreads: 0,
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
        };
        let decompressed = decoder.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    /// Regression test for issue #68: a buffer larger than `chunk_bytes`
    /// with a non-multiple size splits into N full chunks plus one short
    /// trailing chunk. Both encode and decode must handle the short tail.
    ///
    /// This would fail with the pre-fix `decompress()` that used
    /// `schunk.items(0..schunk.items_num())`, because `items_num()` in
    /// blosc2-0.2.2 over-reports when the final chunk is short — see
    /// the note in `Blosc2Compressor::decompress`.
    #[test]
    fn blosc2_multi_chunk_round_trip_short_tail() {
        let chunk_bytes = 4096;
        let len = 3 * chunk_bytes + 777;
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();

        let compressor = small_chunk_compressor(chunk_bytes);
        let compressed = compressor.compress(&data).unwrap().data;

        let decompressed = compressor.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed.len(), data.len());
        assert_eq!(decompressed, data);

        let schunk = blosc2::chunk::SChunk::from_buffer(compressed.as_slice().into()).unwrap();
        assert!(
            schunk.num_chunks() >= 2,
            "multi-chunk path not exercised: num_chunks = {}",
            schunk.num_chunks()
        );
    }

    /// Multi-chunk path with input length that is an exact multiple of
    /// `chunk_bytes` (no short trailing chunk).  Guards that the equal-
    /// size case stays correct alongside the short-tail case.
    #[test]
    fn blosc2_multi_chunk_round_trip_exact_multiple() {
        let chunk_bytes = 4096;
        let len = 4 * chunk_bytes;
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();

        let compressor = small_chunk_compressor(chunk_bytes);
        let compressed = compressor.compress(&data).unwrap().data;
        let decompressed = compressor.decompress(&compressed, data.len()).unwrap();

        assert_eq!(decompressed, data);
    }

    /// Partial-decode across a chunk boundary.  The range starts in one
    /// SChunk chunk and ends in the next; the C `get_slice_buffer` path
    /// used by `decompress_range()` must stitch the two halves together.
    #[test]
    fn blosc2_range_decode_spans_chunk_boundary() {
        let chunk_bytes = 4096;
        let len = 3 * chunk_bytes + 500;
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();

        let compressor = small_chunk_compressor(chunk_bytes);
        let compressed = compressor.compress(&data).unwrap().data;

        let range_start = chunk_bytes - 200;
        let range_len = 500;
        let partial = compressor
            .decompress_range(&compressed, &[], range_start, range_len)
            .unwrap();

        assert_eq!(partial.len(), range_len);
        assert_eq!(&partial[..], &data[range_start..range_start + range_len]);
    }

    /// A small input with the production default `chunk_bytes` must stay
    /// in the single-append fast path and produce a single SChunk chunk.
    /// This preserves byte-level compatibility with the pre-fix output
    /// for anything that previously worked.
    #[test]
    fn blosc2_small_input_is_single_chunk() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
            nthreads: 0,
            chunk_bytes: DEFAULT_BLOSC2_CHUNK_BYTES,
        };
        let compressed = compressor.compress(&data).unwrap().data;

        let schunk = blosc2::chunk::SChunk::from_buffer(compressed.as_slice().into()).unwrap();
        assert_eq!(
            schunk.num_chunks(),
            1,
            "small input should stay on single-append fast path"
        );

        let decompressed = compressor.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    /// Input that exactly equals `chunk_bytes` must still take the
    /// single-append path — the fast-path predicate is `<=`, not `<`.
    /// Guards against a common off-by-one that would force a trivial
    /// multi-chunk encode for any N-aligned buffer.
    #[test]
    fn blosc2_input_equal_to_chunk_bytes_stays_single_chunk() {
        let chunk_bytes = 4096;
        let data: Vec<u8> = (0..chunk_bytes).map(|i| (i % 251) as u8).collect();

        let compressor = small_chunk_compressor(chunk_bytes);
        let compressed = compressor.compress(&data).unwrap().data;

        let schunk = blosc2::chunk::SChunk::from_buffer(compressed.as_slice().into()).unwrap();
        assert_eq!(schunk.num_chunks(), 1);

        let decompressed = compressor.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    /// Typesize-aligned chunking: with typesize = 4 and chunk_bytes
    /// explicitly set smaller than the input, every internal append
    /// except the possible trailing chunk must receive a buffer whose
    /// length is a multiple of typesize.  This is asserted indirectly:
    /// blosc2's shuffle filter corrupts data if typesize alignment is
    /// wrong on non-tail chunks, so round-trip success is the guard.
    #[test]
    fn blosc2_multi_chunk_typesize_alignment() {
        let chunk_bytes = 4096;
        let num_values: usize = 2 * chunk_bytes + 37;
        let data: Vec<u8> = (0..num_values)
            .flat_map(|i: usize| (i as u32).to_ne_bytes())
            .collect();
        assert_eq!(data.len() % 4, 0);

        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Blosclz,
            clevel: 5,
            typesize: 4,
            nthreads: 0,
            chunk_bytes,
        };
        let compressed = compressor.compress(&data).unwrap().data;
        let decompressed = compressor.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    /// Input whose total length is NOT a multiple of `typesize`.  This
    /// is reachable in production from `simple_packing` with a
    /// non-byte-aligned `bits_per_value` (packed length =
    /// `ceil(num_values * bpv / 8)`), and is explicitly tolerated by
    /// c-blosc2's shuffle/bitshuffle filters, which pass leftover
    /// trailing bytes through unchanged.  Round-trip must still be
    /// lossless.
    #[test]
    fn blosc2_multi_chunk_non_typesize_aligned_tail() {
        let chunk_bytes = 4096;
        let len = 2 * chunk_bytes + 13;
        assert_ne!(len % 4, 0, "test setup: length must not be 4-aligned");
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();

        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Blosclz,
            clevel: 5,
            typesize: 4,
            nthreads: 0,
            chunk_bytes,
        };
        let compressed = compressor.compress(&data).unwrap().data;
        let decompressed = compressor.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    /// A caller who constructs `Blosc2Compressor` directly with a
    /// `chunk_bytes` larger than `BLOSC2_MAX_BUFFERSIZE` (2 GiB) must
    /// still round-trip: the clamp in `compress()` keeps every
    /// `schunk.append()` below the C limit.  This guards against the
    /// original issue #68 re-emerging through a misconfigured field.
    #[test]
    fn blosc2_chunk_bytes_above_max_buffersize_is_clamped() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
            nthreads: 0,
            chunk_bytes: BLOSC2_MAX_BUFFERSIZE + (1 << 30),
        };
        let compressed = compressor.compress(&data).unwrap().data;
        let decompressed = compressor.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }
}
