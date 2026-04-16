// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ShuffleError {
    #[error("element_size must not be zero")]
    InvalidElementSize,
    #[error("data length {data_len} is not divisible by element_size {element_size}")]
    Misaligned {
        data_len: usize,
        element_size: usize,
    },
}

/// Minimum input length below which the parallel shuffle skips the
/// rayon split.  Below ~64 KiB the parallel split overhead dominates.
#[cfg(feature = "threads")]
const PARALLEL_SHUFFLE_MIN_BYTES: usize = 64 * 1024;

/// Byte-level shuffle: groups `byte[0]` of all elements, then `byte[1]`, etc.
/// Input length must be divisible by element_size.
/// Returns `Err(ShuffleError::InvalidElementSize)` if element_size is 0.
/// Returns `Err(ShuffleError::Misaligned)` if data.len() % element_size != 0.
pub fn shuffle(data: &[u8], element_size: usize) -> Result<Vec<u8>, ShuffleError> {
    shuffle_with_threads(data, element_size, 0)
}

/// Thread-aware shuffle.
///
/// `threads == 0` preserves the pre-0.13.0 sequential path.
/// `threads > 0` parallelises over the `element_size` outer loop
/// (one worker per byte-plane).  Output bytes are byte-identical to
/// the sequential path.
pub fn shuffle_with_threads(
    data: &[u8],
    element_size: usize,
    #[allow(unused_variables)] threads: u32,
) -> Result<Vec<u8>, ShuffleError> {
    if element_size == 0 {
        return Err(ShuffleError::InvalidElementSize);
    }
    if element_size == 1 || data.is_empty() {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(ShuffleError::Misaligned {
            data_len: data.len(),
            element_size,
        });
    }
    let num_elements = data.len() / element_size;

    #[cfg(feature = "threads")]
    {
        if threads >= 2 && element_size >= 2 && data.len() >= PARALLEL_SHUFFLE_MIN_BYTES {
            use rayon::prelude::*;
            let mut output = vec![0u8; data.len()];
            output
                .par_chunks_exact_mut(num_elements)
                .enumerate()
                .for_each(|(byte_idx, plane)| {
                    for elem in 0..num_elements {
                        plane[elem] = data[elem * element_size + byte_idx];
                    }
                });
            return Ok(output);
        }
    }

    let mut output = vec![0u8; data.len()];
    for byte_idx in 0..element_size {
        for elem in 0..num_elements {
            output[byte_idx * num_elements + elem] = data[elem * element_size + byte_idx];
        }
    }
    Ok(output)
}

/// Reverse shuffle.
/// Returns `Err(ShuffleError::InvalidElementSize)` if element_size is 0.
/// Returns `Err(ShuffleError::Misaligned)` if data.len() % element_size != 0.
pub fn unshuffle(data: &[u8], element_size: usize) -> Result<Vec<u8>, ShuffleError> {
    unshuffle_with_threads(data, element_size, 0)
}

/// Thread-aware reverse shuffle.
///
/// `threads == 0` preserves the pre-0.13.0 sequential path.
/// `threads > 0` parallelises the scatter over input byte-planes
/// using a `chunks_mut` strided write pattern.  Output bytes are
/// byte-identical to the sequential path.
pub fn unshuffle_with_threads(
    data: &[u8],
    element_size: usize,
    #[allow(unused_variables)] threads: u32,
) -> Result<Vec<u8>, ShuffleError> {
    if element_size == 0 {
        return Err(ShuffleError::InvalidElementSize);
    }
    if element_size == 1 || data.is_empty() {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(ShuffleError::Misaligned {
            data_len: data.len(),
            element_size,
        });
    }
    let num_elements = data.len() / element_size;

    #[cfg(feature = "threads")]
    {
        // unshuffle is harder to parallelise cleanly — the output is
        // written with stride `element_size`, meaning each element's
        // bytes are scattered across `element_size` different input
        // planes.  We parallelise on CHUNKS of output elements so that
        // each chunk can read its `element_size` stripes independently.
        if threads >= 2 && element_size >= 2 && data.len() >= PARALLEL_SHUFFLE_MIN_BYTES {
            use rayon::prelude::*;
            let mut output = vec![0u8; data.len()];
            // Split output into element-aligned chunks.  ~4 KiB per
            // chunk amortises rayon split cost; minimum 64 elements so
            // tiny element_size values still get decent chunks.
            //
            // The chunk size is always a multiple of `element_size` and
            // `data.len()` is a multiple of `element_size` (validated
            // above), so within every chunk `chunks_exact_mut` visits
            // every element with no tail.
            let chunk_elems = (4096 / element_size).max(64);
            output
                .par_chunks_mut(chunk_elems * element_size)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let elem_start = chunk_idx * chunk_elems;
                    for (local_elem, dst) in out_chunk.chunks_exact_mut(element_size).enumerate() {
                        let elem = elem_start + local_elem;
                        for byte_idx in 0..element_size {
                            dst[byte_idx] = data[byte_idx * num_elements + elem];
                        }
                    }
                });
            return Ok(output);
        }
    }

    // Sequential fallback — inverse of shuffle: scatter each source
    // byte (from plane `byte_idx`, position `elem`) into the
    // `byte_idx`-th byte of output element `elem`.
    let mut output = vec![0u8; data.len()];
    for byte_idx in 0..element_size {
        for elem in 0..num_elements {
            output[elem * element_size + byte_idx] = data[byte_idx * num_elements + elem];
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_unshuffle_float32() {
        // 3 float32 elements = 12 bytes
        let data: Vec<u8> = (0..12).collect();
        let shuffled = shuffle(&data, 4).unwrap();
        // byte[0] of each element: [0, 4, 8], byte[1]: [1, 5, 9], etc.
        assert_eq!(shuffled, vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
        let unshuffled = unshuffle(&shuffled, 4).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn test_shuffle_unshuffle_float64() {
        let data: Vec<u8> = (0..16).collect(); // 2 float64 elements
        let shuffled = shuffle(&data, 8).unwrap();
        let unshuffled = unshuffle(&shuffled, 8).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn test_shuffle_element_size_1() {
        let data = vec![1, 2, 3, 4];
        assert_eq!(shuffle(&data, 1).unwrap(), data);
        assert_eq!(unshuffle(&data, 1).unwrap(), data);
    }

    #[test]
    fn test_shuffle_empty() {
        let data: Vec<u8> = vec![];
        assert_eq!(shuffle(&data, 4).unwrap(), data);
    }

    #[test]
    fn test_shuffle_element_size_zero() {
        assert!(
            matches!(
                shuffle(&[1, 2, 3], 0),
                Err(ShuffleError::InvalidElementSize)
            ),
            "shuffle with element_size=0 must return Err(InvalidElementSize)"
        );
        assert!(
            matches!(
                unshuffle(&[1, 2, 3], 0),
                Err(ShuffleError::InvalidElementSize)
            ),
            "unshuffle with element_size=0 must return Err(InvalidElementSize)"
        );
    }

    // ── Threading determinism ─────────────────────────────────────────

    #[cfg(feature = "threads")]
    #[test]
    fn shuffle_threads_byte_identical() {
        let data: Vec<u8> = (0..256 * 1024).map(|i| (i % 256) as u8).collect();
        for element_size in [2usize, 4, 8, 16] {
            let seq = shuffle_with_threads(&data, element_size, 0).unwrap();
            for t in [1u32, 2, 4, 8] {
                let par = shuffle_with_threads(&data, element_size, t).unwrap();
                assert_eq!(seq, par, "shuffle threads={t} elem={element_size} mismatch");
                let rt = unshuffle_with_threads(&par, element_size, t).unwrap();
                assert_eq!(
                    rt, data,
                    "round-trip mismatch threads={t} elem={element_size}"
                );
            }
        }
    }

    #[cfg(feature = "threads")]
    #[test]
    fn unshuffle_threads_byte_identical() {
        let data: Vec<u8> = (0..256 * 1024).map(|i| (i % 256) as u8).collect();
        for element_size in [2usize, 4, 8, 16] {
            let shuffled = shuffle(&data, element_size).unwrap();
            let seq = unshuffle_with_threads(&shuffled, element_size, 0).unwrap();
            for t in [1u32, 2, 4, 8] {
                let par = unshuffle_with_threads(&shuffled, element_size, t).unwrap();
                assert_eq!(
                    seq, par,
                    "unshuffle threads={t} elem={element_size} mismatch"
                );
            }
        }
    }

    #[cfg(feature = "threads")]
    #[test]
    fn shuffle_below_threshold_uses_sequential_path() {
        // Tiny inputs should fall back to sequential regardless of the
        // threads argument.
        let data: Vec<u8> = (0..64).collect();
        let seq = shuffle(&data, 4).unwrap();
        let par = shuffle_with_threads(&data, 4, 8).unwrap();
        assert_eq!(seq, par);
    }

    #[test]
    fn test_shuffle_misaligned_data() {
        let result = shuffle(&[1, 2, 3], 2);
        assert!(
            matches!(
                result,
                Err(ShuffleError::Misaligned {
                    data_len: 3,
                    element_size: 2
                })
            ),
            "shuffle with misaligned data must return Err(Misaligned)"
        );
        let result2 = unshuffle(&[1, 2, 3], 2);
        assert!(
            matches!(
                result2,
                Err(ShuffleError::Misaligned {
                    data_len: 3,
                    element_size: 2
                })
            ),
            "unshuffle with misaligned data must return Err(Misaligned)"
        );
    }
}
