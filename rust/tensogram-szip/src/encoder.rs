// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! AEC encoder — CCSDS 121.0-B-3 adaptive entropy encoding.
//!
//! Encodes sample blocks using the optimal coding option per block:
//! zero-block, second extension, split-sample (k=0..kmax), or
//! no-compression. Tracks RSI block offsets for range-decode support.
//!
//! The block layout matches libaec exactly: when preprocessing is
//! enabled, `data_pp[0] = 0` (dummy) and the reference sample is
//! stored separately. Each block is `block_size` entries in `data_pp`,
//! and the encoder processes entries `[ref..block_size)` where `ref=1`
//! for the first block (skipping the dummy) and `ref=0` thereafter.

use crate::AecError;
use crate::AecParams;
use crate::bitstream::BitWriter;
use crate::params;
use crate::preprocessor;

/// Encode data using the CCSDS 121.0-B-3 adaptive entropy coder.
///
/// Returns `(compressed_bytes, rsi_block_bit_offsets)`.
pub(crate) fn encode(
    data: &[u8],
    p: &AecParams,
    track_offsets: bool,
) -> Result<(Vec<u8>, Vec<u64>), AecError> {
    let flags = params::effective_flags(p);
    let bps = p.bits_per_sample;
    let block_size = p.block_size as usize;
    let rsi_blocks = p.rsi as usize;
    let byte_width = params::sample_byte_width(bps, flags);
    let id_len = params::id_len(bps, flags);
    let kmax = params::kmax(id_len) as usize;
    let samples_per_rsi = rsi_blocks * block_size;
    let pp = flags & params::AEC_DATA_PREPROCESS != 0;
    let signed = flags & params::AEC_DATA_SIGNED != 0;

    let xmax = if signed {
        (1u64 << (bps - 1)) as u32 - 1
    } else if bps == 32 {
        u32::MAX
    } else {
        (1u32 << bps) - 1
    };

    if data.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    if !data.len().is_multiple_of(byte_width) {
        return Err(AecError::Data(format!(
            "data length {} is not a multiple of sample byte width {}",
            data.len(),
            byte_width
        )));
    }

    let samples = read_samples(data, bps, byte_width, flags)?;
    let num_samples = samples.len();

    let mut writer = BitWriter::new();
    let mut offsets: Vec<u64> = Vec::new();
    let mut sample_idx = 0;

    while sample_idx < num_samples {
        if track_offsets {
            offsets.push(writer.bit_position());
        }

        let rsi_end = (sample_idx + samples_per_rsi).min(num_samples);
        let rsi_samples = &samples[sample_idx..rsi_end];

        // Pad to full RSI if needed (repeat last sample)
        let mut padded;
        let rsi_data = if rsi_samples.len() < samples_per_rsi {
            padded = rsi_samples.to_vec();
            let last = *padded.last().unwrap_or(&0);
            padded.resize(samples_per_rsi, last);
            &padded
        } else {
            rsi_samples
        };

        let actual_blocks = if rsi_samples.len() < samples_per_rsi {
            rsi_samples.len().div_ceil(block_size)
        } else {
            rsi_blocks
        };

        // Build data_pp array matching libaec layout:
        // - If pp: data_pp[0] = 0 (dummy), data_pp[1..] = mapped deltas
        //          ref_sample stored separately
        // - If !pp: data_pp = raw samples (no transformation)
        let (ref_sample, data_pp) = if pp {
            let (reference, mapped) = if signed {
                preprocessor::preprocess_signed(rsi_data, bps, xmax)
            } else {
                preprocessor::preprocess_unsigned(rsi_data, xmax)
            };
            // Build data_pp: [0, mapped[0], mapped[1], ...]
            let mut dp = Vec::with_capacity(samples_per_rsi);
            dp.push(0u32); // dummy at index 0
            dp.extend_from_slice(&mapped);
            // Pad if needed
            while dp.len() < samples_per_rsi {
                dp.push(0);
            }
            (reference, dp)
        } else {
            (0, rsi_data.to_vec())
        };

        // Encode blocks — matching libaec's m_get_block / m_check_zero_block flow
        let mut k_hint: usize = 0;
        let mut zero_blocks: usize = 0;
        let mut zero_ref: bool = false;
        let mut zero_ref_sample: u32 = 0;
        let mut uncomp_len: u32 = if pp {
            (block_size as u32 - 1) * bps // first block with ref
        } else {
            block_size as u32 * bps
        };

        for block_idx in 0..actual_blocks {
            let is_first_block = block_idx == 0;
            let has_ref = pp && is_first_block;
            let ref_flag: usize = if has_ref { 1 } else { 0 };

            // Block slice from data_pp — always block_size entries
            let block_start = block_idx * block_size;
            let block = &data_pp[block_start..block_start + block_size];

            // The encoded portion is block[ref_flag..block_size]
            let encoded_part = &block[ref_flag..];

            // Check if all entries in the FULL block are zero
            // (libaec checks all block_size entries including dummy)
            let all_zero = block.iter().all(|&v| v == 0);

            if all_zero {
                zero_blocks += 1;
                if zero_blocks == 1 {
                    zero_ref = has_ref;
                    zero_ref_sample = ref_sample;
                }
                // Flush conditions: end of RSI, or 64-block segment boundary
                let at_rsi_end = block_idx + 1 >= actual_blocks;
                let at_segment_boundary = (block_idx + 1) % 64 == 0;
                if at_rsi_end || at_segment_boundary {
                    let is_ros = at_rsi_end && zero_blocks > 4;
                    encode_zero_block(
                        &mut writer,
                        id_len,
                        bps,
                        zero_blocks,
                        zero_ref,
                        zero_ref_sample,
                        is_ros,
                    );
                    zero_blocks = 0;
                }
            } else {
                // Non-zero block — flush pending zero blocks first
                if zero_blocks > 0 {
                    encode_zero_block(
                        &mut writer,
                        id_len,
                        bps,
                        zero_blocks,
                        zero_ref,
                        zero_ref_sample,
                        false,
                    );
                    zero_blocks = 0;
                }

                // Assess coding options on the ENCODED part (excluding ref)
                let (best_k, split_len) = if id_len > 1 {
                    assess_splitting(encoded_part, kmax, k_hint)
                } else {
                    (0, u64::MAX)
                };

                let se_len = if encoded_part.len() % 2 == 0 {
                    assess_second_extension(encoded_part, uncomp_len as u64)
                } else {
                    u64::MAX
                };

                if split_len < uncomp_len as u64 {
                    if split_len <= se_len {
                        encode_split(
                            &mut writer,
                            id_len,
                            bps,
                            best_k,
                            encoded_part,
                            has_ref,
                            ref_sample,
                        );
                        k_hint = best_k;
                    } else {
                        encode_se(&mut writer, id_len, bps, encoded_part, has_ref, ref_sample);
                    }
                } else if (uncomp_len as u64) <= se_len {
                    encode_uncomp(&mut writer, id_len, bps, block, has_ref, ref_sample);
                } else {
                    encode_se(&mut writer, id_len, bps, encoded_part, has_ref, ref_sample);
                }
            }

            // After the first block, switch to full-block uncomp_len
            if has_ref {
                uncomp_len = block_size as u32 * bps;
            }
        }

        // Flush any trailing zero blocks
        if zero_blocks > 0 {
            let is_ros = zero_blocks > 4;
            encode_zero_block(
                &mut writer,
                id_len,
                bps,
                zero_blocks,
                zero_ref,
                zero_ref_sample,
                is_ros,
            );
        }

        sample_idx = rsi_end;
    }

    writer.pad_to_byte();
    Ok((writer.finish(), offsets))
}

// ── Coding option encoders ───────────────────────────────────────────────────

fn encode_zero_block(
    w: &mut BitWriter,
    id_len: u32,
    bps: u32,
    count: usize,
    has_ref: bool,
    ref_sample: u32,
    is_ros: bool,
) {
    // Low-entropy ID + zero-block bit: id_len+1 bits, all zero
    w.emit(0, id_len + 1);

    if has_ref {
        w.emit(ref_sample, bps);
    }

    // FS encoding for block count (matching libaec m_encode_zero exactly)
    let fs_val = if is_ros {
        4u32 // ROS marker
    } else if count >= 5 {
        count as u32 // encode as-is
    } else {
        count as u32 - 1 // 1→0, 2→1, 3→2, 4→3
    };
    w.emit_fs(fs_val);
}

fn encode_split(
    w: &mut BitWriter,
    id_len: u32,
    bps: u32,
    k: usize,
    encoded_part: &[u32],
    has_ref: bool,
    ref_sample: u32,
) {
    // Split ID = k + 1 (matching libaec: emit(state, k + 1, state->id_len))
    w.emit(k as u32 + 1, id_len);

    if has_ref {
        w.emit(ref_sample, bps);
    }

    // FS part: for each sample, emit quotient (value >> k) as fundamental sequence
    for &val in encoded_part {
        w.emit_fs(val >> k);
    }

    // Binary part: for each sample, emit k LSBs
    if k > 0 {
        let mask = (1u32 << k) - 1;
        for &val in encoded_part {
            w.emit(val & mask, k as u32);
        }
    }
}

fn encode_se(
    w: &mut BitWriter,
    id_len: u32,
    bps: u32,
    encoded_part: &[u32],
    has_ref: bool,
    ref_sample: u32,
) {
    // SE ID: id_len zero bits + 1 one bit (matching libaec: emit(state, 1, state->id_len + 1))
    w.emit(1, id_len + 1);

    if has_ref {
        w.emit(ref_sample, bps);
    }

    for i in (0..encoded_part.len()).step_by(2) {
        let a = encoded_part[i] as u64;
        let b = if i + 1 < encoded_part.len() {
            encoded_part[i + 1] as u64
        } else {
            0
        };
        let d = a + b;
        // Use u128 to prevent overflow when d is large (32-bit samples
        // can produce d ≈ 2×u32::MAX, and d*(d+1) overflows u64).
        let fs_val = ((d as u128) * (d as u128 + 1) / 2 + b as u128) as u64;
        w.emit_fs(fs_val as u32);
    }
}

fn encode_uncomp(
    w: &mut BitWriter,
    id_len: u32,
    bps: u32,
    block: &[u32], // full block_size entries
    has_ref: bool,
    ref_sample: u32,
) {
    // Uncomp ID = all-ones (matching libaec: emit(state, (1U << state->id_len) - 1, state->id_len))
    w.emit((1u32 << id_len) - 1, id_len);

    // For uncomp, emit ALL block_size samples with ref substituted at position 0
    // (matching libaec: state->block[0] = state->ref_sample; emitblock(strm, bps, 0))
    if has_ref {
        w.emit(ref_sample, bps); // position 0 = ref instead of dummy
        for &val in &block[1..] {
            w.emit(val, bps);
        }
    } else {
        for &val in block {
            w.emit(val, bps);
        }
    }
}

// ── Coding option assessment ─────────────────────────────────────────────────

/// Find the optimal split parameter `k` for a block using a hill-climbing
/// search seeded by `k_hint` (the best `k` from the previous block).
///
/// The cost of split-sample coding with parameter `k` is:
///   `Σ(v >> k) + block_size * (k + 1)` bits
/// where the first term is the FS (quotient) cost and the second is the
/// binary (remainder) cost.  The function searches in both directions from
/// `k_hint`, stopping as soon as the cost starts rising.
///
/// Returns `(best_k, best_len_in_bits)`.
fn assess_splitting(block: &[u32], kmax: usize, k_hint: usize) -> (usize, u64) {
    let bs = block.len() as u64;
    let mut best_k = 0;
    let mut best_len = u64::MAX;

    let mut k = k_hint.min(kmax);
    let mut no_turn = k == 0;
    let mut dir = true;

    loop {
        let fs_sum: u64 = block.iter().map(|&v| (v >> k) as u64).sum();
        let len = fs_sum + bs * (k as u64 + 1);

        if len < best_len {
            if best_len < u64::MAX {
                no_turn = true;
            }
            best_len = len;
            best_k = k;

            if dir {
                if fs_sum < bs || k >= kmax {
                    if no_turn {
                        break;
                    }
                    k = if k_hint > 0 { k_hint - 1 } else { break };
                    dir = false;
                    no_turn = true;
                } else {
                    k += 1;
                }
            } else {
                if fs_sum >= bs || k == 0 {
                    break;
                }
                k -= 1;
            }
        } else {
            if no_turn {
                break;
            }
            k = if k_hint > 0 { k_hint - 1 } else { break };
            dir = false;
            no_turn = true;
        }
    }

    (best_k, best_len)
}

fn assess_second_extension(block: &[u32], uncomp_len: u64) -> u64 {
    let mut len: u64 = 1;

    for i in (0..block.len()).step_by(2) {
        let a = block[i] as u64;
        let b = if i + 1 < block.len() {
            block[i + 1] as u64
        } else {
            0
        };
        let d = a + b;
        // Use u128 to prevent overflow when d is large (32-bit samples).
        let se_bits = (d as u128) * (d as u128 + 1) / 2 + b as u128 + 1;
        // If the SE length overflows u64 or exceeds uncomp_len, bail out.
        if se_bits > uncomp_len as u128 {
            return u64::MAX;
        }
        len = len.saturating_add(se_bits as u64);
        if len > uncomp_len {
            return u64::MAX;
        }
    }

    len
}

// ── Sample I/O ───────────────────────────────────────────────────────────────

fn read_samples(
    data: &[u8],
    bps: u32,
    byte_width: usize,
    flags: u32,
) -> Result<Vec<u32>, AecError> {
    let msb = flags & params::AEC_DATA_MSB != 0;
    let num = data.len() / byte_width;
    let mut samples = Vec::with_capacity(num);

    for i in 0..num {
        let offset = i * byte_width;
        let raw = match byte_width {
            1 => data[offset] as u32,
            2 => {
                if msb {
                    u16::from_be_bytes([data[offset], data[offset + 1]]) as u32
                } else {
                    u16::from_le_bytes([data[offset], data[offset + 1]]) as u32
                }
            }
            3 => {
                if msb {
                    ((data[offset] as u32) << 16)
                        | ((data[offset + 1] as u32) << 8)
                        | (data[offset + 2] as u32)
                } else {
                    (data[offset] as u32)
                        | ((data[offset + 1] as u32) << 8)
                        | ((data[offset + 2] as u32) << 16)
                }
            }
            4 => {
                if msb {
                    u32::from_be_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ])
                } else {
                    u32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ])
                }
            }
            // byte_width is computed by sample_byte_width() and is always 1-4,
            // but we return an error instead of panicking for safety.
            w => return Err(AecError::Config(format!("unexpected byte width {w}"))),
        };
        let masked = if bps < 32 {
            raw & ((1u32 << bps) - 1)
        } else {
            raw
        };
        samples.push(masked);
    }

    Ok(samples)
}
