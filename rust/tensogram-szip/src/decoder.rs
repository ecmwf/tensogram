// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! AEC decoder — CCSDS 121.0-B-3 adaptive entropy decoding.
//!
//! Decodes a compressed AEC bitstream back into sample values.
//! Supports full decode and partial range decode via RSI block offsets.
//!
//! The decoder matches libaec's state machine exactly:
//! 1. Read ID (id_len bits) → dispatch by ID
//! 2. For split/SE/zero: read reference AFTER ID (if first block)
//! 3. For uncomp: reference is embedded in block data (no separate read)

use crate::AecError;
use crate::AecParams;
use crate::bitstream::BitReader;
use crate::params;
use crate::preprocessor;

/// Second Extension lookup table.
struct SeTable {
    table: [(u32, u32); 91],
}

impl SeTable {
    fn new() -> Self {
        let mut table = [(0u32, 0u32); 91];
        let mut k = 0usize;
        for i in 0..13u32 {
            let ms = k as u32;
            for _j in 0..=i {
                if k < 91 {
                    table[k] = (i, ms);
                }
                k += 1;
            }
        }
        Self { table }
    }

    fn lookup(&self, fs: u32) -> Option<(u32, u32)> {
        if (fs as usize) < self.table.len() {
            Some(self.table[fs as usize])
        } else {
            None
        }
    }
}

/// Decode an entire AEC-compressed stream.
pub(crate) fn decode(
    data: &[u8],
    expected_size: usize,
    p: &AecParams,
) -> Result<Vec<u8>, AecError> {
    let flags = params::effective_flags(p);
    let bps = p.bits_per_sample;
    let block_size = p.block_size as usize;
    let rsi_blocks = p.rsi as usize;
    let byte_width = params::sample_byte_width(bps, flags);
    let id_len = params::id_len(bps, flags);
    let pp = flags & params::AEC_DATA_PREPROCESS != 0;
    let signed = flags & params::AEC_DATA_SIGNED != 0;
    let samples_per_rsi = rsi_blocks * block_size;
    let total_samples = expected_size / byte_width;

    let xmax = if signed {
        (1u64 << (bps - 1)) as u32 - 1
    } else {
        if bps == 32 {
            u32::MAX
        } else {
            (1u32 << bps) - 1
        }
    };

    if data.is_empty() {
        return Ok(Vec::new());
    }

    let se_table = SeTable::new();
    let mut reader = BitReader::new(data);
    let mut all_samples: Vec<u32> = Vec::with_capacity(total_samples);

    while all_samples.len() < total_samples {
        let remaining = total_samples - all_samples.len();
        let rsi_sample_count = remaining.min(samples_per_rsi);

        let rsi_buffer = decode_rsi(
            &mut reader,
            bps,
            block_size,
            rsi_blocks,
            id_len,
            pp,
            rsi_sample_count,
            &se_table,
        )?;

        // Post-process if preprocessing was used
        let final_samples = if pp && !rsi_buffer.is_empty() {
            let reference = rsi_buffer[0];
            let mapped = &rsi_buffer[1..];
            if signed {
                preprocessor::postprocess_signed(reference, mapped, bps, xmax)
            } else {
                preprocessor::postprocess_unsigned(reference, mapped, xmax)
            }
        } else {
            rsi_buffer
        };

        let take = rsi_sample_count.min(final_samples.len());
        all_samples.extend_from_slice(&final_samples[..take]);
    }

    write_samples(&all_samples[..total_samples], byte_width, flags)
}

/// Decode a partial range using RSI block offsets.
pub(crate) fn decode_range(
    data: &[u8],
    block_offsets: &[u64],
    byte_pos: usize,
    byte_size: usize,
    p: &AecParams,
) -> Result<Vec<u8>, AecError> {
    if byte_size == 0 {
        return Ok(Vec::new());
    }
    if data.is_empty() {
        return Err(AecError::Data(
            "cannot decompress range from empty data".into(),
        ));
    }

    let flags = params::effective_flags(p);
    let byte_width = params::sample_byte_width(p.bits_per_sample, flags);
    let rsi_size = p.rsi as usize * p.block_size as usize * byte_width;

    let rsi_n = byte_pos / rsi_size;
    if rsi_n >= block_offsets.len() {
        return Err(AecError::Data(format!(
            "RSI index {} out of range (have {} offsets)",
            rsi_n,
            block_offsets.len()
        )));
    }

    let rsi_byte_offset = rsi_n * rsi_size;
    let local_byte_pos = byte_pos - rsi_byte_offset;

    let bit_offset = block_offsets[rsi_n];
    let mut reader = BitReader::from_bit_offset(data, bit_offset);

    let bps = p.bits_per_sample;
    let block_size = p.block_size as usize;
    let rsi_blocks = p.rsi as usize;
    let id_len = params::id_len(bps, flags);
    let pp = flags & params::AEC_DATA_PREPROCESS != 0;
    let signed = flags & params::AEC_DATA_SIGNED != 0;
    let samples_per_rsi = rsi_blocks * block_size;
    let se_table = SeTable::new();

    let xmax = if signed {
        (1u64 << (bps - 1)) as u32 - 1
    } else {
        if bps == 32 {
            u32::MAX
        } else {
            (1u32 << bps) - 1
        }
    };

    let rsi_buffer = decode_rsi(
        &mut reader,
        bps,
        block_size,
        rsi_blocks,
        id_len,
        pp,
        samples_per_rsi,
        &se_table,
    )?;

    let final_samples = if pp && !rsi_buffer.is_empty() {
        let reference = rsi_buffer[0];
        let mapped = &rsi_buffer[1..];
        if signed {
            preprocessor::postprocess_signed(reference, mapped, bps, xmax)
        } else {
            preprocessor::postprocess_unsigned(reference, mapped, xmax)
        }
    } else {
        rsi_buffer
    };

    let output = write_samples(&final_samples, byte_width, flags)?;

    if local_byte_pos + byte_size > output.len() {
        return Err(AecError::Data(format!(
            "range [{}, {}) exceeds decoded RSI #{} output ({} bytes, {} samples × {} byte_width, expected {})",
            byte_pos,
            byte_pos + byte_size,
            rsi_n,
            output.len(),
            final_samples.len(),
            byte_width,
            samples_per_rsi * byte_width,
        )));
    }

    Ok(output[local_byte_pos..local_byte_pos + byte_size].to_vec())
}

// ── RSI-level decoder ────────────────────────────────────────────────────────

/// Decode one RSI of samples from the bitstream.
///
/// Returns the RSI buffer matching libaec's layout:
/// - If pp: [reference, mapped[0], mapped[1], ...] (for post-processing)
/// - If !pp: [sample[0], sample[1], ...]
#[allow(clippy::too_many_arguments)]
fn decode_rsi(
    reader: &mut BitReader<'_>,
    bps: u32,
    block_size: usize,
    rsi_blocks: usize,
    id_len: u32,
    pp: bool,
    max_samples: usize,
    se_table: &SeTable,
) -> Result<Vec<u32>, AecError> {
    let samples_per_rsi = rsi_blocks * block_size;
    let modi = 1u32 << id_len;
    let mut rsi_buffer: Vec<u32> = Vec::with_capacity(samples_per_rsi);

    // Track block state matching libaec's m_next_cds
    let mut has_ref = pp; // first block has ref if preprocessing
    let mut encoded_block_size = if pp { block_size - 1 } else { block_size };

    while rsi_buffer.len() < max_samples && rsi_buffer.len() < samples_per_rsi {
        // Read coding option ID (id_len bits) — ALWAYS first
        let id = reader
            .read(id_len)
            .ok_or_else(|| AecError::Data("unexpected end of stream reading ID".into()))?;

        if id == 0 {
            // Low-entropy: read 1 extra bit to distinguish zero vs SE
            let low_bit = reader.read(1).ok_or_else(|| {
                AecError::Data("unexpected end of stream in low-entropy bit".into())
            })?;

            // Read reference AFTER ID + low_bit (matching libaec m_low_entropy_ref)
            if has_ref {
                let reference = reader.read(bps).ok_or_else(|| {
                    AecError::Data("unexpected end reading ref in low-entropy".into())
                })?;
                rsi_buffer.push(reference);
            }

            if low_bit == 1 {
                // Second Extension
                decode_se(reader, &mut rsi_buffer, encoded_block_size, se_table)?;
            } else {
                // Zero block
                let fs = reader
                    .read_fs()
                    .ok_or_else(|| AecError::Data("unexpected end in zero-block FS".into()))?;

                // Decode zero block count (matching libaec m_zero_block exactly)
                let mut zero_block_count = fs + 1;
                if zero_block_count == 5 {
                    // ROS: fill rest of 64-block segment or RSI
                    let blocks_used = rsi_buffer.len() / block_size;
                    let remaining_rsi = rsi_blocks - blocks_used;
                    let remaining_seg = 64 - (blocks_used % 64);
                    zero_block_count = remaining_rsi.min(remaining_seg) as u32;
                } else if zero_block_count > 5 {
                    zero_block_count -= 1; // undo encoder's as-is encoding
                }

                // Push zero samples for all zero blocks
                // The first zero block already had its ref pushed (if any)
                let zero_samples =
                    zero_block_count as usize * block_size - if has_ref { 1 } else { 0 }; // ref already pushed for first block
                rsi_buffer.extend(std::iter::repeat_n(0, zero_samples));
            }
        } else if id < modi - 1 {
            // Split-sample: k = id - 1
            let k = id - 1;

            // Read reference AFTER ID (matching libaec m_split)
            if has_ref {
                let reference = reader
                    .read(bps)
                    .ok_or_else(|| AecError::Data("unexpected end reading ref in split".into()))?;
                rsi_buffer.push(reference);
            }

            decode_split(reader, &mut rsi_buffer, encoded_block_size, k)?;
        } else {
            // Uncompressed (id == modi - 1)
            // Reference is embedded as the first sample — no separate read
            // (matching libaec m_uncomp: reads ALL block_size samples)
            decode_uncomp(reader, &mut rsi_buffer, block_size, bps)?;
        }

        // After the first block, switch to non-ref mode
        // (matching libaec m_next_cds: ref=0, encoded_block_size=block_size)
        if has_ref {
            has_ref = false;
            encoded_block_size = block_size;
        }
    }

    Ok(rsi_buffer)
}

fn decode_split(
    reader: &mut BitReader<'_>,
    buffer: &mut Vec<u32>,
    count: usize,
    k: u32,
) -> Result<(), AecError> {
    let start = buffer.len();

    // Phase 1: read FS values (MSBs shifted by k)
    for _ in 0..count {
        let fs = reader
            .read_fs()
            .ok_or_else(|| AecError::Data("unexpected end in split FS".into()))?;
        buffer.push(fs << k);
    }

    // Phase 2: read k-bit binary values (LSBs) and add to MSBs
    if k > 0 {
        for entry in buffer.iter_mut().skip(start).take(count) {
            let lsb = reader
                .read(k)
                .ok_or_else(|| AecError::Data("unexpected end in split binary part".into()))?;
            *entry += lsb;
        }
    }

    Ok(())
}

fn decode_se(
    reader: &mut BitReader<'_>,
    buffer: &mut Vec<u32>,
    count: usize,
    se_table: &SeTable,
) -> Result<(), AecError> {
    let mut decoded = 0usize;
    while decoded < count {
        let m = reader
            .read_fs()
            .ok_or_else(|| AecError::Data("unexpected end in SE FS".into()))?;

        let (i, ms) = se_table
            .lookup(m)
            .ok_or_else(|| AecError::Data(format!("SE lookup out of range: m={m}")))?;

        let d1 = m as i32 - ms as i32;

        if decoded & 1 == 0 {
            buffer.push((i as i32 - d1) as u32);
            decoded += 1;
        }
        if decoded < count {
            buffer.push(d1 as u32);
            decoded += 1;
        }
    }
    Ok(())
}

fn decode_uncomp(
    reader: &mut BitReader<'_>,
    buffer: &mut Vec<u32>,
    block_size: usize,
    bps: u32,
) -> Result<(), AecError> {
    // Read ALL block_size samples (reference is embedded at position 0)
    for _ in 0..block_size {
        let val = reader
            .read(bps)
            .ok_or_else(|| AecError::Data("unexpected end in uncomp block".into()))?;
        buffer.push(val);
    }
    Ok(())
}

// ── Sample output ────────────────────────────────────────────────────────────

fn write_samples(samples: &[u32], byte_width: usize, flags: u32) -> Result<Vec<u8>, AecError> {
    let msb = flags & params::AEC_DATA_MSB != 0;
    let mut out = Vec::with_capacity(samples.len() * byte_width);

    for &val in samples {
        match byte_width {
            1 => out.push(val as u8),
            2 => {
                if msb {
                    out.extend_from_slice(&(val as u16).to_be_bytes());
                } else {
                    out.extend_from_slice(&(val as u16).to_le_bytes());
                }
            }
            3 => {
                if msb {
                    out.push((val >> 16) as u8);
                    out.push((val >> 8) as u8);
                    out.push(val as u8);
                } else {
                    out.push(val as u8);
                    out.push((val >> 8) as u8);
                    out.push((val >> 16) as u8);
                }
            }
            4 => {
                if msb {
                    out.extend_from_slice(&val.to_be_bytes());
                } else {
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
            // byte_width is computed by sample_byte_width() and is always 1-4,
            // but we return an error instead of panicking for safety.
            w => return Err(AecError::Config(format!("unexpected byte width {w}"))),
        }
    }

    Ok(out)
}
