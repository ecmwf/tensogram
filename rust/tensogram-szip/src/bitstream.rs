// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! MSB-first bitstream reader and writer for the AEC codec.
//!
//! The AEC standard specifies MSB-first bit ordering: the first bit
//! emitted is the most-significant bit of the first byte, etc.

// ── Bitstream Writer ─────────────────────────────────────────────────────────

/// MSB-first bitstream writer that accumulates bits into an output buffer.
pub(crate) struct BitWriter {
    out: Vec<u8>,
    /// Accumulator for the current partially-filled byte.
    acc: u8,
    /// Number of free (unused) bits remaining in `acc`, counting from LSB.
    /// Starts at 8 (empty byte) and decreases as bits are written.
    bits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            bits: 8,
        }
    }

    /// Write `n` bits from the LSB of `data` (1 ≤ n ≤ 32).
    pub fn emit(&mut self, data: u32, n: u32) {
        debug_assert!(n > 0 && n <= 32);
        let mut remaining = n;
        let mut val = data;

        // Mask off any upper bits beyond n
        if n < 32 {
            val &= (1u32 << n) - 1;
        }

        if remaining <= self.bits {
            // Fits entirely in the current accumulator byte
            self.bits -= remaining;
            self.acc |= (val as u8) << self.bits;
            if self.bits == 0 {
                self.out.push(self.acc);
                self.acc = 0;
                self.bits = 8;
            }
        } else {
            // Doesn't fit — fill current byte, then emit full bytes, then remainder
            remaining -= self.bits;
            self.acc |= (val >> remaining) as u8;
            self.out.push(self.acc);

            while remaining > 8 {
                remaining -= 8;
                self.out.push((val >> remaining) as u8);
            }

            self.bits = 8 - remaining;
            self.acc = (val as u8) << self.bits;
        }
    }

    /// Emit a Fundamental Sequence: `fs` zero bits followed by one 1-bit.
    pub fn emit_fs(&mut self, fs: u32) {
        let mut zeros = fs;

        loop {
            if zeros < self.bits {
                // Room for the zeros + the terminating 1
                self.bits -= zeros + 1;
                self.acc |= 1u8 << self.bits;
                if self.bits == 0 {
                    self.out.push(self.acc);
                    self.acc = 0;
                    self.bits = 8;
                }
                break;
            } else {
                // Fill remainder of current byte with zeros
                zeros -= self.bits;
                self.out.push(self.acc);
                self.acc = 0;
                self.bits = 8;
            }
        }
    }

    /// Return the current bit position (total bits written so far).
    pub fn bit_position(&self) -> u64 {
        self.out.len() as u64 * 8 + (8 - self.bits) as u64
    }

    /// Flush any remaining partial byte (pad with zeros) and return
    /// the output buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits < 8 {
            // Pad the last byte with zeros in the LSB positions
            self.out.push(self.acc);
        }
        self.out
    }

    /// Flush any partial byte with zero padding (used when finishing
    /// the entire stream).
    pub fn pad_to_byte(&mut self) {
        if self.bits < 8 {
            self.out.push(self.acc);
            self.acc = 0;
            self.bits = 8;
        }
    }
}

// ── Bitstream Reader ─────────────────────────────────────────────────────────

/// MSB-first bitstream reader that consumes bits from an input buffer.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Current byte index in `data`.
    pos: usize,
    /// 64-bit accumulator holding pre-fetched bits.
    acc: u64,
    /// Number of valid bits in `acc` (counted from MSB).
    bitp: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            acc: 0,
            bitp: 0,
        }
    }

    /// Create a reader starting at a specific bit offset.
    pub fn from_bit_offset(data: &'a [u8], bit_offset: u64) -> Self {
        let byte_offset = (bit_offset / 8) as usize;
        let bit_remainder = (bit_offset % 8) as u32;

        let mut reader = Self {
            data,
            pos: byte_offset,
            acc: 0,
            bitp: 0,
        };

        if bit_remainder > 0 && byte_offset < data.len() {
            // Load the partial first byte
            reader.acc = (data[byte_offset] as u64) & ((1u64 << (8 - bit_remainder)) - 1);
            reader.bitp = 8 - bit_remainder;
            reader.pos = byte_offset + 1;
        }

        reader
    }

    /// Ensure at least `n` bits are available in the accumulator.
    /// Returns false if input is exhausted.
    fn fill(&mut self, n: u32) -> bool {
        while self.bitp < n {
            if self.pos >= self.data.len() {
                return false;
            }
            self.acc = (self.acc << 8) | (self.data[self.pos] as u64);
            self.pos += 1;
            self.bitp += 8;
        }
        true
    }

    /// Read `n` bits (1 ≤ n ≤ 32) and return them as u32.
    /// Returns None if input is exhausted.
    pub fn read(&mut self, n: u32) -> Option<u32> {
        debug_assert!(n > 0 && n <= 32);
        if !self.fill(n) {
            return None;
        }
        self.bitp -= n;
        let val = (self.acc >> self.bitp) & ((1u64 << n) - 1);
        Some(val as u32)
    }

    /// Read a Fundamental Sequence: count zero bits until a 1 is found.
    /// Returns the count of zeros (the FS value).
    /// Returns None if input is exhausted before the terminating 1.
    pub fn read_fs(&mut self) -> Option<u32> {
        let mut fs: u32 = 0;

        // Mask off any bits above bitp so leading zeros can be counted
        if self.bitp > 0 {
            self.acc &= (1u64 << self.bitp) - 1;
        } else {
            self.acc = 0;
        }

        // Fast path: scan through zero bytes
        while self.acc == 0 {
            if self.pos >= self.data.len() {
                return None;
            }
            fs += self.bitp;
            // Refill — load as many bytes as possible
            let load = (self.data.len() - self.pos).min(7);
            self.acc = 0;
            for i in 0..load {
                self.acc = (self.acc << 8) | (self.data[self.pos + i] as u64);
            }
            self.pos += load;
            self.bitp = (load as u32) * 8;

            if self.bitp > 0 {
                self.acc &= (1u64 << self.bitp) - 1;
            }
        }

        // Find the position of the highest set bit
        let leading_zeros = self.acc.leading_zeros() - (64 - self.bitp);
        fs += leading_zeros;
        self.bitp -= leading_zeros + 1; // consume zeros + the 1-bit
        Some(fs)
    }

    /// Return the total number of bits consumed so far.
    #[allow(dead_code)] // used in tests and available for future callers
    pub fn bits_consumed(&self) -> u64 {
        self.pos as u64 * 8 - self.bitp as u64
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_bits() {
        let mut w = BitWriter::new();
        w.emit(0b101, 3);
        w.emit(0b1111_0000, 8);
        w.emit(0b1, 1);
        let buf = w.finish();

        let mut r = BitReader::new(&buf);
        assert_eq!(r.read(3), Some(0b101));
        assert_eq!(r.read(8), Some(0b1111_0000));
        assert_eq!(r.read(1), Some(1));
    }

    #[test]
    fn write_then_read_fs() {
        let mut w = BitWriter::new();
        w.emit_fs(0); // just a 1
        w.emit_fs(5); // 00000_1
        w.emit_fs(0); // 1
        w.emit_fs(12); // 000000000000_1
        let buf = w.finish();

        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_fs(), Some(0));
        assert_eq!(r.read_fs(), Some(5));
        assert_eq!(r.read_fs(), Some(0));
        assert_eq!(r.read_fs(), Some(12));
    }

    #[test]
    fn emit_32_bits() {
        let mut w = BitWriter::new();
        w.emit(0xDEADBEEF, 32);
        let buf = w.finish();
        assert_eq!(buf, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn bit_position_tracking() {
        let mut w = BitWriter::new();
        assert_eq!(w.bit_position(), 0);
        w.emit(0, 3);
        assert_eq!(w.bit_position(), 3);
        w.emit(0, 5);
        assert_eq!(w.bit_position(), 8);
        w.emit(0, 1);
        assert_eq!(w.bit_position(), 9);
    }

    #[test]
    fn reader_from_bit_offset() {
        let data = [0b1010_1100, 0b0011_1111];
        // Start reading from bit 4 (middle of first byte)
        let mut r = BitReader::from_bit_offset(&data, 4);
        assert_eq!(r.read(4), Some(0b1100)); // lower nibble of first byte
        assert_eq!(r.read(4), Some(0b0011)); // upper nibble of second byte
    }

    #[test]
    fn round_trip_various_widths() {
        for n in 1..=32 {
            let val = if n == 32 {
                0xFFFF_FFFF
            } else {
                (1u32 << n) - 1
            };
            let mut w = BitWriter::new();
            w.emit(val, n);
            let buf = w.finish();
            let mut r = BitReader::new(&buf);
            assert_eq!(r.read(n), Some(val), "failed for n={n}");
        }
    }

    #[test]
    fn fs_large_value() {
        let mut w = BitWriter::new();
        w.emit_fs(100);
        let buf = w.finish();

        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_fs(), Some(100));
    }

    #[test]
    fn interleaved_fs_and_bits() {
        let mut w = BitWriter::new();
        w.emit(0b110, 3);
        w.emit_fs(3);
        w.emit(0xFF, 8);
        w.emit_fs(0);
        let buf = w.finish();

        let mut r = BitReader::new(&buf);
        assert_eq!(r.read(3), Some(0b110));
        assert_eq!(r.read_fs(), Some(3));
        assert_eq!(r.read(8), Some(0xFF));
        assert_eq!(r.read_fs(), Some(0));
    }
}
