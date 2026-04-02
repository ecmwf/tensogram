use crate::error::{Result, TensogramError};

pub const MAGIC: &[u8; 8] = b"TENSOGRM";
pub const TERMINATOR: &[u8; 8] = b"39277777";
pub const OBJS: &[u8; 4] = b"OBJS";
pub const OBJE: &[u8; 4] = b"OBJE";

/// Fixed portion of binary header: magic(8) + total_len(8) + meta_offset(8) + meta_len(8) + num_objects(8)
pub const FIXED_HEADER_SIZE: usize = 40;

/// Parsed binary header (the fixed-size prefix of every message).
#[derive(Debug, Clone)]
pub struct BinaryHeader {
    pub total_length: u64,
    pub metadata_offset: u64,
    pub metadata_length: u64,
    pub num_objects: u64,
    pub object_offsets: Vec<u64>,
}

impl BinaryHeader {
    /// Total header size including object offsets array.
    pub fn header_size(num_objects: u64) -> usize {
        FIXED_HEADER_SIZE + num_objects as usize * 8
    }

    /// Read a binary header from a buffer. Buffer must start with MAGIC.
    pub fn read_from(buf: &[u8]) -> Result<Self> {
        if buf.len() < FIXED_HEADER_SIZE {
            return Err(TensogramError::Framing(format!(
                "buffer too short for header: {} < {}",
                buf.len(),
                FIXED_HEADER_SIZE
            )));
        }

        if &buf[0..8] != MAGIC {
            return Err(TensogramError::Framing("invalid magic bytes".to_string()));
        }

        let total_length = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        let metadata_offset = u64::from_be_bytes(buf[16..24].try_into().unwrap());
        let metadata_length = u64::from_be_bytes(buf[24..32].try_into().unwrap());
        let num_objects = u64::from_be_bytes(buf[32..40].try_into().unwrap());

        let full_header_size = Self::header_size(num_objects);
        if buf.len() < full_header_size {
            return Err(TensogramError::Framing(format!(
                "buffer too short for {} object offsets: {} < {}",
                num_objects,
                buf.len(),
                full_header_size
            )));
        }

        let object_offsets = (0..num_objects as usize)
            .map(|i| {
                let offset = FIXED_HEADER_SIZE + i * 8;
                u64::from_be_bytes(buf[offset..offset + 8].try_into().unwrap())
            })
            .collect();

        Ok(BinaryHeader {
            total_length,
            metadata_offset,
            metadata_length,
            num_objects,
            object_offsets,
        })
    }

    /// Write the binary header to a byte vector.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.total_length.to_be_bytes());
        out.extend_from_slice(&self.metadata_offset.to_be_bytes());
        out.extend_from_slice(&self.metadata_length.to_be_bytes());
        out.extend_from_slice(&self.num_objects.to_be_bytes());
        for &offset in &self.object_offsets {
            out.extend_from_slice(&offset.to_be_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_round_trip() {
        let header = BinaryHeader {
            total_length: 1024,
            metadata_offset: 56,
            metadata_length: 200,
            num_objects: 2,
            object_offsets: vec![300, 600],
        };
        let mut buf = Vec::new();
        header.write_to(&mut buf);
        assert_eq!(buf.len(), BinaryHeader::header_size(2));

        let parsed = BinaryHeader::read_from(&buf).unwrap();
        assert_eq!(parsed.total_length, 1024);
        assert_eq!(parsed.metadata_offset, 56);
        assert_eq!(parsed.metadata_length, 200);
        assert_eq!(parsed.num_objects, 2);
        assert_eq!(parsed.object_offsets, vec![300, 600]);
    }

    #[test]
    fn test_invalid_magic() {
        let buf = vec![0u8; 100];
        assert!(BinaryHeader::read_from(&buf).is_err());
    }

    #[test]
    fn test_truncated_buffer() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        // Only 8 bytes after magic, need 32 more
        buf.extend_from_slice(&[0u8; 8]);
        assert!(BinaryHeader::read_from(&buf).is_err());
    }
}
