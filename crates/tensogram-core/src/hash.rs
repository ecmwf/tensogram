use crate::error::{Result, TensogramError};
use crate::types::HashDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Xxh3,
    Sha1,
    Md5,
}

impl HashAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            HashAlgorithm::Xxh3 => "xxh3",
            HashAlgorithm::Sha1 => "sha1",
            HashAlgorithm::Md5 => "md5",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "xxh3" => Ok(HashAlgorithm::Xxh3),
            "sha1" => Ok(HashAlgorithm::Sha1),
            "md5" => Ok(HashAlgorithm::Md5),
            _ => Err(TensogramError::Metadata(format!("unknown hash type: {s}"))),
        }
    }
}

/// Compute a hash of the given data, returning the hex-encoded digest.
pub fn compute_hash(data: &[u8], algorithm: HashAlgorithm) -> String {
    match algorithm {
        HashAlgorithm::Xxh3 => {
            let hash = xxhash_rust::xxh3::xxh3_64(data);
            format!("{hash:016x}")
        }
        HashAlgorithm::Sha1 => {
            use sha1::{Digest, Sha1};
            let mut hasher = Sha1::new();
            hasher.update(data);
            let result = hasher.finalize();
            hex_encode(&result)
        }
        HashAlgorithm::Md5 => {
            let digest = md5::compute(data);
            hex_encode(&digest.0)
        }
    }
}

/// Verify a hash descriptor against data.
pub fn verify_hash(data: &[u8], descriptor: &HashDescriptor) -> Result<()> {
    let algorithm = HashAlgorithm::parse(&descriptor.hash_type)?;
    let actual = compute_hash(data, algorithm);
    if actual != descriptor.value {
        return Err(TensogramError::HashMismatch {
            expected: descriptor.value.clone(),
            actual,
        });
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xxh3() {
        let data = b"hello world";
        let hash = compute_hash(data, HashAlgorithm::Xxh3);
        assert_eq!(hash.len(), 16); // 64-bit = 16 hex chars
        // Verify deterministic
        assert_eq!(hash, compute_hash(data, HashAlgorithm::Xxh3));
    }

    #[test]
    fn test_sha1() {
        let data = b"hello world";
        let hash = compute_hash(data, HashAlgorithm::Sha1);
        assert_eq!(hash.len(), 40); // 160-bit = 40 hex chars
    }

    #[test]
    fn test_md5() {
        let data = b"hello world";
        let hash = compute_hash(data, HashAlgorithm::Md5);
        assert_eq!(hash.len(), 32); // 128-bit = 32 hex chars
    }

    #[test]
    fn test_verify_hash() {
        let data = b"test data";
        let hash = compute_hash(data, HashAlgorithm::Xxh3);
        let descriptor = HashDescriptor {
            hash_type: "xxh3".to_string(),
            value: hash,
        };
        assert!(verify_hash(data, &descriptor).is_ok());
    }

    #[test]
    fn test_verify_hash_mismatch() {
        let data = b"test data";
        let descriptor = HashDescriptor {
            hash_type: "xxh3".to_string(),
            value: "0000000000000000".to_string(),
        };
        assert!(verify_hash(data, &descriptor).is_err());
    }
}
