use crate::error::{Result, TensogramError};
use crate::types::HashDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Xxh3,
}

impl HashAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            HashAlgorithm::Xxh3 => "xxh3",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "xxh3" => Ok(HashAlgorithm::Xxh3),
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
    }
}

/// Verify a hash descriptor against data.
///
/// If the hash algorithm is not recognized, a warning is logged and
/// verification is skipped (returns Ok). This ensures forward compatibility
/// when new hash algorithms are added.
pub fn verify_hash(data: &[u8], descriptor: &HashDescriptor) -> Result<()> {
    let algorithm = match HashAlgorithm::parse(&descriptor.hash_type) {
        Ok(algo) => algo,
        Err(_) => {
            eprintln!(
                "warning: unknown hash algorithm '{}', skipping verification",
                descriptor.hash_type
            );
            return Ok(());
        }
    };
    let actual = compute_hash(data, algorithm);
    if actual != descriptor.value {
        return Err(TensogramError::HashMismatch {
            expected: descriptor.value.clone(),
            actual,
        });
    }
    Ok(())
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

    #[test]
    fn test_unknown_hash_type_skips_verification() {
        let data = b"test data";
        let descriptor = HashDescriptor {
            hash_type: "sha256".to_string(),
            value: "abc123".to_string(),
        };
        // Unknown hash algorithms skip verification with a warning (forward compatibility)
        assert!(verify_hash(data, &descriptor).is_ok());
    }
}
