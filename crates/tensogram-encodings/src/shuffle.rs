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

/// Byte-level shuffle: groups byte[0] of all elements, then byte[1], etc.
/// Input length must be divisible by element_size.
/// Returns `Err(ShuffleError::InvalidElementSize)` if element_size is 0.
/// Returns `Err(ShuffleError::Misaligned)` if data.len() % element_size != 0.
pub fn shuffle(data: &[u8], element_size: usize) -> Result<Vec<u8>, ShuffleError> {
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
