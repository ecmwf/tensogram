/// Byte-level shuffle: groups byte[0] of all elements, then byte[1], etc.
/// Input length must be divisible by element_size.
pub fn shuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let num_elements = data.len() / element_size;
    assert_eq!(
        data.len() % element_size,
        0,
        "data length {} not divisible by element_size {}",
        data.len(),
        element_size
    );

    let mut output = vec![0u8; data.len()];
    for byte_idx in 0..element_size {
        for elem in 0..num_elements {
            output[byte_idx * num_elements + elem] = data[elem * element_size + byte_idx];
        }
    }
    output
}

/// Reverse shuffle.
pub fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let num_elements = data.len() / element_size;
    assert_eq!(
        data.len() % element_size,
        0,
        "data length {} not divisible by element_size {}",
        data.len(),
        element_size
    );

    let mut output = vec![0u8; data.len()];
    for byte_idx in 0..element_size {
        for elem in 0..num_elements {
            output[elem * element_size + byte_idx] = data[byte_idx * num_elements + elem];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_unshuffle_float32() {
        // 3 float32 elements = 12 bytes
        let data: Vec<u8> = (0..12).collect();
        let shuffled = shuffle(&data, 4);
        // byte[0] of each element: [0, 4, 8], byte[1]: [1, 5, 9], etc.
        assert_eq!(shuffled, vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
        let unshuffled = unshuffle(&shuffled, 4);
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn test_shuffle_unshuffle_float64() {
        let data: Vec<u8> = (0..16).collect(); // 2 float64 elements
        let shuffled = shuffle(&data, 8);
        let unshuffled = unshuffle(&shuffled, 8);
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn test_shuffle_element_size_1() {
        let data = vec![1, 2, 3, 4];
        assert_eq!(shuffle(&data, 1), data);
        assert_eq!(unshuffle(&data, 1), data);
    }

    #[test]
    fn test_shuffle_empty() {
        let data: Vec<u8> = vec![];
        assert_eq!(shuffle(&data, 4), data);
    }
}
