use super::{CompressResult, CompressionError, Compressor};
use crate::pipeline::{ByteOrder, Sz3ErrorBound};

fn map_err(e: sz3::SZ3Error) -> CompressionError {
    CompressionError::Sz3(format!("{e:?}"))
}

fn to_sz3_bound(bound: &Sz3ErrorBound) -> sz3::ErrorBound {
    match bound {
        Sz3ErrorBound::Absolute(v) => sz3::ErrorBound::Absolute(*v),
        Sz3ErrorBound::Relative(v) => sz3::ErrorBound::Relative(*v),
        Sz3ErrorBound::Psnr(v) => sz3::ErrorBound::PSNR(*v),
    }
}

/// SZ3 error-bounded lossy compressor for floating-point data.
///
/// Like ZFP, SZ3 operates on typed floating-point arrays. Use with
/// `encoding: "none"`, `filter: "none"`, `compression: "sz3"`.
pub struct Sz3Compressor {
    pub error_bound: Sz3ErrorBound,
    pub num_values: usize,
    /// Byte order to use when serialising decompressed f64 values back to
    /// bytes.  Ensures decompressed output matches the wire byte order so
    /// the pipeline's uniform native-endian byteswap step works for lossy
    /// codecs the same as for lossless ones.
    pub byte_order: ByteOrder,
}

impl Compressor for Sz3Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        // Compress side: input bytes are always in the caller's native byte
        // order (per design: "always encode in the endianness of the caller").
        // This intentionally ignores `self.byte_order` — it only governs
        // the decompress output format.  Using the declared byte_order here
        // would produce garbage if the caller provides native-endian data
        // (which is the contract for all encode paths).
        let values = bytes_to_f64_native(data)?;
        let dimensioned = sz3::DimensionedData::<f64, _>::build(&values)
            .dim(values.len())
            .map_err(map_err)?
            .finish()
            .map_err(map_err)?;

        let compressed =
            sz3::compress(&dimensioned, to_sz3_bound(&self.error_bound)).map_err(map_err)?;

        Ok(CompressResult {
            data: compressed,
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        let (_config, dimensioned) = sz3::decompress::<f64, _>(data).map_err(map_err)?;
        let values = dimensioned.into_data();
        // Write in the wire byte order so the pipeline's byteswap step can
        // uniformly convert wire → native without special-casing lossy codecs.
        Ok(f64_to_bytes(&values, self.byte_order))
    }

    fn decompress_range(
        &self,
        _data: &[u8],
        _block_offsets: &[u64],
        _byte_pos: usize,
        _byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        Err(CompressionError::RangeNotSupported)
    }
}

/// Interpret raw bytes as native-endian f64 values (used on the compress side
/// where the caller always provides native-endian data).
fn bytes_to_f64_native(data: &[u8]) -> Result<Vec<f64>, CompressionError> {
    if !data.len().is_multiple_of(8) {
        return Err(CompressionError::Sz3(format!(
            "data length {} is not a multiple of 8",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(8)
        .map(|chunk| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            f64::from_ne_bytes(arr)
        })
        .collect())
}

/// Serialise f64 values to bytes in the specified byte order.
fn f64_to_bytes(values: &[f64], byte_order: ByteOrder) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| match byte_order {
            ByteOrder::Big => v.to_be_bytes(),
            ByteOrder::Little => v.to_le_bytes(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smooth_data(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i as f64 / n as f64 * std::f64::consts::PI).sin())
            .flat_map(|v| v.to_ne_bytes())
            .collect()
    }

    #[test]
    fn sz3_round_trip_absolute() {
        let data = smooth_data(512);
        let tol = 1e-4;
        let compressor = Sz3Compressor {
            error_bound: Sz3ErrorBound::Absolute(tol),
            num_values: 512,
            byte_order: ByteOrder::native(),
        };

        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed.len(), data.len());

        // Verify within tolerance
        let orig: Vec<f64> = data
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        let dec: Vec<f64> = decompressed
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        for (o, d) in orig.iter().zip(dec.iter()) {
            assert!(
                (o - d).abs() <= tol,
                "orig={o}, dec={d}, diff={}",
                (o - d).abs()
            );
        }
    }

    #[test]
    fn sz3_range_not_supported() {
        let compressor = Sz3Compressor {
            error_bound: Sz3ErrorBound::Absolute(1e-4),
            num_values: 100,
            byte_order: ByteOrder::native(),
        };
        let result = compressor.decompress_range(&[0], &[], 0, 1);
        assert!(matches!(result, Err(CompressionError::RangeNotSupported)));
    }
}
