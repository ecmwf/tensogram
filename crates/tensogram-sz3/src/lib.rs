// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! High-level SZ3 compression API for Tensogram.
//!
//! This crate provides the same public API surface as the published `sz3`
//! crate (v0.4.3), but is a clean-room implementation backed by
//! `tensogram-sz3-sys` instead of `sz3-sys`.

// Ensure the zstd native library is linked into the final binary.
// tensogram-sz3-sys compiles C++ code that calls ZSTD_compress etc.;
// the symbols come from zstd-sys's static archive.
extern crate zstd_sys;

use tensogram_sz3_sys::SZ3_Config;

// ---------------------------------------------------------------------------
// CompressionAlgorithm
// ---------------------------------------------------------------------------

/// Which prediction algorithm SZ3 should use during compression.
#[derive(Clone, Debug, Copy)]
#[non_exhaustive]
pub enum CompressionAlgorithm {
    Interpolation,
    InterpolationLorenzo,
    LorenzoRegression {
        lorenzo: bool,
        lorenzo_second_order: bool,
        regression: bool,
    },
    BiologyMolecularData,
    BiologyMolecularDataGromacsXtc,
    NoPrediction,
    Lossless,
}

impl CompressionAlgorithm {
    fn decode(config: SZ3_Config) -> Self {
        match config.cmprAlgo as u32 {
            tensogram_sz3_sys::SZ3::ALGO_ALGO_INTERP => Self::Interpolation,
            tensogram_sz3_sys::SZ3::ALGO_ALGO_INTERP_LORENZO => Self::InterpolationLorenzo,
            tensogram_sz3_sys::SZ3::ALGO_ALGO_LORENZO_REG => Self::LorenzoRegression {
                lorenzo: config.lorenzo,
                lorenzo_second_order: config.lorenzo2,
                regression: config.regression,
            },
            tensogram_sz3_sys::SZ3::ALGO_ALGO_BIOMD => Self::BiologyMolecularData,
            tensogram_sz3_sys::SZ3::ALGO_ALGO_BIOMDXTC => Self::BiologyMolecularDataGromacsXtc,
            tensogram_sz3_sys::SZ3::ALGO_ALGO_NOPRED => Self::NoPrediction,
            tensogram_sz3_sys::SZ3::ALGO_ALGO_LOSSLESS => Self::Lossless,
            algo => panic!("unsupported compression algorithm {}", algo),
        }
    }

    fn code(&self) -> u8 {
        (match self {
            Self::Interpolation => tensogram_sz3_sys::SZ3::ALGO_ALGO_INTERP,
            Self::InterpolationLorenzo => tensogram_sz3_sys::SZ3::ALGO_ALGO_INTERP_LORENZO,
            Self::LorenzoRegression { .. } => tensogram_sz3_sys::SZ3::ALGO_ALGO_LORENZO_REG,
            Self::BiologyMolecularData => tensogram_sz3_sys::SZ3::ALGO_ALGO_BIOMD,
            Self::BiologyMolecularDataGromacsXtc => tensogram_sz3_sys::SZ3::ALGO_ALGO_BIOMDXTC,
            Self::NoPrediction => tensogram_sz3_sys::SZ3::ALGO_ALGO_NOPRED,
            Self::Lossless => tensogram_sz3_sys::SZ3::ALGO_ALGO_LOSSLESS,
        }) as _
    }

    fn lorenzo(&self) -> bool {
        match self {
            Self::LorenzoRegression { lorenzo, .. } => *lorenzo,
            _ => true,
        }
    }

    fn lorenzo_second_order(&self) -> bool {
        match self {
            Self::LorenzoRegression {
                lorenzo_second_order,
                ..
            } => *lorenzo_second_order,
            _ => true,
        }
    }

    fn regression(&self) -> bool {
        match self {
            Self::LorenzoRegression { regression, .. } => *regression,
            _ => true,
        }
    }

    /// Pure interpolation predictor.
    pub fn interpolation() -> Self {
        Self::Interpolation
    }

    /// Interpolation with Lorenzo fallback (the default).
    pub fn interpolation_lorenzo() -> Self {
        Self::InterpolationLorenzo
    }

    /// Lorenzo + regression predictor with default sub-flags.
    pub fn lorenzo_regression() -> Self {
        Self::LorenzoRegression {
            lorenzo: true,
            lorenzo_second_order: false,
            regression: true,
        }
    }

    /// Lorenzo + regression predictor with individually overridable sub-flags.
    pub fn lorenzo_regression_custom(
        lorenzo: Option<bool>,
        lorenzo_second_order: Option<bool>,
        regression: Option<bool>,
    ) -> Self {
        Self::LorenzoRegression {
            lorenzo: lorenzo.unwrap_or(true),
            lorenzo_second_order: lorenzo_second_order.unwrap_or(false),
            regression: regression.unwrap_or(true),
        }
    }

    /// Predictor optimised for molecular-dynamics data.
    pub fn biology_molecular_data() -> Self {
        Self::BiologyMolecularData
    }

    /// Predictor optimised for GROMACS XTC molecular-dynamics data.
    pub fn biology_molecular_data_gromacs_xtc() -> Self {
        Self::BiologyMolecularDataGromacsXtc
    }

    /// Skip prediction entirely; only quantise and encode.
    pub fn no_prediction() -> Self {
        Self::NoPrediction
    }

    /// Fully lossless compression (no quantisation).
    pub fn lossless() -> Self {
        Self::Lossless
    }
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::interpolation_lorenzo()
    }
}

// ---------------------------------------------------------------------------
// ErrorBound
// ---------------------------------------------------------------------------

/// Error-bound mode that controls how much distortion SZ3 is allowed to
/// introduce during lossy compression.
#[derive(Clone, Debug, Copy)]
#[non_exhaustive]
pub enum ErrorBound {
    Absolute(f64),
    Relative(f64),
    PSNR(f64),
    L2Norm(f64),
    AbsoluteAndRelative {
        absolute_bound: f64,
        relative_bound: f64,
    },
    AbsoluteOrRelative {
        absolute_bound: f64,
        relative_bound: f64,
    },
}

impl ErrorBound {
    fn decode(config: SZ3_Config) -> Self {
        match config.errorBoundMode as u32 {
            tensogram_sz3_sys::SZ3::EB_EB_ABS => Self::Absolute(config.absErrorBound),
            tensogram_sz3_sys::SZ3::EB_EB_REL => Self::Relative(config.relErrorBound),
            tensogram_sz3_sys::SZ3::EB_EB_PSNR => Self::PSNR(config.psnrErrorBound),
            tensogram_sz3_sys::SZ3::EB_EB_L2NORM => Self::L2Norm(config.l2normErrorBound),
            tensogram_sz3_sys::SZ3::EB_EB_ABS_OR_REL => Self::AbsoluteOrRelative {
                absolute_bound: config.absErrorBound,
                relative_bound: config.relErrorBound,
            },
            tensogram_sz3_sys::SZ3::EB_EB_ABS_AND_REL => Self::AbsoluteAndRelative {
                absolute_bound: config.absErrorBound,
                relative_bound: config.relErrorBound,
            },
            mode => panic!("unsupported error bound {}", mode),
        }
    }

    fn code(&self) -> u8 {
        (match self {
            Self::Absolute(_) => tensogram_sz3_sys::SZ3::EB_EB_ABS,
            Self::Relative(_) => tensogram_sz3_sys::SZ3::EB_EB_REL,
            Self::PSNR(_) => tensogram_sz3_sys::SZ3::EB_EB_PSNR,
            Self::L2Norm(_) => tensogram_sz3_sys::SZ3::EB_EB_L2NORM,
            Self::AbsoluteAndRelative { .. } => tensogram_sz3_sys::SZ3::EB_EB_ABS_AND_REL,
            Self::AbsoluteOrRelative { .. } => tensogram_sz3_sys::SZ3::EB_EB_ABS_OR_REL,
        }) as _
    }

    fn abs_bound(&self) -> f64 {
        match self {
            Self::Absolute(bound) => *bound,
            Self::AbsoluteOrRelative { absolute_bound, .. }
            | Self::AbsoluteAndRelative { absolute_bound, .. } => *absolute_bound,
            _ => 0.0,
        }
    }

    fn rel_bound(&self) -> f64 {
        match self {
            Self::Relative(bound) => *bound,
            Self::AbsoluteOrRelative { relative_bound, .. }
            | Self::AbsoluteAndRelative { relative_bound, .. } => *relative_bound,
            _ => 0.0,
        }
    }

    fn l2norm_bound(&self) -> f64 {
        match self {
            Self::L2Norm(bound) => *bound,
            _ => 0.0,
        }
    }

    fn psnr_bound(&self) -> f64 {
        match self {
            Self::PSNR(bound) => *bound,
            _ => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Full configuration for an SZ3 compress/decompress round-trip.
///
/// Use the builder-style setters to customise individual parameters; the
/// only required parameter is [`ErrorBound`], passed via [`Config::new`].
#[derive(Clone, Debug)]
pub struct Config {
    compression_algorithm: CompressionAlgorithm,
    error_bound: ErrorBound,
    openmp: bool,
    quantization_bincount: u32,
    block_size: Option<u32>,
}

impl Config {
    /// Create a new configuration with the given error bound and default
    /// settings (interpolation-lorenzo algorithm, 65 536 quantization bins).
    pub fn new(error_bound: ErrorBound) -> Self {
        Self {
            compression_algorithm: CompressionAlgorithm::default(),
            error_bound,
            openmp: false,
            quantization_bincount: 65536,
            block_size: None,
        }
    }

    fn from_decompressed(config: SZ3_Config) -> Self {
        Self {
            compression_algorithm: CompressionAlgorithm::decode(config),
            error_bound: ErrorBound::decode(config),
            openmp: config.openmp,
            quantization_bincount: config.quantbinCnt as _,
            block_size: Some(config.blockSize as _),
        }
    }

    /// Set the prediction algorithm.
    pub fn compression_algorithm(mut self, compression_algorithm: CompressionAlgorithm) -> Self {
        self.compression_algorithm = compression_algorithm;
        self
    }

    /// Override the error bound.
    pub fn error_bound(mut self, error_bound: ErrorBound) -> Self {
        self.error_bound = error_bound;
        self
    }

    /// Enable or disable OpenMP parallelism (requires the `openmp` feature).
    #[cfg(feature = "openmp")]
    pub fn openmp(mut self, openmp: bool) -> Self {
        self.openmp = openmp;
        self
    }

    /// Set the number of quantization bins (default: 65 536).
    pub fn quantization_bincount(mut self, quantization_bincount: u32) -> Self {
        self.quantization_bincount = quantization_bincount;
        self
    }

    /// Set an explicit block size, overriding the automatic default.
    pub fn block_size(mut self, block_size: u32) -> Self {
        self.block_size = Some(block_size);
        self
    }

    /// Revert to the automatic block size (chosen based on dimensionality).
    pub fn automatic_block_size(mut self) -> Self {
        self.block_size = None;
        self
    }
}

// ---------------------------------------------------------------------------
// SZ3Compressible trait + sealed implementation
// ---------------------------------------------------------------------------

/// Marker trait for types that SZ3 can compress and decompress.
///
/// Implemented for `f32`, `f64`, `u8`, `i8`, `u16`, `i16`, `u32`, `i32`,
/// `u64`, and `i64`.  This trait is sealed and cannot be implemented outside
/// this crate.
pub trait SZ3Compressible: private::Sealed + Sized {}
impl SZ3Compressible for f32 {}
impl SZ3Compressible for f64 {}
impl SZ3Compressible for u8 {}
impl SZ3Compressible for i8 {}
impl SZ3Compressible for u16 {}
impl SZ3Compressible for i16 {}
impl SZ3Compressible for u32 {}
impl SZ3Compressible for i32 {}
impl SZ3Compressible for u64 {}
impl SZ3Compressible for i64 {}

mod private {
    pub trait Sealed: Copy {
        const SZ_DATA_TYPE: u8;

        unsafe fn compress_size_bound(config: tensogram_sz3_sys::SZ3_Config) -> usize;

        unsafe fn compress(
            config: tensogram_sz3_sys::SZ3_Config,
            data: *const Self,
            compressed_data: *mut u8,
            compressed_capacity: usize,
        ) -> usize;

        unsafe fn decompress(
            compressed_data: *const u8,
            compressed_len: usize,
            decompressed_data: *mut Self,
        );
    }

    macro_rules! impl_sealed {
        ($($impl_mod:ident),*) => {
            $(impl Sealed for tensogram_sz3_sys::$impl_mod::ty {
                const SZ_DATA_TYPE: u8 = tensogram_sz3_sys::$impl_mod::DATA_TYPE_TYPE;

                unsafe fn compress_size_bound(config: tensogram_sz3_sys::SZ3_Config) -> usize {
                    unsafe { tensogram_sz3_sys::$impl_mod::compress_size_bound(config) }
                }

                unsafe fn compress(
                    config: tensogram_sz3_sys::SZ3_Config,
                    data: *const Self,
                    compressed_data: *mut u8,
                    compressed_capacity: usize,
                ) -> usize {
                    unsafe {
                        tensogram_sz3_sys::$impl_mod::compress(
                            config,
                            data,
                            compressed_data.cast(),
                            compressed_capacity,
                        )
                    }
                }

                unsafe fn decompress(
                    compressed_data: *const u8,
                    compressed_len: usize,
                    decompressed_data: *mut Self,
                ) {
                    unsafe {
                        tensogram_sz3_sys::$impl_mod::decompress(
                            compressed_data.cast(),
                            compressed_len,
                            decompressed_data,
                        )
                    }
                }
            })*
        }
    }

    impl_sealed!(
        impl_f32, impl_f64, impl_u8, impl_i8, impl_u16, impl_i16, impl_u32, impl_i32, impl_u64,
        impl_i64
    );
}

// ---------------------------------------------------------------------------
// DimensionedData + builders
// ---------------------------------------------------------------------------

/// A data buffer together with its N-dimensional shape, ready for SZ3
/// compression or decompression.
///
/// Construct via [`DimensionedData::build`] (immutable) or
/// [`DimensionedData::build_mut`] (mutable).
#[derive(Clone, Debug)]
pub struct DimensionedData<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>> {
    data: T,
    dims: Vec<usize>,
}

/// Builder for an immutable [`DimensionedData`] reference.
#[derive(Clone, Debug)]
pub struct DimensionedDataBuilder<'a, V> {
    data: &'a [V],
    dims: Vec<usize>,
    remainder: usize,
}

/// Builder for a mutable [`DimensionedData`] reference.
#[derive(Debug)]
pub struct DimensionedDataBuilderMut<'a, V> {
    data: &'a mut [V],
    dims: Vec<usize>,
    remainder: usize,
}

impl<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>> DimensionedData<V, T> {
    /// Start building an immutable dimensioned view over `data`.
    pub fn build<'a>(data: &'a T) -> DimensionedDataBuilder<'a, V> {
        DimensionedDataBuilder {
            data,
            dims: vec![],
            remainder: data.len(),
        }
    }

    /// Returns the underlying data as a flat slice.
    pub fn data(&self) -> &[V] {
        &self.data
    }

    /// Consume the wrapper and return the owned data container.
    pub fn into_data(self) -> T {
        self.data
    }

    /// Returns the dimension sizes.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn as_ptr(&self) -> *const V {
        self.data.as_ptr()
    }
}

impl<V: SZ3Compressible, T: std::ops::DerefMut<Target = [V]>> DimensionedData<V, T> {
    /// Start building a mutable dimensioned view over `data`.
    pub fn build_mut<'a>(data: &'a mut T) -> DimensionedDataBuilderMut<'a, V> {
        DimensionedDataBuilderMut {
            remainder: data.len(),
            data,
            dims: vec![],
        }
    }

    /// Returns the underlying data as a mutable flat slice.
    pub fn data_mut(&mut self) -> &mut [V] {
        &mut self.data
    }
}

// ---------------------------------------------------------------------------
// SZ3Error
// ---------------------------------------------------------------------------

/// Errors returned by the SZ3 compression and dimension-building APIs.
#[derive(thiserror::Error, Debug)]
pub enum SZ3Error {
    #[error(
        "invalid dimension specification for data of length {len}: already specified dimensions \
         {dims:?}, and wanted to add dimension with length {wanted}, but this does not divide \
         {remainder} cleanly"
    )]
    InvalidDimensionSize {
        dims: Vec<usize>,
        len: usize,
        wanted: usize,
        remainder: usize,
    },
    #[error("dimension with size one has no use")]
    OneSizedDimension,
    #[error(
        "dimension specification {dims:?} for data of length {len} does not cover whole space, \
         missing a dimension of {remainder}"
    )]
    UnderSpecifiedDimensions {
        dims: Vec<usize>,
        len: usize,
        remainder: usize,
    },
    #[error("cannot decompress to array with a different data type")]
    DecompressedDataTypeMismatch,
    #[error("cannot decompress array with dimensions {found:?} to array with different dimensions {expected:?}")]
    DecompressedDimsMismatch {
        found: Vec<usize>,
        expected: Vec<usize>,
    },
}

type Result<T> = std::result::Result<T, SZ3Error>;

// ---------------------------------------------------------------------------
// Builder implementations (shared logic for immutable and mutable builders)
// ---------------------------------------------------------------------------

macro_rules! impl_dimensioned_data_builder {
    ($($builder:ident => $data:ty),*) => {
        $(impl<'a, V: SZ3Compressible> $builder<'a, V> {
            /// Append a dimension with the given `length`.
            ///
            /// Returns an error if `length` does not evenly divide the
            /// remaining element count.
            pub fn dim(mut self, length: usize) -> Result<Self> {
                if length == 1 {
                    if self.dims.is_empty() && self.remainder == 1 {
                        self.dims.push(1);
                        Ok(self)
                    } else {
                        Err(SZ3Error::OneSizedDimension)
                    }
                } else if self.remainder % length != 0 {
                    Err(SZ3Error::InvalidDimensionSize {
                        dims: self.dims,
                        len: self.data.len(),
                        wanted: length,
                        remainder: self.remainder,
                    })
                } else {
                    self.dims.push(length);
                    self.remainder /= length;
                    Ok(self)
                }
            }

            /// Append a final dimension that consumes all remaining elements.
            pub fn remainder_dim(self) -> Result<$data> {
                let remainder = self.remainder;
                self.dim(remainder)?.finish()
            }

            /// Finalise the builder, returning the dimensioned data.
            ///
            /// Returns an error if the specified dimensions do not exactly
            /// cover the data length.
            pub fn finish(self) -> Result<$data> {
                if self.remainder != 1 {
                    Err(SZ3Error::UnderSpecifiedDimensions {
                        dims: self.dims,
                        len: self.data.len(),
                        remainder: self.remainder,
                    })
                } else {
                    Ok(DimensionedData {
                        data: self.data,
                        dims: self.dims,
                    })
                }
            }
        })*
    };
}

impl_dimensioned_data_builder! {
    DimensionedDataBuilder => DimensionedData<V, &'a [V]>,
    DimensionedDataBuilderMut => DimensionedData<V, &'a mut [V]>
}

// ---------------------------------------------------------------------------
// Internal: read config from compressed blob
// ---------------------------------------------------------------------------

struct ParsedConfig {
    config: Config,
    len: usize,
    dims: Vec<usize>,
    data_type: u8,
}

impl ParsedConfig {
    fn from_compressed(compressed_data: &[u8]) -> Self {
        let raw = unsafe {
            tensogram_sz3_sys::sz3_decompress_config(
                compressed_data.as_ptr().cast(),
                compressed_data.len(),
            )
        };
        let dims: Vec<usize> = (0..raw.N)
            .map(|i| unsafe { std::ptr::read(raw.dims.add(i as usize)) })
            .collect();
        unsafe {
            tensogram_sz3_sys::sz3_dealloc_size_t(raw.dims);
        }
        let SZ3_Config {
            num: len,
            dataType: data_type,
            ..
        } = raw;
        let config = Config::from_decompressed(raw);
        Self {
            config,
            len,
            dims,
            data_type,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API: compress / decompress
// ---------------------------------------------------------------------------

/// Compress `data` with the given error bound, returning a new `Vec<u8>`.
pub fn compress<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>>(
    data: &DimensionedData<V, T>,
    error_bound: ErrorBound,
) -> Result<Vec<u8>> {
    let config = Config::new(error_bound);
    compress_with_config(data, &config)
}

/// Compress `data` with a full [`Config`], returning a new `Vec<u8>`.
pub fn compress_with_config<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>>(
    data: &DimensionedData<V, T>,
    config: &Config,
) -> Result<Vec<u8>> {
    let mut compressed_data = Vec::new();
    compress_into_with_config(data, config, &mut compressed_data)?;
    Ok(compressed_data)
}

/// Compress `data` and **append** the result to `compressed_data`.
pub fn compress_into<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>>(
    data: &DimensionedData<V, T>,
    error_bound: ErrorBound,
    compressed_data: &mut Vec<u8>,
) -> Result<()> {
    let config = Config::new(error_bound);
    compress_into_with_config(data, &config, compressed_data)
}

/// Compress `data` with a full [`Config`] and **append** the result to
/// `compressed_data`.
pub fn compress_into_with_config<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>>(
    data: &DimensionedData<V, T>,
    config: &Config,
    compressed_data: &mut Vec<u8>,
) -> Result<()> {
    let block_size = config.block_size.unwrap_or(match data.dims().len() {
        1 => 128,
        2 => 16,
        _ => 6,
    });

    let raw_config = SZ3_Config {
        N: data.dims().len() as _,
        dims: data.dims.as_ptr() as _,
        num: data.len() as _,
        errorBoundMode: config.error_bound.code(),
        absErrorBound: config.error_bound.abs_bound(),
        relErrorBound: config.error_bound.rel_bound(),
        l2normErrorBound: config.error_bound.l2norm_bound(),
        psnrErrorBound: config.error_bound.psnr_bound(),
        cmprAlgo: config.compression_algorithm.code(),
        lorenzo: config.compression_algorithm.lorenzo(),
        lorenzo2: config.compression_algorithm.lorenzo_second_order(),
        regression: config.compression_algorithm.regression(),
        openmp: config.openmp,
        dataType: V::SZ_DATA_TYPE as _,
        blockSize: block_size as _,
        quantbinCnt: config.quantization_bincount as _,
    };

    let capacity: usize = unsafe { V::compress_size_bound(raw_config) };
    compressed_data.reserve(capacity);

    let len = unsafe {
        V::compress(
            raw_config,
            data.as_ptr(),
            compressed_data
                .spare_capacity_mut()
                .as_mut_ptr()
                .cast::<u8>(),
            capacity,
        )
    };
    unsafe { compressed_data.set_len(compressed_data.len() + len) };

    Ok(())
}

/// Decompress an SZ3 blob into a new `Vec<V>`, returning the recovered
/// [`Config`] and [`DimensionedData`].
pub fn decompress<V: SZ3Compressible, T: std::ops::Deref<Target = [u8]>>(
    compressed_data: T,
) -> Result<(Config, DimensionedData<V, Vec<V>>)> {
    let ParsedConfig {
        config,
        len,
        dims,
        data_type,
    } = ParsedConfig::from_compressed(&compressed_data);

    if data_type != V::SZ_DATA_TYPE {
        return Err(SZ3Error::DecompressedDataTypeMismatch);
    }

    let decompressed_data = unsafe {
        let mut decompressed_data: Vec<V> = Vec::with_capacity(len);

        V::decompress(
            compressed_data.as_ptr(),
            compressed_data.len(),
            decompressed_data
                .spare_capacity_mut()
                .as_mut_ptr()
                .cast::<V>(),
        );

        decompressed_data.set_len(len);
        decompressed_data
    };

    Ok((
        config,
        DimensionedData {
            data: decompressed_data,
            dims,
        },
    ))
}

/// Decompress an SZ3 blob into a pre-allocated [`DimensionedData`] buffer.
///
/// Returns an error if the data type or dimensions of the compressed blob do
/// not match the destination buffer.
pub fn decompress_into_dimensioned<
    V: SZ3Compressible,
    C: std::ops::Deref<Target = [u8]>,
    D: std::ops::DerefMut<Target = [V]>,
>(
    compressed_data: C,
    decompressed_data: &mut DimensionedData<V, D>,
) -> Result<Config> {
    let ParsedConfig {
        config,
        len,
        dims,
        data_type,
    } = ParsedConfig::from_compressed(&compressed_data);

    if data_type != V::SZ_DATA_TYPE {
        return Err(SZ3Error::DecompressedDataTypeMismatch);
    }

    if decompressed_data.dims() != dims.as_slice() {
        return Err(SZ3Error::DecompressedDimsMismatch {
            found: dims,
            expected: decompressed_data.dims.clone(),
        });
    }

    assert_eq!(decompressed_data.len(), len);

    unsafe {
        V::decompress(
            compressed_data.as_ptr(),
            compressed_data.len(),
            decompressed_data.data.as_mut_ptr(),
        );
    }

    Ok(config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_f64() {
        let data: Vec<f64> = (0..256)
            .map(|i| (i as f64 / 256.0 * std::f64::consts::PI).sin())
            .collect();
        let dimensioned = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&dimensioned, ErrorBound::Absolute(1e-6)).unwrap();
        let (_config, decompressed) = decompress::<f64, _>(&*compressed).unwrap();
        assert_eq!(decompressed.data().len(), data.len());
        for (orig, dec) in data.iter().zip(decompressed.data()) {
            assert!(
                (orig - dec).abs() <= 1e-6,
                "orig={orig}, dec={dec}, diff={}",
                (orig - dec).abs()
            );
        }
    }

    #[test]
    fn round_trip_f32() {
        let data: Vec<f32> = (0..256)
            .map(|i| (i as f32 / 256.0 * std::f32::consts::PI).sin())
            .collect();
        let dimensioned = DimensionedData::<f32, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&dimensioned, ErrorBound::Absolute(1e-4)).unwrap();
        let (_config, decompressed) = decompress::<f32, _>(&*compressed).unwrap();
        assert_eq!(decompressed.data().len(), data.len());
        for (orig, dec) in data.iter().zip(decompressed.data()) {
            assert!(
                (orig - dec).abs() <= 1e-4,
                "orig={orig}, dec={dec}, diff={}",
                (orig - dec).abs()
            );
        }
    }

    #[test]
    fn dimension_errors() {
        let data: Vec<f64> = vec![1.0; 100];
        let err = DimensionedData::<f64, _>::build(&data).dim(1);
        assert!(matches!(err.unwrap_err(), SZ3Error::OneSizedDimension));
        let err = DimensionedData::<f64, _>::build(&data).dim(7);
        assert!(matches!(
            err.unwrap_err(),
            SZ3Error::InvalidDimensionSize { .. }
        ));
        let err = DimensionedData::<f64, _>::build(&data)
            .dim(10)
            .unwrap()
            .finish();
        assert!(matches!(
            err.unwrap_err(),
            SZ3Error::UnderSpecifiedDimensions { .. }
        ));
    }

    #[test]
    fn round_trip_u8() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let d = DimensionedData::<u8, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(0.0)).unwrap();
        let (_, dec) = decompress::<u8, _>(&*compressed).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn round_trip_i8() {
        let data: Vec<i8> = (-128..127).map(|i| i as i8).collect();
        let d = DimensionedData::<i8, _>::build(&data)
            .dim(data.len())
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(0.0)).unwrap();
        let (_, dec) = decompress::<i8, _>(&*compressed).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn round_trip_u16() {
        let data: Vec<u16> = (0..256).map(|i| i as u16 * 100).collect();
        let d = DimensionedData::<u16, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(0.0)).unwrap();
        let (_, dec) = decompress::<u16, _>(&*compressed).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn round_trip_i16() {
        let data: Vec<i16> = (-128..128).map(|i| i as i16 * 50).collect();
        let d = DimensionedData::<i16, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(0.0)).unwrap();
        let (_, dec) = decompress::<i16, _>(&*compressed).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn round_trip_u32() {
        let data: Vec<u32> = (0..256).map(|i| i as u32 * 1000).collect();
        let d = DimensionedData::<u32, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(0.0)).unwrap();
        let (_, dec) = decompress::<u32, _>(&*compressed).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn round_trip_i32() {
        let data: Vec<i32> = (-128..128).map(|i: i32| i * 1000).collect();
        let d = DimensionedData::<i32, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(0.0)).unwrap();
        let (_, dec) = decompress::<i32, _>(&*compressed).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn round_trip_u64() {
        let data: Vec<u64> = (0..256).map(|i| i as u64 * 10000).collect();
        let d = DimensionedData::<u64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(0.0)).unwrap();
        let (_, dec) = decompress::<u64, _>(&*compressed).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn round_trip_i64() {
        let data: Vec<i64> = (-128..128).map(|i| i as i64 * 10000).collect();
        let d = DimensionedData::<i64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(0.0)).unwrap();
        let (_, dec) = decompress::<i64, _>(&*compressed).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn error_bound_relative() {
        let data: Vec<f64> = (0..256)
            .map(|i| (i as f64 / 256.0 * std::f64::consts::PI).sin() + 2.0)
            .collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::Relative(1e-4));
        let c = compress_with_config(&d, &cfg).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data().len(), data.len());
    }

    #[test]
    fn error_bound_psnr() {
        let data: Vec<f64> = (0..256)
            .map(|i| (i as f64 / 256.0 * std::f64::consts::PI).sin())
            .collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::PSNR(80.0));
        let c = compress_with_config(&d, &cfg).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data().len(), data.len());
    }

    #[test]
    fn error_bound_l2norm() {
        let data: Vec<f64> = (0..256)
            .map(|i| (i as f64 / 256.0 * std::f64::consts::PI).sin())
            .collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::L2Norm(1e-3));
        let c = compress_with_config(&d, &cfg).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data().len(), data.len());
    }

    #[test]
    fn error_bound_abs_and_rel() {
        let data: Vec<f64> = (0..256)
            .map(|i| (i as f64 / 256.0 * std::f64::consts::PI).sin() + 2.0)
            .collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::AbsoluteAndRelative {
            absolute_bound: 1e-4,
            relative_bound: 1e-3,
        });
        let c = compress_with_config(&d, &cfg).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data().len(), data.len());
    }

    #[test]
    fn error_bound_abs_or_rel() {
        let data: Vec<f64> = (0..256)
            .map(|i| (i as f64 / 256.0 * std::f64::consts::PI).sin() + 2.0)
            .collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::AbsoluteOrRelative {
            absolute_bound: 1e-4,
            relative_bound: 1e-3,
        });
        let c = compress_with_config(&d, &cfg).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data().len(), data.len());
    }

    #[test]
    fn algo_interpolation() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::Absolute(1e-6))
            .compression_algorithm(CompressionAlgorithm::interpolation());
        let c = compress_with_config(&d, &cfg).unwrap();
        let (dc, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data().len(), data.len());
        assert!(matches!(
            dc.compression_algorithm,
            CompressionAlgorithm::Interpolation
        ));
    }

    #[test]
    fn algo_lorenzo_regression() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::Absolute(1e-6))
            .compression_algorithm(CompressionAlgorithm::lorenzo_regression());
        let c = compress_with_config(&d, &cfg).unwrap();
        let (dc, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data().len(), data.len());
        assert!(matches!(
            dc.compression_algorithm,
            CompressionAlgorithm::LorenzoRegression { .. }
        ));
    }

    #[test]
    fn algo_lossless() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::Absolute(0.0))
            .compression_algorithm(CompressionAlgorithm::lossless());
        let c = compress_with_config(&d, &cfg).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn config_quantization_bincount() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::Absolute(1e-6)).quantization_bincount(1024);
        let c = compress_with_config(&d, &cfg).unwrap();
        let (dc, _) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dc.quantization_bincount, 1024);
    }

    #[test]
    fn config_block_size() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::Absolute(1e-6)).block_size(64);
        let c = compress_with_config(&d, &cfg).unwrap();
        let (dc, _) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dc.block_size, Some(64));
    }

    #[test]
    fn config_automatic_block_size() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::Absolute(1e-6))
            .block_size(64)
            .automatic_block_size();
        assert!(cfg.block_size.is_none());
        let c = compress_with_config(&d, &cfg).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.data().len(), data.len());
    }

    #[test]
    fn config_error_bound_setter() {
        let cfg = Config::new(ErrorBound::Absolute(1.0)).error_bound(ErrorBound::Relative(0.5));
        assert!(matches!(cfg.error_bound, ErrorBound::Relative(_)));
    }

    #[test]
    fn dimensioned_data_accessors() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dd = DimensionedData::<f64, _>::build(&data)
            .dim(2)
            .unwrap()
            .dim(3)
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(dd.dims(), &[2, 3]);
        assert_eq!(dd.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let owned = dd.into_data();
        assert_eq!(owned, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn dimensioned_data_build_mut() {
        let mut data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut dd = DimensionedData::<f64, _>::build_mut(&mut data)
            .dim(4)
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(dd.data(), &[1.0, 2.0, 3.0, 4.0]);
        dd.data_mut()[0] = 99.0;
        assert_eq!(dd.data()[0], 99.0);
    }

    #[test]
    fn dimensioned_data_remainder_dim() {
        let data: Vec<f64> = vec![1.0; 120];
        let dd = DimensionedData::<f64, _>::build(&data)
            .dim(10)
            .unwrap()
            .remainder_dim()
            .unwrap();
        assert_eq!(dd.dims(), &[10, 12]);
        assert_eq!(dd.data().len(), 120);
    }

    #[test]
    fn dimensioned_data_singleton() {
        let data: Vec<f64> = vec![42.0];
        let dd = DimensionedData::<f64, _>::build(&data)
            .dim(1)
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(dd.dims(), &[1]);
        assert_eq!(dd.data(), &[42.0]);
    }

    #[test]
    fn decompress_into_dimensioned_round_trip() {
        let data: Vec<f64> = (0..256)
            .map(|i| (i as f64 / 256.0 * std::f64::consts::PI).sin())
            .collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(1e-6)).unwrap();
        let mut output = vec![0.0f64; 256];
        let mut out_dim = DimensionedData::<f64, _>::build_mut(&mut output)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let _cfg = decompress_into_dimensioned(&*compressed, &mut out_dim).unwrap();
        for (orig, dec) in data.iter().zip(out_dim.data()) {
            assert!((orig - dec).abs() <= 1e-6);
        }
    }

    #[test]
    fn decompress_data_type_mismatch() {
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let d = DimensionedData::<f32, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(1e-4)).unwrap();
        let result = decompress::<f64, _>(&*compressed);
        assert!(matches!(
            result.unwrap_err(),
            SZ3Error::DecompressedDataTypeMismatch
        ));
    }

    #[test]
    fn decompress_into_dims_mismatch() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(1e-6)).unwrap();
        let mut output = vec![0.0f64; 256];
        let mut out_dim = DimensionedData::<f64, _>::build_mut(&mut output)
            .dim(16)
            .unwrap()
            .dim(16)
            .unwrap()
            .finish()
            .unwrap();
        let result = decompress_into_dimensioned(&*compressed, &mut out_dim);
        assert!(matches!(
            result.unwrap_err(),
            SZ3Error::DecompressedDimsMismatch { .. }
        ));
    }

    #[test]
    fn decompress_into_type_mismatch() {
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let d = DimensionedData::<f32, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let compressed = compress(&d, ErrorBound::Absolute(1e-4)).unwrap();
        let mut output = vec![0.0f64; 256];
        let mut out_dim = DimensionedData::<f64, _>::build_mut(&mut output)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let result = decompress_into_dimensioned(&*compressed, &mut out_dim);
        assert!(matches!(
            result.unwrap_err(),
            SZ3Error::DecompressedDataTypeMismatch
        ));
    }

    #[test]
    fn compress_into_appends() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let mut buf = vec![0xAA, 0xBB, 0xCC];
        compress_into(&d, ErrorBound::Absolute(1e-6), &mut buf).unwrap();
        assert_eq!(&buf[..3], &[0xAA, 0xBB, 0xCC]);
        assert!(buf.len() > 3);
        let (_, dec) = decompress::<f64, _>(&buf[3..]).unwrap();
        assert_eq!(dec.data().len(), data.len());
    }

    #[test]
    fn compress_into_with_config_appends() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(256)
            .unwrap()
            .finish()
            .unwrap();
        let cfg = Config::new(ErrorBound::Absolute(1e-6))
            .compression_algorithm(CompressionAlgorithm::lossless());
        let mut buf = vec![0xDD, 0xEE];
        compress_into_with_config(&d, &cfg, &mut buf).unwrap();
        assert_eq!(&buf[..2], &[0xDD, 0xEE]);
        let (_, dec) = decompress::<f64, _>(&buf[2..]).unwrap();
        assert_eq!(dec.data(), data.as_slice());
    }

    #[test]
    fn round_trip_2d() {
        let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(16)
            .unwrap()
            .dim(16)
            .unwrap()
            .finish()
            .unwrap();
        let c = compress(&d, ErrorBound::Absolute(1e-6)).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.dims(), &[16, 16]);
    }

    #[test]
    fn round_trip_3d() {
        let data: Vec<f64> = (0..216).map(|i| i as f64).collect();
        let d = DimensionedData::<f64, _>::build(&data)
            .dim(6)
            .unwrap()
            .dim(6)
            .unwrap()
            .dim(6)
            .unwrap()
            .finish()
            .unwrap();
        let c = compress(&d, ErrorBound::Absolute(1e-6)).unwrap();
        let (_, dec) = decompress::<f64, _>(&*c).unwrap();
        assert_eq!(dec.dims(), &[6, 6, 6]);
    }

    #[test]
    fn algo_constructors() {
        assert!(matches!(
            CompressionAlgorithm::interpolation(),
            CompressionAlgorithm::Interpolation
        ));
        assert!(matches!(
            CompressionAlgorithm::interpolation_lorenzo(),
            CompressionAlgorithm::InterpolationLorenzo
        ));
        assert!(matches!(
            CompressionAlgorithm::lorenzo_regression(),
            CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: false,
                regression: true
            }
        ));
        let a =
            CompressionAlgorithm::lorenzo_regression_custom(Some(false), Some(true), Some(false));
        assert!(matches!(
            a,
            CompressionAlgorithm::LorenzoRegression {
                lorenzo: false,
                lorenzo_second_order: true,
                regression: false
            }
        ));
        let a = CompressionAlgorithm::lorenzo_regression_custom(None, None, None);
        assert!(matches!(
            a,
            CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: false,
                regression: true
            }
        ));
        assert!(matches!(
            CompressionAlgorithm::biology_molecular_data(),
            CompressionAlgorithm::BiologyMolecularData
        ));
        assert!(matches!(
            CompressionAlgorithm::biology_molecular_data_gromacs_xtc(),
            CompressionAlgorithm::BiologyMolecularDataGromacsXtc
        ));
        assert!(matches!(
            CompressionAlgorithm::no_prediction(),
            CompressionAlgorithm::NoPrediction
        ));
        assert!(matches!(
            CompressionAlgorithm::lossless(),
            CompressionAlgorithm::Lossless
        ));
        assert!(matches!(
            CompressionAlgorithm::default(),
            CompressionAlgorithm::InterpolationLorenzo
        ));
    }

    #[test]
    fn error_display_messages() {
        let e = SZ3Error::OneSizedDimension;
        assert!(format!("{e}").contains("size one"));
        let e = SZ3Error::DecompressedDataTypeMismatch;
        assert!(format!("{e}").contains("different data type"));
        let e = SZ3Error::DecompressedDimsMismatch {
            found: vec![256],
            expected: vec![16, 16],
        };
        assert!(format!("{e}").contains("[256]"));
        let e = SZ3Error::InvalidDimensionSize {
            dims: vec![10],
            len: 100,
            wanted: 7,
            remainder: 10,
        };
        assert!(format!("{e}").contains("7"));
        let e = SZ3Error::UnderSpecifiedDimensions {
            dims: vec![10],
            len: 100,
            remainder: 10,
        };
        assert!(format!("{e}").contains("10"));
    }
}
