// (C) Copyright 2024- ECMWF and individual contributors.
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
            Self::Interpolation { .. } => tensogram_sz3_sys::SZ3::ALGO_ALGO_INTERP,
            Self::InterpolationLorenzo { .. } => tensogram_sz3_sys::SZ3::ALGO_ALGO_INTERP_LORENZO,
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

    pub fn interpolation() -> Self {
        Self::Interpolation
    }

    pub fn interpolation_lorenzo() -> Self {
        Self::InterpolationLorenzo
    }

    pub fn lorenzo_regression() -> Self {
        Self::LorenzoRegression {
            lorenzo: true,
            lorenzo_second_order: false,
            regression: true,
        }
    }

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

    pub fn biology_molecular_data() -> Self {
        Self::BiologyMolecularData
    }

    pub fn biology_molecular_data_gromacs_xtc() -> Self {
        Self::BiologyMolecularDataGromacsXtc
    }

    pub fn no_prediction() -> Self {
        Self::NoPrediction
    }

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
            Self::AbsoluteOrRelative { absolute_bound, .. } => *absolute_bound,
            Self::AbsoluteAndRelative { absolute_bound, .. } => *absolute_bound,
            _ => 0.0,
        }
    }

    fn rel_bound(&self) -> f64 {
        match self {
            Self::Relative(bound) => *bound,
            Self::AbsoluteOrRelative { relative_bound, .. } => *relative_bound,
            Self::AbsoluteAndRelative { relative_bound, .. } => *relative_bound,
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

#[derive(Clone, Debug)]
pub struct Config {
    compression_algorithm: CompressionAlgorithm,
    error_bound: ErrorBound,
    openmp: bool,
    quantization_bincount: u32,
    block_size: Option<u32>,
}

impl Config {
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

    pub fn compression_algorithm(mut self, compression_algorithm: CompressionAlgorithm) -> Self {
        self.compression_algorithm = compression_algorithm;
        self
    }

    pub fn error_bound(mut self, error_bound: ErrorBound) -> Self {
        self.error_bound = error_bound;
        self
    }

    #[cfg(feature = "openmp")]
    pub fn openmp(mut self, openmp: bool) -> Self {
        self.openmp = openmp;
        self
    }

    pub fn quantization_bincount(mut self, quantization_bincount: u32) -> Self {
        self.quantization_bincount = quantization_bincount;
        self
    }

    pub fn block_size(mut self, block_size: u32) -> Self {
        self.block_size = Some(block_size);
        self
    }

    pub fn automatic_block_size(mut self) -> Self {
        self.block_size = None;
        self
    }
}

// ---------------------------------------------------------------------------
// SZ3Compressible trait + sealed implementation
// ---------------------------------------------------------------------------

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

#[derive(Clone, Debug)]
pub struct DimensionedData<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>> {
    data: T,
    dims: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct DimensionedDataBuilder<'a, V> {
    data: &'a [V],
    dims: Vec<usize>,
    remainder: usize,
}

#[derive(Debug)]
pub struct DimensionedDataBuilderMut<'a, V> {
    data: &'a mut [V],
    dims: Vec<usize>,
    remainder: usize,
}

impl<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>> DimensionedData<V, T> {
    pub fn build<'a>(data: &'a T) -> DimensionedDataBuilder<'a, V> {
        DimensionedDataBuilder {
            data,
            dims: vec![],
            remainder: data.len(),
        }
    }

    pub fn data(&self) -> &[V] {
        &self.data
    }

    pub fn into_data(self) -> T {
        self.data
    }

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
    pub fn build_mut<'a>(data: &'a mut T) -> DimensionedDataBuilderMut<'a, V> {
        DimensionedDataBuilderMut {
            remainder: data.len(),
            data,
            dims: vec![],
        }
    }

    pub fn data_mut(&mut self) -> &mut [V] {
        &mut self.data
    }
}

// ---------------------------------------------------------------------------
// SZ3Error
// ---------------------------------------------------------------------------

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
            pub fn dim(mut self, length: usize) -> Result<Self> {
                if length == 1 {
                    if self.dims.is_empty() && self.remainder == 1 {
                        self.dims.push(1);
                        Ok(self)
                    } else {
                        Err(SZ3Error::OneSizedDimension)
                    }
                } else if self.remainder.rem_euclid(length) != 0 {
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

            pub fn remainder_dim(self) -> Result<$data> {
                let remainder = self.remainder;
                self.dim(remainder)?.finish()
            }

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

struct DecompressedConfig {
    config: Config,
    len: usize,
    dims: Vec<usize>,
    data_type: u8,
}

impl DecompressedConfig {
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

pub fn compress<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>>(
    data: &DimensionedData<V, T>,
    error_bound: ErrorBound,
) -> Result<Vec<u8>> {
    let config = Config::new(error_bound);
    compress_with_config(data, &config)
}

pub fn compress_with_config<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>>(
    data: &DimensionedData<V, T>,
    config: &Config,
) -> Result<Vec<u8>> {
    let mut compressed_data = Vec::new();
    compress_into_with_config(data, config, &mut compressed_data)?;
    Ok(compressed_data)
}

pub fn compress_into<V: SZ3Compressible, T: std::ops::Deref<Target = [V]>>(
    data: &DimensionedData<V, T>,
    error_bound: ErrorBound,
    compressed_data: &mut Vec<u8>,
) -> Result<()> {
    let config = Config::new(error_bound);
    compress_into_with_config(data, &config, compressed_data)
}

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

pub fn decompress<V: SZ3Compressible, T: std::ops::Deref<Target = [u8]>>(
    compressed_data: T,
) -> Result<(Config, DimensionedData<V, Vec<V>>)> {
    let DecompressedConfig {
        config,
        len,
        dims,
        data_type,
    } = DecompressedConfig::from_compressed(&compressed_data);

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

pub fn decompress_into_dimensioned<
    V: SZ3Compressible,
    C: std::ops::Deref<Target = [u8]>,
    D: std::ops::DerefMut<Target = [V]>,
>(
    compressed_data: C,
    decompressed_data: &mut DimensionedData<V, D>,
) -> Result<Config> {
    let DecompressedConfig {
        config,
        len,
        dims,
        data_type,
    } = DecompressedConfig::from_compressed(&compressed_data);

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
        // Size 1 dimension on non-singleton data is an error
        let err = DimensionedData::<f64, _>::build(&data).dim(1);
        assert!(err.is_err());

        // Non-dividing dimension
        let err = DimensionedData::<f64, _>::build(&data).dim(7);
        assert!(err.is_err());

        // Under-specified
        let err = DimensionedData::<f64, _>::build(&data)
            .dim(10)
            .unwrap()
            .finish();
        assert!(err.is_err());
    }
}
