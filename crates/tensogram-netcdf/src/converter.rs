// NetCDF → Tensogram conversion logic.
// Populated incrementally: Task 6 (basic), Task 7 (dtypes),
// Task 8 (metadata), Task 9 (packing/fill/time), Task 10 (split-by),
// Task 11 (--cf), Task 12 (edge cases).

use std::path::Path;

use crate::error::NetcdfError;

/// How to group NetCDF variables into Tensogram messages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SplitBy {
    /// All variables from one file → one Tensogram message with N objects.
    /// This is the default.
    #[default]
    File,
    /// Each variable → its own Tensogram message.
    Variable,
    /// Split along the unlimited (record) dimension — one message per record.
    /// Errors if the file has no unlimited dimension.
    Record,
}

/// Options for NetCDF → Tensogram conversion.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    pub split_by: SplitBy,
    pub encode_options: tensogram_core::EncodeOptions,
    pub cf: bool,
}

/// Convert all variables from a NetCDF file into Tensogram wire bytes.
///
/// Each variable in the file becomes one data object in the output message(s).
/// The grouping strategy is controlled by `options.split_by`.
///
/// NetCDF attributes are stored under `base[i]["netcdf"]` for each object.
/// When `options.cf` is `true`, CF convention attributes are also parsed
/// into `base[i]["cf"]`.
///
/// Sub-groups (NetCDF-4) are not processed — a warning is emitted to stderr
/// and only root-group variables are converted.
pub fn convert_netcdf_file(
    path: &Path,
    _options: &ConvertOptions,
) -> Result<Vec<Vec<u8>>, NetcdfError> {
    // TODO (Task 6): implement basic conversion
    let _ = netcdf::open(path)?;
    Err(NetcdfError::NoVariables)
}
