// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Tensogram Python bindings via PyO3.
//!
//! Exposes the tensogram library as a native Python module with
//! numpy integration. All tensor data crosses the boundary as numpy arrays.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;

use numpy::PyArrayMethods;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyUserWarning, PyValueError};
// `PyFileNotFoundError` is only referenced from the GRIB/NetCDF converter
// error-mapping helpers, which are themselves cfg-gated.  Gate the import
// so `--no-default-features --features async` builds do not warn.
#[cfg(any(feature = "grib", feature = "netcdf"))]
use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyDict, PyList};

#[cfg(feature = "async")]
use std::sync::Arc;

use tensogram_lib::validate::{
    ValidateOptions, ValidationLevel, validate_file as core_validate_file, validate_message,
};
use tensogram_lib::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
    HashAlgorithm, RESERVED_KEY, RemoteScanOptions, StreamingEncoder, TensogramError,
    TensogramFile, decode, decode_descriptors, decode_metadata, decode_object, decode_range,
    encode, encode_pre_encoded, scan,
};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn to_py_err(e: TensogramError) -> PyErr {
    match e {
        TensogramError::Framing(msg) => PyValueError::new_err(format!("FramingError: {msg}")),
        TensogramError::Metadata(msg) => PyValueError::new_err(format!("MetadataError: {msg}")),
        TensogramError::Encoding(msg) => PyValueError::new_err(format!("EncodingError: {msg}")),
        TensogramError::Compression(msg) => {
            PyValueError::new_err(format!("CompressionError: {msg}"))
        }
        TensogramError::Object(msg) => PyValueError::new_err(format!("ObjectError: {msg}")),
        TensogramError::Io(e) => PyIOError::new_err(format!("{e}")),
        TensogramError::HashMismatch { expected, actual } => PyRuntimeError::new_err(format!(
            "HashMismatch: expected={expected}, actual={actual}"
        )),
        TensogramError::Remote(msg) => PyIOError::new_err(format!("RemoteError: {msg}")),
    }
}

// ---------------------------------------------------------------------------
// Open-method helpers
// ---------------------------------------------------------------------------

/// Parse a `storage_options` PyDict into the `BTreeMap<String, String>` shape
/// expected by `TensogramFile::open_remote*`.  Errors at the call site so
/// async entry points fail before any await.
fn parse_storage_options(
    storage_options: Option<&Bound<'_, PyDict>>,
) -> PyResult<BTreeMap<String, String>> {
    match storage_options {
        Some(dict) => {
            let mut map = BTreeMap::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let val: String = v.extract::<String>().or_else(|_| {
                    v.str().map(|s| s.to_string()).map_err(|_| {
                        PyValueError::new_err(format!(
                            "storage_options value for key '{key}' must be convertible to string"
                        ))
                    })
                })?;
                map.insert(key, val);
            }
            Ok(map)
        }
        None => Ok(BTreeMap::new()),
    }
}

/// Always pass explicit `RemoteScanOptions` to the Rust layer so the
/// caller's `bidirectional=False` opt-out survives any future change
/// to `RemoteScanOptions::default()` — collapsing `false` to `None`
/// would silently flip with the default.
fn scan_opts_for(bidirectional: bool) -> Option<RemoteScanOptions> {
    Some(RemoteScanOptions { bidirectional })
}

// ---------------------------------------------------------------------------
// CBOR ↔ Python conversion
// ---------------------------------------------------------------------------

fn cbor_to_py(py: Python<'_>, val: &ciborium::Value) -> PyResult<PyObject> {
    match val {
        ciborium::Value::Text(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            let v = i64::try_from(n)
                .map_err(|_| PyValueError::new_err(format!("CBOR integer {n} out of i64 range")))?;
            Ok(v.into_pyobject(py)?.into_any().unbind())
        }
        ciborium::Value::Float(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        ciborium::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        ciborium::Value::Null => Ok(py.None()),
        ciborium::Value::Array(arr) => {
            let items: PyResult<Vec<PyObject>> = arr.iter().map(|v| cbor_to_py(py, v)).collect();
            let list = PyList::new(py, items?)?;
            Ok(list.into_any().unbind())
        }
        ciborium::Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries {
                let key = cbor_to_py(py, k)?;
                let val = cbor_to_py(py, v)?;
                dict.set_item(key, val)?;
            }
            Ok(dict.into_any().unbind())
        }
        ciborium::Value::Bytes(b) => Ok(PyBytes::new(py, b).into_any().unbind()),
        _ => Ok(format!("{val:?}").into_pyobject(py)?.into_any().unbind()),
    }
}

fn py_to_cbor(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<ciborium::Value> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(ciborium::Value::Text(s));
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(ciborium::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ciborium::Value::Integer(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(ciborium::Value::Float(f));
    }
    if obj.is_none() {
        return Ok(ciborium::Value::Null);
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut entries = Vec::new();
        for (k, v) in dict.iter() {
            entries.push((py_to_cbor(&k)?, py_to_cbor(&v)?));
        }
        return Ok(ciborium::Value::Map(entries));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let items: PyResult<Vec<_>> = list.iter().map(|v| py_to_cbor(&v)).collect();
        return Ok(ciborium::Value::Array(items?));
    }
    Err(PyValueError::new_err(format!(
        "cannot convert {} to CBOR",
        obj.get_type().name()?
    )))
}

fn extra_to_py(py: Python<'_>, extra: &BTreeMap<String, ciborium::Value>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (k, v) in extra {
        dict.set_item(k, cbor_to_py(py, v)?)?;
    }
    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// PyDataObjectDescriptor — wraps DataObjectDescriptor (merged tensor + encoding)
// ---------------------------------------------------------------------------

/// Describes a single data object in a Tensogram message.
///
/// Contains both tensor metadata (shape, dtype, strides) and encoding
/// pipeline info (byte_order, encoding, filter, compression, params).
/// Returned alongside decoded numpy arrays from ``decode()`` and friends.
#[pyclass(name = "DataObjectDescriptor", from_py_object)]
#[derive(Clone)]
struct PyDataObjectDescriptor {
    inner: DataObjectDescriptor,
}

#[pymethods]
impl PyDataObjectDescriptor {
    // ── Tensor metadata ──

    #[getter]
    fn obj_type(&self) -> &str {
        &self.inner.obj_type
    }

    #[getter]
    fn ndim(&self) -> u64 {
        self.inner.ndim
    }

    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.inner.shape.clone()
    }

    #[getter]
    fn strides(&self) -> Vec<u64> {
        self.inner.strides.clone()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype.to_string()
    }

    // ── Encoding pipeline ──

    #[getter]
    fn byte_order(&self) -> &str {
        match self.inner.byte_order {
            ByteOrder::Big => "big",
            ByteOrder::Little => "little",
        }
    }

    #[getter]
    fn encoding(&self) -> &str {
        &self.inner.encoding
    }

    #[getter]
    fn filter(&self) -> &str {
        &self.inner.filter
    }

    #[getter]
    fn compression(&self) -> &str {
        &self.inner.compression
    }

    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<PyObject> {
        extra_to_py(py, &self.inner.params)
    }

    /// **Deprecated in v3.**  The per-object hash moved from the
    /// CBOR descriptor to the frame footer's inline slot (see
    /// `plans/WIRE_FORMAT.md` §2.4).  This getter returns `None`
    /// unconditionally; use `Message.object_inline_hashes()` or
    /// `Message.object_hash(i)` to read the inline slot.
    #[getter]
    fn hash(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(py.None())
    }

    fn __repr__(&self) -> String {
        format!(
            "DataObjectDescriptor(type='{}', shape={:?}, dtype='{}', encoding='{}', compression='{}')",
            self.inner.obj_type,
            self.inner.shape,
            self.inner.dtype,
            self.inner.encoding,
            self.inner.compression,
        )
    }
}

// ---------------------------------------------------------------------------
// PyMetadata — wraps GlobalMetadata
//
// Python name kept as "Metadata" for backward compatibility with callers that
// already do `tensogram.Metadata`. Internals now use GlobalMetadata (v2).
// Per-object metadata lives in `base[i]`, with `_reserved_.tensor` auto-populated
// by the encoder. Client-writable message-level metadata goes in `extra`.
// ---------------------------------------------------------------------------

/// Message-level metadata.
///
/// Access wire-format version via ``meta.version`` (always ``3`` in the
/// current library — sourced from the preamble, not from any CBOR
/// field).  Per-object metadata lives in ``meta.base``; extra keys are
/// reachable via dict syntax (``meta["mars"]``, ``"mars" in meta``,
/// ``meta.extra``).
#[pyclass(name = "Metadata", from_py_object)]
#[derive(Clone)]
struct PyMetadata {
    inner: GlobalMetadata,
}

#[pymethods]
impl PyMetadata {
    /// Wire-format version of the message this metadata came from.
    ///
    /// The value is sourced from the preamble (see
    /// `plans/WIRE_FORMAT.md` §3) — not from any CBOR key.  In the
    /// current library this always returns :data:`tensogram.WIRE_VERSION`
    /// (``3``) because the decoder rejects any other preamble version.
    #[getter]
    fn version(&self) -> u16 {
        tensogram_lib::WIRE_VERSION
    }

    /// Per-object metadata list.  ``meta.base[i]`` is a dict of
    /// ALL metadata for the *i*-th data object.
    #[getter]
    fn base(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items: Vec<PyObject> = self
            .inner
            .base
            .iter()
            .map(|m| extra_to_py(py, m))
            .collect::<PyResult<Vec<_>>>()?;
        let list = pyo3::types::PyList::new(py, items)?;
        Ok(list.into_any().unbind())
    }

    /// Library-reserved metadata (read-only — provenance info).
    #[getter]
    fn reserved(&self, py: Python<'_>) -> PyResult<PyObject> {
        extra_to_py(py, &self.inner.reserved)
    }

    /// Client-writable extra metadata (message-level annotations).
    #[getter]
    fn extra(&self, py: Python<'_>) -> PyResult<PyObject> {
        extra_to_py(py, &self.inner.extra)
    }

    /// Dictionary-style access: ``meta["key"]``.
    ///
    /// Searches ``base`` entries first (first match, skipping ``_reserved_``
    /// keys within each entry), then ``extra``.  Raises ``KeyError`` if the
    /// key is not found.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        // Search base entries (skip the "_reserved_" key within each entry)
        for entry in &self.inner.base {
            if key == RESERVED_KEY {
                continue;
            }
            if let Some(v) = entry.get(key) {
                return cbor_to_py(py, v);
            }
        }
        // Then search extra
        match self.inner.extra.get(key) {
            Some(v) => cbor_to_py(py, v),
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
        }
    }

    /// Membership test: ``"key" in meta``.
    fn __contains__(&self, key: &str) -> bool {
        if key != RESERVED_KEY {
            for entry in &self.inner.base {
                if entry.contains_key(key) {
                    return true;
                }
            }
        }
        self.inner.extra.contains_key(key)
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(version={}, base_len={}, extra_keys={:?})",
            tensogram_lib::WIRE_VERSION,
            self.inner.base.len(),
            self.inner.extra.keys().collect::<Vec<_>>()
        )
    }
}

// ---------------------------------------------------------------------------
// PyTensogramFile — wraps TensogramFile
// ---------------------------------------------------------------------------

/// File-based Tensogram container.
///
/// Supports context manager protocol::
///
///     with tensogram.TensogramFile.create("out.tgm") as f:
///         f.append(meta, [(desc, data)])
///
///     with tensogram.TensogramFile.open("out.tgm") as f:
///         meta, objects = f.decode_message(0)
///
/// Use ``len(f)`` to get the message count.
#[pyclass(name = "TensogramFile")]
struct PyTensogramFile {
    file: TensogramFile,
}

#[pymethods]
impl PyTensogramFile {
    #[staticmethod]
    #[pyo3(signature = (source, *, bidirectional=true))]
    fn open(py: Python<'_>, source: &str, bidirectional: bool) -> PyResult<Self> {
        let scan_opts = scan_opts_for(bidirectional);
        let source = source.to_string();
        let file = py
            .detach(|| TensogramFile::open_source(&source, scan_opts))
            .map_err(to_py_err)?;
        Ok(PyTensogramFile { file })
    }

    #[staticmethod]
    #[pyo3(signature = (source, storage_options=None, *, bidirectional=true))]
    fn open_remote(
        py: Python<'_>,
        source: &str,
        storage_options: Option<&Bound<'_, PyDict>>,
        bidirectional: bool,
    ) -> PyResult<Self> {
        let opts = parse_storage_options(storage_options)?;
        let scan_opts = scan_opts_for(bidirectional);
        let source = source.to_string();
        let file = py
            .detach(|| TensogramFile::open_remote(&source, &opts, scan_opts))
            .map_err(to_py_err)?;
        Ok(PyTensogramFile { file })
    }

    /// Create a new file (truncates if exists).
    #[staticmethod]
    fn create(path: &str) -> PyResult<Self> {
        let file = TensogramFile::create(path).map_err(to_py_err)?;
        Ok(PyTensogramFile { file })
    }

    /// Number of valid messages in the file.
    fn message_count(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.file.message_count()).map_err(to_py_err)
    }

    /// Append one message.
    ///
    /// Args:
    ///     global_meta_dict: ``{"base": [...], ...}`` with any extra keys.
    ///     descriptors_and_data: list of ``(descriptor_dict, data)`` pairs.
    ///         Each descriptor_dict requires ``type``, ``shape``, ``dtype`` and
    ///         optionally ``byte_order``, ``encoding``, ``filter``, ``compression``.
    ///     hash: ``"xxh3"`` (default) or ``None`` to skip hashing.
    ///     threads: thread budget (0 = sequential / env fallback).
    #[pyo3(
        signature = (
            global_meta_dict,
            descriptors_and_data,
            hash=Some("xxh3"),
            threads=0,
            allow_nan=false,
            allow_inf=false,
            nan_mask_method=None,
            pos_inf_mask_method=None,
            neg_inf_mask_method=None,
            small_mask_threshold_bytes=None,
            create_header_hashes=None,
            create_footer_hashes=None,
        )
    )]
    #[allow(clippy::too_many_arguments)]
    fn append(
        &mut self,
        py: Python<'_>,
        global_meta_dict: &Bound<'_, PyDict>,
        descriptors_and_data: &Bound<'_, PyList>,
        hash: Option<&str>,
        threads: u32,
        allow_nan: bool,
        allow_inf: bool,
        nan_mask_method: Option<&str>,
        pos_inf_mask_method: Option<&str>,
        neg_inf_mask_method: Option<&str>,
        small_mask_threshold_bytes: Option<usize>,
        create_header_hashes: Option<bool>,
        create_footer_hashes: Option<bool>,
    ) -> PyResult<()> {
        let global_meta = dict_to_global_metadata(global_meta_dict)?;
        let pairs = extract_descriptor_data_pairs(py, descriptors_and_data)?;
        let refs: Vec<(&DataObjectDescriptor, &[u8])> =
            pairs.iter().map(|(d, b)| (d, b.as_slice())).collect();

        let options = make_encode_options_full(
            hash,
            threads,
            allow_nan,
            allow_inf,
            nan_mask_method,
            pos_inf_mask_method,
            neg_inf_mask_method,
            small_mask_threshold_bytes,
            create_header_hashes,
            create_footer_hashes,
        )?;
        py.detach(|| self.file.append(&global_meta, &refs, &options))
            .map_err(to_py_err)
    }

    /// Decode message at *index* → ``Message(metadata, objects)``.
    ///
    /// Set *verify_hash* to ``True`` to verify payload integrity (default ``False``).
    /// Set *native_byte_order* to ``False`` to get wire-order bytes (default ``True``).
    /// Set *threads* to ``N`` to spend a budget of ``N`` threads on decoding
    /// (0 = sequential / env fallback).
    #[pyo3(
        signature = (
            index,
            verify_hash=None,
            native_byte_order=true,
            threads=0,
            restore_non_finite=true,
        )
    )]
    fn decode_message(
        &self,
        py: Python<'_>,
        index: usize,
        verify_hash: Option<bool>,
        native_byte_order: bool,
        threads: u32,
        restore_non_finite: bool,
    ) -> PyResult<PyObject> {
        let options = DecodeOptions {
            verify_hash: verify_hash.unwrap_or(false),
            native_byte_order,
            threads,
            restore_non_finite,
            ..Default::default()
        };
        let (global_meta, data_objects) = py
            .detach(|| self.file.decode_message(index, &options))
            .map_err(to_py_err)?;
        let result_list = data_objects_to_python(py, &data_objects)?;
        pack_message(py, PyMetadata { inner: global_meta }, result_list)
    }

    fn file_decode_metadata(&self, py: Python<'_>, msg_index: usize) -> PyResult<PyObject> {
        let meta = py
            .detach(|| self.file.decode_metadata(msg_index))
            .map_err(to_py_err)?;
        Ok(PyMetadata { inner: meta }
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    fn file_decode_descriptors(&self, py: Python<'_>, msg_index: usize) -> PyResult<PyObject> {
        let (meta, descriptors) = py
            .detach(|| self.file.decode_descriptors(msg_index))
            .map_err(to_py_err)?;
        let desc_list: Vec<PyObject> = descriptors
            .iter()
            .map(|d| {
                Ok(PyDataObjectDescriptor { inner: d.clone() }
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
            })
            .collect::<PyResult<_>>()?;
        let result = PyDict::new(py);
        result.set_item("metadata", PyMetadata { inner: meta }.into_pyobject(py)?)?;
        result.set_item("descriptors", PyList::new(py, desc_list)?)?;
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature = (msg_index, obj_index, verify_hash=false, native_byte_order=true, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn file_decode_object(
        &self,
        py: Python<'_>,
        msg_index: usize,
        obj_index: usize,
        verify_hash: bool,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<PyObject> {
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            threads,
            ..Default::default()
        };
        let (meta, desc, data) = py
            .detach(|| self.file.decode_object(msg_index, obj_index, &options))
            .map_err(to_py_err)?;
        let arr = bytes_to_numpy(py, &desc, &data)?;
        let py_desc = PyDataObjectDescriptor { inner: desc }
            .into_pyobject(py)?
            .into_any()
            .unbind();
        let result = PyDict::new(py);
        result.set_item("metadata", PyMetadata { inner: meta }.into_pyobject(py)?)?;
        result.set_item("descriptor", py_desc)?;
        result.set_item("data", arr)?;
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature = (msg_index, obj_index, ranges, join=false, verify_hash=false, native_byte_order=true, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn file_decode_range(
        &self,
        py: Python<'_>,
        msg_index: usize,
        obj_index: usize,
        ranges: Vec<(u64, u64)>,
        join: bool,
        verify_hash: bool,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<PyObject> {
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            threads,
            ..Default::default()
        };

        let (desc, parts) = py
            .detach(|| {
                self.file
                    .decode_range(msg_index, obj_index, &ranges, &options)
            })
            .map_err(to_py_err)?;

        build_range_result(py, desc.dtype, parts, &ranges, join)
    }

    /// Batch-decode full objects across multiple messages. Remote only.
    #[pyo3(signature = (msg_indices, obj_index, verify_hash=false, native_byte_order=true, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn file_decode_object_batch(
        &self,
        py: Python<'_>,
        msg_indices: Vec<usize>,
        obj_index: usize,
        verify_hash: bool,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<PyObject> {
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            threads,
            ..Default::default()
        };

        let batch = py
            .detach(|| {
                self.file
                    .decode_object_batch(&msg_indices, obj_index, &options)
            })
            .map_err(to_py_err)?;

        let py_results: Vec<PyObject> = batch
            .into_iter()
            .map(|(meta, desc, data)| {
                let arr = bytes_to_numpy(py, &desc, &data)?;
                let py_desc = PyDataObjectDescriptor { inner: desc }
                    .into_pyobject(py)?
                    .into_any()
                    .unbind();
                let result = PyDict::new(py);
                result.set_item("metadata", PyMetadata { inner: meta }.into_pyobject(py)?)?;
                result.set_item("descriptor", py_desc)?;
                result.set_item("data", arr)?;
                Ok(result.into_any().unbind())
            })
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, py_results)?.into_any().unbind())
    }

    /// Batch-decode a sub-array range from the same object across multiple
    /// messages via batched HTTP. Remote only.
    #[pyo3(signature = (msg_indices, obj_index, ranges, join=false, verify_hash=false, native_byte_order=true, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn file_decode_range_batch(
        &self,
        py: Python<'_>,
        msg_indices: Vec<usize>,
        obj_index: usize,
        ranges: Vec<(u64, u64)>,
        join: bool,
        verify_hash: bool,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<PyObject> {
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            threads,
            ..Default::default()
        };

        let batch = py
            .detach(|| {
                self.file
                    .decode_range_batch(&msg_indices, obj_index, &ranges, &options)
            })
            .map_err(to_py_err)?;

        let py_results: Vec<PyObject> = batch
            .into_iter()
            .map(|(desc, parts)| build_range_result(py, desc.dtype, parts, &ranges, join))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, py_results)?.into_any().unbind())
    }

    fn is_remote(&self) -> bool {
        self.file.is_remote()
    }

    fn source(&self) -> String {
        self.file.source()
    }

    /// Raw wire-format bytes for the message at *index*.
    fn read_message<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| self.file.read_message(index))
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// All raw message bytes as a list of ``bytes`` objects.
    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        #[allow(deprecated)]
        let msgs = py.detach(|| self.file.messages()).map_err(to_py_err)?;
        let items: Vec<PyObject> = msgs
            .iter()
            .map(|m| PyBytes::new(py, m).into_any().unbind())
            .collect();
        PyList::new(py, items)
    }

    fn __repr__(&self) -> String {
        format!("TensogramFile(source='{}')", self.file.source())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, pyo3::PyAny>>,
        _exc_val: Option<&Bound<'_, pyo3::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> bool {
        false // do not suppress exceptions
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.file.message_count()).map_err(to_py_err)
    }

    /// Iterate over all messages in the file.
    ///
    /// Yields ``Message(metadata, objects)`` namedtuples,
    /// one per message, in file order.
    ///
    /// Example::
    ///
    ///     with tensogram.TensogramFile.open("data.tgm") as f:
    ///         for meta, objects in f:
    ///             desc, arr = objects[0]
    ///             print(arr.shape)
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyFileIter> {
        let path = self
            .file
            .path()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("iteration not supported on remote files")
            })?
            .to_path_buf();
        let (iter_file, count) = py
            .detach(|| {
                let f = TensogramFile::open(&path)?;
                let c = f.message_count()?;
                Ok::<_, tensogram_lib::TensogramError>((f, c))
            })
            .map_err(to_py_err)?;
        Ok(PyFileIter {
            file: iter_file,
            index: 0,
            count,
        })
    }

    /// Index or slice messages.
    ///
    /// - ``file[i]`` returns ``Message(metadata, objects)``
    /// - ``file[-1]`` returns the last message
    /// - ``file[1:4]`` returns ``list[Message]``
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, pyo3::PyAny>) -> PyResult<PyObject> {
        let count = py.detach(|| self.file.message_count()).map_err(to_py_err)?;

        if let Ok(index) = key.extract::<isize>() {
            let idx = if index < 0 {
                count as isize + index
            } else {
                index
            };
            if idx < 0 || idx >= count as isize {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "message index {index} out of range for file with {count} messages"
                )));
            }
            return self.decode_message(py, idx as usize, None, true, 0, true);
        }

        if let Ok(slice) = key.cast::<pyo3::types::PySlice>() {
            let indices = slice.indices(count as isize)?;
            let mut items: Vec<PyObject> = Vec::with_capacity(indices.slicelength as usize);
            let mut i = indices.start;
            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                items.push(self.decode_message(py, i as usize, None, true, 0, true)?);
                i += indices.step;
            }
            return Ok(PyList::new(py, items)?.into_any().unbind());
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be integers or slices",
        ))
    }
}

/// Construct a ``Message(metadata, objects)`` namedtuple.
///
/// The Message class is cached after first lookup to avoid repeated
/// import + getattr on every decode call.
fn pack_message(py: Python<'_>, meta: PyMetadata, objects: PyObject) -> PyResult<PyObject> {
    use pyo3::sync::PyOnceLock;
    static MESSAGE_TYPE: PyOnceLock<PyObject> = PyOnceLock::new();
    let msg_type = MESSAGE_TYPE
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import("tensogram")?.getattr("Message")?.unbind())
        })?
        .bind(py);
    let meta_obj = meta.into_pyobject(py)?.into_any();
    let objs_obj = objects.bind(py).clone().into_any();
    Ok(msg_type.call1((meta_obj, objs_obj))?.unbind())
}

// ---------------------------------------------------------------------------
// File iterator
// ---------------------------------------------------------------------------

/// Iterator over messages in a TensogramFile.
///
/// Owns an independent file handle — safe under free-threaded Python.
/// Yields ``Message(metadata, objects)`` namedtuples.
/// Created by ``iter(file)`` or ``for msg in file:``.
#[pyclass(name = "TensogramFileIter")]
struct PyFileIter {
    file: TensogramFile,
    index: usize,
    count: usize,
}

#[pymethods]
impl PyFileIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.index >= self.count {
            return Ok(None);
        }
        let i = self.index;
        self.index += 1;
        let options = DecodeOptions::default();
        let (global_meta, data_objects) = py
            .detach(|| self.file.decode_message(i, &options))
            .map_err(to_py_err)?;
        let result_list = data_objects_to_python(py, &data_objects)?;
        Ok(Some(pack_message(
            py,
            PyMetadata { inner: global_meta },
            result_list,
        )?))
    }

    fn __len__(&self) -> usize {
        self.count.saturating_sub(self.index)
    }

    fn __repr__(&self) -> String {
        let remaining = self.count.saturating_sub(self.index);
        format!(
            "TensogramFileIter(position={}, remaining={})",
            self.index, remaining
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Encode arrays into a Tensogram wire-format message.
///
/// Args:
///     global_meta_dict: ``{"base": [...], ...}`` with any extra keys.
///     descriptors_and_data: list of ``(descriptor_dict, numpy_array)`` pairs.
///     hash: ``"xxh3"`` (default) or ``None`` to skip integrity hashing.
///
/// Returns:
///     ``bytes`` — the complete wire-format message.
#[pyfunction]
#[pyo3(
    name = "encode",
    signature = (
        global_meta_dict,
        descriptors_and_data,
        hash=Some("xxh3"),
        threads=0,
        allow_nan=false,
        allow_inf=false,
        nan_mask_method=None,
        pos_inf_mask_method=None,
        neg_inf_mask_method=None,
        small_mask_threshold_bytes=None,
        create_header_hashes=None,
        create_footer_hashes=None,
    )
)]
#[allow(clippy::too_many_arguments)]
fn py_encode<'py>(
    py: Python<'py>,
    global_meta_dict: &Bound<'_, PyDict>,
    descriptors_and_data: &Bound<'_, PyList>,
    hash: Option<&str>,
    threads: u32,
    allow_nan: bool,
    allow_inf: bool,
    nan_mask_method: Option<&str>,
    pos_inf_mask_method: Option<&str>,
    neg_inf_mask_method: Option<&str>,
    small_mask_threshold_bytes: Option<usize>,
    create_header_hashes: Option<bool>,
    create_footer_hashes: Option<bool>,
) -> PyResult<Bound<'py, PyBytes>> {
    let global_meta = dict_to_global_metadata(global_meta_dict)?;
    let pairs = extract_descriptor_data_pairs(py, descriptors_and_data)?;
    let refs: Vec<(&DataObjectDescriptor, &[u8])> =
        pairs.iter().map(|(d, b)| (d, b.as_slice())).collect();

    let options = make_encode_options_full(
        hash,
        threads,
        allow_nan,
        allow_inf,
        nan_mask_method,
        pos_inf_mask_method,
        neg_inf_mask_method,
        small_mask_threshold_bytes,
        create_header_hashes,
        create_footer_hashes,
    )?;
    let msg = py.detach(|| encode(&global_meta, &refs, &options).map_err(to_py_err))?;
    Ok(PyBytes::new(py, &msg))
}

/// Encode already-encoded payloads into a Tensogram wire-format message.
///
/// Unlike :func:`encode`, this function does **not** run the encoding pipeline
/// (encoding/filter/compression/szip).  Each *data* element must be a
/// ``bytes`` object already carrying the encoded payload that matches the
/// descriptor's declared ``encoding`` / ``filter`` / ``compression`` and any
/// codec-specific ``params`` (e.g. ``szip_block_offsets``).  The library still
/// computes and stamps the payload hash.
///
/// Args:
///     global_meta_dict: ``{"base": [...], ...}`` with any extra keys.
///     descriptors_and_data: list of ``(descriptor_dict, bytes)`` pairs.  The
///         second element of each pair **must** be a ``bytes``-like object —
///         numpy arrays are rejected because pre-encoded payloads are
///         already in their final wire form.
///     hash: ``"xxh3"`` (default) or ``None`` to skip integrity hashing.
///
/// Returns:
///     ``bytes`` — the complete wire-format message.
#[pyfunction]
#[pyo3(name = "encode_pre_encoded", signature = (global_meta_dict, descriptors_and_data, hash=Some("xxh3"), threads=0))]
fn py_encode_pre_encoded<'py>(
    py: Python<'py>,
    global_meta_dict: &Bound<'_, PyDict>,
    descriptors_and_data: &Bound<'_, PyList>,
    hash: Option<&str>,
    threads: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let global_meta = dict_to_global_metadata(global_meta_dict)?;
    let pairs = extract_pre_encoded_pairs(descriptors_and_data)?;
    let refs: Vec<(&DataObjectDescriptor, &[u8])> =
        pairs.iter().map(|(d, b)| (d, b.as_slice())).collect();

    let options = make_encode_options(hash, threads)?;
    let msg = py.detach(|| encode_pre_encoded(&global_meta, &refs, &options).map_err(to_py_err))?;
    Ok(PyBytes::new(py, &msg))
}

/// Decode a wire-format message → ``Message(metadata, objects)``.
///
/// Set *verify_hash* to ``True`` to verify payload integrity.
#[pyfunction]
#[pyo3(
    name = "decode",
    signature = (
        buf,
        verify_hash=false,
        native_byte_order=true,
        threads=0,
        restore_non_finite=true,
    )
)]
fn py_decode(
    py: Python<'_>,
    buf: PyBackedBytes,
    verify_hash: bool,
    native_byte_order: bool,
    threads: u32,
    restore_non_finite: bool,
) -> PyResult<PyObject> {
    let options = DecodeOptions {
        verify_hash,
        native_byte_order,
        threads,
        restore_non_finite,
        ..Default::default()
    };
    let (global_meta, data_objects) = py.detach(|| decode(&buf, &options).map_err(to_py_err))?;
    let result_list = data_objects_to_python(py, &data_objects)?;
    pack_message(py, PyMetadata { inner: global_meta }, result_list)
}

/// Decode a wire-format message, returning raw NaN / Inf bitmasks
/// alongside the `0.0`-substituted payload for advanced callers.
///
/// Unlike :func:`decode`, this entry point never writes canonical
/// NaN / ±Inf bits into the payload — callers see the substituted
/// zeros exactly as on disk, plus the raw mask booleans.  Use it
/// to aggregate missing-count statistics without materialising
/// canonical non-finite bytes, or to convert to a domain-specific
/// missing-value representation.
///
/// Returns a ``Message(metadata, objects)`` where each object is a
/// ``(descriptor, payload_array, masks_dict)`` tuple.  ``masks_dict``
/// holds any subset of ``"nan"``, ``"inf+"``, ``"inf-"`` keys, each
/// mapped to a boolean numpy array of length ``n_elements``.  An
/// empty dict means the frame carried no mask companion.
///
/// See :doc:`nan-inf-handling` and ``plans/WIRE_FORMAT.md`` §6.5.
#[pyfunction]
#[pyo3(
    name = "decode_with_masks",
    signature = (
        buf,
        verify_hash=false,
        native_byte_order=true,
        threads=0,
    )
)]
fn py_decode_with_masks(
    py: Python<'_>,
    buf: PyBackedBytes,
    verify_hash: bool,
    native_byte_order: bool,
    threads: u32,
) -> PyResult<PyObject> {
    let options = DecodeOptions {
        verify_hash,
        native_byte_order,
        threads,
        // Forced false by the underlying `decode_with_masks`
        // regardless of what we pass; stated explicitly for
        // symmetry with the Rust API contract.
        restore_non_finite: false,
        ..Default::default()
    };
    let (global_meta, objects) = py.detach(|| {
        tensogram_lib::decode_with_masks(&buf, &options).map_err(to_py_err)
    })?;
    let result_list = objects_with_masks_to_python(py, &objects)?;
    pack_message(py, PyMetadata { inner: global_meta }, result_list)
}

/// Decode only metadata (no payload decompression).
///
/// Faster than ``decode()`` when you only need metadata for filtering.
/// Hash verification is not available in this mode.
#[pyfunction]
#[pyo3(name = "decode_metadata")]
fn py_decode_metadata(buf: &[u8]) -> PyResult<PyMetadata> {
    let meta = decode_metadata(buf).map_err(to_py_err)?;
    Ok(PyMetadata { inner: meta })
}

/// Decode global metadata **and** per-object descriptors without decoding
/// any payload data.
///
/// Returns ``(Metadata, list[DataObjectDescriptor])``.  Cheaper than
/// :func:`decode` because no decompression or filter reversal is performed.
#[pyfunction]
#[pyo3(name = "decode_descriptors")]
fn py_decode_descriptors(py: Python<'_>, buf: &[u8]) -> PyResult<(PyMetadata, PyObject)> {
    let (gm, descs) = decode_descriptors(buf).map_err(to_py_err)?;
    let py_descs: Vec<PyObject> = descs
        .into_iter()
        .map(|d| {
            let obj = PyDataObjectDescriptor { inner: d };
            Ok(Py::new(py, obj)?.into_any())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let list = PyList::new(py, py_descs)?;
    Ok((PyMetadata { inner: gm }, list.into_any().unbind()))
}

/// Decode a single object by *index* -> ``(Metadata, DataObjectDescriptor, ndarray)``.
///
/// Only the header and the requested object's payload are read.
/// Raises ``ValueError`` if *index* is out of range.
#[pyfunction]
#[pyo3(
    name = "decode_object",
    signature = (
        buf,
        index,
        verify_hash=false,
        native_byte_order=true,
        threads=0,
        restore_non_finite=true,
    )
)]
#[allow(clippy::too_many_arguments)]
fn py_decode_object(
    py: Python<'_>,
    buf: PyBackedBytes,
    index: usize,
    verify_hash: bool,
    native_byte_order: bool,
    threads: u32,
    restore_non_finite: bool,
) -> PyResult<(PyMetadata, PyDataObjectDescriptor, PyObject)> {
    let options = DecodeOptions {
        verify_hash,
        native_byte_order,
        threads,
        restore_non_finite,
        ..Default::default()
    };
    let (global_meta, desc, obj_bytes) =
        py.detach(|| decode_object(&buf, index, &options).map_err(to_py_err))?;
    let arr = bytes_to_numpy(py, &desc, &obj_bytes)?;
    Ok((
        PyMetadata { inner: global_meta },
        PyDataObjectDescriptor { inner: desc },
        arr,
    ))
}

/// Extract sub-ranges from a data object.
///
/// Args:
///     buf: wire-format message bytes.
///     object_index: which object in the message (0-based).
///     ranges: list of ``(start, count)`` element-offset tuples,
///         e.g. ``[(0, 10), (50, 5)]`` reads elements 0..10 and 50..55.
///     join: when ``True``, return a single concatenated 1-d ndarray
///         (like the pre-0.6 behaviour).  When ``False`` (default),
///         return a ``list`` of 1-d ndarrays, one per range.
///     verify_hash: verify payload hash before extraction.
///
/// Returns:
///     ``list[ndarray]`` (default) or ``ndarray`` (when ``join=True``).
#[pyfunction]
#[pyo3(
    name = "decode_range",
    signature = (
        buf,
        object_index,
        ranges,
        join=false,
        verify_hash=false,
        native_byte_order=true,
        threads=0,
        restore_non_finite=true,
    )
)]
// The argument list is the public Python ABI — each one is a documented
// keyword argument that other bindings (Rust core, FFI, WASM) also
// expose.  Collapsing them into an options struct would break the
// Pythonic kwargs calling convention.
#[allow(clippy::too_many_arguments)]
fn py_decode_range(
    py: Python<'_>,
    buf: PyBackedBytes,
    object_index: usize,
    ranges: Vec<(u64, u64)>,
    join: bool,
    verify_hash: bool,
    native_byte_order: bool,
    threads: u32,
    restore_non_finite: bool,
) -> PyResult<PyObject> {
    let options = DecodeOptions {
        verify_hash,
        native_byte_order,
        threads,
        restore_non_finite,
        ..Default::default()
    };
    let (desc, parts) =
        py.detach(|| decode_range(&buf, object_index, &ranges, &options).map_err(to_py_err))?;

    build_range_result(py, desc.dtype, parts, &ranges, join)
}

/// Scan *buf* for message boundaries → ``list[(offset, length)]``.
///
/// Each tuple identifies one complete Tensogram message within the buffer.
/// Useful for multi-message buffers (e.g. from a network socket or mmap).
#[pyfunction]
#[pyo3(name = "scan")]
fn py_scan(py: Python<'_>, buf: PyBackedBytes) -> Vec<(usize, usize)> {
    py.detach(|| scan(&buf))
}

/// Iterate over messages in a byte buffer.
///
/// Scans for message boundaries, then decodes each message on demand.
/// Equivalent to calling :func:`scan` then :func:`decode` on each entry,
/// but more convenient.
///
/// **Note:** The iterator copies the entire buffer into its own memory.
/// For large files (> 100 MB), prefer ``TensogramFile.open()`` for
/// stream-based iteration without the full buffer copy.
///
/// Args:
///     buf: bytes containing one or more wire-format messages.
///     verify_hash: verify payload hashes during decode (default ``False``).
///
/// Yields:
///     ``Message(metadata, objects)`` namedtuples per message.
///
/// Example::
///
///     buf = open("data.tgm", "rb").read()
///     for msg in tensogram.iter_messages(buf):
///         desc, arr = msg.objects[0]
#[pyfunction]
#[pyo3(name = "iter_messages", signature = (buf, verify_hash=false))]
fn py_iter_messages(buf: &[u8], verify_hash: bool) -> PyBufferIter {
    let offsets = scan(buf);
    PyBufferIter {
        buf: buf.to_vec(),
        offsets,
        index: 0,
        verify_hash,
    }
}

/// Iterator over messages in a byte buffer.
///
/// Created by :func:`iter_messages`. Owns a copy of the buffer.
#[pyclass(name = "MessageIter")]
struct PyBufferIter {
    buf: Vec<u8>,
    offsets: Vec<(usize, usize)>,
    index: usize,
    verify_hash: bool,
}

#[pymethods]
impl PyBufferIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.index >= self.offsets.len() {
            return Ok(None);
        }
        let (offset, length) = self.offsets[self.index];
        self.index += 1;
        let msg_bytes = self.buf[offset..offset + length].to_vec();
        let options = DecodeOptions {
            verify_hash: self.verify_hash,
            ..Default::default()
        };
        let (global_meta, data_objects) =
            py.detach(|| decode(&msg_bytes, &options).map_err(to_py_err))?;
        let result_list = data_objects_to_python(py, &data_objects)?;
        Ok(Some(pack_message(
            py,
            PyMetadata { inner: global_meta },
            result_list,
        )?))
    }

    fn __len__(&self) -> usize {
        self.offsets.len().saturating_sub(self.index)
    }

    fn __repr__(&self) -> String {
        let remaining = self.offsets.len().saturating_sub(self.index);
        format!(
            "MessageIter(position={}, remaining={})",
            self.index, remaining
        )
    }
}

/// Compute a hash digest over arbitrary bytes.
///
/// Mirrors Rust :func:`tensogram::compute_hash`, the WASM
/// ``compute_hash`` export, and ``tgm_compute_hash`` in the C FFI.
/// Useful for verifying an encoded payload against the hash recorded
/// in a descriptor, or for pre-computing a hash before calling
/// :func:`encode_pre_encoded`.
///
/// Args:
///     data: Bytes to hash — ``bytes`` or ``bytearray``.  Zero-copy
///         for ``bytes`` via :class:`PyBackedBytes`.  For other
///         buffer-protocol objects (``memoryview``, ``numpy.ndarray``,
///         etc.) call ``bytes(obj)`` / ``obj.tobytes()`` first.
///     algo: Algorithm name.  ``"xxh3"`` is the only supported value
///         today and the default.
///
/// Returns:
///     The hex-encoded digest as a ``str`` (16 characters for xxh3-64).
///
/// Raises:
///     TypeError: If ``data`` is not ``bytes`` or ``bytearray``.
///     ValueError: If ``algo`` is not a recognised algorithm name.
///
/// Example::
///
///     >>> tensogram.compute_hash(b"hello world")
///     'd447b1ea40e6988b'
#[pyfunction]
#[pyo3(name = "compute_hash", signature = (data, algo="xxh3"))]
fn py_compute_hash(data: PyBackedBytes, algo: &str) -> PyResult<String> {
    let algorithm = HashAlgorithm::parse(algo).map_err(to_py_err)?;
    Ok(tensogram_lib::compute_hash(&data, algorithm))
}

/// Run environment diagnostics and return the report as a Python dict.
///
/// Mirrors the Rust :func:`tensogram::doctor::run_diagnostics`, the
/// WASM ``doctor()`` export, and the ``tensogram doctor`` CLI
/// subcommand.  The Python build does **not** run the GRIB or NetCDF
/// converter self-tests — those features are CLI-only — so the
/// ``self_test`` array covers only the core encode/decode pipeline
/// plus the codecs that were compiled into this Python wheel.
///
/// Returns:
///     A ``dict`` with three top-level keys:
///       - ``build`` (dict): version, wire-format version, target
///         triple, and build profile.
///       - ``features`` (list[dict]): one entry per known feature.
///         ``state == "on"`` rows additionally carry ``backend``,
///         ``linkage``, and (where available) ``version``.
///       - ``self_test`` (list[dict]): one entry per self-test step
///         with ``label`` and ``outcome`` (``"ok"`` / ``"failed"``
///         / ``"skipped"``).
///
///     The shape matches the JSON schema documented in
///     ``docs/src/cli/doctor.md``.
///
/// Example::
///
///     >>> import tensogram
///     >>> report = tensogram.doctor()
///     >>> report["build"]["version"]
///     '0.19.0'
///     >>> [f["name"] for f in report["features"] if f["state"] == "on"][:3]
///     ['szip', 'zstd', 'lz4']
#[pyfunction]
#[pyo3(name = "doctor")]
fn py_doctor(py: Python<'_>) -> PyResult<PyObject> {
    let report = tensogram_lib::doctor::run_diagnostics();
    // Round-trip through JSON: serde already produces the canonical
    // shape we promise across all language bindings, and Python's
    // built-in `json.loads` returns the matching native types
    // (`dict`/`list`/`str`/`int`) without pulling in pythonize.
    let json = serde_json::to_string(&report).map_err(|e| {
        PyValueError::new_err(format!("failed to serialise doctor report: {e}"))
    })?;
    let json_module = py.import("json")?;
    json_module.call_method1("loads", (json,)).map(|v| v.unbind())
}

/// Compute simple-packing parameters for a float64 array.
///
/// Args:
///     values: 1-d float64 numpy array (must not contain NaN).
///     bits_per_value: quantization depth (e.g. 16).
///     decimal_scale_factor: decimal scaling (usually 0).
///
/// Returns a dict with ``sp_``-prefixed keys:
/// ``sp_reference_value``, ``sp_binary_scale_factor``,
/// ``sp_decimal_scale_factor``, ``sp_bits_per_value``.
///
/// The dict can be spread directly into a descriptor literal::
///
///     params = tensogram.compute_packing_params(data.ravel(), 16, 0)
///     desc = {"type": "ntensor", "encoding": "simple_packing",
///             "shape": [...], "dtype": "float64", **params}
///
/// The encoder also auto-computes these values when the descriptor
/// carries only ``sp_bits_per_value`` (and optionally
/// ``sp_decimal_scale_factor``) — calling this function explicitly
/// is only needed if the caller wants to cache or inspect the derived
/// params across multiple encodes (for example to pin
/// ``sp_reference_value`` across a time-series).
#[pyfunction]
fn compute_packing_params(
    py: Python<'_>,
    values: numpy::PyReadonlyArray1<'_, f64>,
    bits_per_value: u32,
    decimal_scale_factor: i32,
) -> PyResult<PyObject> {
    let slice = values
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let params = tensogram_encodings::simple_packing::compute_params(
        slice,
        bits_per_value,
        decimal_scale_factor,
    )
    .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("sp_reference_value", params.reference_value)?;
    dict.set_item("sp_binary_scale_factor", params.binary_scale_factor)?;
    dict.set_item("sp_decimal_scale_factor", params.decimal_scale_factor)?;
    dict.set_item("sp_bits_per_value", params.bits_per_value)?;
    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// PyStreamingEncoder — wraps StreamingEncoder<Cursor<Vec<u8>>>
// ---------------------------------------------------------------------------

/// Progressive, frame-at-a-time encoder backed by an in-memory ``bytes`` buffer.
///
/// Unlike :func:`encode`, which builds a complete message in one shot, the
/// streaming encoder writes each object frame as soon as
/// :meth:`write_object` (or :meth:`write_object_pre_encoded`) is called.
/// Call :meth:`finish` to flush the footer frames + postamble and retrieve the
/// complete wire-format message.
///
/// Two finish modes:
///
/// - :meth:`finish` writes the postamble with ``total_length = 0`` and the
///   preamble with ``total_length = 0`` — fastest path, but readers must
///   forward-scan to find the next message boundary because the
///   backward-locatability invariant in the wire format §7 is not met.
/// - :meth:`finish_backfilled` seeks back into the in-memory cursor and
///   patches both length slots with the real message length so readers can
///   backward-scan from EOF in O(1) using the postamble's mirrored
///   ``total_length``.  Required for fixtures or workloads that exercise
///   the bidirectional remote walker.
///
/// Example::
///
///     enc = tensogram.StreamingEncoder({})
///     enc.write_object({"type": "ntensor", "shape": [4], "dtype": "float32"},
///                      np.ones(4, dtype=np.float32))
///     msg = enc.finish()              # streaming-mode (total_length=0)
///     # or
///     msg = enc.finish_backfilled()   # backfilled (full backward locatability)
#[pyclass(name = "StreamingEncoder")]
struct PyStreamingEncoder {
    inner: Option<StreamingEncoder<std::io::Cursor<Vec<u8>>>>,
}

#[pymethods]
impl PyStreamingEncoder {
    /// Begin a new streaming message.
    ///
    /// Args:
    ///     global_meta_dict: ``{"base": [...], ...}`` with any extra keys.
    ///     hash: ``"xxh3"`` (default) or ``None`` to skip integrity hashing.
    ///     threads: thread budget for intra-codec parallelism inside
    ///         :meth:`write_object` (axis B only — streaming encoding
    ///         does not have cross-object parallelism by design).
    ///         Default ``0`` preserves the sequential path.
    #[new]
    #[pyo3(
        signature = (
            global_meta_dict,
            hash=Some("xxh3"),
            threads=0,
            allow_nan=false,
            allow_inf=false,
            nan_mask_method=None,
            pos_inf_mask_method=None,
            neg_inf_mask_method=None,
            small_mask_threshold_bytes=None,
            create_header_hashes=None,
            create_footer_hashes=None,
        )
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        global_meta_dict: &Bound<'_, PyDict>,
        hash: Option<&str>,
        threads: u32,
        allow_nan: bool,
        allow_inf: bool,
        nan_mask_method: Option<&str>,
        pos_inf_mask_method: Option<&str>,
        neg_inf_mask_method: Option<&str>,
        small_mask_threshold_bytes: Option<usize>,
        create_header_hashes: Option<bool>,
        create_footer_hashes: Option<bool>,
    ) -> PyResult<Self> {
        let global_meta = dict_to_global_metadata(global_meta_dict)?;
        let options = make_encode_options_full(
            hash,
            threads,
            allow_nan,
            allow_inf,
            nan_mask_method,
            pos_inf_mask_method,
            neg_inf_mask_method,
            small_mask_threshold_bytes,
            create_header_hashes,
            create_footer_hashes,
        )?;
        let inner = StreamingEncoder::new(std::io::Cursor::new(Vec::new()), &global_meta, &options)
            .map_err(to_py_err)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Encode and write a single data object frame.
    ///
    /// Args:
    ///     descriptor: dict with ``type``/``shape``/``dtype`` plus encoding
    ///         pipeline fields (``encoding``, ``filter``, ``compression``,
    ///         ``params``, etc.).
    ///     data: numpy array or ``bytes``-like object carrying the raw
    ///         native-endian payload to encode.
    fn write_object(
        &mut self,
        descriptor: &Bound<'_, PyDict>,
        data: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        let desc = dict_to_data_object_descriptor(descriptor)?;
        let bytes = extract_single_data_item(data)?;
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("StreamingEncoder already finished"))?;
        inner.write_object(&desc, &bytes).map_err(to_py_err)
    }

    /// Write a pre-encoded data object frame directly (no pipeline).
    ///
    /// ``data`` must already contain the encoded payload matching the
    /// descriptor's ``encoding`` / ``filter`` / ``compression`` / ``params``
    /// (e.g. ``szip_block_offsets`` as a ``list[int]``).  The library still
    /// computes and stamps the payload hash.
    fn write_object_pre_encoded(
        &mut self,
        descriptor: &Bound<'_, PyDict>,
        data: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        let desc = dict_to_data_object_descriptor(descriptor)?;
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("StreamingEncoder already finished"))?;
        // Zero-copy fast path for PyBytes; fall back to extract for bytearray / memoryview.
        if let Ok(py_bytes) = data.cast::<PyBytes>() {
            inner
                .write_object_pre_encoded(&desc, py_bytes.as_bytes())
                .map_err(to_py_err)
        } else {
            let bytes = data.extract::<Vec<u8>>()?;
            inner
                .write_object_pre_encoded(&desc, &bytes)
                .map_err(to_py_err)
        }
    }

    /// Finalize the stream and return the complete wire-format ``bytes``.
    ///
    /// Streaming mode: writes ``total_length = 0`` in both the preamble and
    /// postamble.  Readers fall back to forward-scan to find message
    /// boundaries.  See :meth:`finish_backfilled` for the backward-locatable
    /// alternative.
    ///
    /// Subsequent calls on this encoder raise ``RuntimeError``.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("StreamingEncoder already finished"))?;
        let cursor = inner.finish().map_err(to_py_err)?;
        Ok(PyBytes::new(py, cursor.get_ref()))
    }

    /// Finalize the stream and back-fill ``total_length`` in both the
    /// preamble and postamble before returning the complete wire-format
    /// ``bytes``.
    ///
    /// Equivalent to :meth:`finish` but the produced message satisfies the
    /// backward-locatability invariant in the wire format §7: readers can
    /// O(1) seek to ``end - 16`` to read the mirrored ``total_length`` and
    /// jump straight to the message start.  Required for the bidirectional
    /// remote walker and any workload that needs efficient backward access.
    ///
    /// Subsequent calls on this encoder raise ``RuntimeError``.
    fn finish_backfilled<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("StreamingEncoder already finished"))?;
        let cursor = inner.finish_with_backfill().map_err(to_py_err)?;
        Ok(PyBytes::new(py, cursor.get_ref()))
    }
}

/// Parse a Python level string into ValidateOptions.
fn parse_validate_options(level: &str, check_canonical: bool) -> PyResult<ValidateOptions> {
    let (max_level, checksum_only) = match level {
        "quick" => (ValidationLevel::Structure, false),
        "default" => (ValidationLevel::Integrity, false),
        "checksum" => (ValidationLevel::Integrity, true),
        "full" => (ValidationLevel::Fidelity, false),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown validation level: '{}', expected one of: quick, default, checksum, full",
                other
            )));
        }
    };
    Ok(ValidateOptions {
        max_level,
        check_canonical,
        checksum_only,
    })
}

/// Convert a serde_json::Value to a Python object.
fn json_value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Err(PyValueError::new_err(format!(
                    "JSON number {n} cannot be represented as i64, u64, or f64"
                )))
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let items: PyResult<Vec<PyObject>> =
                arr.iter().map(|v| json_value_to_py(py, v)).collect();
            let list = PyList::new(py, items?)?;
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Validate a single Tensogram message buffer.
///
/// Checks structural integrity, metadata correctness, hash verification,
/// and optionally data fidelity (NaN/Inf detection) depending on the
/// validation level.
///
/// Args:
///     buf: Wire-format message bytes (a single message, not a file).
///     level: Validation depth — ``"quick"`` (structure only),
///         ``"default"`` (up to hash verification), ``"checksum"``
///         (hash verification, suppress structural warnings),
///         ``"full"`` (full decode + NaN/Inf scan).
///     check_canonical: Check RFC 8949 deterministic CBOR key ordering.
///
/// Returns:
///     A dict with keys ``"issues"`` (list of issue dicts), ``"object_count"``
///     (int), and ``"hash_verified"`` (bool).  Each issue dict contains
///     ``"code"``, ``"level"``, ``"severity"``, ``"description"``, and
///     optionally ``"object_index"`` and ``"byte_offset"``.
#[pyfunction]
#[pyo3(name = "validate", signature = (buf, level="default", check_canonical=false))]
fn py_validate(
    py: Python<'_>,
    buf: PyBackedBytes,
    level: &str,
    check_canonical: bool,
) -> PyResult<PyObject> {
    let options = parse_validate_options(level, check_canonical)?;
    let report = py.detach(|| validate_message(&buf, &options));
    let json_val = serde_json::to_value(&report)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")))?;
    json_value_to_py(py, &json_val)
}

/// Validate all messages in a ``.tgm`` file.
///
/// Uses streaming I/O — only one message is in memory at a time.
/// Detects gaps, trailing bytes, and truncated messages between valid
/// messages.
///
/// Args:
///     path: Path to a ``.tgm`` file.
///     level: Validation depth (same as :func:`validate`).
///     check_canonical: Check RFC 8949 deterministic CBOR key ordering.
///
/// Returns:
///     A dict with keys ``"file_issues"`` (list of file-level issue dicts)
///     and ``"messages"`` (list of per-message validation report dicts).
///     Each file-level issue has ``"byte_offset"``, ``"length"``, and
///     ``"description"``.  Each message report has the same structure as
///     the return value of :func:`validate`.
///
/// Raises:
///     OSError: If the file cannot be opened or read.
#[pyfunction]
#[pyo3(name = "validate_file", signature = (path, level="default", check_canonical=false))]
fn py_validate_file(
    py: Python<'_>,
    path: &str,
    level: &str,
    check_canonical: bool,
) -> PyResult<PyObject> {
    let options = parse_validate_options(level, check_canonical)?;
    let path = path.to_string();
    let report = py
        .detach(|| core_validate_file(Path::new(&path), &options))
        .map_err(|e| PyIOError::new_err(format!("{e}")))?;
    let json_val = serde_json::to_value(&report)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")))?;
    json_value_to_py(py, &json_val)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// PyAsyncTensogramFile — async wrapper around TensogramFile
// ---------------------------------------------------------------------------

/// Async file-based Tensogram container for use with Python ``asyncio``.
///
/// Wraps :class:`TensogramFile` for non-blocking I/O.  Decode methods
/// return coroutines that can be composed with ``asyncio.gather``::
///
///     f = await tensogram.AsyncTensogramFile.open("data.tgm")
///     results = await asyncio.gather(
///         f.file_decode_object(0, 0),
///         f.file_decode_object(0, 1),
///     )
///
/// A single handle supports truly concurrent operations (no mutex).
#[cfg(feature = "async")]
#[pyclass(name = "AsyncTensogramFile")]
struct PyAsyncTensogramFile {
    file: Arc<TensogramFile>,
    cached_source: String,
    cached_is_remote: bool,
    cached_message_count: Arc<std::sync::OnceLock<usize>>,
}

#[cfg(feature = "async")]
#[pymethods]
impl PyAsyncTensogramFile {
    /// Open a local file or remote URL asynchronously.
    ///
    /// Auto-detects remote URLs (``s3://``, ``gs://``, ``http://``, etc.)
    /// when the ``remote`` feature is enabled.
    ///
    /// Returns a coroutine; use ``await``::
    ///
    ///     f = await AsyncTensogramFile.open("data.tgm")
    #[staticmethod]
    #[pyo3(signature = (source, *, bidirectional=true))]
    fn open<'py>(
        py: Python<'py>,
        source: &str,
        bidirectional: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scan_opts = scan_opts_for(bidirectional);
        let source = source.to_string();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let file = TensogramFile::open_source_async(&source, scan_opts)
                .await
                .map_err(to_py_err)?;
            let is_remote = file.is_remote();
            let source_str = file.source();
            let wrapped = PyAsyncTensogramFile {
                file: Arc::new(file),
                cached_source: source_str,
                cached_is_remote: is_remote,
                cached_message_count: Arc::new(std::sync::OnceLock::new()),
            };
            Python::attach(|py| Ok(wrapped.into_pyobject(py)?.into_any().unbind()))
        })
    }

    /// Open a remote URL with explicit storage options.
    ///
    /// Args:
    ///     source: Remote URL (``s3://``, ``gs://``, ``http://``, etc.).
    ///     storage_options: Optional dict of provider credentials / config.
    ///     bidirectional: Default ``True`` — pipelined bidirectional
    ///         remote scan walker.  Pass ``False`` to force a
    ///         forward-only walk.
    ///
    /// Type errors on ``storage_options`` and ``bidirectional`` surface
    /// at the call site, before any I/O.
    ///
    /// Returns a coroutine; use ``await``::
    ///
    ///     f = await AsyncTensogramFile.open_remote("s3://bucket/data.tgm",
    ///                                               {"region": "eu-west-1"})
    #[staticmethod]
    #[pyo3(signature = (source, storage_options=None, *, bidirectional=true))]
    fn open_remote<'py>(
        py: Python<'py>,
        source: &str,
        storage_options: Option<&Bound<'_, PyDict>>,
        bidirectional: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let opts = parse_storage_options(storage_options)?;
        let scan_opts = scan_opts_for(bidirectional);
        let source = source.to_string();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let file =
                TensogramFile::open_remote_async(&source, &opts, scan_opts)
                    .await
                    .map_err(to_py_err)?;
            let is_remote = file.is_remote();
            let source_str = file.source();
            let wrapped = PyAsyncTensogramFile {
                file: Arc::new(file),
                cached_source: source_str,
                cached_is_remote: is_remote,
                cached_message_count: Arc::new(std::sync::OnceLock::new()),
            };
            Python::attach(|py| Ok(wrapped.into_pyobject(py)?.into_any().unbind()))
        })
    }

    // ── Async decode methods ─────────────────────────────────────────────

    /// Decode message at *index* asynchronously.
    ///
    /// Returns ``Message(metadata, objects)`` — identical to the sync
    /// :meth:`TensogramFile.decode_message`.
    #[pyo3(signature = (index, verify_hash=None, native_byte_order=true, threads=0))]
    fn decode_message<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        verify_hash: Option<bool>,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);
        let options = DecodeOptions {
            verify_hash: verify_hash.unwrap_or(false),
            native_byte_order,
            threads,
            ..Default::default()
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (global_meta, data_objects) = file
                .decode_message_async(index, &options)
                .await
                .map_err(to_py_err)?;
            Python::attach(|py| {
                let result_list = data_objects_to_python(py, &data_objects)?;
                let msg = pack_message(py, PyMetadata { inner: global_meta }, result_list)?;
                Ok(msg)
            })
        })
    }

    /// Decode only metadata for message *msg_index* asynchronously.
    fn file_decode_metadata<'py>(
        &self,
        py: Python<'py>,
        msg_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let meta = file
                .decode_metadata_async(msg_index)
                .await
                .map_err(to_py_err)?;
            Python::attach(|py| {
                Ok(PyMetadata { inner: meta }
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Decode metadata and descriptors for message *msg_index* asynchronously.
    fn file_decode_descriptors<'py>(
        &self,
        py: Python<'py>,
        msg_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (meta, descriptors) = file
                .decode_descriptors_async(msg_index)
                .await
                .map_err(to_py_err)?;
            Python::attach(|py| {
                let desc_list: Vec<Py<PyAny>> = descriptors
                    .iter()
                    .map(|d| {
                        Ok(PyDataObjectDescriptor { inner: d.clone() }
                            .into_pyobject(py)?
                            .into_any()
                            .unbind())
                    })
                    .collect::<PyResult<_>>()?;
                let result = PyDict::new(py);
                result.set_item("metadata", PyMetadata { inner: meta }.into_pyobject(py)?)?;
                result.set_item("descriptors", PyList::new(py, desc_list)?)?;
                Ok(result.into_any().unbind())
            })
        })
    }

    /// Decode a single data object asynchronously.
    ///
    /// Returns ``dict(metadata=Metadata, descriptor=DataObjectDescriptor, data=ndarray)``.
    #[pyo3(signature = (msg_index, obj_index, verify_hash=false, native_byte_order=true, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn file_decode_object<'py>(
        &self,
        py: Python<'py>,
        msg_index: usize,
        obj_index: usize,
        verify_hash: bool,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            threads,
            ..Default::default()
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (meta, desc, data) = file
                .decode_object_async(msg_index, obj_index, &options)
                .await
                .map_err(to_py_err)?;
            Python::attach(|py| {
                let arr = bytes_to_numpy(py, &desc, &data)?;
                let py_desc = PyDataObjectDescriptor { inner: desc }
                    .into_pyobject(py)?
                    .into_any()
                    .unbind();
                let result = PyDict::new(py);
                result.set_item("metadata", PyMetadata { inner: meta }.into_pyobject(py)?)?;
                result.set_item("descriptor", py_desc)?;
                result.set_item("data", arr)?;
                Ok(result.into_any().unbind())
            })
        })
    }

    // ── Async read ───────────────────────────────────────────────────────

    /// Raw wire-format bytes for the message at *index* (async).
    fn read_message<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let bytes = file.read_message_async(index).await.map_err(to_py_err)?;
            Python::attach(|py| Ok(PyBytes::new(py, &bytes).into_any().unbind()))
        })
    }

    // ── Sync utility methods (no I/O, cached fields) ─────────────────────

    /// Number of messages in the file (async, lazy scan on first call).
    fn message_count<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(&count) = self.cached_message_count.get() {
            return pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(count) });
        }
        let file = Arc::clone(&self.file);
        let cache = Arc::clone(&self.cached_message_count);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let count = file.message_count_async().await.map_err(to_py_err)?;
            let _ = cache.set(count);
            Ok(count)
        })
    }

    /// Whether this file was opened from a remote URL.
    fn is_remote(&self) -> bool {
        self.cached_is_remote
    }

    /// The source path or URL this file was opened from.
    fn source(&self) -> &str {
        &self.cached_source
    }

    /// Batch-prefetch layouts for the given message indices. No-op for local files.
    fn prefetch_layouts<'py>(
        &self,
        py: Python<'py>,
        msg_indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            file.prefetch_layouts_async(&msg_indices)
                .await
                .map_err(to_py_err)?;
            Ok(())
        })
    }

    // ── file_decode_range (native async for remote, spawn_blocking for local) ─

    #[pyo3(signature = (msg_index, obj_index, ranges, join=false, verify_hash=false, native_byte_order=true, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn file_decode_range<'py>(
        &self,
        py: Python<'py>,
        msg_index: usize,
        obj_index: usize,
        ranges: Vec<(u64, u64)>,
        join: bool,
        verify_hash: bool,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            threads,
            ..Default::default()
        };
        let ranges_for_result = ranges.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (desc, parts) = file
                .decode_range_async(msg_index, obj_index, &ranges, &options)
                .await
                .map_err(to_py_err)?;
            let dtype = desc.dtype;
            Python::attach(|py| build_range_result(py, dtype, parts, &ranges_for_result, join))
        })
    }

    /// Batch-decode full objects across multiple messages. Remote only.
    #[pyo3(signature = (msg_indices, obj_index, verify_hash=false, native_byte_order=true, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn file_decode_object_batch<'py>(
        &self,
        py: Python<'py>,
        msg_indices: Vec<usize>,
        obj_index: usize,
        verify_hash: bool,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            threads,
            ..Default::default()
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let batch = file
                .decode_object_batch_async(&msg_indices, obj_index, &options)
                .await
                .map_err(to_py_err)?;

            Python::attach(|py| {
                let py_results: Vec<PyObject> = batch
                    .into_iter()
                    .map(|(meta, desc, data)| {
                        let arr = bytes_to_numpy(py, &desc, &data)?;
                        let py_desc = PyDataObjectDescriptor { inner: desc }
                            .into_pyobject(py)?
                            .into_any()
                            .unbind();
                        let result = PyDict::new(py);
                        result
                            .set_item("metadata", PyMetadata { inner: meta }.into_pyobject(py)?)?;
                        result.set_item("descriptor", py_desc)?;
                        result.set_item("data", arr)?;
                        Ok(result.into_any().unbind())
                    })
                    .collect::<PyResult<_>>()?;
                Ok(PyList::new(py, py_results)?.into_any().unbind())
            })
        })
    }

    /// Batch-decode sub-array ranges across multiple messages via batched HTTP. Remote only. Call ``prefetch_layouts`` first to avoid per-message discovery overhead.
    #[pyo3(signature = (msg_indices, obj_index, ranges, join=false, verify_hash=false, native_byte_order=true, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn file_decode_range_batch<'py>(
        &self,
        py: Python<'py>,
        msg_indices: Vec<usize>,
        obj_index: usize,
        ranges: Vec<(u64, u64)>,
        join: bool,
        verify_hash: bool,
        native_byte_order: bool,
        threads: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            threads,
            ..Default::default()
        };
        let ranges_for_result = ranges.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let batch = file
                .decode_range_batch_async(&msg_indices, obj_index, &ranges, &options)
                .await
                .map_err(to_py_err)?;

            Python::attach(|py| {
                let py_results: Vec<PyObject> = batch
                    .into_iter()
                    .map(|(desc, parts)| {
                        build_range_result(py, desc.dtype, parts, &ranges_for_result, join)
                    })
                    .collect::<PyResult<_>>()?;
                Ok(PyList::new(py, py_results)?.into_any().unbind())
            })
        })
    }

    /// All raw message bytes as a list of ``bytes`` objects (async).
    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let file = Arc::clone(&self.file);
        let cache = Arc::clone(&self.cached_message_count);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let count = file.message_count_async().await.map_err(to_py_err)?;
            let _ = cache.set(count);
            let mut raw_msgs = Vec::with_capacity(count);
            for i in 0..count {
                let bytes = file.read_message_async(i).await.map_err(to_py_err)?;
                raw_msgs.push(bytes);
            }
            Python::attach(|py| {
                let items: Vec<PyObject> = raw_msgs
                    .iter()
                    .map(|m| PyBytes::new(py, m).into_any().unbind())
                    .collect();
                Ok(PyList::new(py, items)?.into_any().unbind())
            })
        })
    }

    // ── Async context manager ────────────────────────────────────────────

    fn __aenter__<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let obj = slf.unbind();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(obj) })
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __aexit__<'py>(
        &self,
        py: Python<'py>,
        _exc_type: Option<&Bound<'_, pyo3::PyAny>>,
        _exc_val: Option<&Bound<'_, pyo3::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(false) })
    }

    // ── __len__ (cached after first access via OnceLock) ───────────────

    fn __len__(&self) -> PyResult<usize> {
        if let Some(&count) = self.cached_message_count.get() {
            return Ok(count);
        }
        Err(PyRuntimeError::new_err(
            "message count not yet known; call 'await f.message_count()' first",
        ))
    }

    // ── Async iteration ──────────────────────────────────────────────────

    fn __aiter__(&self, _py: Python<'_>) -> PyResult<PyAsyncTensogramFileIter> {
        let count = self.__len__()?;
        Ok(PyAsyncTensogramFileIter {
            file: Arc::clone(&self.file),
            index: std::sync::atomic::AtomicUsize::new(0),
            count,
        })
    }

    fn __repr__(&self) -> String {
        format!("AsyncTensogramFile(source='{}')", self.cached_source)
    }
}

#[cfg(feature = "async")]
#[pyclass(name = "AsyncTensogramFileIter")]
struct PyAsyncTensogramFileIter {
    file: Arc<TensogramFile>,
    index: std::sync::atomic::AtomicUsize,
    count: usize,
}

#[cfg(feature = "async")]
#[pymethods]
impl PyAsyncTensogramFileIter {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let idx = self
            .index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if idx >= self.count {
            return Ok(None);
        }
        let file = Arc::clone(&self.file);
        let options = DecodeOptions::default();
        Ok(Some(pyo3_async_runtimes::tokio::future_into_py(
            py,
            async move {
                let (meta, objects) = file
                    .decode_message_async(idx, &options)
                    .await
                    .map_err(to_py_err)?;
                Python::attach(|py| {
                    let result_list = data_objects_to_python(py, &objects)?;
                    pack_message(py, PyMetadata { inner: meta }, result_list)
                })
            },
        )?))
    }

    fn __len__(&self) -> usize {
        let current = self.index.load(std::sync::atomic::Ordering::Relaxed);
        self.count.saturating_sub(current)
    }

    fn __repr__(&self) -> String {
        let current = self.index.load(std::sync::atomic::Ordering::Relaxed);
        let remaining = self.count.saturating_sub(current);
        format!("AsyncTensogramFileIter(position={current}, remaining={remaining})")
    }
}

/// Check whether a source string looks like a remote URL.
///
/// Returns ``True`` for sources starting with ``http://``, ``https://``,
/// ``s3://``, ``gs://``, or ``az://``; ``False`` for everything else
/// (local paths, unrecognised schemes).  Used by
/// :class:`TensogramFile` to dispatch between the local-file and
/// remote backends.
///
/// Args:
///     source: The source string to classify.
///
/// Returns:
///     ``True`` if ``source`` has a recognised remote scheme prefix.
///
/// Example::
///
///     >>> tensogram.is_remote_url("s3://bucket/key.tgm")
///     True
///     >>> tensogram.is_remote_url("/local/path.tgm")
///     False
#[pyfunction]
#[pyo3(name = "is_remote_url")]
fn py_is_remote_url(source: &str) -> bool {
    tensogram_lib::is_remote_url(source)
}

// ---------------------------------------------------------------------------
// GRIB / NetCDF conversion wrappers
//
// These wrap `tensogram-grib::convert_grib_file` / `convert_grib_buffer` and
// `tensogram-netcdf::convert_netcdf_file`.  They are *always* compiled into
// the Python module — when the `grib` / `netcdf` cargo feature is off, the
// function raises a `RuntimeError` at call time explaining how to rebuild
// the wheel with support enabled.  This matches the convention established
// by `tensogram.encode()` / `tensogram.decode()` (no feature-gated visibility
// on the Python side).
//
// The official manylinux PyPI wheels do NOT ship with these features enabled
// because libeccodes / libnetcdf are not part of the `manylinux_2_28` base
// image.  Users who need GRIB / NetCDF conversion either:
//   - install the C libraries + rebuild from source
//     (`maturin develop --features grib,netcdf`), or
//   - use the `tensogram` CLI binary which is distributed with these
//     features enabled on the platforms where they are supported.
// ---------------------------------------------------------------------------

/// Short-hand for the three converter-stub branches used when a Cargo
/// feature is off.  Eats the parameters (so clippy does not flag them
/// as unused) and returns the documented `RuntimeError`.
///
/// The macro is unused when BOTH `grib` and `netcdf` are on, which is the
/// common case — silence the warning rather than gate the definition on
/// a complex cfg expression.
#[allow(unused_macros)]
macro_rules! feature_disabled_stub {
    ($msg:expr, $( $arg:ident ),* $(,)?) => {{
        $( let _ = $arg; )*
        Err(PyRuntimeError::new_err($msg))
    }};
}

/// Build a [`tensogram_lib::DataPipeline`] from the shared pipeline
/// arguments (`encoding`, `bits`, `filter`, `compression`,
/// `compression_level`) exposed by the `convert_grib` and
/// `convert_netcdf` PyO3 wrappers.
///
/// Pure structural construction — all value-level validation (e.g.
/// rejecting an unknown codec name) happens later inside
/// `tensogram_encodings::pipeline::apply_pipeline`.  This helper only
/// converts the caller's `&str` arguments into the owned `String`
/// fields that `DataPipeline` stores.
///
/// Only compiled when at least one of the GRIB / NetCDF converters is
/// enabled; without those features the public `convert_grib` /
/// `convert_netcdf` entry points are stubs that never reach this code
/// path.
#[cfg(any(feature = "grib", feature = "netcdf"))]
fn build_data_pipeline(
    encoding: &str,
    bits: Option<u32>,
    filter: &str,
    compression: &str,
    compression_level: Option<i32>,
) -> tensogram_lib::DataPipeline {
    tensogram_lib::DataPipeline {
        encoding: encoding.to_string(),
        bits,
        filter: filter.to_string(),
        compression: compression.to_string(),
        compression_level,
    }
}

/// Build a `tensogram_grib::ConvertOptions` from the PyO3-level arguments,
/// rejecting unknown `grouping` / `hash` strings up front.
#[cfg(feature = "grib")]
#[allow(clippy::too_many_arguments)]
fn build_grib_options(
    grouping: &str,
    preserve_all_keys: bool,
    encoding: &str,
    bits: Option<u32>,
    filter: &str,
    compression: &str,
    compression_level: Option<i32>,
    threads: u32,
    hash: Option<&str>,
) -> PyResult<tensogram_grib::ConvertOptions> {
    let grouping = match grouping {
        "one_to_one" => tensogram_grib::Grouping::OneToOne,
        "merge_all" => tensogram_grib::Grouping::MergeAll,
        other => {
            return Err(PyValueError::new_err(format!(
                "grouping must be 'one_to_one' or 'merge_all', got {other:?}"
            )));
        }
    };

    Ok(tensogram_grib::ConvertOptions {
        grouping,
        preserve_all_keys,
        pipeline: build_data_pipeline(encoding, bits, filter, compression, compression_level),
        encode_options: make_encode_options(hash, threads)?,
    })
}

/// Build a `tensogram_netcdf::ConvertOptions` from the PyO3-level arguments,
/// rejecting unknown `split_by` / `hash` strings up front.
#[cfg(feature = "netcdf")]
#[allow(clippy::too_many_arguments)]
fn build_netcdf_options(
    split_by: &str,
    cf: bool,
    encoding: &str,
    bits: Option<u32>,
    filter: &str,
    compression: &str,
    compression_level: Option<i32>,
    threads: u32,
    hash: Option<&str>,
) -> PyResult<tensogram_netcdf::ConvertOptions> {
    let split_by = match split_by {
        "file" => tensogram_netcdf::SplitBy::File,
        "variable" => tensogram_netcdf::SplitBy::Variable,
        "record" => tensogram_netcdf::SplitBy::Record,
        other => {
            return Err(PyValueError::new_err(format!(
                "split_by must be 'file', 'variable' or 'record', got {other:?}"
            )));
        }
    };

    Ok(tensogram_netcdf::ConvertOptions {
        split_by,
        cf,
        pipeline: build_data_pipeline(encoding, bits, filter, compression, compression_level),
        encode_options: make_encode_options(hash, threads)?,
    })
}

/// Map a [`tensogram_grib::GribError`] to the most appropriate Python
/// exception. The routing follows the same convention as the rest of
/// the bindings (see `map_tensogram_error`):
///
/// - [`std::io::ErrorKind::NotFound`] → [`FileNotFoundError`]
///   (subclass of [`OSError`])
/// - Other [`GribError::Io`] variants → [`OSError`]
/// - Caller-input problems ([`GribError::NoMessages`],
///   [`GribError::InvalidData`], [`GribError::Encode`]) → [`ValueError`]
/// - Everything else (ecCodes C-library internal failures) → [`RuntimeError`]
#[cfg(feature = "grib")]
fn grib_error_to_pyerr(e: tensogram_grib::GribError) -> PyErr {
    use tensogram_grib::GribError;
    match &e {
        GribError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(e.to_string())
        }
        GribError::Io(_) => PyIOError::new_err(e.to_string()),
        GribError::NoMessages | GribError::InvalidData(_) | GribError::Encode(_) => {
            PyValueError::new_err(e.to_string())
        }
        GribError::EcCodes(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Map a [`tensogram_netcdf::NetcdfError`] to the most appropriate Python
/// exception. Same routing convention as [`grib_error_to_pyerr`]:
///
/// - [`std::io::ErrorKind::NotFound`] → [`FileNotFoundError`]
/// - Other [`NetcdfError::Io`] variants → [`OSError`]
/// - Caller-input problems → [`ValueError`] (including
///   [`NetcdfError::NoUnlimitedDimension`] which is raised when the
///   caller asks for `split_by="record"` against a file without one)
/// - libnetcdf C-library errors → [`RuntimeError`]
#[cfg(feature = "netcdf")]
fn netcdf_error_to_pyerr(e: tensogram_netcdf::NetcdfError) -> PyErr {
    use tensogram_netcdf::NetcdfError;
    match &e {
        NetcdfError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(e.to_string())
        }
        NetcdfError::Io(_) => PyIOError::new_err(e.to_string()),
        NetcdfError::NoVariables
        | NetcdfError::InvalidData(_)
        | NetcdfError::Encode(_)
        | NetcdfError::UnsupportedType { .. }
        | NetcdfError::NoUnlimitedDimension { .. } => PyValueError::new_err(e.to_string()),
        NetcdfError::Netcdf(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Wrap a list of Tensogram wire-format messages into a Python
/// `list[bytes]` for return from the `convert_grib` / `convert_netcdf`
/// PyO3 wrappers.
#[cfg(any(feature = "grib", feature = "netcdf"))]
fn messages_to_pybytes_list(py: Python<'_>, messages: Vec<Vec<u8>>) -> PyResult<PyObject> {
    let items: Vec<Bound<'_, PyBytes>> =
        messages.into_iter().map(|m| PyBytes::new(py, &m)).collect();
    Ok(PyList::new(py, items)?.into_any().unbind())
}

/// Convert a GRIB file to Tensogram wire-format messages.
///
/// Returns a list of `bytes`, one per produced Tensogram message.
/// Concatenate them (`b"".join(result)`) to get a complete multi-message
/// file, or write them sequentially to an open file handle.
///
/// # Python exceptions raised
/// - [`FileNotFoundError`] — `path` does not exist.
/// - [`OSError`] — other I/O failure (permission denied, etc.).
/// - [`ValueError`] — unknown `grouping` / `hash`; unknown codec name or
///   bit width in the pipeline; the buffer contained zero GRIB messages;
///   any other caller-input problem.
/// - [`RuntimeError`] — the `grib` Cargo feature is disabled, or an
///   ecCodes C-library internal error bubbled up.
///
/// # Arguments
/// - `path`: filesystem path to the GRIB file.
/// - `grouping`: `"merge_all"` (default — one Tensogram message per file
///   with N data objects) or `"one_to_one"` (one Tensogram message per
///   GRIB message).
/// - `preserve_all_keys`: when `True`, lift every ecCodes namespace key
///   into `base[i]["grib"]`.  Default `False` (MARS keys only).
/// - `encoding`: `"none"` (default) or `"simple_packing"`.
/// - `bits`: bits-per-value for `simple_packing` (1..=64). Defaults to
///   16 when `encoding="simple_packing"` and `bits=None`; ignored for
///   `encoding="none"`. Values outside `1..=64` cause the encoder to
///   emit a warning to stderr and silently fall back to `encoding="none"`.
/// - `filter`: `"none"` (default) or `"shuffle"`.
/// - `compression`: `"none"` (default), `"zstd"`, `"lz4"`, `"blosc2"`,
///   `"szip"`.
/// - `compression_level`: codec-specific integer (e.g. zstd 1..=22).
///   Defaults to the codec's own default when `None`.
/// - `threads`: parallelism budget. `0` uses the default behaviour and may
///   be overridden by `TENSOGRAM_THREADS=N`; use `1` to force
///   single-threaded execution.
/// - `hash`: `"xxh3"` (default) or `None` to skip hashing.
#[pyfunction]
#[pyo3(name = "convert_grib", signature = (
    path,
    *,
    grouping = "merge_all",
    preserve_all_keys = false,
    encoding = "none",
    bits = None,
    filter = "none",
    compression = "none",
    compression_level = None,
    threads = 0,
    hash = Some("xxh3"),
))]
#[allow(clippy::too_many_arguments)]
fn py_convert_grib(
    py: Python<'_>,
    path: &str,
    grouping: &str,
    preserve_all_keys: bool,
    encoding: &str,
    bits: Option<u32>,
    filter: &str,
    compression: &str,
    compression_level: Option<i32>,
    threads: u32,
    hash: Option<&str>,
) -> PyResult<PyObject> {
    #[cfg(feature = "grib")]
    {
        let options = build_grib_options(
            grouping,
            preserve_all_keys,
            encoding,
            bits,
            filter,
            compression,
            compression_level,
            threads,
            hash,
        )?;
        let path_buf = std::path::PathBuf::from(path);

        // Pre-check path existence so callers receive a standard Python
        // FileNotFoundError rather than whatever ecCodes happens to
        // surface on a missing path (typically CodesError::LibcNonZero,
        // which would otherwise map to RuntimeError).
        if !path_buf.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "no such file: {path}"
            )));
        }

        let messages = py
            .detach(|| tensogram_grib::convert_grib_file(&path_buf, &options))
            .map_err(grib_error_to_pyerr)?;
        messages_to_pybytes_list(py, messages)
    }
    #[cfg(not(feature = "grib"))]
    feature_disabled_stub!(
        "tensogram was built without GRIB support. Install libeccodes \
         (`brew install eccodes` or `apt install libeccodes-dev`) and \
         rebuild the Python bindings with `maturin develop --features grib`.",
        py,
        path,
        grouping,
        preserve_all_keys,
        encoding,
        bits,
        filter,
        compression,
        compression_level,
        threads,
        hash,
    )
}

/// Convert a GRIB byte buffer to Tensogram wire-format messages.
///
/// In-memory equivalent of [`convert_grib`].  Accepts any Python
/// bytes-like object (`bytes`, `bytearray`, memoryview, `numpy.uint8[:]`)
/// and releases the GIL during the conversion itself.  Accepts the
/// same `reject_nan` / `reject_inf` kwargs as [`convert_grib`].
///
/// # Python exceptions raised
/// - [`ValueError`] — `buffer` is not a bytes-like object; the buffer
///   contained zero GRIB messages; or any other caller-input problem
///   (same taxonomy as [`convert_grib`]).
/// - [`RuntimeError`] — the `grib` Cargo feature is disabled, or an
///   ecCodes C-library internal error bubbled up.
///
/// Useful when:
/// - you already have GRIB bytes in memory (e.g. from a byte-range
///   HTTP fetch), and
/// - you do not want to stage the bytes through the filesystem.
#[pyfunction]
#[pyo3(name = "convert_grib_buffer", signature = (
    buffer,
    *,
    grouping = "merge_all",
    preserve_all_keys = false,
    encoding = "none",
    bits = None,
    filter = "none",
    compression = "none",
    compression_level = None,
    threads = 0,
    hash = Some("xxh3"),
))]
#[allow(clippy::too_many_arguments)]
fn py_convert_grib_buffer(
    py: Python<'_>,
    buffer: &Bound<'_, PyAny>,
    grouping: &str,
    preserve_all_keys: bool,
    encoding: &str,
    bits: Option<u32>,
    filter: &str,
    compression: &str,
    compression_level: Option<i32>,
    threads: u32,
    hash: Option<&str>,
) -> PyResult<PyObject> {
    #[cfg(feature = "grib")]
    {
        // Accept any Python bytes-like object. PyO3's `extract::<Vec<u8>>`
        // goes through the buffer protocol, so `bytes`, `bytearray`,
        // `memoryview`, and numpy `uint8` arrays all work. We need an owned
        // `Vec<u8>` because ecCodes' `new_from_memory` takes ownership
        // (it hands the buffer to `fmemopen(3)`).
        let owned: Vec<u8> = buffer.extract().map_err(|_| {
            PyValueError::new_err(
                "convert_grib_buffer expects a bytes-like object \
                 (bytes, bytearray, memoryview, numpy.uint8[:])",
            )
        })?;

        let options = build_grib_options(
            grouping,
            preserve_all_keys,
            encoding,
            bits,
            filter,
            compression,
            compression_level,
            threads,
            hash,
        )?;

        let messages = py
            .detach(|| tensogram_grib::convert_grib_buffer(owned, &options))
            .map_err(grib_error_to_pyerr)?;
        messages_to_pybytes_list(py, messages)
    }
    #[cfg(not(feature = "grib"))]
    feature_disabled_stub!(
        "tensogram was built without GRIB support. Install libeccodes \
         (`brew install eccodes` or `apt install libeccodes-dev`) and \
         rebuild the Python bindings with `maturin develop --features grib`.",
        py,
        buffer,
        grouping,
        preserve_all_keys,
        encoding,
        bits,
        filter,
        compression,
        compression_level,
        threads,
        hash,
    )
}

/// Convert a NetCDF file to Tensogram wire-format messages.
///
/// Returns a list of `bytes`, one per produced Tensogram message.
///
/// # Python exceptions raised
/// - [`FileNotFoundError`] — `path` does not exist.
/// - [`OSError`] — other I/O failure.
/// - [`ValueError`] — unknown `split_by` / `hash`; unknown codec name or
///   bit width; the file contains no variables; `split_by="record"` was
///   requested on a file without an unlimited dimension; any other
///   caller-input problem.
/// - [`RuntimeError`] — the `netcdf` Cargo feature is disabled, or a
///   libnetcdf C-library internal error bubbled up.
///
/// # Arguments
/// - `path`: filesystem path to the NetCDF file (`.nc`, NetCDF-3 classic
///   or NetCDF-4/HDF5).
/// - `split_by`: `"file"` (default — one Tensogram message with N
///   variables), `"variable"` (one message per variable), or `"record"`
///   (one message per unlimited-dimension record).
/// - `cf`: when `True`, lift the 16 allow-listed CF attributes into
///   `base[i]["cf"]`.
/// - `encoding`: `"none"` (default) or `"simple_packing"`.
/// - `bits`: bits-per-value for `simple_packing` (1..=64). Defaults to
///   16 when `encoding="simple_packing"` and `bits=None`. See
///   [`convert_grib`] for the full behaviour.
/// - `filter` / `compression` / `compression_level`: see [`convert_grib`].
/// - `threads`: parallelism budget. `0` uses the default behaviour and may
///   be overridden by `TENSOGRAM_THREADS=N`; use `1` to force
///   single-threaded execution.
/// - `hash`: `"xxh3"` (default) or `None` to skip hashing.
#[pyfunction]
#[pyo3(name = "convert_netcdf", signature = (
    path,
    *,
    split_by = "file",
    cf = false,
    encoding = "none",
    bits = None,
    filter = "none",
    compression = "none",
    compression_level = None,
    threads = 0,
    hash = Some("xxh3"),
))]
#[allow(clippy::too_many_arguments)]
fn py_convert_netcdf(
    py: Python<'_>,
    path: &str,
    split_by: &str,
    cf: bool,
    encoding: &str,
    bits: Option<u32>,
    filter: &str,
    compression: &str,
    compression_level: Option<i32>,
    threads: u32,
    hash: Option<&str>,
) -> PyResult<PyObject> {
    #[cfg(feature = "netcdf")]
    {
        let options = build_netcdf_options(
            split_by,
            cf,
            encoding,
            bits,
            filter,
            compression,
            compression_level,
            threads,
            hash,
        )?;
        let path_buf = std::path::PathBuf::from(path);

        // Pre-check path existence — same reasoning as `py_convert_grib`.
        if !path_buf.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "no such file: {path}"
            )));
        }

        let messages = py
            .detach(|| tensogram_netcdf::convert_netcdf_file(&path_buf, &options))
            .map_err(netcdf_error_to_pyerr)?;
        messages_to_pybytes_list(py, messages)
    }
    #[cfg(not(feature = "netcdf"))]
    feature_disabled_stub!(
        "tensogram was built without NetCDF support. Install libnetcdf \
         (`brew install netcdf` or `apt install libnetcdf-dev`) and \
         rebuild the Python bindings with `maturin develop --features netcdf`.",
        py,
        path,
        split_by,
        cf,
        encoding,
        bits,
        filter,
        compression,
        compression_level,
        threads,
        hash,
    )
}

#[pymodule(gil_used = false)]
fn tensogram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_is_remote_url, m)?)?;
    m.add_function(wrap_pyfunction!(py_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_encode_pre_encoded, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_with_masks, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_object, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_range, m)?)?;
    m.add_function(wrap_pyfunction!(py_scan, m)?)?;
    m.add_function(wrap_pyfunction!(py_iter_messages, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_hash, m)?)?;
    m.add_function(wrap_pyfunction!(compute_packing_params, m)?)?;
    m.add_function(wrap_pyfunction!(py_convert_grib, m)?)?;
    m.add_function(wrap_pyfunction!(py_convert_grib_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(py_convert_netcdf, m)?)?;
    m.add_function(wrap_pyfunction!(py_doctor, m)?)?;
    m.add_class::<PyMetadata>()?;
    m.add_class::<PyDataObjectDescriptor>()?;
    m.add_class::<PyTensogramFile>()?;
    m.add_class::<PyFileIter>()?;
    m.add_class::<PyBufferIter>()?;
    m.add_class::<PyStreamingEncoder>()?;
    #[cfg(feature = "async")]
    m.add_class::<PyAsyncTensogramFile>()?;
    #[cfg(feature = "async")]
    m.add_class::<PyAsyncTensogramFileIter>()?;

    // Feature-flag probes for Python consumers. Useful for tests and for
    // branching behaviour in notebooks / examples so they can skip cleanly
    // when the wheel was built without a converter feature:
    //   if tensogram.__has_grib__:
    //       msgs = tensogram.convert_grib(...)
    m.add("__has_grib__", cfg!(feature = "grib"))?;
    m.add("__has_netcdf__", cfg!(feature = "netcdf"))?;

    // Wire-format version constant.  See `plans/WIRE_FORMAT.md` §3 —
    // the version lives in the preamble of every message, never in the
    // CBOR metadata frame.  Exposed here so tooling and notebooks can
    // read it without having to decode a message first.
    m.add("WIRE_VERSION", tensogram_lib::WIRE_VERSION)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_encode_options(hash: Option<&str>, threads: u32) -> PyResult<EncodeOptions> {
    make_encode_options_full(
        hash, threads, false, false, None, None, None, None, None, None,
    )
}

/// Build an [`EncodeOptions`] from the full kwargs set exposed by the
/// Python-facing `encode` / `append` / `StreamingEncoder.create`
/// entry points.
///
/// Mask method names follow [`plans/WIRE_FORMAT.md` §6.5.1]:
/// `"none"` | `"rle"` | `"roaring"` | `"lz4"` | `"zstd"` |
/// `"blosc2"`.  Missing sentinels use the library defaults
/// (`Roaring` for methods, `128` for the small-mask fallback
/// threshold).
///
/// `create_header_hashes` / `create_footer_hashes` are v3 opt-in
/// flags controlling the aggregate HashFrame emission.  When
/// `None`, the library default applies (buffered mode: header-
/// only; streaming mode: footer-only).  Either or both may be
/// set explicitly — streaming mode silently folds
/// `create_header_hashes = true` into `create_footer_hashes`
/// because a streaming header is emitted before any data object.
#[allow(clippy::too_many_arguments)]
fn make_encode_options_full(
    hash: Option<&str>,
    threads: u32,
    allow_nan: bool,
    allow_inf: bool,
    nan_mask_method: Option<&str>,
    pos_inf_mask_method: Option<&str>,
    neg_inf_mask_method: Option<&str>,
    small_mask_threshold_bytes: Option<usize>,
    create_header_hashes: Option<bool>,
    create_footer_hashes: Option<bool>,
) -> PyResult<EncodeOptions> {
    use tensogram_lib::encode::MaskMethod;

    let hash_algorithm = match hash {
        None => None,
        Some("xxh3") => Some(HashAlgorithm::Xxh3),
        Some(other) => return Err(PyValueError::new_err(format!("unknown hash: {other}"))),
    };
    let parse_method = |s: Option<&str>, default: MaskMethod| -> PyResult<MaskMethod> {
        let Some(name) = s else {
            return Ok(default);
        };
        // Delegate the error message to `MaskError::UnknownMethod`'s
        // Display so the accepted-names list stays in one place
        // across every binding.
        MaskMethod::from_name(name).map_err(|e| PyValueError::new_err(e.to_string()))
    };

    let defaults = EncodeOptions::default();
    Ok(EncodeOptions {
        hash_algorithm,
        threads,
        allow_nan,
        allow_inf,
        nan_mask_method: parse_method(nan_mask_method, defaults.nan_mask_method)?,
        pos_inf_mask_method: parse_method(pos_inf_mask_method, defaults.pos_inf_mask_method)?,
        neg_inf_mask_method: parse_method(neg_inf_mask_method, defaults.neg_inf_mask_method)?,
        small_mask_threshold_bytes: small_mask_threshold_bytes
            .unwrap_or(defaults.small_mask_threshold_bytes),
        create_header_hashes: create_header_hashes.unwrap_or(defaults.create_header_hashes),
        create_footer_hashes: create_footer_hashes.unwrap_or(defaults.create_footer_hashes),
        ..defaults
    })
}

/// Extract a list of (descriptor, raw_bytes) pairs from a Python list of
/// (descriptor_dict, data) tuples used by `encode` and `append`.
fn extract_descriptor_data_pairs(
    _py: Python<'_>,
    pairs_list: &Bound<'_, PyList>,
) -> PyResult<Vec<(DataObjectDescriptor, Vec<u8>)>> {
    let mut result = Vec::new();
    for item in pairs_list.iter() {
        let tuple = item.cast::<pyo3::types::PyTuple>().map_err(|_| {
            PyValueError::new_err("each element must be a (descriptor_dict, data) tuple")
        })?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "each element must be a (descriptor_dict, data) tuple of length 2",
            ));
        }
        let desc_dict = tuple.get_item(0)?.cast_into::<PyDict>().map_err(|_| {
            PyValueError::new_err("first element of each pair must be a descriptor dict")
        })?;
        let desc = dict_to_data_object_descriptor(&desc_dict)?;
        let data_item = tuple.get_item(1)?;
        let data = extract_single_data_item(&data_item)?;
        result.push((desc, data));
    }
    Ok(result)
}

/// Extract raw native-endian bytes from a numpy array (any supported dtype)
/// or a Python `bytes` object.
///
/// Tries each numpy scalar type in turn. Uses `to_ne_bytes()` to produce
/// the byte representation the Rust encoder expects.
fn extract_single_data_item(item: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<u8>> {
    // Macro: try to extract a numpy array of type $T, convert via to_ne_bytes.
    macro_rules! try_numpy {
        ($T:ty) => {
            if let Ok(arr) = item.extract::<numpy::PyReadonlyArrayDyn<'_, $T>>() {
                let s = arr
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                return Ok(s.iter().flat_map(|v| v.to_ne_bytes()).collect());
            }
        };
    }

    // u8 is special — no byte-swap needed, just copy.
    if let Ok(arr) = item.extract::<numpy::PyReadonlyArrayDyn<'_, u8>>() {
        return arr
            .as_slice()
            .map(|s| s.to_vec())
            .map_err(|e| PyValueError::new_err(format!("{e}")));
    }

    try_numpy!(u16);
    try_numpy!(u32);
    try_numpy!(u64);
    try_numpy!(i8);
    try_numpy!(i16);
    try_numpy!(i32);
    try_numpy!(i64);
    try_numpy!(f32);
    try_numpy!(f64);

    // Raw bytes fallback
    if let Ok(b) = item.extract::<Vec<u8>>() {
        return Ok(b);
    }
    Err(PyValueError::new_err("data must be a numpy array or bytes"))
}

/// Extract a list of (descriptor, raw_bytes) pairs for `encode_pre_encoded`.
///
/// Unlike [`extract_descriptor_data_pairs`] this function **only** accepts
/// `bytes`-like objects as the data element — numpy arrays are rejected because
/// pre-encoded payloads are already in their final wire form.
fn extract_pre_encoded_pairs(
    pairs_list: &Bound<'_, PyList>,
) -> PyResult<Vec<(DataObjectDescriptor, Vec<u8>)>> {
    let mut result = Vec::new();
    for item in pairs_list.iter() {
        let tuple = item.cast::<pyo3::types::PyTuple>().map_err(|_| {
            PyValueError::new_err("each element must be a (descriptor_dict, bytes) tuple")
        })?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "each element must be a (descriptor_dict, bytes) tuple of length 2",
            ));
        }
        let desc_dict = tuple.get_item(0)?.cast_into::<PyDict>().map_err(|_| {
            PyValueError::new_err("first element of each pair must be a descriptor dict")
        })?;
        let desc = dict_to_data_object_descriptor(&desc_dict)?;
        let data_item = tuple.get_item(1)?;
        // Only accept bytes-like objects, not numpy arrays.
        let data = data_item.extract::<Vec<u8>>().map_err(|_| {
            let type_name = data_item
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<unknown>".to_string());
            PyValueError::new_err(format!(
                "encode_pre_encoded requires bytes data (got {type_name}) — \
                 the payload must already be in its final wire form",
            ))
        })?;
        result.push((desc, data));
    }
    Ok(result)
}

/// Convert decoded data objects to a Python list of (descriptor, ndarray) tuples.
fn data_objects_to_python(
    py: Python<'_>,
    data_objects: &[(DataObjectDescriptor, Vec<u8>)],
) -> PyResult<PyObject> {
    let items: PyResult<Vec<PyObject>> = data_objects
        .iter()
        .map(|(desc, bytes)| {
            let arr = bytes_to_numpy(py, desc, bytes)?;
            let py_desc = PyDataObjectDescriptor {
                inner: desc.clone(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind();
            let pair = pyo3::types::PyTuple::new(py, [py_desc, arr])?
                .into_any()
                .unbind();
            Ok(pair)
        })
        .collect();
    Ok(PyList::new(py, items?)?.into_any().unbind())
}

/// Build a Python list of `(descriptor, payload_array, masks_dict)`
/// tuples for [`py_decode_with_masks`].  Each `masks_dict` contains
/// only the kinds that were actually present in the frame (empty
/// dict for frames without a mask companion).
fn objects_with_masks_to_python(
    py: Python<'_>,
    objects: &[tensogram_lib::DecodedObjectWithMasks],
) -> PyResult<PyObject> {
    use numpy::PyArray1;
    let items: PyResult<Vec<PyObject>> = objects
        .iter()
        .map(|obj| {
            let arr = bytes_to_numpy(py, &obj.descriptor, &obj.payload)?;
            let py_desc = PyDataObjectDescriptor {
                inner: obj.descriptor.clone(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind();
            let masks_dict = PyDict::new(py);
            let add_kind = |key: &str, bits: &Option<Vec<bool>>| -> PyResult<()> {
                if let Some(bits) = bits {
                    let arr = PyArray1::<bool>::from_slice(py, bits.as_slice());
                    masks_dict.set_item(key, arr)?;
                }
                Ok(())
            };
            add_kind("nan", &obj.masks.nan)?;
            add_kind("inf+", &obj.masks.pos_inf)?;
            add_kind("inf-", &obj.masks.neg_inf)?;
            let triple = pyo3::types::PyTuple::new(
                py,
                [py_desc, arr, masks_dict.into_any().unbind()],
            )?
            .into_any()
            .unbind();
            Ok(triple)
        })
        .collect();
    Ok(PyList::new(py, items?)?.into_any().unbind())
}

/// Build a GlobalMetadata from a Python dict.
///
/// The CBOR metadata frame is **fully free-form** (see
/// `plans/WIRE_FORMAT.md` §6.1).  The only top-level keys the library
/// interprets are ``"base"`` (list of per-object dicts) and ``"_extra_"``
/// / ``"extra"`` (convenience alias for message-level free-form keys).
///
/// Anything else supplied by the caller — including a stray legacy
/// ``"version"`` key — flows into ``_extra_``.  The wire-format version
/// lives exclusively in the preamble and is not settable from here.
///
/// ``"_reserved_"`` remains protected: it is the library-managed
/// provenance namespace and cannot be written by client code.
///
/// # Priority rule for `_extra_` collisions
///
/// When a caller supplies both an explicit ``_extra_`` (or ``extra``)
/// section AND a free-form top-level key with the same name (e.g.
/// ``{"_extra_": {"version": 1}, "version": 99}``), the **explicit
/// section wins**.  Free-form top-level keys only fill slots that
/// ``_extra_`` did not already claim — matching the "explicit beats
/// implicit" principle.
fn dict_to_global_metadata(dict: &Bound<'_, PyDict>) -> PyResult<GlobalMetadata> {
    let base = match dict.get_item("base")? {
        Some(v) => {
            let list = v
                .cast::<PyList>()
                .map_err(|_| PyValueError::new_err("'base' must be a list of dicts"))?;
            let mut entries = Vec::with_capacity(list.len());
            for item in list.iter() {
                let entry = py_dict_to_btree(&item)?;
                // Validate: no _reserved_ keys in base entries
                if entry.contains_key(RESERVED_KEY) {
                    return Err(PyValueError::new_err(format!(
                        "base entries must not contain '{RESERVED_KEY}' key — the encoder populates it",
                    )));
                }
                entries.push(entry);
            }
            entries
        }
        None => Vec::new(),
    };

    // Accept both "_extra_" (wire name) and "extra" (convenience alias).
    // Either must be a dict when present — the error carries the key the
    // caller actually supplied so they can fix the right place.
    let explicit_extra = match dict.get_item("_extra_")? {
        Some(v) => {
            if v.cast::<PyDict>().is_err() {
                return Err(PyValueError::new_err("'_extra_' must be a dict"));
            }
            py_dict_to_btree(&v)?
        }
        None => match dict.get_item("extra")? {
            Some(v) => {
                if v.cast::<PyDict>().is_err() {
                    return Err(PyValueError::new_err(
                        "'extra' must be a dict when provided as a convenience alias for '_extra_'",
                    ));
                }
                py_dict_to_btree(&v)?
            }
            None => BTreeMap::new(),
        },
    };

    // Reject explicit _reserved_ — library-managed, encoder populates it
    if dict.get_item(RESERVED_KEY)?.is_some() {
        return Err(PyValueError::new_err(format!(
            "'{RESERVED_KEY}' must not be set by client code — the encoder populates it",
        )));
    }

    // Known (library-interpreted) top-level keys.  `"version"` is NOT
    // here: it is a free-form key like any other and flows into
    // ``_extra_`` when present.
    let known_keys = ["base", "_extra_", "extra", RESERVED_KEY];
    // Explicit `_extra_` beats implicit free-form top-level keys on
    // collision — only fill slots that `_extra_` did not already claim.
    let mut extra = explicit_extra;
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if !known_keys.contains(&key.as_str()) && !extra.contains_key(&key) {
            extra.insert(key, py_to_cbor(&v)?);
        }
    }

    Ok(GlobalMetadata {
        base,
        reserved: BTreeMap::new(),
        extra,
    })
}

/// Convert a Python dict (or dict-like CBOR map value) to `BTreeMap<String, CborValue>`.
fn py_dict_to_btree(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<BTreeMap<String, ciborium::Value>> {
    let dict = obj
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("expected a dict"))?;
    let mut map = BTreeMap::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        map.insert(key, py_to_cbor(&v)?);
    }
    Ok(map)
}

/// Extract an optional string value from a Python dict, returning *default* if absent.
fn get_string_or_default(dict: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    dict.get_item(key)?
        .map(|v| v.extract::<String>())
        .transpose()?
        .map_or_else(|| Ok(default.to_string()), Ok)
}

/// Build a DataObjectDescriptor from a Python dict.
///
/// Required keys: `type`, `shape`, `dtype`.
/// Optional keys: `strides` (defaults to row-major from `shape`),
///   `byte_order` (defaults to native), `encoding` / `filter` /
///   `compression` (default "none").
/// All other keys are stored in `params`.
fn dict_to_data_object_descriptor(dict: &Bound<'_, PyDict>) -> PyResult<DataObjectDescriptor> {
    let obj_type: String = dict
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("descriptor missing 'type'"))?
        .extract()?;
    let shape: Vec<u64> = dict
        .get_item("shape")?
        .ok_or_else(|| PyValueError::new_err("descriptor missing 'shape'"))?
        .extract()?;
    let dtype_str: String = dict
        .get_item("dtype")?
        .ok_or_else(|| PyValueError::new_err("descriptor missing 'dtype'"))?
        .extract()?;

    let ndim = shape.len() as u64;

    let strides: Vec<u64> = if let Some(s) = dict.get_item("strides")? {
        s.extract()?
    } else {
        compute_strides(&shape)?
    };

    let dtype = parse_dtype(&dtype_str)?;

    let byte_order = match dict.get_item("byte_order")? {
        None => ByteOrder::native(),
        Some(v) => match v.extract::<String>()?.as_str() {
            "little" => ByteOrder::Little,
            "big" => ByteOrder::Big,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown byte_order: '{other}', expected 'little' or 'big'"
                )));
            }
        },
    };

    let encoding = get_string_or_default(dict, "encoding", "none")?;
    let filter = get_string_or_default(dict, "filter", "none")?;
    let compression = get_string_or_default(dict, "compression", "none")?;

    // All remaining keys → params
    let reserved_keys = [
        "type",
        "ndim",
        "shape",
        "strides",
        "dtype",
        "byte_order",
        "encoding",
        "filter",
        "compression",
        "hash",
    ];
    let mut params = BTreeMap::new();
    let mut misplaced_metadata_keys: Vec<String> = Vec::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if !reserved_keys.contains(&key.as_str()) {
            if is_metadata_like_key(&key) {
                misplaced_metadata_keys.push(key.clone());
            }
            params.insert(key, py_to_cbor(&v)?);
        }
    }
    if !misplaced_metadata_keys.is_empty() {
        warn_misplaced_metadata(dict.py(), &misplaced_metadata_keys)?;
    }

    Ok(DataObjectDescriptor {
        obj_type,
        ndim,
        shape,
        strides,
        dtype,
        byte_order,
        encoding,
        filter,
        compression,
        params,
        masks: None,
    })
}

/// Keys that look like application metadata rather than encoding parameters.
///
/// When a user places one of these in the descriptor dict, it is still
/// captured into ``DataObjectDescriptor.params`` for wire compatibility,
/// but a ``UserWarning`` is emitted pointing at ``meta['base'][i]`` as
/// the correct location.  Keys fall into three groups:
///
/// * Consumed by zarr/xarray's naming chain (``resolve_variable_name``).
/// * Consumed by xarray's dimension logic (``base[i]["dim_names"]``).
/// * Obvious namespace roots from established vocabularies.
///
/// See issue #67 and ``python/tensogram-zarr/src/tensogram_zarr/mapping.py``
/// for the naming-chain key list.
const METADATA_LIKE_DESCRIPTOR_KEYS: &[&str] = &[
    "name",
    "param",
    "shortName",
    "long_name",
    "description",
    "units",
    "dim_names",
    "mars",
    "cf",
    "product",
    "instrument",
];

fn is_metadata_like_key(key: &str) -> bool {
    METADATA_LIKE_DESCRIPTOR_KEYS.contains(&key)
}

/// Emit a single aggregated ``UserWarning`` listing descriptor keys that
/// look like application metadata.  Multi-object messages thus warn at
/// most once per descriptor rather than once per key.
fn warn_misplaced_metadata(py: Python<'_>, keys: &[String]) -> PyResult<()> {
    let key_list = keys
        .iter()
        .map(|k| format!("'{k}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let message = format!(
        "descriptor contains application-metadata-like keys ({key_list}) \
         that were captured into DataObjectDescriptor.params.  Application \
         metadata belongs in meta['base'][i] — placing it in the descriptor \
         dict works (xarray/zarr fall back to descriptor params) but is not \
         canonical and may produce unexpected results in downstream tools. \
         See examples/python/02b_generic_metadata.py for the recommended \
         pattern."
    );
    let c_message = CString::new(message)
        .map_err(|_| PyValueError::new_err("internal: warning message contained NUL"))?;
    let warning_type = py.get_type::<PyUserWarning>();
    PyErr::warn(py, &warning_type, &c_message, 1)?;
    Ok(())
}

fn parse_dtype(s: &str) -> PyResult<Dtype> {
    match s {
        "float16" => Ok(Dtype::Float16),
        "bfloat16" => Ok(Dtype::Bfloat16),
        "float32" => Ok(Dtype::Float32),
        "float64" => Ok(Dtype::Float64),
        "complex64" => Ok(Dtype::Complex64),
        "complex128" => Ok(Dtype::Complex128),
        "int8" => Ok(Dtype::Int8),
        "int16" => Ok(Dtype::Int16),
        "int32" => Ok(Dtype::Int32),
        "int64" => Ok(Dtype::Int64),
        "uint8" => Ok(Dtype::Uint8),
        "uint16" => Ok(Dtype::Uint16),
        "uint32" => Ok(Dtype::Uint32),
        "uint64" => Ok(Dtype::Uint64),
        "bitmask" => Ok(Dtype::Bitmask),
        _ => Err(PyValueError::new_err(format!("unknown dtype: {s}"))),
    }
}

fn compute_strides(shape: &[u64]) -> PyResult<Vec<u64>> {
    if shape.is_empty() {
        return Ok(vec![]);
    }
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1]).ok_or_else(|| {
            PyValueError::new_err(format!(
                "strides overflow: cannot compute strides for shape {:?}",
                shape
            ))
        })?;
    }
    Ok(strides)
}

// ---------------------------------------------------------------------------
// Numpy conversion — macro-driven to avoid repeating the same pattern
// for each numeric dtype.
//
// `chunks_exact(N)` guarantees each chunk is exactly N bytes, so
// `try_into()` on the chunk→fixed-array conversion cannot fail.
// We still propagate as `PyValueError` instead of panicking because
// this code sits behind the FFI boundary (`panic = "abort"`).
// ---------------------------------------------------------------------------

/// Decode raw bytes into a flat Vec<$T> by reinterpreting via from_ne_bytes.
/// Returns a `PyValueError` if `bytes.len()` is not an exact multiple of `$width`.
macro_rules! decode_ne_vec {
    ($bytes:expr, $T:ty, $width:expr) => {{
        if $bytes.len() % $width != 0 {
            return Err(PyValueError::new_err(format!(
                "byte length {} is not a multiple of element width {}",
                $bytes.len(),
                $width,
            )));
        }
        $bytes
            .chunks_exact($width)
            .map(|c| -> Result<$T, PyErr> {
                let arr: [u8; $width] = c
                    .try_into()
                    .map_err(|_| PyValueError::new_err("internal: chunk size mismatch"))?;
                Ok(<$T>::from_ne_bytes(arr))
            })
            .collect::<Result<Vec<$T>, PyErr>>()?
    }};
}

/// Build a shaped numpy array from raw bytes for a given dtype.
macro_rules! numpy_from_ne {
    ($py:expr, $bytes:expr, $shape:expr, $T:ty, $width:expr) => {{
        let values = decode_ne_vec!($bytes, $T, $width);
        let arr = numpy::PyArray::from_vec($py, values).reshape($shape)?;
        Ok(arr.into_any().unbind())
    }};
}

/// Build a flat (1-d) numpy array from raw bytes for a given dtype.
macro_rules! numpy_flat_from_ne {
    ($py:expr, $bytes:expr, $T:ty, $width:expr) => {{
        let values = decode_ne_vec!($bytes, $T, $width);
        let arr = numpy::PyArray::from_vec($py, values);
        Ok(arr.into_any().unbind())
    }};
}

/// Create a numpy array via ``numpy.frombuffer(...).reshape(shape).copy()``.
///
/// Works for any dtype string that numpy understands (``"float16"``,
/// ``"complex64"``, etc.).
fn numpy_via_frombuffer(
    py: Python<'_>,
    bytes: &[u8],
    dtype_str: &str,
    shape: &[usize],
) -> PyResult<PyObject> {
    let np = py.import("numpy")?;
    let py_bytes = PyBytes::new(py, bytes);
    let dtype = np.getattr("dtype")?.call1((dtype_str,))?;
    let arr = np.call_method1("frombuffer", (py_bytes, dtype))?;
    let py_shape = pyo3::types::PyTuple::new(py, shape.iter().map(|&s| s as isize))?;
    let reshaped = arr.call_method1("reshape", (py_shape,))?;
    let copied = reshaped.call_method0("copy")?;
    Ok(copied.unbind())
}

/// Create a numpy array with ``ml_dtypes.bfloat16`` dtype.
///
/// Falls back to ``uint16`` view if ``ml_dtypes`` is not installed.
fn numpy_via_frombuffer_bfloat16(
    py: Python<'_>,
    bytes: &[u8],
    shape: &[usize],
) -> PyResult<PyObject> {
    // Try ml_dtypes first.
    if let Ok(ml) = py.import("ml_dtypes") {
        let np = py.import("numpy")?;
        let py_bytes = PyBytes::new(py, bytes);
        let dtype = ml.getattr("bfloat16")?;
        let arr = np.call_method1("frombuffer", (py_bytes, dtype))?;
        let py_shape = pyo3::types::PyTuple::new(py, shape.iter().map(|&s| s as isize))?;
        let reshaped = arr.call_method1("reshape", (py_shape,))?;
        let copied = reshaped.call_method0("copy")?;
        return Ok(copied.unbind());
    }
    // Fallback: return as uint16.
    numpy_via_frombuffer(py, bytes, "uint16", shape)
}

/// Convert decoded object bytes to a shaped numpy array.
fn bytes_to_numpy(py: Python<'_>, desc: &DataObjectDescriptor, bytes: &[u8]) -> PyResult<PyObject> {
    // simple_packing always produces f64 output regardless of declared dtype.
    let effective_dtype = if desc.encoding == "simple_packing" {
        Dtype::Float64
    } else {
        desc.dtype
    };

    let shape: Vec<usize> = desc.shape.iter().map(|&s| s as usize).collect();

    match effective_dtype {
        Dtype::Float32 => numpy_from_ne!(py, bytes, shape, f32, 4),
        Dtype::Float64 => numpy_from_ne!(py, bytes, shape, f64, 8),
        Dtype::Int8 => {
            // Safety: i8 and u8 have identical in-memory layout. Cast the slice
            // to &[i8] to avoid an intermediate Vec allocation; from_slice still
            // copies the data into the NumPy-owned buffer.
            let values: &[i8] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) };
            let arr = numpy::PyArray::from_slice(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Int16 => numpy_from_ne!(py, bytes, shape, i16, 2),
        Dtype::Int32 => numpy_from_ne!(py, bytes, shape, i32, 4),
        Dtype::Int64 => numpy_from_ne!(py, bytes, shape, i64, 8),
        Dtype::Uint8 => {
            // Avoids intermediate Vec allocation; from_slice copies directly into
            // the NumPy-owned buffer.
            let arr = numpy::PyArray::from_slice(py, bytes).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Uint16 => numpy_from_ne!(py, bytes, shape, u16, 2),
        Dtype::Uint32 => numpy_from_ne!(py, bytes, shape, u32, 4),
        Dtype::Uint64 => numpy_from_ne!(py, bytes, shape, u64, 8),
        Dtype::Float16 => {
            // numpy natively supports float16: reinterpret raw bytes.
            numpy_via_frombuffer(py, bytes, "float16", &shape)
        }
        Dtype::Bfloat16 => {
            // Try ml_dtypes.bfloat16 first, fall back to uint16.
            numpy_via_frombuffer_bfloat16(py, bytes, &shape)
        }
        Dtype::Complex64 => {
            // complex64 = two float32 per element.
            numpy_via_frombuffer(py, bytes, "complex64", &shape)
        }
        Dtype::Complex128 => {
            // complex128 = two float64 per element.
            numpy_via_frombuffer(py, bytes, "complex128", &shape)
        }
        _ => {
            // Bitmask has byte_width=0 (sub-byte); return raw bytes as uint8.
            let arr = numpy::PyArray::from_slice(py, bytes).reshape(vec![bytes.len()])?;
            Ok(arr.into_any().unbind())
        }
    }
}

/// Convert raw bytes to a flat 1-d numpy array of the correct dtype.
///
/// Used by `decode_range` where the result is always flat.
/// Validates that the byte count matches `expected_elements * byte_width`.
fn build_range_result(
    py: Python<'_>,
    dtype: Dtype,
    parts: Vec<Vec<u8>>,
    ranges: &[(u64, u64)],
    join: bool,
) -> PyResult<PyObject> {
    if join {
        let total_bytes: Vec<u8> = parts.into_iter().flatten().collect();
        let total_elements: u64 = ranges.iter().map(|(_, c)| c).sum();
        raw_bytes_to_numpy_flat(py, dtype, &total_bytes, total_elements as usize)
    } else {
        let mut arrays = Vec::with_capacity(parts.len());
        for (part, &(_, count)) in parts.iter().zip(ranges.iter()) {
            let arr = raw_bytes_to_numpy_flat(py, dtype, part, count as usize)?;
            arrays.push(arr);
        }
        let list = pyo3::types::PyList::new(py, arrays)?;
        Ok(list.into_any().unbind())
    }
}

fn raw_bytes_to_numpy_flat(
    py: Python<'_>,
    dtype: Dtype,
    bytes: &[u8],
    expected_elements: usize,
) -> PyResult<PyObject> {
    let bw = dtype.byte_width();
    // Bitmask (bw=0) skips the check — sub-byte packing, raw bytes returned.
    if bw > 0 {
        let expected_bytes = expected_elements
            .checked_mul(bw)
            .ok_or_else(|| PyValueError::new_err("element count overflows"))?;
        if bytes.len() != expected_bytes {
            return Err(PyValueError::new_err(format!(
                "decode_range returned {} bytes but expected {} elements * {} bytes = {}",
                bytes.len(),
                expected_elements,
                bw,
                expected_bytes,
            )));
        }
    }

    match dtype {
        Dtype::Float32 => numpy_flat_from_ne!(py, bytes, f32, 4),
        Dtype::Float64 => numpy_flat_from_ne!(py, bytes, f64, 8),
        Dtype::Int8 => {
            let values: &[i8] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) };
            Ok(numpy::PyArray::from_slice(py, values).into_any().unbind())
        }
        Dtype::Int16 => numpy_flat_from_ne!(py, bytes, i16, 2),
        Dtype::Int32 => numpy_flat_from_ne!(py, bytes, i32, 4),
        Dtype::Int64 => numpy_flat_from_ne!(py, bytes, i64, 8),
        Dtype::Uint8 => Ok(numpy::PyArray::from_slice(py, bytes).into_any().unbind()),
        Dtype::Uint16 => numpy_flat_from_ne!(py, bytes, u16, 2),
        Dtype::Uint32 => numpy_flat_from_ne!(py, bytes, u32, 4),
        Dtype::Uint64 => numpy_flat_from_ne!(py, bytes, u64, 8),
        Dtype::Float16 => numpy_via_frombuffer(py, bytes, "float16", &[expected_elements]),
        Dtype::Bfloat16 => numpy_via_frombuffer_bfloat16(py, bytes, &[expected_elements]),
        Dtype::Complex64 => numpy_via_frombuffer(py, bytes, "complex64", &[expected_elements]),
        Dtype::Complex128 => numpy_via_frombuffer(py, bytes, "complex128", &[expected_elements]),
        _ => {
            // Bitmask — raw bytes as uint8.
            Ok(numpy::PyArray::from_slice(py, bytes).into_any().unbind())
        }
    }
}
