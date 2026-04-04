//! Tensogram Python bindings via PyO3.
//!
//! Exposes the tensogram-core library as a native Python module with
//! numpy integration. All tensor data crosses the boundary as numpy arrays.

use std::collections::BTreeMap;

use numpy::PyArrayMethods;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use tensogram_core::{
    decode, decode_metadata, decode_object, decode_range, encode, scan, ByteOrder,
    DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata, HashAlgorithm,
    TensogramError, TensogramFile,
};

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
    }
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
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut entries = Vec::new();
        for (k, v) in dict.iter() {
            entries.push((py_to_cbor(&k)?, py_to_cbor(&v)?));
        }
        return Ok(ciborium::Value::Map(entries));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
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
#[pyclass(name = "DataObjectDescriptor")]
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

    #[getter]
    fn hash(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.hash {
            Some(h) => {
                let d = PyDict::new(py);
                d.set_item("type", &h.hash_type)?;
                d.set_item("value", &h.value)?;
                Ok(d.into_any().unbind())
            }
            None => Ok(py.None()),
        }
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
// The `objects` and `payload` attributes are gone; per-object info lives on
// each DataObjectDescriptor returned alongside the decoded bytes.
// ---------------------------------------------------------------------------

/// Message-level metadata.
///
/// Access version via ``meta.version`` and extra keys via dict syntax:
/// ``meta["mars"]``, ``"mars" in meta``, ``meta.extra``.
#[pyclass(name = "Metadata")]
#[derive(Clone)]
struct PyMetadata {
    inner: GlobalMetadata,
}

#[pymethods]
impl PyMetadata {
    #[getter]
    fn version(&self) -> u16 {
        self.inner.version
    }

    #[getter]
    fn extra(&self, py: Python<'_>) -> PyResult<PyObject> {
        extra_to_py(py, &self.inner.extra)
    }

    /// Dictionary-style access: ``meta["key"]``.  Raises ``KeyError`` if missing.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.extra.get(key) {
            Some(v) => cbor_to_py(py, v),
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
        }
    }

    /// Membership test: ``"key" in meta``.
    fn __contains__(&self, key: &str) -> bool {
        self.inner.extra.contains_key(key)
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(version={}, extra_keys={:?})",
            self.inner.version,
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
    /// Open an existing file for reading.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let file = TensogramFile::open(path).map_err(to_py_err)?;
        Ok(PyTensogramFile { file })
    }

    /// Create a new file (truncates if exists).
    #[staticmethod]
    fn create(path: &str) -> PyResult<Self> {
        let file = TensogramFile::create(path).map_err(to_py_err)?;
        Ok(PyTensogramFile { file })
    }

    /// Number of valid messages in the file.
    fn message_count(&mut self) -> PyResult<usize> {
        self.file.message_count().map_err(to_py_err)
    }

    /// Append one message.
    ///
    /// Args:
    ///     global_meta_dict: ``{"version": 2, ...}`` with any extra keys.
    ///     descriptors_and_data: list of ``(descriptor_dict, data)`` pairs.
    ///         Each descriptor_dict requires ``type``, ``shape``, ``dtype`` and
    ///         optionally ``byte_order``, ``encoding``, ``filter``, ``compression``.
    ///     hash: ``"xxh3"`` (default) or ``None`` to skip hashing.
    #[pyo3(signature = (global_meta_dict, descriptors_and_data, hash=Some("xxh3")))]
    fn append(
        &mut self,
        py: Python<'_>,
        global_meta_dict: &Bound<'_, PyDict>,
        descriptors_and_data: &Bound<'_, PyList>,
        hash: Option<&str>,
    ) -> PyResult<()> {
        let global_meta = dict_to_global_metadata(global_meta_dict)?;
        let pairs = extract_descriptor_data_pairs(py, descriptors_and_data)?;
        let refs: Vec<(&DataObjectDescriptor, &[u8])> =
            pairs.iter().map(|(d, b)| (d, b.as_slice())).collect();

        let options = make_encode_options(hash)?;
        self.file
            .append(&global_meta, &refs, &options)
            .map_err(to_py_err)
    }

    /// Decode message at *index* → ``(Metadata, list[(DataObjectDescriptor, ndarray)])``.
    ///
    /// Set *verify_hash* to ``True`` to verify payload integrity (default ``False``).
    #[pyo3(signature = (index, verify_hash=None))]
    fn decode_message(
        &mut self,
        py: Python<'_>,
        index: usize,
        verify_hash: Option<bool>,
    ) -> PyResult<(PyMetadata, PyObject)> {
        let options = DecodeOptions {
            verify_hash: verify_hash.unwrap_or(false),
        };
        let (global_meta, data_objects) = self
            .file
            .decode_message(index, &options)
            .map_err(to_py_err)?;
        let result_list = data_objects_to_python(py, &data_objects)?;
        Ok((PyMetadata { inner: global_meta }, result_list))
    }

    /// Raw wire-format bytes for the message at *index*.
    fn read_message<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.file.read_message(index).map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// All raw message bytes as a list of ``bytes`` objects.
    fn messages<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        #[allow(deprecated)]
        let msgs = self.file.messages().map_err(to_py_err)?;
        let items: Vec<PyObject> = msgs
            .iter()
            .map(|m| PyBytes::new(py, m).into_any().unbind())
            .collect();
        PyList::new(py, items)
    }

    fn __repr__(&self) -> String {
        format!("TensogramFile(path='{}')", self.file.path().display())
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

    fn __len__(&mut self) -> PyResult<usize> {
        self.file.message_count().map_err(to_py_err)
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Encode arrays into a Tensogram wire-format message.
///
/// Args:
///     global_meta_dict: ``{"version": 2, ...}`` with any extra keys.
///     descriptors_and_data: list of ``(descriptor_dict, numpy_array)`` pairs.
///     hash: ``"xxh3"`` (default) or ``None`` to skip integrity hashing.
///
/// Returns:
///     ``bytes`` — the complete wire-format message.
#[pyfunction]
#[pyo3(name = "encode", signature = (global_meta_dict, descriptors_and_data, hash=Some("xxh3")))]
fn py_encode<'py>(
    py: Python<'py>,
    global_meta_dict: &Bound<'_, PyDict>,
    descriptors_and_data: &Bound<'_, PyList>,
    hash: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let global_meta = dict_to_global_metadata(global_meta_dict)?;
    let pairs = extract_descriptor_data_pairs(py, descriptors_and_data)?;
    let refs: Vec<(&DataObjectDescriptor, &[u8])> =
        pairs.iter().map(|(d, b)| (d, b.as_slice())).collect();

    let options = make_encode_options(hash)?;
    let msg = encode(&global_meta, &refs, &options).map_err(to_py_err)?;
    Ok(PyBytes::new(py, &msg))
}

/// Decode a wire-format message → ``(Metadata, list[(DataObjectDescriptor, ndarray)])``.
///
/// Set *verify_hash* to ``True`` to verify payload integrity.
#[pyfunction]
#[pyo3(name = "decode", signature = (buf, verify_hash=false))]
fn py_decode(py: Python<'_>, buf: &[u8], verify_hash: bool) -> PyResult<(PyMetadata, PyObject)> {
    let options = DecodeOptions { verify_hash };
    let (global_meta, data_objects) = decode(buf, &options).map_err(to_py_err)?;
    let result_list = data_objects_to_python(py, &data_objects)?;
    Ok((PyMetadata { inner: global_meta }, result_list))
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

/// Decode a single object by *index* → ``(Metadata, DataObjectDescriptor, ndarray)``.
///
/// Only the header and the requested object's payload are read.
/// Raises ``ValueError`` if *index* is out of range.
#[pyfunction]
#[pyo3(name = "decode_object", signature = (buf, index, verify_hash=false))]
fn py_decode_object(
    py: Python<'_>,
    buf: &[u8],
    index: usize,
    verify_hash: bool,
) -> PyResult<(PyMetadata, PyDataObjectDescriptor, PyObject)> {
    let options = DecodeOptions { verify_hash };
    let (global_meta, desc, obj_bytes) = decode_object(buf, index, &options).map_err(to_py_err)?;
    let arr = bytes_to_numpy(py, &desc, &obj_bytes)?;
    Ok((
        PyMetadata { inner: global_meta },
        PyDataObjectDescriptor { inner: desc },
        arr,
    ))
}

/// Extract a sub-range from an uncompressed object → flat ``ndarray``.
///
/// Args:
///     buf: wire-format message bytes.
///     object_index: which object in the message (0-based).
///     ranges: list of ``(start, count)`` element-offset tuples,
///         e.g. ``[(0, 10), (50, 5)]`` reads elements 0..10 and 50..55.
///     verify_hash: verify payload hash before extraction.
///
/// Returns a 1-d numpy array with the concatenated requested elements.
/// Only works with ``encoding="none"`` and ``compression="none"``.
#[pyfunction]
#[pyo3(name = "decode_range", signature = (buf, object_index, ranges, verify_hash=false))]
fn py_decode_range(
    py: Python<'_>,
    buf: &[u8],
    object_index: usize,
    ranges: Vec<(u64, u64)>,
    verify_hash: bool,
) -> PyResult<PyObject> {
    let options = DecodeOptions { verify_hash };
    let bytes = decode_range(buf, object_index, &ranges, &options).map_err(to_py_err)?;

    // Look up the dtype so we produce the correct numpy array type.
    let (_gm, desc, _) = decode_object(buf, object_index, &DecodeOptions { verify_hash: false })
        .map_err(to_py_err)?;
    let total_elements: u64 = ranges.iter().map(|(_, c)| c).sum();
    raw_bytes_to_numpy_flat(py, desc.dtype, &bytes, total_elements as usize)
}

/// Scan *buf* for message boundaries → ``list[(offset, length)]``.
///
/// Each tuple identifies one complete Tensogram message within the buffer.
/// Useful for multi-message buffers (e.g. from a network socket or mmap).
#[pyfunction]
#[pyo3(name = "scan")]
fn py_scan(buf: &[u8]) -> Vec<(usize, usize)> {
    scan(buf)
}

/// Compute simple-packing parameters for a float64 array.
///
/// Args:
///     values: 1-d float64 numpy array (must not contain NaN).
///     bits_per_value: quantization depth (e.g. 16).
///     decimal_scale_factor: decimal scaling (usually 0).
///
/// Returns a dict with keys ``reference_value``, ``binary_scale_factor``,
/// ``decimal_scale_factor``, ``bits_per_value``.
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
    dict.set_item("reference_value", params.reference_value)?;
    dict.set_item("binary_scale_factor", params.binary_scale_factor)?;
    dict.set_item("decimal_scale_factor", params.decimal_scale_factor)?;
    dict.set_item("bits_per_value", params.bits_per_value)?;
    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn tensogram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_object, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_range, m)?)?;
    m.add_function(wrap_pyfunction!(py_scan, m)?)?;
    m.add_function(wrap_pyfunction!(compute_packing_params, m)?)?;
    m.add_class::<PyMetadata>()?;
    m.add_class::<PyDataObjectDescriptor>()?;
    m.add_class::<PyTensogramFile>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_encode_options(hash: Option<&str>) -> PyResult<EncodeOptions> {
    let hash_algorithm = match hash {
        None => None,
        Some("xxh3") => Some(HashAlgorithm::Xxh3),
        Some(other) => return Err(PyValueError::new_err(format!("unknown hash: {other}"))),
    };
    Ok(EncodeOptions { hash_algorithm })
}

/// Extract a list of (descriptor, raw_bytes) pairs from a Python list of
/// (descriptor_dict, data) tuples used by `encode` and `append`.
fn extract_descriptor_data_pairs(
    _py: Python<'_>,
    pairs_list: &Bound<'_, PyList>,
) -> PyResult<Vec<(DataObjectDescriptor, Vec<u8>)>> {
    let mut result = Vec::new();
    for item in pairs_list.iter() {
        let tuple = item.downcast::<pyo3::types::PyTuple>().map_err(|_| {
            PyValueError::new_err("each element must be a (descriptor_dict, data) tuple")
        })?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "each element must be a (descriptor_dict, data) tuple of length 2",
            ));
        }
        let desc_dict = tuple.get_item(0)?.downcast_into::<PyDict>().map_err(|_| {
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

/// Build a GlobalMetadata from a Python dict.
///
/// Required key: `version` (int). All other keys → `extra`.
fn dict_to_global_metadata(dict: &Bound<'_, PyDict>) -> PyResult<GlobalMetadata> {
    let version: u16 = dict
        .get_item("version")?
        .ok_or_else(|| PyValueError::new_err("missing 'version'"))?
        .extract()?;

    let mut extra = BTreeMap::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if key != "version" {
            extra.insert(key, py_to_cbor(&v)?);
        }
    }

    Ok(GlobalMetadata {
        version,
        common: BTreeMap::new(),
        payload: Vec::new(),
        reserved: BTreeMap::new(),
        extra,
    })
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
/// Optional keys: `strides`, `byte_order` (default "little"),
///   `encoding` / `filter` / `compression` (default "none").
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
        compute_strides(&shape)
    };

    let dtype = parse_dtype(&dtype_str)?;

    let byte_order = match dict.get_item("byte_order")? {
        None => ByteOrder::Little,
        Some(v) => match v.extract::<String>()?.as_str() {
            "little" => ByteOrder::Little,
            "big" => ByteOrder::Big,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown byte_order: '{other}', expected 'little' or 'big'"
                )))
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
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if !reserved_keys.contains(&key.as_str()) {
            params.insert(key, py_to_cbor(&v)?);
        }
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
        hash: None,
    })
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

fn compute_strides(shape: &[u64]) -> Vec<u64> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// Numpy conversion — macro-driven to avoid repeating the same pattern
// for each numeric dtype.
//
// `chunks_exact(N)` guarantees each chunk is exactly N bytes, so
// `try_into().unwrap()` on the chunk→fixed-array conversion is safe.
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
            .map(|c| <$T>::from_ne_bytes(c.try_into().unwrap()))
            .collect::<Vec<$T>>()
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
        _ => {
            // float16, bfloat16, complex, bitmask — return raw bytes as uint8.
            // Bitmask has byte_width=0 (sub-byte); just return all bytes as-is.
            let arr = numpy::PyArray::from_slice(py, bytes).reshape(vec![bytes.len()])?;
            Ok(arr.into_any().unbind())
        }
    }
}

/// Convert raw bytes to a flat 1-d numpy array of the correct dtype.
///
/// Used by `decode_range` where the result is always flat.
/// Validates that the byte count matches `expected_elements * byte_width`.
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
        _ => {
            // float16, bfloat16, complex, bitmask — raw bytes as uint8
            Ok(numpy::PyArray::from_slice(py, bytes).into_any().unbind())
        }
    }
}
