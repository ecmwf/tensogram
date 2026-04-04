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
            Ok((n as i64).into_pyobject(py)?.into_any().unbind())
        }
        ciborium::Value::Float(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        ciborium::Value::Bool(b) => Ok((*b)
            .into_pyobject(py)?
            .to_owned()
            .into_any()
            .unbind()),
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

/// Python class exposing all fields of a DataObjectDescriptor.
///
/// In v2 this single descriptor replaces the old split between
/// `ObjectDescriptor` (tensor info) and `PayloadDescriptor` (encoding info).
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

    /// Dictionary-style access to top-level extra keys.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.extra.get(key) {
            Some(v) => cbor_to_py(py, v),
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
        }
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

    /// How many valid messages are in the file.
    fn message_count(&mut self) -> PyResult<usize> {
        self.file.message_count().map_err(to_py_err)
    }

    /// Append one message.
    ///
    /// `global_meta_dict` describes version and any extra top-level keys.
    /// `descriptors_and_data` is a list of (descriptor_dict, data) pairs where
    /// each descriptor_dict contains both tensor info (type, shape, dtype) and
    /// encoding info (byte_order, encoding, filter, compression, params).
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

    /// Decode message at index → (Metadata, list[(DataObjectDescriptor, numpy.ndarray)]).
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

    /// Raw message bytes at index.
    fn read_message<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.file.read_message(index).map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// All raw message bytes as a list of bytes objects.
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
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Encode a Tensogram message.
///
/// global_meta_dict: dict with 'version' (int) and any extra top-level keys.
/// descriptors_and_data: list of (descriptor_dict, data) pairs.
///   Each descriptor_dict has tensor fields (type, shape, dtype) and
///   encoding fields (byte_order, encoding, filter, compression, and params as
///   additional keys).
/// hash: hash algorithm name ("xxh3") or None.
///
/// Returns bytes.
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

/// Decode a complete message → (Metadata, list[(DataObjectDescriptor, numpy.ndarray)]).
#[pyfunction]
#[pyo3(name = "decode", signature = (buf, verify_hash=false))]
fn py_decode(py: Python<'_>, buf: &[u8], verify_hash: bool) -> PyResult<(PyMetadata, PyObject)> {
    let options = DecodeOptions { verify_hash };
    let (global_meta, data_objects) = decode(buf, &options).map_err(to_py_err)?;
    let result_list = data_objects_to_python(py, &data_objects)?;
    Ok((PyMetadata { inner: global_meta }, result_list))
}

/// Decode only metadata (no payload bytes touched).
#[pyfunction]
#[pyo3(name = "decode_metadata")]
fn py_decode_metadata(buf: &[u8]) -> PyResult<PyMetadata> {
    let meta = decode_metadata(buf).map_err(to_py_err)?;
    Ok(PyMetadata { inner: meta })
}

/// Decode a single object by index → (Metadata, DataObjectDescriptor, numpy.ndarray).
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

/// Decode a partial range from an uncompressed object → numpy.ndarray.
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

    // decode_range returns raw bytes in the logical dtype — look up the dtype
    // from the metadata so we can reconstruct the correct numpy array type.
    // We need the DataObjectDescriptor for the dtype; re-decode the object header.
    let (_gm, desc, _) =
        decode_object(buf, object_index, &DecodeOptions { verify_hash: false })
            .map_err(to_py_err)?;
    let total_elements: u64 = ranges.iter().map(|(_, c)| c).sum();
    let arr = raw_bytes_to_numpy_flat(py, desc.dtype, &bytes, total_elements as usize)?;
    Ok(arr)
}

/// Scan a buffer for message boundaries → list[(offset, length)].
#[pyfunction]
#[pyo3(name = "scan")]
fn py_scan(buf: &[u8]) -> Vec<(usize, usize)> {
    scan(buf)
}

/// Compute simple_packing parameters.
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
    py: Python<'_>,
    pairs_list: &Bound<'_, PyList>,
) -> PyResult<Vec<(DataObjectDescriptor, Vec<u8>)>> {
    let mut result = Vec::new();
    for item in pairs_list.iter() {
        // Accept a 2-tuple (descriptor_dict, data)
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

        // Extract data bytes from the second element
        let data_item = tuple.get_item(1)?;
        let data = extract_single_data_item(py, &data_item)?;

        result.push((desc, data));
    }
    Ok(result)
}

/// Extract raw bytes from a single data item (numpy array or bytes).
fn extract_single_data_item(_py: Python<'_>, item: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(arr) = item.extract::<numpy::PyReadonlyArrayDyn<'_, u8>>() {
        return arr
            .as_slice()
            .map(|s| s.to_vec())
            .map_err(|e| PyValueError::new_err(format!("{e}")));
    }
    if let Ok(arr) = item.extract::<numpy::PyReadonlyArrayDyn<'_, f32>>() {
        let s = arr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        return Ok(s.iter().flat_map(|v| v.to_ne_bytes()).collect());
    }
    if let Ok(arr) = item.extract::<numpy::PyReadonlyArrayDyn<'_, f64>>() {
        let s = arr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        return Ok(s.iter().flat_map(|v| v.to_ne_bytes()).collect());
    }
    if let Ok(b) = item.extract::<Vec<u8>>() {
        return Ok(b);
    }
    Err(PyValueError::new_err("data must be a numpy array or bytes"))
}

/// Convert GlobalMetadata + decoded data objects list to a Python list of
/// (DataObjectDescriptor, numpy.ndarray) tuples.
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
/// Expected keys: 'version' (int). All other keys are stored as `extra`.
fn dict_to_global_metadata(dict: &Bound<'_, PyDict>) -> PyResult<GlobalMetadata> {
    let version: u16 = dict
        .get_item("version")?
        .ok_or_else(|| PyValueError::new_err("missing 'version'"))?
        .extract()?;

    // Collect extra keys (everything except 'version')
    let mut extra = BTreeMap::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if key != "version" {
            extra.insert(key, py_to_cbor(&v)?);
        }
    }

    Ok(GlobalMetadata { version, extra })
}

/// Build a DataObjectDescriptor from a Python dict.
///
/// Tensor fields: 'type', 'shape', 'dtype', optionally 'strides', 'ndim'.
/// Encoding fields: 'byte_order', 'encoding', 'filter', 'compression'.
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

    let byte_order = match dict.get_item("byte_order")?.map(|v| v.extract::<String>()) {
        Some(Ok(s)) if s == "little" => ByteOrder::Little,
        _ => ByteOrder::Big,
    };
    let encoding = dict
        .get_item("encoding")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "none".to_string());
    let filter = dict
        .get_item("filter")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "none".to_string());
    let compression = dict
        .get_item("compression")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "none".to_string());

    // All remaining keys go into params (replaces the old separate params dict)
    let reserved = [
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
        if !reserved.contains(&key.as_str()) {
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

/// Convert a single decoded object's bytes to a numpy array.
///
/// Uses `DataObjectDescriptor` for both shape/dtype and encoding info
/// (the v2 unified descriptor).
fn bytes_to_numpy(py: Python<'_>, desc: &DataObjectDescriptor, bytes: &[u8]) -> PyResult<PyObject> {
    // If simple_packing was used, decoded bytes are f64 regardless of declared dtype
    let effective_dtype = if desc.encoding == "simple_packing" {
        Dtype::Float64
    } else {
        desc.dtype
    };

    let shape: Vec<usize> = desc.shape.iter().map(|&s| s as usize).collect();
    let total_elements: usize = shape.iter().product();

    match effective_dtype {
        Dtype::Float32 => {
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Float64 => {
            let values: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Int32 => {
            let values: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Int64 => {
            let values: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Uint8 => {
            let arr = numpy::PyArray::from_vec(py, bytes.to_vec()).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Uint16 => {
            let values: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Uint32 => {
            let values: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Uint64 => {
            let values: Vec<u64> = bytes
                .chunks_exact(8)
                .map(|c| u64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Int8 => {
            let values: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        Dtype::Int16 => {
            let values: Vec<i16> = bytes
                .chunks_exact(2)
                .map(|c| i16::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values).reshape(shape)?;
            Ok(arr.into_any().unbind())
        }
        _ => {
            // For types without direct numpy mapping (float16, bfloat16, complex),
            // return raw bytes as uint8 and let the caller reinterpret.
            let arr = numpy::PyArray::from_vec(py, bytes.to_vec())
                .reshape(vec![total_elements * desc.dtype.byte_width()])?;
            Ok(arr.into_any().unbind())
        }
    }
}

fn raw_bytes_to_numpy_flat(
    py: Python<'_>,
    dtype: Dtype,
    bytes: &[u8],
    _num_elements: usize,
) -> PyResult<PyObject> {
    match dtype {
        Dtype::Float32 => {
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values);
            Ok(arr.into_any().unbind())
        }
        Dtype::Float64 => {
            let values: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let arr = numpy::PyArray::from_vec(py, values);
            Ok(arr.into_any().unbind())
        }
        Dtype::Uint8 => {
            let arr = numpy::PyArray::from_vec(py, bytes.to_vec());
            Ok(arr.into_any().unbind())
        }
        _ => {
            // Return raw bytes
            let arr = numpy::PyArray::from_vec(py, bytes.to_vec());
            Ok(arr.into_any().unbind())
        }
    }
}
