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
    decode, decode_metadata, decode_object, decode_range, encode, scan, ByteOrder, DecodeOptions,
    Dtype, EncodeOptions, HashAlgorithm, Metadata, ObjectDescriptor, PayloadDescriptor,
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

fn cbor_to_py(py: Python<'_>, val: &ciborium::Value) -> PyObject {
    match val {
        ciborium::Value::Text(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            (n as i64).into_pyobject(py).unwrap().into_any().unbind()
        }
        ciborium::Value::Float(f) => f.into_pyobject(py).unwrap().into_any().unbind(),
        ciborium::Value::Bool(b) => (*b)
            .into_pyobject(py)
            .unwrap()
            .to_owned()
            .into_any()
            .unbind(),
        ciborium::Value::Null => py.None(),
        ciborium::Value::Array(arr) => {
            let list = PyList::new(py, arr.iter().map(|v| cbor_to_py(py, v))).unwrap();
            list.into_any().unbind()
        }
        ciborium::Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries {
                let key = cbor_to_py(py, k);
                let val = cbor_to_py(py, v);
                dict.set_item(key, val).unwrap();
            }
            dict.into_any().unbind()
        }
        ciborium::Value::Bytes(b) => PyBytes::new(py, b).into_any().unbind(),
        _ => format!("{val:?}")
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
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

fn extra_to_py(py: Python<'_>, extra: &BTreeMap<String, ciborium::Value>) -> PyObject {
    let dict = PyDict::new(py);
    for (k, v) in extra {
        dict.set_item(k, cbor_to_py(py, v)).unwrap();
    }
    dict.into_any().unbind()
}

fn py_to_extra(dict: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, ciborium::Value>> {
    let mut map = BTreeMap::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        map.insert(key, py_to_cbor(&v)?);
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// PyObjectDescriptor — wraps ObjectDescriptor for Python access
// ---------------------------------------------------------------------------

#[pyclass(name = "ObjectDescriptor")]
#[derive(Clone)]
struct PyObjectDescriptor {
    inner: ObjectDescriptor,
}

#[pymethods]
impl PyObjectDescriptor {
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

    #[getter]
    fn extra(&self, py: Python<'_>) -> PyObject {
        extra_to_py(py, &self.inner.extra)
    }

    fn __repr__(&self) -> String {
        format!(
            "ObjectDescriptor(type='{}', shape={:?}, dtype='{}')",
            self.inner.obj_type, self.inner.shape, self.inner.dtype
        )
    }
}

// ---------------------------------------------------------------------------
// PyPayloadDescriptor
// ---------------------------------------------------------------------------

#[pyclass(name = "PayloadDescriptor")]
#[derive(Clone)]
struct PyPayloadDescriptor {
    inner: PayloadDescriptor,
}

#[pymethods]
impl PyPayloadDescriptor {
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
    fn hash(&self, py: Python<'_>) -> PyObject {
        match &self.inner.hash {
            Some(h) => {
                let d = PyDict::new(py);
                d.set_item("type", &h.hash_type).unwrap();
                d.set_item("value", &h.value).unwrap();
                d.into_any().unbind()
            }
            None => py.None(),
        }
    }

    #[getter]
    fn params(&self, py: Python<'_>) -> PyObject {
        extra_to_py(py, &self.inner.params)
    }

    fn __repr__(&self) -> String {
        format!(
            "PayloadDescriptor(encoding='{}', filter='{}', compression='{}')",
            self.inner.encoding, self.inner.filter, self.inner.compression
        )
    }
}

// ---------------------------------------------------------------------------
// PyMetadata — wraps Metadata
// ---------------------------------------------------------------------------

#[pyclass(name = "Metadata")]
#[derive(Clone)]
struct PyMetadata {
    inner: Metadata,
}

#[pymethods]
impl PyMetadata {
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version
    }

    #[getter]
    fn objects(&self) -> Vec<PyObjectDescriptor> {
        self.inner
            .objects
            .iter()
            .map(|o| PyObjectDescriptor { inner: o.clone() })
            .collect()
    }

    #[getter]
    fn payload(&self) -> Vec<PyPayloadDescriptor> {
        self.inner
            .payload
            .iter()
            .map(|p| PyPayloadDescriptor { inner: p.clone() })
            .collect()
    }

    #[getter]
    fn extra(&self, py: Python<'_>) -> PyObject {
        extra_to_py(py, &self.inner.extra)
    }

    /// Dictionary-style access to top-level extra keys.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.extra.get(key) {
            Some(v) => Ok(cbor_to_py(py, v)),
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(version={}, objects={}, extra_keys={:?})",
            self.inner.version,
            self.inner.objects.len(),
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

    /// Append one message. `data_list` is a list of numpy arrays (one per object).
    #[pyo3(signature = (metadata_dict, data_list, hash=Some("xxh3")))]
    fn append(
        &mut self,
        py: Python<'_>,
        metadata_dict: &Bound<'_, PyDict>,
        data_list: &Bound<'_, PyList>,
        hash: Option<&str>,
    ) -> PyResult<()> {
        let metadata = dict_to_metadata(metadata_dict)?;
        let bufs = extract_data_list(py, data_list)?;
        let refs: Vec<&[u8]> = bufs.iter().map(|v| v.as_slice()).collect();

        let options = make_encode_options(hash)?;
        self.file
            .append(&metadata, &refs, &options)
            .map_err(to_py_err)
    }

    /// Decode message at index → (Metadata, list[bytes]).
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
        let (meta, objects) = self
            .file
            .decode_message(index, &options)
            .map_err(to_py_err)?;
        let arrays = objects_to_numpy(py, &meta, &objects)?;
        Ok((PyMetadata { inner: meta }, arrays))
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
        Ok(PyList::new(py, items)?)
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
/// metadata_dict: dict describing the message (version, objects, payload, extras).
/// data_list: list of numpy arrays, one per object.
/// hash: hash algorithm name ("xxh3", "sha1", "md5") or None.
///
/// Returns bytes.
#[pyfunction]
#[pyo3(name = "encode", signature = (metadata_dict, data_list, hash=Some("xxh3")))]
fn py_encode<'py>(
    py: Python<'py>,
    metadata_dict: &Bound<'_, PyDict>,
    data_list: &Bound<'_, PyList>,
    hash: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let metadata = dict_to_metadata(metadata_dict)?;
    let bufs = extract_data_list(py, data_list)?;
    let refs: Vec<&[u8]> = bufs.iter().map(|v| v.as_slice()).collect();

    let options = make_encode_options(hash)?;
    let msg = encode(&metadata, &refs, &options).map_err(to_py_err)?;
    Ok(PyBytes::new(py, &msg))
}

/// Decode a complete message → (Metadata, list[numpy.ndarray]).
#[pyfunction]
#[pyo3(name = "decode", signature = (buf, verify_hash=false))]
fn py_decode(py: Python<'_>, buf: &[u8], verify_hash: bool) -> PyResult<(PyMetadata, PyObject)> {
    let options = DecodeOptions { verify_hash };
    let (meta, objects) = decode(buf, &options).map_err(to_py_err)?;
    let arrays = objects_to_numpy(py, &meta, &objects)?;
    Ok((PyMetadata { inner: meta }, arrays))
}

/// Decode only metadata (no payload bytes touched).
#[pyfunction]
#[pyo3(name = "decode_metadata")]
fn py_decode_metadata(buf: &[u8]) -> PyResult<PyMetadata> {
    let meta = decode_metadata(buf).map_err(to_py_err)?;
    Ok(PyMetadata { inner: meta })
}

/// Decode a single object by index → (ObjectDescriptor, numpy.ndarray).
#[pyfunction]
#[pyo3(name = "decode_object", signature = (buf, index, verify_hash=false))]
fn py_decode_object(
    py: Python<'_>,
    buf: &[u8],
    index: usize,
    verify_hash: bool,
) -> PyResult<(PyObjectDescriptor, PyObject)> {
    let options = DecodeOptions { verify_hash };
    let (meta, obj_bytes) = decode_object(buf, index, &options).map_err(to_py_err)?;
    let arr = bytes_to_numpy(py, &meta.objects[index], &obj_bytes, &meta.payload[index])?;
    Ok((
        PyObjectDescriptor {
            inner: meta.objects[index].clone(),
        },
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

    // decode_range returns raw bytes in the logical dtype
    let meta = decode_metadata(buf).map_err(to_py_err)?;
    let desc = &meta.objects[object_index];
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
    m.add_class::<PyObjectDescriptor>()?;
    m.add_class::<PyPayloadDescriptor>()?;
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
        Some("sha1") => Some(HashAlgorithm::Sha1),
        Some("md5") => Some(HashAlgorithm::Md5),
        Some(other) => return Err(PyValueError::new_err(format!("unknown hash: {other}"))),
    };
    Ok(EncodeOptions { hash_algorithm })
}

fn extract_data_list(_py: Python<'_>, data_list: &Bound<'_, PyList>) -> PyResult<Vec<Vec<u8>>> {
    let mut bufs = Vec::new();
    for item in data_list.iter() {
        // Try as numpy array first, then as bytes
        if let Ok(arr) = item.extract::<numpy::PyReadonlyArrayDyn<'_, u8>>() {
            bufs.push(
                arr.as_slice()
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?
                    .to_vec(),
            );
        } else if let Ok(arr) = item.extract::<numpy::PyReadonlyArrayDyn<'_, f32>>() {
            let s = arr
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_ne_bytes()).collect();
            bufs.push(bytes);
        } else if let Ok(arr) = item.extract::<numpy::PyReadonlyArrayDyn<'_, f64>>() {
            let s = arr
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_ne_bytes()).collect();
            bufs.push(bytes);
        } else if let Ok(b) = item.extract::<Vec<u8>>() {
            bufs.push(b);
        } else {
            return Err(PyValueError::new_err(
                "data_list items must be numpy arrays or bytes",
            ));
        }
    }
    Ok(bufs)
}

fn dict_to_metadata(dict: &Bound<'_, PyDict>) -> PyResult<Metadata> {
    let version: u64 = dict
        .get_item("version")?
        .ok_or_else(|| PyValueError::new_err("missing 'version'"))?
        .extract()?;

    let objects_list = dict
        .get_item("objects")?
        .ok_or_else(|| PyValueError::new_err("missing 'objects'"))?;
    let objects_list = objects_list.downcast::<PyList>()?;

    let mut objects = Vec::new();
    for obj_any in objects_list.iter() {
        let obj_dict = obj_any.downcast::<PyDict>()?;
        objects.push(dict_to_object_descriptor(obj_dict)?);
    }

    let mut payload = Vec::new();
    if let Some(payload_any) = dict.get_item("payload")? {
        let payload_list = payload_any.downcast::<PyList>()?;
        for p_any in payload_list.iter() {
            let p_dict = p_any.downcast::<PyDict>()?;
            payload.push(dict_to_payload_descriptor(p_dict)?);
        }
    } else {
        // Default: one "none" payload per object
        for _ in &objects {
            payload.push(PayloadDescriptor {
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            });
        }
    }

    // Collect extra keys (everything except version, objects, payload)
    let mut extra = BTreeMap::new();
    let reserved = ["version", "objects", "payload"];
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if !reserved.contains(&key.as_str()) {
            extra.insert(key, py_to_cbor(&v)?);
        }
    }

    Ok(Metadata {
        version,
        objects,
        payload,
        extra,
    })
}

fn dict_to_object_descriptor(dict: &Bound<'_, PyDict>) -> PyResult<ObjectDescriptor> {
    let obj_type: String = dict
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("object missing 'type'"))?
        .extract()?;
    let shape: Vec<u64> = dict
        .get_item("shape")?
        .ok_or_else(|| PyValueError::new_err("object missing 'shape'"))?
        .extract()?;
    let dtype_str: String = dict
        .get_item("dtype")?
        .ok_or_else(|| PyValueError::new_err("object missing 'dtype'"))?
        .extract()?;

    let ndim = shape.len() as u64;

    // Compute strides if not provided
    let strides: Vec<u64> = if let Some(s) = dict.get_item("strides")? {
        s.extract()?
    } else {
        compute_strides(&shape)
    };

    let dtype = parse_dtype(&dtype_str)?;

    // Extra keys
    let reserved = ["type", "ndim", "shape", "strides", "dtype"];
    let mut extra = BTreeMap::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if !reserved.contains(&key.as_str()) {
            extra.insert(key, py_to_cbor(&v)?);
        }
    }

    Ok(ObjectDescriptor {
        obj_type,
        ndim,
        shape,
        strides,
        dtype,
        extra,
    })
}

fn dict_to_payload_descriptor(dict: &Bound<'_, PyDict>) -> PyResult<PayloadDescriptor> {
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

    // Collect params (everything except the fixed keys)
    let reserved = ["byte_order", "encoding", "filter", "compression", "hash"];
    let mut params = BTreeMap::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if !reserved.contains(&key.as_str()) {
            params.insert(key, py_to_cbor(&v)?);
        }
    }

    Ok(PayloadDescriptor {
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

/// Convert decoded object bytes to numpy arrays.
fn objects_to_numpy(py: Python<'_>, meta: &Metadata, objects: &[Vec<u8>]) -> PyResult<PyObject> {
    let mut arrays: Vec<PyObject> = Vec::new();
    for (i, obj_bytes) in objects.iter().enumerate() {
        let desc = &meta.objects[i];
        let payload_desc = &meta.payload[i];
        let arr = bytes_to_numpy(py, desc, obj_bytes, payload_desc)?;
        arrays.push(arr);
    }
    Ok(PyList::new(py, arrays)?.into_any().unbind())
}

fn bytes_to_numpy(
    py: Python<'_>,
    desc: &ObjectDescriptor,
    bytes: &[u8],
    payload_desc: &PayloadDescriptor,
) -> PyResult<PyObject> {
    // If simple_packing was used, decoded bytes are f64 regardless of dtype
    let effective_dtype = if payload_desc.encoding == "simple_packing" {
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
