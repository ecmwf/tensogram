//! Tensogram Python bindings via PyO3.
//!
//! Exposes the tensogram-core library as a native Python module with
//! numpy integration. All tensor data crosses the boundary as numpy arrays.

use std::collections::BTreeMap;
use std::path::Path;

use numpy::PyArrayMethods;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use tensogram_core::validate::{
    validate_file as core_validate_file, validate_message, ValidateOptions, ValidationLevel,
};
use tensogram_core::{
    decode, decode_descriptors, decode_metadata, decode_object, decode_range, encode,
    encode_pre_encoded, scan, ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions,
    GlobalMetadata, HashAlgorithm, StreamingEncoder, TensogramError, TensogramFile, RESERVED_KEY,
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
        TensogramError::Remote(msg) => PyIOError::new_err(format!("RemoteError: {msg}")),
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
// Per-object metadata lives in `base[i]`, with `_reserved_.tensor` auto-populated
// by the encoder. Client-writable message-level metadata goes in `extra`.
// ---------------------------------------------------------------------------

/// Message-level metadata.
///
/// Access version via ``meta.version``, per-object metadata via ``meta.base``,
/// and extra keys via dict syntax:
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
            self.inner.version,
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
    fn open(py: Python<'_>, source: &str) -> PyResult<Self> {
        let source = source.to_string();
        let file = py
            .allow_threads(|| TensogramFile::open_source(&source))
            .map_err(to_py_err)?;
        Ok(PyTensogramFile { file })
    }

    #[staticmethod]
    #[pyo3(signature = (source, storage_options=None))]
    fn open_remote(
        py: Python<'_>,
        source: &str,
        storage_options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let opts = match storage_options {
            Some(dict) => {
                let mut map = BTreeMap::new();
                for (k, v) in dict.iter() {
                    let key: String = k.extract()?;
                    let val: String = v.extract::<String>().or_else(|_| {
                        v.str()
                            .map(|s| s.to_string())
                            .map_err(|_| PyValueError::new_err(format!(
                                "storage_options value for key '{key}' must be convertible to string"
                            )))
                    })?;
                    map.insert(key, val);
                }
                map
            }
            None => BTreeMap::new(),
        };
        let source = source.to_string();
        let file = py
            .allow_threads(|| TensogramFile::open_remote(&source, &opts))
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
    fn message_count(&mut self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.file.message_count())
            .map_err(to_py_err)
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

    /// Decode message at *index* → ``Message(metadata, objects)``.
    ///
    /// Set *verify_hash* to ``True`` to verify payload integrity (default ``False``).
    /// Set *native_byte_order* to ``False`` to get wire-order bytes (default ``True``).
    #[pyo3(signature = (index, verify_hash=None, native_byte_order=true))]
    fn decode_message(
        &mut self,
        py: Python<'_>,
        index: usize,
        verify_hash: Option<bool>,
        native_byte_order: bool,
    ) -> PyResult<PyObject> {
        let options = DecodeOptions {
            verify_hash: verify_hash.unwrap_or(false),
            native_byte_order,
            ..Default::default()
        };
        let (global_meta, data_objects) = py
            .allow_threads(|| self.file.decode_message(index, &options))
            .map_err(to_py_err)?;
        let result_list = data_objects_to_python(py, &data_objects)?;
        pack_message(py, PyMetadata { inner: global_meta }, result_list)
    }

    fn file_decode_metadata(&mut self, py: Python<'_>, msg_index: usize) -> PyResult<PyObject> {
        let meta = py
            .allow_threads(|| self.file.decode_metadata(msg_index))
            .map_err(to_py_err)?;
        Ok(PyMetadata { inner: meta }
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    fn file_decode_descriptors(&mut self, py: Python<'_>, msg_index: usize) -> PyResult<PyObject> {
        let (meta, descriptors) = py
            .allow_threads(|| self.file.decode_descriptors(msg_index))
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

    #[pyo3(signature = (msg_index, obj_index, verify_hash=false, native_byte_order=true))]
    fn file_decode_object(
        &mut self,
        py: Python<'_>,
        msg_index: usize,
        obj_index: usize,
        verify_hash: bool,
        native_byte_order: bool,
    ) -> PyResult<PyObject> {
        let options = DecodeOptions {
            verify_hash,
            native_byte_order,
            ..Default::default()
        };
        let (meta, desc, data) = py
            .allow_threads(|| self.file.decode_object(msg_index, obj_index, &options))
            .map_err(to_py_err)?;
        let arr = bytes_to_numpy(py, &desc, &data)?;
        let py_desc = PyDataObjectDescriptor {
            inner: desc.clone(),
        }
        .into_pyobject(py)?
        .into_any()
        .unbind();
        let result = PyDict::new(py);
        result.set_item("metadata", PyMetadata { inner: meta }.into_pyobject(py)?)?;
        result.set_item("descriptor", py_desc)?;
        result.set_item("data", arr)?;
        Ok(result.into_any().unbind())
    }

    fn is_remote(&self) -> bool {
        self.file.is_remote()
    }

    fn source(&self) -> String {
        self.file.source()
    }

    /// Raw wire-format bytes for the message at *index*.
    fn read_message<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .allow_threads(|| self.file.read_message(index))
            .map_err(to_py_err)?;
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

    fn __len__(&mut self) -> PyResult<usize> {
        self.file.message_count().map_err(to_py_err)
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
    fn __iter__(&mut self) -> PyResult<PyFileIter> {
        // Open an independent file handle so the iterator owns its state.
        // Safe under free-threaded Python (no shared mutable borrows).
        let path = self.file.path().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("iteration not supported on remote files")
        })?;
        let mut iter_file = TensogramFile::open(path).map_err(to_py_err)?;
        // Read count from the *iterator's* handle to avoid TOCTOU race
        // if the file is modified between open() and first next() call.
        let count = iter_file.message_count().map_err(to_py_err)?;
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
    fn __getitem__(&mut self, py: Python<'_>, key: &Bound<'_, pyo3::PyAny>) -> PyResult<PyObject> {
        let count = self.file.message_count().map_err(to_py_err)?;

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
            return self.decode_message(py, idx as usize, None, true);
        }

        if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            let indices = slice.indices(count as isize)?;
            let mut items: Vec<PyObject> = Vec::with_capacity(indices.slicelength as usize);
            let mut i = indices.start;
            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                items.push(self.decode_message(py, i as usize, None, true)?);
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
    use pyo3::sync::GILOnceCell;
    static MESSAGE_TYPE: GILOnceCell<PyObject> = GILOnceCell::new();
    let msg_type = MESSAGE_TYPE
        .get_or_try_init::<_, PyErr>(py, || {
            Ok(py.import("tensogram")?.getattr("Message")?.unbind())
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
        let (global_meta, data_objects) =
            self.file.decode_message(i, &options).map_err(to_py_err)?;
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
///     global_meta_dict: ``{"version": 2, ...}`` with any extra keys.
///     descriptors_and_data: list of ``(descriptor_dict, bytes)`` pairs.  The
///         second element of each pair **must** be a ``bytes``-like object —
///         numpy arrays are rejected because pre-encoded payloads are
///         already in their final wire form.
///     hash: ``"xxh3"`` (default) or ``None`` to skip integrity hashing.
///
/// Returns:
///     ``bytes`` — the complete wire-format message.
#[pyfunction]
#[pyo3(name = "encode_pre_encoded", signature = (global_meta_dict, descriptors_and_data, hash=Some("xxh3")))]
fn py_encode_pre_encoded<'py>(
    py: Python<'py>,
    global_meta_dict: &Bound<'_, PyDict>,
    descriptors_and_data: &Bound<'_, PyList>,
    hash: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let global_meta = dict_to_global_metadata(global_meta_dict)?;
    let pairs = extract_pre_encoded_pairs(descriptors_and_data)?;
    let refs: Vec<(&DataObjectDescriptor, &[u8])> =
        pairs.iter().map(|(d, b)| (d, b.as_slice())).collect();

    let options = make_encode_options(hash)?;
    let msg = encode_pre_encoded(&global_meta, &refs, &options).map_err(to_py_err)?;
    Ok(PyBytes::new(py, &msg))
}

/// Decode a wire-format message → ``Message(metadata, objects)``.
///
/// Set *verify_hash* to ``True`` to verify payload integrity.
#[pyfunction]
#[pyo3(name = "decode", signature = (buf, verify_hash=false, native_byte_order=true))]
fn py_decode(
    py: Python<'_>,
    buf: &[u8],
    verify_hash: bool,
    native_byte_order: bool,
) -> PyResult<PyObject> {
    let options = DecodeOptions {
        verify_hash,
        native_byte_order,
        ..Default::default()
    };
    let (global_meta, data_objects) = decode(buf, &options).map_err(to_py_err)?;
    let result_list = data_objects_to_python(py, &data_objects)?;
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
#[pyo3(name = "decode_object", signature = (buf, index, verify_hash=false, native_byte_order=true))]
fn py_decode_object(
    py: Python<'_>,
    buf: &[u8],
    index: usize,
    verify_hash: bool,
    native_byte_order: bool,
) -> PyResult<(PyMetadata, PyDataObjectDescriptor, PyObject)> {
    let options = DecodeOptions {
        verify_hash,
        native_byte_order,
        ..Default::default()
    };
    let (global_meta, desc, obj_bytes) = decode_object(buf, index, &options).map_err(to_py_err)?;
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
#[pyo3(name = "decode_range", signature = (buf, object_index, ranges, join=false, verify_hash=false, native_byte_order=true))]
fn py_decode_range(
    py: Python<'_>,
    buf: &[u8],
    object_index: usize,
    ranges: Vec<(u64, u64)>,
    join: bool,
    verify_hash: bool,
    native_byte_order: bool,
) -> PyResult<PyObject> {
    let options = DecodeOptions {
        verify_hash,
        native_byte_order,
        ..Default::default()
    };
    let parts = decode_range(buf, object_index, &ranges, &options).map_err(to_py_err)?;

    // Look up the dtype from descriptors only — no payload decode.
    let (_gm, descriptors) = decode_descriptors(buf).map_err(to_py_err)?;
    let desc = descriptors.get(object_index).ok_or_else(|| {
        PyValueError::new_err(format!(
            "object_index {object_index} out of range (num_objects={})",
            descriptors.len()
        ))
    })?;

    if join {
        // Concatenate all parts into a single flat array.
        let total_bytes: Vec<u8> = parts.into_iter().flatten().collect();
        let total_elements: u64 = ranges.iter().map(|(_, c)| c).sum();
        raw_bytes_to_numpy_flat(py, desc.dtype, &total_bytes, total_elements as usize)
    } else {
        // Return a list of arrays, one per range.
        let mut arrays = Vec::with_capacity(parts.len());
        for (part, &(_, count)) in parts.iter().zip(ranges.iter()) {
            let arr = raw_bytes_to_numpy_flat(py, desc.dtype, part, count as usize)?;
            arrays.push(arr);
        }
        let list = pyo3::types::PyList::new(py, arrays)?;
        Ok(list.into_any().unbind())
    }
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
        let msg_bytes = &self.buf[offset..offset + length];
        let options = DecodeOptions {
            verify_hash: self.verify_hash,
            ..Default::default()
        };
        let (global_meta, data_objects) = decode(msg_bytes, &options).map_err(to_py_err)?;
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
// PyStreamingEncoder — wraps StreamingEncoder<Vec<u8>>
// ---------------------------------------------------------------------------

/// Progressive, frame-at-a-time encoder backed by an in-memory ``bytes`` buffer.
///
/// Unlike :func:`encode`, which builds a complete message in one shot, the
/// streaming encoder writes each object frame as soon as
/// :meth:`write_object` (or :meth:`write_object_pre_encoded`) is called.
/// Call :meth:`finish` to flush the footer frames + postamble and retrieve the
/// complete wire-format message.
///
/// Example::
///
///     enc = tensogram.StreamingEncoder({"version": 2})
///     enc.write_object({"type": "ntensor", "shape": [4], "dtype": "float32"},
///                      np.ones(4, dtype=np.float32))
///     msg = enc.finish()
#[pyclass(name = "StreamingEncoder")]
struct PyStreamingEncoder {
    inner: Option<StreamingEncoder<Vec<u8>>>,
}

#[pymethods]
impl PyStreamingEncoder {
    /// Begin a new streaming message.
    ///
    /// Args:
    ///     global_meta_dict: ``{"version": 2, ...}`` with any extra keys.
    ///     hash: ``"xxh3"`` (default) or ``None`` to skip integrity hashing.
    #[new]
    #[pyo3(signature = (global_meta_dict, hash=Some("xxh3")))]
    fn new(global_meta_dict: &Bound<'_, PyDict>, hash: Option<&str>) -> PyResult<Self> {
        let global_meta = dict_to_global_metadata(global_meta_dict)?;
        let options = make_encode_options(hash)?;
        let inner = StreamingEncoder::new(Vec::new(), &global_meta, &options).map_err(to_py_err)?;
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
        if let Ok(py_bytes) = data.downcast::<PyBytes>() {
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
    /// Subsequent calls on this encoder raise ``RuntimeError``.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("StreamingEncoder already finished"))?;
        let buf = inner.finish().map_err(to_py_err)?;
        Ok(PyBytes::new(py, &buf))
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
    buf: &[u8],
    level: &str,
    check_canonical: bool,
) -> PyResult<PyObject> {
    let options = parse_validate_options(level, check_canonical)?;
    let report = validate_message(buf, &options);
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
    let report = core_validate_file(Path::new(path), &options)
        .map_err(|e| PyIOError::new_err(format!("{e}")))?;
    let json_val = serde_json::to_value(&report)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")))?;
    json_value_to_py(py, &json_val)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn tensogram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_encode_pre_encoded, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_object, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_range, m)?)?;
    m.add_function(wrap_pyfunction!(py_scan, m)?)?;
    m.add_function(wrap_pyfunction!(py_iter_messages, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate_file, m)?)?;
    m.add_function(wrap_pyfunction!(compute_packing_params, m)?)?;
    m.add_class::<PyMetadata>()?;
    m.add_class::<PyDataObjectDescriptor>()?;
    m.add_class::<PyTensogramFile>()?;
    m.add_class::<PyFileIter>()?;
    m.add_class::<PyBufferIter>()?;
    m.add_class::<PyStreamingEncoder>()?;
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
    Ok(EncodeOptions {
        hash_algorithm,
        ..Default::default()
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
        let tuple = item.downcast::<pyo3::types::PyTuple>().map_err(|_| {
            PyValueError::new_err("each element must be a (descriptor_dict, bytes) tuple")
        })?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "each element must be a (descriptor_dict, bytes) tuple of length 2",
            ));
        }
        let desc_dict = tuple.get_item(0)?.downcast_into::<PyDict>().map_err(|_| {
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

/// Build a GlobalMetadata from a Python dict.
///
/// Required key: `version` (int).
/// Optional keys: `base` (list of dicts), `_extra_` or `extra` (dict).
/// All other keys → `extra`.
fn dict_to_global_metadata(dict: &Bound<'_, PyDict>) -> PyResult<GlobalMetadata> {
    let version: u16 = dict
        .get_item("version")?
        .ok_or_else(|| PyValueError::new_err("missing 'version'"))?
        .extract()?;

    let base = match dict.get_item("base")? {
        Some(v) => {
            let list = v
                .downcast::<PyList>()
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

    // Accept both "_extra_" (wire name) and "extra" (convenience alias)
    let explicit_extra = match dict.get_item("_extra_")? {
        Some(v) => py_dict_to_btree(&v)?,
        None => match dict.get_item("extra")? {
            Some(v) => {
                if v.downcast::<PyDict>().is_ok() {
                    py_dict_to_btree(&v)?
                } else {
                    return Err(PyValueError::new_err(
                        "'extra' must be a dict when provided as a convenience alias for '_extra_'",
                    ));
                }
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

    let known_keys = ["version", "base", "_extra_", "extra", RESERVED_KEY];
    let mut extra = explicit_extra;
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if !known_keys.contains(&key.as_str()) {
            extra.insert(key, py_to_cbor(&v)?);
        }
    }

    Ok(GlobalMetadata {
        version,
        base,
        reserved: BTreeMap::new(),
        extra,
    })
}

/// Convert a Python dict (or dict-like CBOR map value) to `BTreeMap<String, CborValue>`.
fn py_dict_to_btree(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<BTreeMap<String, ciborium::Value>> {
    let dict = obj
        .downcast::<PyDict>()
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
        None => ByteOrder::native(),
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
