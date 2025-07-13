//! PyWavers - Python bindings for the `wavers` crate.
//!
//! This module exposes a high-performance WAV audio interface to Python
//! built using `pyo3`, `numpy`, and `ndarray`. It allows users to:
//!
//! - Read WAV files into NumPy arrays with configurable sample types
//! - Write NumPy arrays to WAV files with metadata control
//! - Inspect WAV file headers and metadata as structured Python objects
//! - Gracefully handle conversion, parsing, and I/O errors from Rust
//!
//! # Example Usage (Python)
//!
//! ```python
//! import pywavers
//! import numpy as np
//! audio, sr = pywavers.read("file.wav", dtype=np.float32)
//! # or pywavers.read("file.wav", dtype="float32")
//! pywavers.write(audio, "out.wav", sr, channels=2)
//! info = pywavers.info("file.wav")
//! print(info.sample_rate, info.channels)
//! ```

use audio_sample::{ChannelIterator, Samples, FrameIterator, BlockIterator};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, wrap_pyfunction};

use crate::{WavInfo, WaversErrorKind, WaversResult};
use ndarray::Array2;
use numpy::{PyArrayMethods, PyReadonlyArray2};
create_exception!(wavers, WaversError, PyException);

use pyo3::IntoPyObject; // <-- make sure this is imported

create_exception!(wavers, IoError, PyException);
create_exception!(wavers, ParseError, PyException);
create_exception!(wavers, ValidationError, PyException);
create_exception!(wavers, ConversionError, PyException);
create_exception!(wavers, OperationError, PyException);
create_exception!(wavers, SampleError, PyException);
create_exception!(wavers, InternalError, PyException);

/// Maps `WaversErrorKind` into specific Python exceptions
fn map_wavers_error(err: WaversErrorKind) -> PyErr {
    match err {
        WaversErrorKind::Io(_) => IoError::new_err(format!("{err:?}")),
        WaversErrorKind::Parse(_) => ParseError::new_err(format!("{err:?}")),
        WaversErrorKind::Validate(_) => ValidationError::new_err(format!("{err:?}")),
        WaversErrorKind::Convert(_) => ConversionError::new_err(format!("{err:?}")),
        WaversErrorKind::Operation(_) => OperationError::new_err(format!("{err:?}")),
        WaversErrorKind::SampleError(_) => SampleError::new_err(format!("{err:?}")),
        WaversErrorKind::Internal(_) => InternalError::new_err(format!("{err:?}")),
    }
}

/// Converts a `WaversResult` into a `PyResult` with Python exceptions
fn into_pyresult<T>(res: WaversResult<T>) -> PyResult<T> {
    res.map_err(map_wavers_error)
}

/// Tries to extract a human-readable dtype name from a Python object
fn extract_dtype_name<'py>(dtype: Bound<'py, PyAny>) -> PyResult<String> {
    dtype
        .getattr("name")
        .or_else(|_| dtype.getattr("__name__"))?
        .extract()
        .map_err(|_| ValidationError::new_err("Could not extract dtype name"))
}

/// Macro to dispatch reading based on the sample type
macro_rules! read_dispatch {
    ($py:expr, $path:expr, $t:ty) => {{
        let (data, sr): (Array2<$t>, u32) =
            crate::read($path).map_err(|e| WaversError::new_err(format!("{:?}", e)))?;

        let py_array = &data.into_pyarray($py);
        Ok((py_array, sr.into_pyobject($py)?.into()))
    }};
}

/// Read a WAV file into a NumPy array and return (data, sample_rate)
#[pyfunction]
#[pyo3(signature = (path, dtype=None))]
fn read(
    py: Python<'_>,
    path: &str,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyObject, PyObject)> {
    let dtype_name = dtype
        .map(|d| extract_dtype_name(d))
        .transpose()?
        .unwrap_or_else(|| "int16".to_string());

    match dtype_name.as_str() {
        "int16" => read_dispatch!(py, path, i16),
        "int32" => read_dispatch!(py, path, i32),
        "float32" => read_dispatch!(py, path, f32),
        "float64" => read_dispatch!(py, path, f64),
        _ => Err(WaversError::new_err(format!(
            "Unsupported dtype: {}",
            dtype_name
        ))),
    }
}

/// Macro to dispatch writing based on the sample type
macro_rules! write_dispatch {
    ($samples:expr, $path:expr, $sr:expr, $channels:expr, $t:ty) => {{
        let samples_array: &Bound<'_, PyArray2<$t>> = $samples.downcast()?;
        let readonly: PyReadonlyArray2<$t> = samples_array.readonly();
        let data: Array2<$t> = readonly.as_array().to_owned();
        into_pyresult(crate::write::<&str, $t>($path, &data, $sr, $channels))
    }};
}

/// Write a NumPy array to a WAV file with the given sample rate, channels, and optional dtype
#[pyfunction]
#[pyo3(signature = (samples, path, sr, channels=1, dtype=None))]
fn write(
    samples: &Bound<'_, PyAny>,
    path: &str,
    sr: u32,
    channels: u16,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    let dtype_name = dtype
        .map(|d| extract_dtype_name(d))
        .transpose()?
        .unwrap_or_else(|| "int16".to_string());

    match dtype_name.as_str() {
        "int16" => write_dispatch!(samples, path, sr, Some(channels), i16),
        "int32" => write_dispatch!(samples, path, sr, Some(channels), i32),
        "float32" => write_dispatch!(samples, path, sr, Some(channels), f32),
        "float64" => write_dispatch!(samples, path, sr, Some(channels), f64),
        _ => Err(ValidationError::new_err(format!(
            "Unsupported dtype: {}",
            dtype_name
        ))),
    }
}

/// Python class representing WAV metadata (sample rate, channels, format, etc.)
#[pyclass(name = "WavInfo")]
#[repr(transparent)]
pub struct PyWavInfo(pub WavInfo);

#[pymethods]
impl PyWavInfo {
    #[getter]
    fn sample_rate(&self) -> PyResult<u32> {
        Ok(self.0.sample_rate)
    }

    #[getter]
    fn channels(&self) -> PyResult<u16> {
        Ok(self.0.channels)
    }

    #[getter]
    fn format(&self) -> PyResult<String> {
        Ok(self.0.format.to_string())
    }

    #[getter]
    fn bits_per_sample(&self) -> PyResult<u16> {
        Ok(self.0.bits_per_sample)
    }

    #[getter]
    fn bytes_per_sample(&self) -> PyResult<u16> {
        Ok(self.0.bytes_per_sample)
    }

    #[getter]
    fn n_samples(&self) -> PyResult<u32> {
        Ok(self.0.n_samples)
    }

    #[getter]
    fn duration(&self) -> PyResult<f32> {
        Ok(self.0.duration)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "sample_rate={}, channels={}, format={}, bits_per_sample={}, bytes_per_sample={}, n_samples={}, duration={}",
            self.0.sample_rate,
            self.0.channels,
            self.0.format,
            self.0.bits_per_sample,
            self.0.bytes_per_sample,
            self.0.n_samples,
            self.0.duration
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "WavInfo(sample_rate={}, channels={}, format={}, bits_per_sample={}, bytes_per_sample={}, n_samples={}, duration={})",
            self.0.sample_rate,
            self.0.channels,
            self.0.format,
            self.0.bits_per_sample,
            self.0.bytes_per_sample,
            self.0.n_samples,
            self.0.duration
        ))
    }
}

/// Get metadata about a WAV file
#[pyfunction]
pub fn info(path: &str) -> PyResult<PyWavInfo> {
    let info = crate::info(path).map_err(|e| WaversError::new_err(format!("{:?}", e)))?;
    Ok(PyWavInfo(info))
}

/// PyWavers module definition
#[pymodule]
fn wavers(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IoError", py.get_type::<IoError>())?;
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    m.add("ConversionError", py.get_type::<ConversionError>())?;
    m.add("OperationError", py.get_type::<OperationError>())?;
    m.add("SampleError", py.get_type::<SampleError>())?;
    m.add("InternalError", py.get_type::<InternalError>())?;

    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_class::<PyWavInfo>()?;
    Ok(())
}
