//! Utility functions for Python bindings.
//!
//! This module provides utility functions for NumPy integration, type conversion,
//! and common operations needed across the Python bindings.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, Element};
use ndarray::{Array1, Array2};
use audio_samples::{AudioSample, AudioSamples};
use crate::python::errors::*;

/// Get the list of supported sample types
#[pyfunction]
pub fn supported_sample_types() -> Vec<&'static str> {
    vec!["i16", "i32", "f32", "f64"]
}

/// Get the library version
#[pyfunction]
pub fn library_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Convert a Python sequence to a Rust vector
pub fn py_sequence_to_vec<T>(py_seq: &Bound<PyAny>) -> PyResult<Vec<T>>
where
    T: for<'a> FromPyObject<'a>,
{
    if let Ok(py_list) = py_seq.downcast::<PyList>() {
        let mut vec = Vec::with_capacity(py_list.len());
        for item in py_list.iter() {
            vec.push(item.extract::<T>()?);
        }
        Ok(vec)
    } else if let Ok(py_tuple) = py_seq.downcast::<PyTuple>() {
        let mut vec = Vec::with_capacity(py_tuple.len());
        for item in py_tuple.iter() {
            vec.push(item.extract::<T>()?);
        }
        Ok(vec)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected a list or tuple",
        ))
    }
}

/// Convert a Rust vector to a Python list
pub fn vec_to_py_list<T>(py: Python<'_>, vec: Vec<T>) -> PyResult<PyObject>
where
    T: IntoPy<PyObject>,
{
    let py_list = PyList::new(py, vec)?;
    Ok(py_list.into())
}

/// Convert a numpy array to AudioSamples, preserving the sample type
pub fn numpy_to_audio_samples<T>(
    array: &Bound<PyAny>,
    sample_rate: u32,
    channels: Option<usize>,
) -> PyResult<AudioSamples<T>>
where
    T: AudioSample + Element + Clone,
{
    // Try 1D array first (mono)
    if let Ok(array_1d) = array.extract::<PyReadonlyArray1<T>>() {
        let ndarray = array_1d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != 1 {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    "1 channel (mono array)",
                ));
            }
        }
        return Ok(AudioSamples::new_mono(ndarray, sample_rate));
    }

    // Try 2D array (multi-channel)
    if let Ok(array_2d) = array.extract::<PyReadonlyArray2<T>>() {
        let ndarray = array_2d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    &format!("{} channels", ndarray.nrows()),
                ));
            }
        }
        return Ok(AudioSamples::new_multi_channel(ndarray, sample_rate));
    }

    Err(unsupported_type_error(
        "numpy array",
        "audio samples conversion",
    ))
}

/// Convert AudioSamples to a numpy array
pub fn audio_samples_to_numpy<T>(
    py: Python<'_>,
    audio: &AudioSamples<T>,
) -> PyResult<PyObject>
where
    T: AudioSample + Element + Clone,
{
    match audio.channels() {
        1 => {
            let mono_data = audio.as_mono().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Failed to extract mono data from AudioSamples",
                )
            })?;
            let array = PyArray1::from_owned_array(py, mono_data.to_owned());
            Ok(array.into())
        }
        _ => {
            let multi_data = audio.as_multi_channel().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Failed to extract multi-channel data from AudioSamples",
                )
            })?;
            let array = PyArray2::from_owned_array(py, multi_data.to_owned());
            Ok(array.into())
        }
    }
}

/// Convert a 1D numpy array to AudioSamples (mono)
pub fn numpy_1d_to_audio_samples<T>(
    array: PyReadonlyArray1<T>,
    sample_rate: u32,
) -> AudioSamples<T>
where
    T: AudioSample + Element + Clone,
{
    let ndarray = array.as_array().to_owned();
    AudioSamples::new_mono(ndarray, sample_rate)
}

/// Convert a 2D numpy array to AudioSamples (multi-channel)
pub fn numpy_2d_to_audio_samples<T>(
    array: PyReadonlyArray2<T>,
    sample_rate: u32,
) -> AudioSamples<T>
where
    T: AudioSample + Element + Clone,
{
    let ndarray = array.as_array().to_owned();
    AudioSamples::new_multi_channel(ndarray, sample_rate)
}

/// Create a 1D numpy array from an ndarray
pub fn ndarray_1d_to_numpy<T>(py: Python<'_>, array: Array1<T>) -> PyResult<PyObject>
where
    T: Element + Clone,
{
    let py_array = PyArray1::from_owned_array(py, array);
    Ok(py_array.into())
}

/// Create a 2D numpy array from an ndarray
pub fn ndarray_2d_to_numpy<T>(py: Python<'_>, array: Array2<T>) -> PyResult<PyObject>
where
    T: Element + Clone,
{
    let py_array = PyArray2::from_owned_array(py, array);
    Ok(py_array.into())
}

/// Parse sample type string to appropriate enum
pub fn parse_sample_type(dtype: &str) -> PyResult<SampleType> {
    match dtype {
        "i16" => Ok(SampleType::I16),
        "i32" => Ok(SampleType::I32),
        "f32" => Ok(SampleType::F32),
        "f64" => Ok(SampleType::F64),
        _ => Err(invalid_parameter_error(
            "dtype",
            dtype,
            "one of ['i16', 'i32', 'f32', 'f64']",
        )),
    }
}

/// Enum for sample types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleType {
    I16,
    I32,
    F32,
    F64,
}

impl SampleType {
    pub fn to_str(&self) -> &'static str {
        match self {
            SampleType::I16 => "i16",
            SampleType::I32 => "i32",
            SampleType::F32 => "f32",
            SampleType::F64 => "f64",
        }
    }
}

/// Parse window function string to appropriate enum
pub fn parse_window_function(window: &str) -> PyResult<WindowFunction> {
    match window {
        "hanning" | "hann" => Ok(WindowFunction::Hanning),
        "hamming" => Ok(WindowFunction::Hamming),
        "blackman" => Ok(WindowFunction::Blackman),
        "bartlett" => Ok(WindowFunction::Bartlett),
        "rectangular" | "rect" => Ok(WindowFunction::Rectangular),
        _ => Err(invalid_parameter_error(
            "window",
            window,
            "one of ['hanning', 'hamming', 'blackman', 'bartlett', 'rectangular']",
        )),
    }
}

/// Enum for window functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowFunction {
    Hanning,
    Hamming,
    Blackman,
    Bartlett,
    Rectangular,
}

/// Parse normalization method string
pub fn parse_normalization_method(method: &str) -> PyResult<audio_samples::NormalizationMethod> {
    match method {
        "minmax" => Ok(audio_samples::NormalizationMethod::MinMax),
        "peak" => Ok(audio_samples::NormalizationMethod::Peak),
        "rms" => Ok(audio_samples::NormalizationMethod::RMS),
        "zscore" => Ok(audio_samples::NormalizationMethod::ZScore),
        _ => Err(invalid_parameter_error(
            "method",
            method,
            "one of ['minmax', 'peak', 'rms', 'zscore']",
        )),
    }
}

/// Parse fade curve string
pub fn parse_fade_curve(curve: &str) -> PyResult<audio_samples::operations::FadeCurve> {
    match curve {
        "linear" => Ok(audio_samples::operations::FadeCurve::Linear),
        "exponential" => Ok(audio_samples::operations::FadeCurve::Exponential),
        "logarithmic" => Ok(audio_samples::operations::FadeCurve::Logarithmic),
        _ => Err(invalid_parameter_error(
            "curve",
            curve,
            "one of ['linear', 'exponential', 'logarithmic']",
        )),
    }
}

/// Parse resampling quality string
pub fn parse_resampling_quality(quality: &str) -> PyResult<audio_samples::operations::types::ResamplingQuality> {
    match quality {
        "low" => Ok(audio_samples::operations::types::ResamplingQuality::Low),
        "medium" => Ok(audio_samples::operations::types::ResamplingQuality::Medium),
        "high" => Ok(audio_samples::operations::types::ResamplingQuality::High),
        _ => Err(invalid_parameter_error(
            "quality",
            quality,
            "one of ['low', 'medium', 'high']",
        )),
    }
}

/// Get the default hop size for a given window size
pub fn default_hop_size(window_size: usize) -> usize {
    window_size / 4 // 75% overlap
}

/// Get the default maximum frequency for a given sample rate
pub fn default_fmax(sample_rate: u32) -> f64 {
    sample_rate as f64 / 2.0 // Nyquist frequency
}

/// Validate that a path is a valid file path
pub fn validate_file_path(path: &str) -> PyResult<()> {
    if path.is_empty() {
        return Err(invalid_parameter_error("path", path, "non-empty string"));
    }
    
    // Check for common invalid characters (basic validation)
    if path.contains('\0') {
        return Err(invalid_parameter_error(
            "path",
            path,
            "path without null characters",
        ));
    }
    
    Ok(())
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Format duration in seconds as human-readable string
pub fn format_duration(seconds: f64) -> String {
    if seconds < 1.0 {
        format!("{:.3} ms", seconds * 1000.0)
    } else if seconds < 60.0 {
        format!("{:.2} s", seconds)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor();
        let remaining_seconds = seconds % 60.0;
        format!("{}:{:05.2} (mm:ss)", minutes, remaining_seconds)
    } else {
        let hours = (seconds / 3600.0).floor();
        let remaining_minutes = ((seconds % 3600.0) / 60.0).floor();
        let remaining_seconds = seconds % 60.0;
        format!("{}:{:02}:{:05.2} (hh:mm:ss)", hours, remaining_minutes, remaining_seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sample_type() {
        assert_eq!(parse_sample_type("i16").unwrap(), SampleType::I16);
        assert_eq!(parse_sample_type("f32").unwrap(), SampleType::F32);
        assert!(parse_sample_type("invalid").is_err());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.5), "500.000 ms");
        assert_eq!(format_duration(1.5), "1.50 s");
        assert_eq!(format_duration(65.0), "1:05.00 (mm:ss)");
        assert_eq!(format_duration(3665.0), "1:01:05.00 (hh:mm:ss)");
    }

    #[test]
    fn test_validate_file_path() {
        assert!(validate_file_path("test.wav").is_ok());
        assert!(validate_file_path("").is_err());
        assert!(validate_file_path("file\0name").is_err());
    }
}