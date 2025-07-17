//! Core WAV file operations for Python bindings.
//!
//! This module provides the main Python API for reading and writing WAV files
//! using the AudioSamples-based functions from the core Wavers library.

use crate::python::utils::map_wavers_error;
use crate::{info, read_audio, write_audio};
use audio_samples::python::PyAudioSamples;
use audio_samples::{AudioSample, AudioSamples};
use numpy::{PyArray1, PyArray2, PyArrayDescr, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::path::Path;

/// Internal dtype representation for audio sample types
#[derive(Debug, Clone, Copy, PartialEq)]
enum DType {
    F64,
    F32,
    I32,
    I16,
}

/// Parse a numpy dtype descriptor to our internal DType enum
fn parse_numpy_dtype(dtype: &Bound<PyArrayDescr>) -> PyResult<DType> {
    // Get the type name from the dtype
    let type_name = dtype.get_item("name")?;
    let name_str: String = type_name.extract()?;

    match name_str.as_str() {
        "float64" | "f8" => Ok(DType::F64),
        "float32" | "f4" => Ok(DType::F32),
        "int32" | "i4" => Ok(DType::I32),
        "int16" | "i2" => Ok(DType::I16),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported dtype '{}'. Supported types: float64, float32, int32, int16",
            name_str
        ))),
    }
}

/// Parse a string dtype to our internal DType enum (fallback for convenience)
fn parse_string_dtype(dtype_str: &str) -> PyResult<DType> {
    match dtype_str {
        "f64" | "float64" => Ok(DType::F64),
        "f32" | "float32" => Ok(DType::F32),
        "i32" | "int32" => Ok(DType::I32),
        "i16" | "int16" => Ok(DType::I16),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported dtype string '{}'. Use numpy dtypes (np.float32, np.int16) or supported strings: f64, f32, i32, i16",
            dtype_str
        ))),
    }
}

/// Create PyAudioSamples from AudioSamples using the public Python API
fn create_py_audio_samples<T: AudioSample + numpy::Element>(
    audio_samples: AudioSamples<T>,
) -> PyResult<PyAudioSamples>
where
    T: Clone + Send + 'static,
{
    // Convert to numpy array first, then create PyAudioSamples using the public Python constructor
    Python::with_gil(|py| {
        let numpy_array = if audio_samples.is_mono() {
            let mono_data = audio_samples.as_mono().unwrap();
            let samples_vec = mono_data.to_vec();
            PyArray1::from_vec(py, samples_vec).to_object(py)
        } else {
            let multi_data = audio_samples.as_multi_channel().unwrap();
            PyArray2::from_owned_array(py, multi_data.clone()).to_object(py)
        };

        // Import and call the PyAudioSamples constructor
        let audio_samples_module = PyModule::import(py, "audio_samples")?;
        let py_audio_samples_class = audio_samples_module.getattr("AudioSamples")?;
        let py_audio_samples =
            py_audio_samples_class.call1((numpy_array, audio_samples.sample_rate()))?;

        // Extract the PyAudioSamples from the Python object
        py_audio_samples.extract::<PyAudioSamples>()
    })
}

/// Read a WAV file and return an AudioSamples object.
///
/// This function reads a WAV file and returns an AudioSamples object that can be used
/// with the audio_samples Python bindings for further processing.
///
/// # Arguments
/// * `path` - Path to the WAV file to read
/// * `dtype` - Target data type (numpy.dtype object or string). Defaults to numpy.float32.
///
/// # Returns
/// An AudioSamples object containing the audio data with metadata.
///
/// # Examples
/// ```python
/// import wavers
/// import numpy as np
///
/// # Read as float32 (default)
/// audio = wavers.read_wav("audio.wav")
///
/// # Read with numpy dtype (recommended)
/// audio = wavers.read_wav("audio.wav", dtype=np.float64)
/// audio = wavers.read_wav("audio.wav", dtype=np.int16)
///
/// # String fallback (not recommended)
/// audio = wavers.read_wav("audio.wav", dtype="f64")
/// ```
#[pyfunction]
#[pyo3(signature = (path, *, dtype=None))]
pub fn read_wav(path: &str, dtype: Option<&Bound<PyAny>>) -> PyResult<PyAudioSamples> {
    let path = Path::new(path);

    // Parse dtype parameter (numpy dtype or string, default to f32)
    let target_type = if let Some(dt) = dtype {
        if let Ok(numpy_dtype) = dt.downcast::<numpy::PyArrayDescr>() {
            parse_numpy_dtype(numpy_dtype)?
        } else if let Ok(dtype_str) = dt.extract::<String>() {
            parse_string_dtype(&dtype_str)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dtype must be a numpy dtype (e.g., np.float32, np.int16) or string",
            ));
        }
    } else {
        DType::F32 // Default to f32
    };

    match target_type {
        DType::F64 => {
            let audio_samples = read_audio::<f64, _>(path).map_err(map_wavers_error)?;
            create_py_audio_samples(audio_samples)
        }
        DType::F32 => {
            let audio_samples = read_audio::<f32, _>(path).map_err(map_wavers_error)?;
            create_py_audio_samples(audio_samples)
        }
        DType::I32 => {
            let audio_samples = read_audio::<i32, _>(path).map_err(map_wavers_error)?;
            create_py_audio_samples(audio_samples)
        }
        DType::I16 => {
            let audio_samples = read_audio::<i16, _>(path).map_err(map_wavers_error)?;
            create_py_audio_samples(audio_samples)
        }
    }
}

/// Write an AudioSamples object to a WAV file.
///
/// This function writes an AudioSamples object to a WAV file, preserving the original
/// sample type and metadata.
///
/// # Arguments
/// * `path` - Path where the WAV file will be written
/// * `audio` - AudioSamples object to write
///
/// # Examples
/// ```python
/// import wavers
///
/// # Read and write back
/// audio = wavers.read_wav("input.wav")
/// wavers.write_wav("output.wav", audio)
///
/// # Works with processed audio
/// processed = audio.normalize()
/// wavers.write_wav("processed.wav", processed)
/// ```
#[pyfunction]
pub fn write_wav(path: &str, audio: &PyAudioSamples) -> PyResult<()> {
    let path = Path::new(path);
    // Access sample rate through the public API using Python
    let sample_rate: u32 = Python::with_gil(|py| {
        let audio_py = Py::new(py, audio.clone())?;
        let sample_rate_attr = audio_py.getattr(py, "sample_rate")?;
        sample_rate_attr.extract(py)
    })?;

    // Convert PyAudioSamples to numpy array, then to AudioSamples
    Python::with_gil(|py| {
        // Call to_numpy using the public Python API
        let audio_py = Py::new(py, audio.clone())?;
        let numpy_array = audio_py.call_method1(py, "to_numpy", ())?;

        // Try to extract as different NumPy array types and write directly
        if let Ok(array_f64) = numpy_array.downcast_bound::<PyArray1<f64>>(py) {
            let ndarray = array_f64.readonly().as_array().to_owned();
            let audio_samples = AudioSamples::new_mono(ndarray, sample_rate);
            write_audio(path, &audio_samples).map_err(map_wavers_error)?;
            return Ok(());
        }

        if let Ok(array_f64) = numpy_array.downcast_bound::<PyArray2<f64>>(py) {
            let ndarray = array_f64.readonly().as_array().to_owned();
            let audio_samples = AudioSamples::new_multi_channel(ndarray, sample_rate);
            write_audio(path, &audio_samples).map_err(map_wavers_error)?;
            return Ok(());
        }

        if let Ok(array_f32) = numpy_array.downcast_bound::<PyArray1<f32>>(py) {
            let ndarray = array_f32.readonly().as_array().to_owned();
            let audio_samples = AudioSamples::new_mono(ndarray, sample_rate);
            write_audio(path, &audio_samples).map_err(map_wavers_error)?;
            return Ok(());
        }

        if let Ok(array_f32) = numpy_array.downcast_bound::<PyArray2<f32>>(py) {
            let ndarray = array_f32.readonly().as_array().to_owned();
            let audio_samples = AudioSamples::new_multi_channel(ndarray, sample_rate);
            write_audio(path, &audio_samples).map_err(map_wavers_error)?;
            return Ok(());
        }

        if let Ok(array_i32) = numpy_array.downcast_bound::<PyArray1<i32>>(py) {
            let ndarray = array_i32.readonly().as_array().to_owned();
            let audio_samples = AudioSamples::new_mono(ndarray, sample_rate);
            write_audio(path, &audio_samples).map_err(map_wavers_error)?;
            return Ok(());
        }

        if let Ok(array_i32) = numpy_array.downcast_bound::<PyArray2<i32>>(py) {
            let ndarray = array_i32.readonly().as_array().to_owned();
            let audio_samples = AudioSamples::new_multi_channel(ndarray, sample_rate);
            write_audio(path, &audio_samples).map_err(map_wavers_error)?;
            return Ok(());
        }

        if let Ok(array_i16) = numpy_array.downcast_bound::<PyArray1<i16>>(py) {
            let ndarray = array_i16.readonly().as_array().to_owned();
            let audio_samples = AudioSamples::new_mono(ndarray, sample_rate);
            write_audio(path, &audio_samples).map_err(map_wavers_error)?;
            return Ok(());
        }

        if let Ok(array_i16) = numpy_array.downcast_bound::<PyArray2<i16>>(py) {
            let ndarray = array_i16.readonly().as_array().to_owned();
            let audio_samples = AudioSamples::new_multi_channel(ndarray, sample_rate);
            write_audio(path, &audio_samples).map_err(map_wavers_error)?;
            return Ok(());
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported array type from AudioSamples",
        ))
    })
}

/// Python wrapper for WAV file information.
#[pyclass]
pub struct WavInfo {
    /// Sample rate in Hz
    #[pyo3(get)]
    pub sample_rate: u32,

    /// Number of channels
    #[pyo3(get)]
    pub channels: u16,

    /// Bits per sample
    #[pyo3(get)]
    pub bits_per_sample: u16,

    /// Bytes per sample
    #[pyo3(get)]
    pub bytes_per_sample: u16,

    /// Total number of samples
    #[pyo3(get)]
    pub n_samples: u32,

    /// Duration in seconds
    #[pyo3(get)]
    pub duration: f32,
}

#[pymethods]
impl WavInfo {
    /// String representation of WAV info.
    fn __repr__(&self) -> String {
        format!(
            "WavInfo(sample_rate={}, channels={}, bits_per_sample={}, duration={:.3}s)",
            self.sample_rate, self.channels, self.bits_per_sample, self.duration
        )
    }

    /// Brief string representation.
    fn __str__(&self) -> String {
        format!(
            "WAV: {} ch, {:.3}s @ {}Hz, {}-bit",
            self.channels, self.duration, self.sample_rate, self.bits_per_sample
        )
    }
}

/// Get information about a WAV file without loading the audio data.
///
/// This function quickly reads the header information from a WAV file,
/// providing metadata without loading the actual audio samples.
///
/// # Arguments
/// * `path` - Path to the WAV file
///
/// # Returns
/// A WavInfo object containing metadata about the WAV file.
///
/// # Examples
/// ```python
/// import wavers
///
/// info = wavers.wav_info("audio.wav")
/// print(f"Sample rate: {info.sample_rate} Hz")
/// print(f"Channels: {info.channels}")
/// print(f"Duration: {info.duration:.3f} seconds")
/// ```
#[pyfunction]
pub fn wav_info(path: &str) -> PyResult<WavInfo> {
    let path = Path::new(path);
    let wav_info = info(path).map_err(map_wavers_error)?;

    Ok(WavInfo {
        sample_rate: wav_info.sample_rate,
        channels: wav_info.channels,
        bits_per_sample: wav_info.bits_per_sample,
        bytes_per_sample: wav_info.bytes_per_sample,
        n_samples: wav_info.n_samples,
        duration: wav_info.duration,
    })
}

/// Read a WAV file and return a NumPy array.
///
/// This function reads a WAV file and returns the audio data as a NumPy array,
/// providing a more direct interface for users who prefer working with NumPy arrays.
///
/// # Arguments
/// * `path` - Path to the WAV file to read
/// * `dtype` - Target data type (numpy.dtype object or string). Defaults to numpy.float32.
///
/// # Returns
/// A tuple containing (numpy_array, sample_rate) where numpy_array has shape:
/// - (samples,) for mono audio
/// - (channels, samples) for multi-channel audio
///
/// # Examples
/// ```python
/// import wavers
/// import numpy as np
///
/// # Read as NumPy array with numpy dtype (recommended)
/// audio_data, sample_rate = wavers.read_wav_numpy("audio.wav", dtype=np.float32)
/// audio_data, sample_rate = wavers.read_wav_numpy("audio.wav", dtype=np.int16)
///
/// # Default (float32)
/// audio_data, sample_rate = wavers.read_wav_numpy("audio.wav")
/// ```
#[pyfunction]
#[pyo3(signature = (path, *, dtype=None))]
pub fn read_wav_numpy(
    py: Python,
    path: &str,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<(PyObject, u32)> {
    let path = Path::new(path);

    // Parse dtype parameter (numpy dtype or string, default to f32)
    let target_type = if let Some(dt) = dtype {
        if let Ok(numpy_dtype) = dt.downcast::<numpy::PyArrayDescr>() {
            parse_numpy_dtype(numpy_dtype)?
        } else if let Ok(dtype_str) = dt.extract::<String>() {
            parse_string_dtype(&dtype_str)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dtype must be a numpy dtype (e.g., np.float32, np.int16) or string",
            ));
        }
    } else {
        DType::F32 // Default to f32
    };

    match target_type {
        DType::F64 => {
            let audio_samples = read_audio::<f64, _>(path).map_err(map_wavers_error)?;
            let sample_rate = audio_samples.sample_rate();

            let numpy_array = if audio_samples.is_mono() {
                let mono_data = audio_samples.as_mono().unwrap();
                PyArray1::from_slice(py, mono_data.as_slice().unwrap()).into()
            } else {
                let multi_data = audio_samples.as_multi_channel().unwrap();
                PyArray2::from_owned_array(py, multi_data.clone()).into()
            };

            Ok((numpy_array, sample_rate))
        }
        DType::F32 => {
            let audio_samples = read_audio::<f32, _>(path).map_err(map_wavers_error)?;
            let sample_rate = audio_samples.sample_rate();

            let numpy_array = if audio_samples.is_mono() {
                let mono_data = audio_samples.as_mono().unwrap();
                PyArray1::from_slice(py, mono_data.as_slice().unwrap()).into()
            } else {
                let multi_data = audio_samples.as_multi_channel().unwrap();
                PyArray2::from_owned_array(py, multi_data.clone()).into()
            };

            Ok((numpy_array, sample_rate))
        }
        DType::I32 => {
            let audio_samples = read_audio::<i32, _>(path).map_err(map_wavers_error)?;
            let sample_rate = audio_samples.sample_rate();

            let numpy_array = if audio_samples.is_mono() {
                let mono_data = audio_samples.as_mono().unwrap();
                PyArray1::from_slice(py, mono_data.as_slice().unwrap()).into()
            } else {
                let multi_data = audio_samples.as_multi_channel().unwrap();
                PyArray2::from_owned_array(py, multi_data.clone()).into()
            };

            Ok((numpy_array, sample_rate))
        }
        DType::I16 => {
            let audio_samples = read_audio::<i16, _>(path).map_err(map_wavers_error)?;
            let sample_rate = audio_samples.sample_rate();

            let numpy_array = if audio_samples.is_mono() {
                let mono_data = audio_samples.as_mono().unwrap();
                PyArray1::from_slice(py, mono_data.as_slice().unwrap()).into()
            } else {
                let multi_data = audio_samples.as_multi_channel().unwrap();
                PyArray2::from_owned_array(py, multi_data.clone()).into()
            };

            Ok((numpy_array, sample_rate))
        }
    }
}

/// Write a NumPy array to a WAV file.
///
/// This function writes a NumPy array to a WAV file, providing a direct interface
/// for users who prefer working with NumPy arrays.
///
/// # Arguments
/// * `path` - Path where the WAV file will be written
/// * `data` - NumPy array containing audio data
/// * `sample_rate` - Sample rate in Hz
///
/// # Examples
/// ```python
/// import wavers
/// import numpy as np
///
/// # Create some audio data
/// sample_rate = 44100
/// duration = 1.0
/// t = np.linspace(0, duration, int(sample_rate * duration))
/// audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
///
/// # Write to file
/// wavers.write_wav_numpy("sine.wav", audio_data, sample_rate)
/// ```
#[pyfunction]
pub fn write_wav_numpy(path: &str, data: &Bound<PyAny>, sample_rate: u32) -> PyResult<()> {
    let path = Path::new(path);

    // Try to extract as different NumPy array types
    if let Ok(array_f64) = data.extract::<PyReadonlyArray1<f64>>() {
        let ndarray = array_f64.as_array().to_owned();
        let audio_samples = AudioSamples::new_mono(ndarray, sample_rate);
        write_audio(path, &audio_samples).map_err(map_wavers_error)?;
        return Ok(());
    }

    if let Ok(array_f64) = data.extract::<PyReadonlyArray2<f64>>() {
        let ndarray = array_f64.as_array().to_owned();
        let audio_samples = AudioSamples::new_multi_channel(ndarray, sample_rate);
        write_audio(path, &audio_samples).map_err(map_wavers_error)?;
        return Ok(());
    }

    if let Ok(array_f32) = data.extract::<PyReadonlyArray1<f32>>() {
        let ndarray = array_f32.as_array().to_owned();
        let audio_samples = AudioSamples::new_mono(ndarray, sample_rate);
        write_audio(path, &audio_samples).map_err(map_wavers_error)?;
        return Ok(());
    }

    if let Ok(array_f32) = data.extract::<PyReadonlyArray2<f32>>() {
        let ndarray = array_f32.as_array().to_owned();
        let audio_samples = AudioSamples::new_multi_channel(ndarray, sample_rate);
        write_audio(path, &audio_samples).map_err(map_wavers_error)?;
        return Ok(());
    }

    if let Ok(array_i32) = data.extract::<PyReadonlyArray1<i32>>() {
        let ndarray = array_i32.as_array().to_owned();
        let audio_samples = AudioSamples::new_mono(ndarray, sample_rate);
        write_audio(path, &audio_samples).map_err(map_wavers_error)?;
        return Ok(());
    }

    if let Ok(array_i32) = data.extract::<PyReadonlyArray2<i32>>() {
        let ndarray = array_i32.as_array().to_owned();
        let audio_samples = AudioSamples::new_multi_channel(ndarray, sample_rate);
        write_audio(path, &audio_samples).map_err(map_wavers_error)?;
        return Ok(());
    }

    if let Ok(array_i16) = data.extract::<PyReadonlyArray1<i16>>() {
        let ndarray = array_i16.as_array().to_owned();
        let audio_samples = AudioSamples::new_mono(ndarray, sample_rate);
        write_audio(path, &audio_samples).map_err(map_wavers_error)?;
        return Ok(());
    }

    if let Ok(array_i16) = data.extract::<PyReadonlyArray2<i16>>() {
        let ndarray = array_i16.as_array().to_owned();
        let audio_samples = AudioSamples::new_multi_channel(ndarray, sample_rate);
        write_audio(path, &audio_samples).map_err(map_wavers_error)?;
        return Ok(());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported array type. Supported types: f64, f32, i32, i16",
    ))
}
