//! Type conversion utilities for Python bindings.
//!
//! This module provides efficient type conversions between Python types,
//! NumPy arrays, and Rust audio types, with support for various sample formats.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, Element};
use audio_samples::{AudioSample, AudioSamples, I24};
use crate::python::errors::*;
use crate::python::utils::SampleType;

/// Enum for different target types in conversions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetType {
    I16,
    I32,
    F32,
    F64,
}

impl TargetType {
    pub fn from_sample_type(sample_type: SampleType) -> Self {
        match sample_type {
            SampleType::I16 => TargetType::I16,
            SampleType::I32 => TargetType::I32,
            SampleType::F32 => TargetType::F32,
            SampleType::F64 => TargetType::F64,
        }
    }
}

/// Trait for converting between different audio sample types
pub trait AudioSampleConversion<T> {
    fn convert_to_sample_type(self, target_type: TargetType) -> PyResult<T>;
}

/// Convert a Python object to AudioSamples with automatic type detection
pub fn py_to_audio_samples(
    py_obj: &Bound<PyAny>,
    sample_rate: u32,
    channels: Option<usize>,
) -> PyResult<AudioSamplesWrapper> {
    // Try different numpy array types in order of preference
    
    // Try f64 first (highest precision)
    if let Ok(array_1d) = py_obj.extract::<PyReadonlyArray1<f64>>() {
        let ndarray = array_1d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != 1 {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    "1 channel (1D array)",
                ));
            }
        }
        let audio = AudioSamples::new_mono(ndarray, sample_rate);
        return Ok(AudioSamplesWrapper::F64(audio));
    }

    if let Ok(array_2d) = py_obj.extract::<PyReadonlyArray2<f64>>() {
        let ndarray = array_2d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    &format!("{} channels", ndarray.nrows()),
                ));
            }
        }
        let audio = AudioSamples::new_multi_channel(ndarray, sample_rate);
        return Ok(AudioSamplesWrapper::F64(audio));
    }

    // Try f32
    if let Ok(array_1d) = py_obj.extract::<PyReadonlyArray1<f32>>() {
        let ndarray = array_1d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != 1 {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    "1 channel (1D array)",
                ));
            }
        }
        let audio = AudioSamples::new_mono(ndarray, sample_rate);
        return Ok(AudioSamplesWrapper::F32(audio));
    }

    if let Ok(array_2d) = py_obj.extract::<PyReadonlyArray2<f32>>() {
        let ndarray = array_2d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    &format!("{} channels", ndarray.nrows()),
                ));
            }
        }
        let audio = AudioSamples::new_multi_channel(ndarray, sample_rate);
        return Ok(AudioSamplesWrapper::F32(audio));
    }

    // Try i32
    if let Ok(array_1d) = py_obj.extract::<PyReadonlyArray1<i32>>() {
        let ndarray = array_1d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != 1 {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    "1 channel (1D array)",
                ));
            }
        }
        let audio = AudioSamples::new_mono(ndarray, sample_rate);
        return Ok(AudioSamplesWrapper::I32(audio));
    }

    if let Ok(array_2d) = py_obj.extract::<PyReadonlyArray2<i32>>() {
        let ndarray = array_2d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    &format!("{} channels", ndarray.nrows()),
                ));
            }
        }
        let audio = AudioSamples::new_multi_channel(ndarray, sample_rate);
        return Ok(AudioSamplesWrapper::I32(audio));
    }

    // Try i16
    if let Ok(array_1d) = py_obj.extract::<PyReadonlyArray1<i16>>() {
        let ndarray = array_1d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != 1 {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    "1 channel (1D array)",
                ));
            }
        }
        let audio = AudioSamples::new_mono(ndarray, sample_rate);
        return Ok(AudioSamplesWrapper::I16(audio));
    }

    if let Ok(array_2d) = py_obj.extract::<PyReadonlyArray2<i16>>() {
        let ndarray = array_2d.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(dimension_mismatch_error(
                    &format!("{} channels", ch),
                    &format!("{} channels", ndarray.nrows()),
                ));
            }
        }
        let audio = AudioSamples::new_multi_channel(ndarray, sample_rate);
        return Ok(AudioSamplesWrapper::I16(audio));
    }

    Err(unsupported_type_error(
        "numpy array",
        "audio samples conversion",
    ))
}

/// Wrapper enum for different AudioSamples types
#[derive(Debug, Clone)]
pub enum AudioSamplesWrapper {
    I16(AudioSamples<i16>),
    I32(AudioSamples<i32>),
    F32(AudioSamples<f32>),
    F64(AudioSamples<f64>),
}

impl AudioSamplesWrapper {
    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        match self {
            AudioSamplesWrapper::I16(audio) => audio.sample_rate(),
            AudioSamplesWrapper::I32(audio) => audio.sample_rate(),
            AudioSamplesWrapper::F32(audio) => audio.sample_rate(),
            AudioSamplesWrapper::F64(audio) => audio.sample_rate(),
        }
    }

    /// Get the number of channels
    pub fn channels(&self) -> usize {
        match self {
            AudioSamplesWrapper::I16(audio) => audio.channels(),
            AudioSamplesWrapper::I32(audio) => audio.channels(),
            AudioSamplesWrapper::F32(audio) => audio.channels(),
            AudioSamplesWrapper::F64(audio) => audio.channels(),
        }
    }

    /// Get the number of samples per channel
    pub fn samples_per_channel(&self) -> usize {
        match self {
            AudioSamplesWrapper::I16(audio) => audio.samples_per_channel(),
            AudioSamplesWrapper::I32(audio) => audio.samples_per_channel(),
            AudioSamplesWrapper::F32(audio) => audio.samples_per_channel(),
            AudioSamplesWrapper::F64(audio) => audio.samples_per_channel(),
        }
    }

    /// Get the duration in seconds
    pub fn duration_seconds(&self) -> f64 {
        match self {
            AudioSamplesWrapper::I16(audio) => audio.duration_seconds(),
            AudioSamplesWrapper::I32(audio) => audio.duration_seconds(),
            AudioSamplesWrapper::F32(audio) => audio.duration_seconds(),
            AudioSamplesWrapper::F64(audio) => audio.duration_seconds(),
        }
    }

    /// Convert to a specific sample type
    pub fn convert_to_type(&self, target_type: TargetType) -> PyResult<AudioSamplesWrapper> {
        use audio_samples::operations::AudioTypeConversion;
        
        match (self, target_type) {
            // I16 conversions
            (AudioSamplesWrapper::I16(audio), TargetType::I16) => Ok(AudioSamplesWrapper::I16(audio.clone())),
            (AudioSamplesWrapper::I16(audio), TargetType::I32) => {
                let converted = audio.to_i32().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::I32(converted))
            }
            (AudioSamplesWrapper::I16(audio), TargetType::F32) => {
                let converted = audio.to_f32().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::F32(converted))
            }
            (AudioSamplesWrapper::I16(audio), TargetType::F64) => {
                let converted = audio.to_f64().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::F64(converted))
            }
            
            // I32 conversions
            (AudioSamplesWrapper::I32(audio), TargetType::I16) => {
                let converted = audio.to_i16().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::I16(converted))
            }
            (AudioSamplesWrapper::I32(audio), TargetType::I32) => Ok(AudioSamplesWrapper::I32(audio.clone())),
            (AudioSamplesWrapper::I32(audio), TargetType::F32) => {
                let converted = audio.to_f32().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::F32(converted))
            }
            (AudioSamplesWrapper::I32(audio), TargetType::F64) => {
                let converted = audio.to_f64().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::F64(converted))
            }
            
            // F32 conversions
            (AudioSamplesWrapper::F32(audio), TargetType::I16) => {
                let converted = audio.to_i16().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::I16(converted))
            }
            (AudioSamplesWrapper::F32(audio), TargetType::I32) => {
                let converted = audio.to_i32().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::I32(converted))
            }
            (AudioSamplesWrapper::F32(audio), TargetType::F32) => Ok(AudioSamplesWrapper::F32(audio.clone())),
            (AudioSamplesWrapper::F32(audio), TargetType::F64) => {
                let converted = audio.to_f64().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::F64(converted))
            }
            
            // F64 conversions
            (AudioSamplesWrapper::F64(audio), TargetType::I16) => {
                let converted = audio.to_i16().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::I16(converted))
            }
            (AudioSamplesWrapper::F64(audio), TargetType::I32) => {
                let converted = audio.to_i32().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::I32(converted))
            }
            (AudioSamplesWrapper::F64(audio), TargetType::F32) => {
                let converted = audio.to_f32().map_err(|e| map_wavers_error(crate::WaversErrorKind::SampleError(e)))?;
                Ok(AudioSamplesWrapper::F32(converted))
            }
            (AudioSamplesWrapper::F64(audio), TargetType::F64) => Ok(AudioSamplesWrapper::F64(audio.clone())),
        }
    }

    /// Convert to numpy array
    pub fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self {
            AudioSamplesWrapper::I16(audio) => audio_samples_to_numpy(py, audio),
            AudioSamplesWrapper::I32(audio) => audio_samples_to_numpy(py, audio),
            AudioSamplesWrapper::F32(audio) => audio_samples_to_numpy(py, audio),
            AudioSamplesWrapper::F64(audio) => audio_samples_to_numpy(py, audio),
        }
    }

    /// Convert to numpy array with a specific type
    pub fn to_numpy_with_type(&self, py: Python<'_>, target_type: TargetType) -> PyResult<PyObject> {
        let converted = self.convert_to_type(target_type)?;
        converted.to_numpy(py)
    }

    /// Get the current sample type
    pub fn sample_type(&self) -> SampleType {
        match self {
            AudioSamplesWrapper::I16(_) => SampleType::I16,
            AudioSamplesWrapper::I32(_) => SampleType::I32,
            AudioSamplesWrapper::F32(_) => SampleType::F32,
            AudioSamplesWrapper::F64(_) => SampleType::F64,
        }
    }
}

/// Convert AudioSamples to numpy array
fn audio_samples_to_numpy<T>(py: Python<'_>, audio: &AudioSamples<T>) -> PyResult<PyObject>
where
    T: Element + Clone,
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

/// Convert raw samples to numpy array
pub fn raw_samples_to_numpy<T>(py: Python<'_>, samples: &[T], channels: usize) -> PyResult<PyObject>
where
    T: Element + Clone,
{
    if channels == 1 {
        let array = PyArray1::from_slice(py, samples);
        Ok(array.into())
    } else {
        let samples_per_channel = samples.len() / channels;
        let shape = (channels, samples_per_channel);
        let array = PyArray2::from_slice(py, samples, shape);
        Ok(array.into())
    }
}

/// Convert numpy array to raw samples
pub fn numpy_to_raw_samples<T>(py_obj: &Bound<PyAny>) -> PyResult<(Vec<T>, usize)>
where
    T: Element + Clone,
{
    // Try 1D array first
    if let Ok(array_1d) = py_obj.extract::<PyReadonlyArray1<T>>() {
        let samples = array_1d.as_array().to_vec();
        return Ok((samples, 1));
    }

    // Try 2D array
    if let Ok(array_2d) = py_obj.extract::<PyReadonlyArray2<T>>() {
        let ndarray = array_2d.as_array();
        let channels = ndarray.nrows();
        let samples_per_channel = ndarray.ncols();
        
        // Convert to interleaved format
        let mut samples = Vec::with_capacity(channels * samples_per_channel);
        for sample_idx in 0..samples_per_channel {
            for channel_idx in 0..channels {
                samples.push(ndarray[(channel_idx, sample_idx)].clone());
            }
        }
        
        return Ok((samples, channels));
    }

    Err(unsupported_type_error(
        "numpy array",
        "raw samples conversion",
    ))
}

/// Parse dtype string or object to TargetType
pub fn parse_dtype_to_target_type(dtype: Option<&Bound<PyAny>>) -> PyResult<Option<TargetType>> {
    match dtype {
        Some(dt) => {
            // Try to extract as string first
            if let Ok(dtype_str) = dt.extract::<String>() {
                let sample_type = crate::python::utils::parse_sample_type(&dtype_str)?;
                Ok(Some(TargetType::from_sample_type(sample_type)))
            } else {
                // Could be a numpy dtype object - for now, just return None
                // TODO: Add proper numpy dtype parsing
                Ok(None)
            }
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::Python;

    #[test]
    fn test_target_type_conversion() {
        assert_eq!(
            TargetType::from_sample_type(SampleType::I16),
            TargetType::I16
        );
        assert_eq!(
            TargetType::from_sample_type(SampleType::F32),
            TargetType::F32
        );
    }

    #[test]
    fn test_audio_samples_wrapper_properties() {
        let audio = AudioSamples::new_mono(ndarray::Array1::from_vec(vec![1.0f32, 2.0, 3.0]), 44100);
        let wrapper = AudioSamplesWrapper::F32(audio);
        
        assert_eq!(wrapper.sample_rate(), 44100);
        assert_eq!(wrapper.channels(), 1);
        assert_eq!(wrapper.samples_per_channel(), 3);
        assert_eq!(wrapper.sample_type(), SampleType::F32);
    }
}