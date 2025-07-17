//! Utility functions for Python bindings.
//!
//! This module provides helper functions for error handling and conversion
//! between Rust and Python types.

use crate::WaversErrorKind;
use pyo3::prelude::*;

/// Convert a Wavers error to a Python exception.
pub fn map_wavers_error(error: WaversErrorKind) -> PyErr {
    use crate::error::{ConversionError, InternalError, ParseError, ValidationError};

    match error {
        WaversErrorKind::Io(e) => {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("IO error: {}", e))
        }
        WaversErrorKind::Parse(parse_err) => match parse_err {
            ParseError::NoChunks => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("No RIFF chunks found in WAV file")
            }
            ParseError::ChunkNotFound { id } => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Required chunk {:?} not found", id),
            ),
            ParseError::Nom { offset, msg } => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Parse error at byte {}: {}", offset, msg),
            ),
        },
        WaversErrorKind::Validate(validation_err) => match validation_err {
            ValidationError::Header(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid WAV header: {}", msg),
            ),
            _ => PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Validation error: {}",
                validation_err
            )),
        },
        WaversErrorKind::Convert(conversion_err) => match conversion_err {
            ConversionError::Overflow { idx } => {
                PyErr::new::<pyo3::exceptions::PyOverflowError, _>(format!(
                    "Conversion overflow at index {}",
                    idx
                ))
            }
            _ => PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Conversion error: {}",
                conversion_err
            )),
        },
        WaversErrorKind::Operation(operation_err) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Operation error: {}",
                operation_err
            ))
        }
        WaversErrorKind::Internal(internal_err) => match internal_err {
            InternalError::Bug(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Internal error: {}",
                msg
            )),
        },
        WaversErrorKind::SampleError(audio_err) => {
            // Map audio_samples errors to Python exceptions
            match audio_err {
                audio_samples::AudioSampleError::ConversionError(
                    value,
                    from_type,
                    to_type,
                    reason,
                ) => PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Audio conversion failed: {} ({} -> {}): {}",
                    value, from_type, to_type, reason
                )),
                audio_samples::AudioSampleError::InvalidRange(msg) => {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid range: {}",
                        msg
                    ))
                }
                audio_samples::AudioSampleError::InvalidParameter(msg) => {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid parameter: {}",
                        msg
                    ))
                }
                audio_samples::AudioSampleError::DimensionMismatch(msg) => {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Dimension mismatch: {}",
                        msg
                    ))
                }
                audio_samples::AudioSampleError::InvalidInput { msg } => {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid input: {}",
                        msg
                    ))
                }
                audio_samples::AudioSampleError::ProcessingError { msg } => {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Processing error: {}",
                        msg
                    ))
                }
                audio_samples::AudioSampleError::FeatureNotEnabled { feature } => {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Feature '{}' is not enabled",
                        feature
                    ))
                }
            }
        }
    }
}
