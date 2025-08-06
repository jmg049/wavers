//! Error handling and mapping for Python bindings.
//!
//! This module provides comprehensive error mapping from Rust wavers errors
//! to appropriate Python exceptions, ensuring clear error messages and
//! proper exception types for different kinds of errors.

use pyo3::exceptions;
use pyo3::prelude::*;
use crate::{WaversErrorKind, WaversResult};

/// Map Rust WaversErrorKind to appropriate Python exceptions.
///
/// This function provides detailed error mapping that preserves the original
/// error information while using appropriate Python exception types.
///
/// # Arguments
/// * `error` - The Rust WaversErrorKind to map
///
/// # Returns
/// A PyErr that can be returned to Python
pub fn map_wavers_error(error: WaversErrorKind) -> PyErr {
    match error {
        WaversErrorKind::Io(io_error) => {
            PyErr::new::<exceptions::PyIOError, _>(format!("I/O error: {}", io_error))
        }
        WaversErrorKind::Parse(parse_error) => {
            PyErr::new::<exceptions::PyValueError, _>(format!("Parse error: {}", parse_error))
        }
        WaversErrorKind::Validate(validation_error) => {
            PyErr::new::<exceptions::PyValueError, _>(format!("Validation error: {}", validation_error))
        }
        WaversErrorKind::Convert(conversion_error) => {
            PyErr::new::<exceptions::PyValueError, _>(format!("Conversion error: {}", conversion_error))
        }
        WaversErrorKind::Operation(operation_error) => {
            PyErr::new::<exceptions::PyRuntimeError, _>(format!("Operation error: {}", operation_error))
        }
        WaversErrorKind::Internal(internal_error) => {
            PyErr::new::<exceptions::PyRuntimeError, _>(format!("Internal error: {}", internal_error))
        }
        WaversErrorKind::SampleError(sample_error) => {
            // Map audio_samples errors to appropriate Python exceptions
            match sample_error {
                audio_samples::AudioSampleError::ConversionError(value, from_type, to_type, reason) => {
                    PyErr::new::<exceptions::PyValueError, _>(format!(
                        "Audio sample conversion failed: {} ({} -> {}): {}",
                        value, from_type, to_type, reason
                    ))
                }
                audio_samples::AudioSampleError::InvalidRange(msg) => {
                    PyErr::new::<exceptions::PyValueError, _>(format!("Invalid range: {}", msg))
                }
                audio_samples::AudioSampleError::InvalidParameter(msg) => {
                    PyErr::new::<exceptions::PyValueError, _>(format!("Invalid parameter: {}", msg))
                }
                audio_samples::AudioSampleError::DimensionMismatch(msg) => {
                    PyErr::new::<exceptions::PyValueError, _>(format!("Dimension mismatch: {}", msg))
                }
                audio_samples::AudioSampleError::InvalidInput { msg } => {
                    PyErr::new::<exceptions::PyValueError, _>(format!("Invalid input: {}", msg))
                }
                audio_samples::AudioSampleError::ProcessingError { msg } => {
                    PyErr::new::<exceptions::PyRuntimeError, _>(format!("Processing error: {}", msg))
                }
                audio_samples::AudioSampleError::FeatureNotEnabled { feature } => {
                    PyErr::new::<exceptions::PyRuntimeError, _>(format!(
                        "Feature '{}' is not enabled in the wavers library",
                        feature
                    ))
                }
            }
        }
    }
}

/// Helper trait to convert WaversResult to PyResult
pub trait ToPyResult<T> {
    fn to_py_result(self) -> PyResult<T>;
}

impl<T> ToPyResult<T> for WaversResult<T> {
    fn to_py_result(self) -> PyResult<T> {
        self.map_err(map_wavers_error)
    }
}

/// Map generic std::io::Error to PyIOError
pub fn map_io_error(error: std::io::Error) -> PyErr {
    PyErr::new::<exceptions::PyIOError, _>(format!("I/O error: {}", error))
}

/// Create a PyValueError for invalid parameter values
pub fn invalid_parameter_error(param_name: &str, value: &str, expected: &str) -> PyErr {
    PyErr::new::<exceptions::PyValueError, _>(format!(
        "Invalid parameter '{}': got '{}', expected {}",
        param_name, value, expected
    ))
}

/// Create a PyTypeError for unsupported types
pub fn unsupported_type_error(type_name: &str, operation: &str) -> PyErr {
    PyErr::new::<exceptions::PyTypeError, _>(format!(
        "Unsupported type '{}' for operation '{}'",
        type_name, operation
    ))
}

/// Create a PyRuntimeError for feature not enabled
pub fn feature_not_enabled_error(feature: &str) -> PyErr {
    PyErr::new::<exceptions::PyRuntimeError, _>(format!(
        "Feature '{}' is not enabled. Please recompile wavers with the appropriate feature flag.",
        feature
    ))
}

/// Create a PyValueError for dimension mismatch
pub fn dimension_mismatch_error(expected: &str, actual: &str) -> PyErr {
    PyErr::new::<exceptions::PyValueError, _>(format!(
        "Dimension mismatch: expected {}, got {}",
        expected, actual
    ))
}

/// Create a PyValueError for invalid file format
pub fn invalid_file_format_error(filename: &str, reason: &str) -> PyErr {
    PyErr::new::<exceptions::PyValueError, _>(format!(
        "Invalid WAV file format in '{}': {}",
        filename, reason
    ))
}

/// Create a PyValueError for unsupported sample format
pub fn unsupported_sample_format_error(format: &str) -> PyErr {
    PyErr::new::<exceptions::PyValueError, _>(format!(
        "Unsupported sample format: {}. Supported formats: PCM 16/24/32-bit, IEEE Float 32/64-bit",
        format
    ))
}

/// Create a PyOverflowError for sample value overflow
pub fn sample_overflow_error(value: f64, target_type: &str) -> PyErr {
    PyErr::new::<exceptions::PyOverflowError, _>(format!(
        "Sample value {} cannot be represented in target type {}",
        value, target_type
    ))
}

/// Create a PyMemoryError for memory allocation failures
pub fn memory_allocation_error(size: usize) -> PyErr {
    PyErr::new::<exceptions::PyMemoryError, _>(format!(
        "Failed to allocate {} bytes of memory",
        size
    ))
}

/// Validate string parameter against allowed values
pub fn validate_string_param(param_name: &str, value: &str, allowed: &[&str]) -> PyResult<()> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(invalid_parameter_error(
            param_name,
            value,
            &format!("one of {:?}", allowed),
        ))
    }
}

/// Validate numeric parameter range
pub fn validate_numeric_range<T>(param_name: &str, value: T, min: T, max: T) -> PyResult<()>
where
    T: PartialOrd + std::fmt::Display,
{
    if value >= min && value <= max {
        Ok(())
    } else {
        Err(invalid_parameter_error(
            param_name,
            &value.to_string(),
            &format!("value between {} and {}", min, max),
        ))
    }
}

/// Validate that a value is positive
pub fn validate_positive<T>(param_name: &str, value: T) -> PyResult<()>
where
    T: PartialOrd + Default + std::fmt::Display,
{
    if value > T::default() {
        Ok(())
    } else {
        Err(invalid_parameter_error(
            param_name,
            &value.to_string(),
            "positive value",
        ))
    }
}

/// Validate that a value is non-negative
pub fn validate_non_negative<T>(param_name: &str, value: T) -> PyResult<()>
where
    T: PartialOrd + Default + std::fmt::Display,
{
    if value >= T::default() {
        Ok(())
    } else {
        Err(invalid_parameter_error(
            param_name,
            &value.to_string(),
            "non-negative value",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::*;

    #[test]
    fn test_error_mapping() {
        // Test various error types are mapped correctly
        let io_error = WaversErrorKind::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "File not found",
        ));
        let py_err = map_wavers_error(io_error);
        assert!(py_err.is_instance_of::<exceptions::PyIOError>());
        
        let parse_error = WaversErrorKind::Parse(ParseError::InvalidChunkSize { 
            chunk_id: crate::chunks::DATA,
            size: 0 
        });
        let py_err = map_wavers_error(parse_error);
        assert!(py_err.is_instance_of::<exceptions::PyValueError>());
    }

    #[test]
    fn test_parameter_validation() {
        // Test string parameter validation
        assert!(validate_string_param("method", "linear", &["linear", "exponential"]).is_ok());
        assert!(validate_string_param("method", "invalid", &["linear", "exponential"]).is_err());
        
        // Test numeric range validation
        assert!(validate_numeric_range("frequency", 1000.0, 0.0, 22050.0).is_ok());
        assert!(validate_numeric_range("frequency", -100.0, 0.0, 22050.0).is_err());
        
        // Test positive validation
        assert!(validate_positive("sample_rate", 44100u32).is_ok());
        assert!(validate_positive("sample_rate", 0u32).is_err());
    }
}