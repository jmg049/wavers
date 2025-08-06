//! Python bindings for the wavers library.
//!
//! This module provides a comprehensive Python interface to the Rust wavers library,
//! offering seamless access to WAV file I/O, audio processing, and analysis capabilities.
//!
//! ## Key Design Features
//!
//! - **Seamless Integration**: Built on top of the audio_samples crate for rich audio processing
//! - **Type Safety**: Rust's type system ensures safe operations across the Python boundary
//! - **Performance**: Efficient zero-copy operations where possible
//! - **Pythonic API**: Intuitive method names and optional parameters with sensible defaults
//! - **Error Handling**: Comprehensive error mapping from Rust to Python exceptions
//!
//! ## Module Structure
//!
//! - **io**: Core I/O functions for reading and writing WAV files
//! - **wav**: PyWav class providing the main interface for WAV file manipulation
//! - **processing**: Audio processing methods (filters, normalization, etc.)
//! - **analysis**: Audio analysis methods (RMS, spectral analysis, etc.)
//! - **editing**: Audio editing operations (trim, pad, concatenate, etc.)
//! - **conversions**: Type conversion utilities between Rust and Python
//! - **utils**: Utility functions for NumPy integration and error handling
//! - **errors**: Error handling and mapping from Rust to Python exceptions

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// Import submodules
mod errors;
mod utils;
mod conversions;
mod io;
mod wav;
mod processing;
mod analysis;
mod editing;

// Re-export key components
pub use errors::*;
pub use utils::*;
pub use conversions::*;

/// Python module for wavers - High-performance WAV file processing
///
/// This module provides comprehensive WAV file I/O, audio processing, and analysis
/// capabilities built on top of the fast Rust wavers library.
///
/// # Examples
/// ```python
/// import wavers
/// 
/// # Read a WAV file
/// audio = wavers.read_wav("audio.wav")
/// print(f"Sample rate: {audio.sample_rate}")
/// print(f"Duration: {audio.duration:.2f} seconds")
/// 
/// # Process audio
/// audio_normalized = audio.normalize()
/// audio_filtered = audio.low_pass_filter(1000.0)
/// 
/// # Write processed audio
/// wavers.write_wav("output.wav", audio_filtered)
/// ```
#[pymodule]
fn wavers(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the main PyWav class
    m.add_class::<wav::PyWav>()?;
    
    // Register I/O functions
    m.add_function(wrap_pyfunction!(io::read_wav, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_wav_raw, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_wav_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_wav, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_wav_raw, m)?)?;
    m.add_function(wrap_pyfunction!(io::get_wav_info, m)?)?;
    m.add_function(wrap_pyfunction!(io::get_wav_metadata, m)?)?;
    
    // Register utility functions
    m.add_function(wrap_pyfunction!(utils::supported_sample_types, m)?)?;
    m.add_function(wrap_pyfunction!(utils::library_version, m)?)?;
    
    // Add module constants
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SUPPORTED_SAMPLE_TYPES", vec!["i16", "i32", "f32", "f64"])?;
    m.add("SUPPORTED_FORMATS", vec!["PCM", "IEEE_FLOAT"])?;
    
    // Add module documentation
    m.add(
        "__doc__",
        "High-performance WAV file processing library with comprehensive audio analysis capabilities",
    )?;
    
    Ok(())
}

/// Register the wavers module with Python
pub fn register_module(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let wavers_module = PyModule::new(py, "wavers")?;
    wavers(py, &wavers_module)?;
    parent_module.add_submodule(&wavers_module)?;
    Ok(())
}