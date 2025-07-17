//! Python bindings for the Wavers library.
//!
//! This module provides Python bindings for Wavers' core WAV file reading and writing
//! functionality, focusing on the AudioSamples-based API for seamless integration with
//! the audio_samples crate's Python bindings.
//!
//! ## Features
//!
//! - **AudioSamples Integration**: Direct integration with the audio_samples crate for
//!   powerful audio processing capabilities.
//! - **NumPy Support**: Efficient conversion between Rust and NumPy arrays.
//! - **Multiple Data Types**: Support for i16, i32, f32, and f64 audio formats.
//! - **High Performance**: Leverages Rust's performance for fast WAV file operations.
//!
//! ## Usage Examples
//!
//! ### Reading WAV Files
//!
//! ```python
//! import wavers
//! import numpy as np
//!
//! # Read as AudioSamples (recommended)
//! audio = wavers.read_wav("audio.wav")
//! print(f"Sample rate: {audio.sample_rate}")
//! print(f"Channels: {audio.channels}")
//! print(f"Duration: {audio.duration} seconds")
//!
//! # Read as NumPy array
//! data, sample_rate = wavers.read_wav_numpy("audio.wav")
//! print(f"Shape: {data.shape}")
//! print(f"Sample rate: {sample_rate} Hz")
//! print(f"Type: {type(data)})
//! ```
//!
//! ### Writing WAV Files
//!
//! ```python
//! import wavers
//! import numpy as np
//!
//! # Write AudioSamples
//! audio = wavers.read_wav("input.wav")
//! processed = audio.normalize()  # Use audio_samples processing
//! wavers.write_wav("output.wav", processed)
//!
//! # Write from NumPy array
//! sample_rate = 44100
//! duration = 1.0
//! t = np.linspace(0, duration, int(sample_rate * duration))
//! sine_wave = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
//! wavers.write_wav_numpy("sine.wav", sine_wave, sample_rate)
//! ```
//!
//! ### Getting WAV File Information
//!
//! ```python
//! import wavers
//!
//! info = wavers.wav_info("audio.wav")
//! print(f"Sample rate: {info.sample_rate} Hz")
//! print(f"Channels: {info.channels}")
//! print(f"Bits per sample: {info.bits_per_sample}")
//! print(f"Duration: {info.duration:.3f} seconds")
//! ```

use pyo3::prelude::*;

mod utils;
mod wav;

pub use wav::*;

/// Register the wavers Python module.
#[pymodule]
fn wavers(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Add the main functions
    m.add_function(wrap_pyfunction!(read_wav, m)?)?;
    m.add_function(wrap_pyfunction!(write_wav, m)?)?;
    m.add_function(wrap_pyfunction!(wav_info, m)?)?;

    // Add NumPy integration functions
    m.add_function(wrap_pyfunction!(read_wav_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(write_wav_numpy, m)?)?;

    // Add the WavInfo class
    m.add_class::<WavInfo>()?;

    // Add module metadata
    m.add("__version__", "0.1.0")?;
    m.add(
        "__doc__",
        "High-performance WAV file reading and writing for Python",
    )?;

    Ok(())
}
