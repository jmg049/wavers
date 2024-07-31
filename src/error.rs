//! Module containing the error types for Wavers
use thiserror::Error;

use crate::{FormatCode, WavType};

/// Result type for Wavers
pub type WaversResult<T> = Result<T, WaversError>;

/// Error types for Wavers
#[derive(Error, Debug)]
pub enum WaversError {
    /// Errors related to WAV format
    #[error("Format error: {0}")]
    Format(#[from] FormatError),

    /// IO error with file
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// UTF-8 conversion error
    #[error("UTF-8 conversion error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    /// Invalid seek operation
    #[error("Invalid seek operation: current position {current}, max data position {max}, attempted to read {attempted} bytes")]
    InvalidSeekOperation {
        current: u64,
        max: u64,
        attempted: u64,
    },

    /// NdArray error (when 'ndarray' feature is enabled)
    #[cfg(feature = "ndarray")]
    #[error("NdArray error: {0}")]
    NdArrayError(#[from] ndarray::ShapeError),
}

/// Errors specific to WAV format
#[derive(Error, Debug)]
pub enum FormatError {
    /// Invalid type specification
    #[error(
        "Invalid type: main format code {main}, bits per sample {bits}, sub-format code {sub}"
    )]
    InvalidType {
        main: FormatCode,
        bits: u16,
        sub: FormatCode,
    },

    /// Invalid FMT chunk size
    #[error("Invalid FMT chunk size: {0}")]
    InvalidFmtChunkSize(usize),

    /// Invalid number of bits for a Wav file to have for each sample.
    #[error("Invalid number of bits per sample: {0}")]
    InvalidBitsPerSample(u16),

    /// Invalid type ID given
    #[error("Invalid type ID '{0}'")]
    InvalidTypeId(&'static str),

    /// Invalid WAV type specified
    #[error("Invalid WAV type specified: {0}")]
    InvalidWavType(WavType),

    /// Unsupported write format
    #[error("Unsupported write format: main format {main}, sub-format {sub}")]
    UnsupportedWriteFormat { main: FormatCode, sub: FormatCode },
}
