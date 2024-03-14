/// Module containing the error types for Wavers
use thiserror::Error;

use crate::{FormatCode, WavType};

pub type WaversResult<T> = Result<T, WaversError>;

/// Error types for Wavers
#[derive(Error, Debug)]
pub enum WaversError {
    #[error(
        "Invalid type specified: Main Format Code: {0} | Bits per sample {1} | Sub-Format Code {1}"
    )]
    InvalidType(FormatCode, u16, FormatCode),
    #[error("Invalid type specified: {0}")]
    InvalidWavType(WavType),
    #[error("Invalid Fmt Chunk Size: {0}")]
    InvalidFmtChunkSize(usize),
    #[error("Invalid number of bits to convert to WavType: {0}")]
    InvalidBitsPerSample(u16),
    #[error("Invalid type id given: {0}")]
    InvalidTypeId(String),
    #[error("IO error with file{0}")]
    IOError(#[from] std::io::Error),
    #[error("FromUTF8 Error {0}")]
    UTF8Error(#[from] std::str::Utf8Error),
    #[error("Unsupported write format ({0}, {1})")]
    UnsupportedWriteFormat(FormatCode, FormatCode),
    #[cfg(feature = "ndarray")]
    #[error("IO error with ndarray")]
    NdArrayError(#[from] ndarray::ShapeError),
}
