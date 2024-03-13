/// Module containing the error types for Wavers
use thiserror::Error;

use crate::FormatCode;

pub type WaversResult<T> = Result<T, WaversError>;

/// Error types for Wavers
#[derive(Error, Debug)]
pub enum WaversError {
    #[error("Invalid type specified: {0}")]
    InvalidType(String),
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
