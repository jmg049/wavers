/// Module containing the error types for Wavers
use thiserror::Error;

pub type WaversResult<T> = Result<T, WaversError>;

/// Error types for Wavers
#[derive(Error, Debug)]
pub enum WaversError {
    #[error("Invalid type specified: {0}")]
    InvalidType(String),
    #[error("IO error with file")]
    IOError(#[from] std::io::Error),
}
