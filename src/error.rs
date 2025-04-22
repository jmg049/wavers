//! error.rs – error types and helpers for **Wavers**
//!
//! This module organises every failure the crate can emit and gives you a
//! handful of helpers so downstream applications can *work* with those
//! errors instead of just printing them.
//!
//! # At a glance
//!
//! 1. **[`WaversErrorKind`]** – umbrella enum combining all sub‑domains.  
//! 2. **[`Domain`]** and **[`Code`]** – quick, classifications.  
//! 3. **[`WaversResult`]** – `core::result::Result` alias with the default error.  
//! 4. Tiny helper methods like [`WaversErrorKind::domain`] and [`WaversErrorKind::code`].

use thiserror::Error;
use crate::{chunks::ChunkID, wav_type::WavType, FormatCode};

/// Crate‑wide result alias.
pub type WaversResult<T, E = WaversErrorKind> = core::result::Result<T, E>;

/// Coarse error *domain* – handy for log filters, metrics, coloured output…
///
/// Marked **non‑exhaustive** so that adding new domains is *not* a breaking
/// change. Always include a wildcard branch when pattern‑matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Domain {
    Io,
    Parse,
    Validate,
    Convert,
    Operation,
    #[cfg(feature = "ndarray")]
    Sample,
    Internal,
}

/// Fine‑grained, stable error *code*.
///
/// Each concrete variant in the sub‑enums maps to exactly **one** code so
/// callers can switch on it without depending on the variant names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Code {
    Io,

    ParseNoChunks,
    ParseChunkNotFound,
    ParseNom,

    ValidateHeader,
    ValidateUnsupportedFormat,
    ValidateFormatBlock,
    ValidateBitDepth,
    ValidateMalformedChunk,
    ValidateSpecViolation,

    ConvertUnsupported,
    ConvertOverflow,
    ConvertPrecision,

    EmptyChunks,
    BadIndex,
    IllegalState,
    InvalidOp,
    #[cfg(feature = "ndarray")]
    SampleError,

    InternalBug,
}

/// Top‑level umbrella error.
#[derive(Error, Debug)]
pub enum WaversErrorKind {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Parse(#[from] ParseError),

    #[error(transparent)]
    Validate(#[from] ValidationError),

    #[error(transparent)]
    Convert(#[from] ConversionError),

    #[error(transparent)]
    Operation(#[from] OperationError),

    #[cfg(feature = "ndarray")]
    #[error("Audio Sample Error: {0}")]
    SampleError(#[from] audio_sample::AudioSampleError),

    #[error(transparent)]
    Internal(#[from] InternalError),
}

impl WaversErrorKind {
    /// Returns a coarse domain classification.
    #[inline]
    pub fn domain(&self) -> Domain {
        use WaversErrorKind::*;
        match self {
            Io(_) => Domain::Io,
            Parse(_) => Domain::Parse,
            Validate(_) => Domain::Validate,
            Convert(_) => Domain::Convert,
            Operation(_) => Domain::Operation,  
            #[cfg(feature = "ndarray")]
            SampleError(_) => Domain::Sample,
            Internal(_) => Domain::Internal,
        }
    }

    /// Stable discriminant suitable for log aggregation or FFI.
    #[inline]
    pub fn code(&self) -> Code {
        use WaversErrorKind::*;
        match self {
            Io(_) => Code::Io,
            Parse(e) => e.code(),
            Validate(e) => e.code(),
            Convert(e) => e.code(),
            Operation(e) => e.code(),
            #[cfg(feature = "ndarray")]
            SampleError(e) =>Code::SampleError,
            Internal(_) => Code::InternalBug,
        }
    }

    /// Returns `true` if the error is an **internal** logic bug.
    #[inline]
    pub fn is_internal(&self) -> bool {
        matches!(self, WaversErrorKind::Internal(_))
    }
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("no RIFF/RIFF64 chunks found in stream")]
    NoChunks,

    #[error("chunk {id:?} not found")]
    ChunkNotFound { id: ChunkID },

    #[error("nom parser failed at byte {offset}: {msg}")]
    Nom { offset: usize, msg: String },
}

impl ParseError {
    #[inline]
    pub fn code(&self) -> Code {
        use ParseError::*;
        match self {
            NoChunks => Code::ParseNoChunks,
            ChunkNotFound { .. } => Code::ParseChunkNotFound,
            Nom { .. } => Code::ParseNom,
        }
    }
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("invalid RIFF header: {0}")]
    Header(String),

    #[error("unsupported format code: {0:?}")]
    UnsupportedFormat(FormatCode),

    #[error("inconsistent format block – {what}")]
    FormatBlock { what: String },

    #[error("bit‑depth {bits} invalid for {format:?}")]
    BitDepth { bits: u16, format: FormatCode },

    #[error("unsupported chunk {id:?}")]
    UnsupportedChunk { id: ChunkID },

    #[error("Chunk error for {chunk_id:?}: {reason}")]
    Chunk { chunk_id: ChunkID, reason: String },

    #[error("WAV does not meet spec: {reason}")]
    InvalidWavFile { reason: String },

    #[error("Invalid target sample type {sample_type:?} ")]
    InvalidTargetSampleType { sample_type: String },
}

impl ValidationError {
    #[inline]
    pub fn code(&self) -> Code {
        use ValidationError::*;
        match self {
            Header(_) => Code::ValidateHeader,
            UnsupportedFormat(_) => Code::ValidateUnsupportedFormat,
            FormatBlock { .. } => Code::ValidateFormatBlock,
            BitDepth { .. } => Code::ValidateBitDepth,
            UnsupportedChunk { .. } => Code::ValidateMalformedChunk,
            InvalidWavFile { .. } => Code::ValidateSpecViolation,
            InvalidTargetSampleType { .. } => Code::ValidateSpecViolation,
            Chunk { .. } => Code::ValidateMalformedChunk,
        }
    }
}

#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("cannot convert from {src:?} to {dst}")]
    Unsupported { src: WavType, dst: String },

    #[error("lossy conversion would overflow (sample index {idx})")]
    Overflow { idx: usize },

    #[error("conversion precision error: {details}")]
    Precision { details: String },
}

impl ConversionError {
    #[inline]
    pub fn code(&self) -> Code {
        use ConversionError::*;
        match self {
            Unsupported { .. } => Code::ConvertUnsupported,
            Overflow { .. } => Code::ConvertOverflow,
            Precision { .. } => Code::ConvertPrecision,
        }
    }
}

#[derive(Error, Debug)]
pub enum OperationError {
    #[error("attempted {op} on an empty chunk list")]
    EmptyChunks { op: &'static str },

    #[error("chunk index {idx} out of bounds (total {total})")]
    BadIndex { idx: usize, total: usize },

    #[error("operation invalid: {reason}")]
    InvalidOp { reason: String },
}

impl OperationError {
    #[inline]
    pub fn code(&self) -> Code {
        use OperationError::*;
        match self {
            EmptyChunks { .. } => Code::EmptyChunks,
            BadIndex { .. } => Code::BadIndex,
            InvalidOp { .. } => Code::InvalidOp,
        }
    }
}

#[derive(Error, Debug)]
pub enum InternalError {
    #[error("logic bug: {0} — please file an issue")]
    Bug(String),
}

impl InternalError {
    #[inline]
    pub fn code(&self) -> Code {
        Code::InternalBug
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_and_code_roundtrip() {
        let err = WaversErrorKind::Parse(ParseError::NoChunks);
        assert_eq!(err.domain(), Domain::Parse);
        assert_eq!(err.code(), Code::ParseNoChunks);
    }
}
