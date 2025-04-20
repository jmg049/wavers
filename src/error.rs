use thiserror::Error;

use crate::{chunks::ChunkID, wav_type::WavType, FormatCode};

pub type WaversResult<T> = std::result::Result<T, WaversErrorKind>;

#[derive(Error, Debug)]
pub enum WaversErrorKind {
    #[error("IO error: {0}")]
    IOErrorKind(#[from] std::io::Error),
    #[error("No chunks found")]
    NoChunksErrorKind,
    #[error("Conversion error: {0}")]
    ConversionErrorKind(String),
    #[error("Chunk {0} not found")]
    ChunkNotFound(ChunkID),
    #[error("Nom parsing error {0}")]
    NomErrorKind(String),
    #[error("Extended FMT Error: {0}")]
    ExtFmtErrorKind(String),
    #[error("Invalid chunk data: {0}")]
    InvalidChunkData(String),
    #[error("Invalid chunk type")]
    InvalidChunkType,
    #[error("Invalid chunk index passed to chunk manager: {0}")]
    InvalidChunkIndex(usize),
    #[error("Invalid header: {0}")]
    InvalidHeader(String),
    #[error("Invalid format info: {0} {1} {2}")]
    InvalidFormatInfo(FormatCode, usize, FormatCode),
    #[error("Invalid bit size: {0}")]
    InvalidBitSize(u16),
    #[error("Invalid target type: native type = {0}, target type = {1}")]
    InvalidTargetType(WavType, String),
    #[error("Invalid WAV file. Reason: {0}")]
    InvalidWavFile(String),
    #[error("Invalid operation: {0}")]
    InvalidOp(String),
}
