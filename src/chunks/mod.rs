//! This module contains the ``Chunk`` trait and the constants relating to the different chunks in a wav file.

pub mod fact;
pub mod fmt;
pub mod list;

use std::fmt::Display;

pub use crate::chunks::fact::FactChunk;
pub use crate::chunks::fmt::FmtChunk;
pub use crate::chunks::list::ListChunk;
use crate::{header::HeaderChunkInfo, ReadSeek, WaversResult};

// 100% necessary to have these chunks

///The RIFF chunk ID "RIFF"
pub const RIFF: [u8; 4] = *b"RIFF";
/// The WAVE chunk ID "WAVE"
pub const WAVE: [u8; 4] = *b"WAVE";
/// The data chunk ID "data"
pub const DATA: [u8; 4] = *b"data";
/// The fmt chunk ID "fmt "
pub const FMT: [u8; 4] = *b"fmt ";

// Optional chunks
/// The fact chunk ID "fact"
pub const LIST: [u8; 4] = *b"LIST";
/// The fact chunk ID "fact"
pub const FACT: [u8; 4] = *b"fact";

/// A trait representing a chunk in a wav file.
/// Allows for the common creation of several chunks that are found in a wav file.
/// Requires them to implement common decoding and encoding methods.
pub trait Chunk: Display {
    fn id(&self) -> &[u8; 4];
    fn size(&self) -> u32;
    fn as_bytes(&self) -> Box<[u8]>;
    fn from_reader(reader: &mut Box<dyn ReadSeek>, info: &HeaderChunkInfo) -> WaversResult<Self>
    where
        Self: Sized;
}

/// Read a chunk from a wav file.
pub fn read_chunk<T: Chunk>(
    reader: &mut Box<dyn ReadSeek>,
    info: &HeaderChunkInfo,
) -> WaversResult<T> {
    T::from_reader(reader, info)
}
