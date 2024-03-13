pub mod fact;
pub mod fmt;
pub mod list;

use std::fmt::Display;

pub use crate::chunks::fact::FactChunk;
pub use crate::chunks::fmt::FmtChunk;
pub use crate::chunks::list::ListChunk;
use crate::{header::HeaderChunkInfo, ReadSeek, WaversResult};

// 100% necessary to have these chunks
pub const RIFF: [u8; 4] = *b"RIFF";
pub const WAVE: [u8; 4] = *b"WAVE";
pub const DATA: [u8; 4] = *b"data";
pub const FMT: [u8; 4] = *b"fmt ";

// Optional chunks
pub const LIST: [u8; 4] = *b"LIST";
pub const FACT: [u8; 4] = *b"fact";

#[allow(unused)]
pub const AVAILABLE_CHUNKS: [[u8; 4]; 5] = [RIFF, WAVE, FMT, LIST, FACT];

pub trait Chunk: Display {
    fn id(&self) -> &[u8; 4];
    fn size(&self) -> u32;
    fn as_bytes(&self) -> Box<[u8]>;
    fn from_reader(reader: &mut Box<dyn ReadSeek>, info: &HeaderChunkInfo) -> WaversResult<Self>
    where
        Self: Sized;
}

pub fn read_chunk<T: Chunk>(
    reader: &mut Box<dyn ReadSeek>,
    info: &HeaderChunkInfo,
) -> WaversResult<T> {
    T::from_reader(reader, info)
}
