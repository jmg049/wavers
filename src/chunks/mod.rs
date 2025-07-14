//! Chunk parsing and metadata helpers for RIFF/WAV formats.
//!
//! This module defines the fundamental building blocks for working with WAV files,
//! including chunk identifiers, descriptors, and a container for managing chunk metadata.
//!
//! WAV files (a form of RIFF files) are composed of ordered "chunks", each prefixed
//! with an identifier and size field. This module provides:
//!
//! - Constants for known and supported chunk identifiers ([`RIFF`], [`FMT`], [`DATA`], etc).
//! - The [`Chunk`] and [`ChunkInfo`] structures for in-memory chunk access.
//! - The [`ChunkManager`] which tracks metadata for all discovered chunks.
//! - [`validate_tag`] and related helpers to ensure correctness when parsing.

use crate::{
    WaversErrorKind, WaversResult,
    error::{OperationError, ValidationError},
};
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    ops::Deref,
};

pub(crate) mod fmt;
pub mod list;
pub(crate) mod riff;

pub(crate) use riff::RIFFChunk;
pub use list::{ListChunk, SubChunk, WavMetadata};

/// Standard RIFF chunk identifier
pub const RIFF: ChunkID = ChunkID::new(*b"RIFF");
/// WAV format identifier
pub const WAVE: ChunkID = ChunkID::new(*b"WAVE");
/// Format chunk identifier
pub const FMT: ChunkID = ChunkID::new(*b"fmt ");
/// Data chunk identifier
pub const DATA: ChunkID = ChunkID::new(*b"data");
/// List chunk identifier
pub const LIST: ChunkID = ChunkID::new(*b"LIST");
/// Fact chunk identifier
pub const FACT: ChunkID = ChunkID::new(*b"fact");
/// Info chunk identifier
pub const INFO: ChunkID = ChunkID::new(*b"INFO");
/// Peak chunk identifier
pub const PEAK: ChunkID = ChunkID::new(*b"PEAK");

/// Returns `Ok(())` if the tag is one of the known WAV/RIFF chunk types.
pub const fn validate_tag(tag: ChunkID) -> WaversResult<()> {
    if chunk_eq(tag.id, RIFF.id) {
        return Ok(());
    }

    if chunk_eq(tag.id, WAVE.id) {
        return Ok(());
    }

    if chunk_eq(tag.id, FMT.id) {
        return Ok(());
    }

    if chunk_eq(tag.id, DATA.id) {
        return Ok(());
    }

    if chunk_eq(tag.id, LIST.id) {
        return Ok(());
    }

    if chunk_eq(tag.id, FACT.id) {
        return Ok(());
    }

    if chunk_eq(tag.id, INFO.id) {
        return Ok(());
    }

    if chunk_eq(tag.id, PEAK.id) {
        return Ok(());
    }

    Err(WaversErrorKind::Validate(
        ValidationError::UnsupportedChunk { id: tag },
    ))
}

/// A WAV file chunk, consisting of an identifier, size, and data
///
/// Chunks are the fundamental building blocks of RIFF/WAV files.
/// Each chunk contains a 4-byte identifier, a 4-byte size field,
/// and the actual chunk data.
#[derive(Debug)]
pub struct Chunk<'a> {
    /// The 4-byte identifier of the chunk
    pub id: &'static ChunkID,
    /// The actual chunk data
    pub data: &'a [u8],
}

impl<'a> Chunk<'a> {
    /// Creates a new chunk with the given identifier, size, and data
    ///
    /// # Arguments
    ///
    /// * `id` - The chunk identifier
    /// * `data` - The chunk data
    ///
    /// # Returns
    ///
    /// A new Chunk instance
    pub const fn new(id: &'static ChunkID, data: &'a [u8]) -> Self {
        Self { id, data }
    }

    /// Gets the total size of the chunk including the 8-byte header
    ///
    /// # Returns
    ///
    /// The total size in bytes
    pub const fn total_size(&self) -> u32 {
        // 4 bytes for ID + 4 bytes for size + actual data size
        8 + self.data.len() as u32
    }

    /// Checks if this chunk has a specific ID
    ///
    /// # Arguments
    ///
    /// * `id` - The chunk ID to check against
    ///
    /// # Returns
    ///
    /// `true` if the chunk has the specified ID, otherwise `false`
    pub fn has_id(&self, id: &ChunkID) -> bool {
        self.id == id
    }

    pub fn size(&self) -> u32 {
        self.data.len() as u32
    }
}

impl<'a> Display for Chunk<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Chunk {{ id: {}, size: {} bytes }}",
            self.id,
            self.data.len()
        )
    }
}

/// A 4-byte chunk identifier
///
/// Chunk identifiers in RIFF/WAV files are 4-byte ASCII codes that
/// identify the type of chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkID {
    /// The 4-byte identifier
    pub id: [u8; 4],
}

pub const fn chunk_eq(a: [u8; 4], b: [u8; 4]) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
}

impl Deref for ChunkID {
    type Target = [u8; 4];
    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

impl AsRef<[u8]> for ChunkID {
    fn as_ref(&self) -> &[u8] {
        &self.id
    }
}

impl ChunkID {
    /// Creates a new chunk identifier
    ///
    /// # Arguments
    ///
    /// * `id` - The 4-byte identifier
    ///
    /// # Returns
    ///
    /// A new ChunkID instance
    pub const fn new(id: [u8; 4]) -> Self {
        Self { id }
    }

    /// Returns the identifier as a string, if it's valid ASCII
    ///
    /// # Returns
    ///
    /// The identifier as a string
    pub fn as_str(&self) -> String {
        String::from_utf8_lossy(&self.id).to_string()
    }

    /// Checks if this identifier matches a specific 4-byte sequence
    ///
    /// # Arguments
    ///
    /// * `id` - The 4-byte sequence to check against
    ///
    /// # Returns
    ///
    /// `true` if the identifier matches, otherwise `false`
    pub fn matches(&self, id: &[u8; 4]) -> bool {
        self.id == *id
    }
}

impl Display for ChunkID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.id))
    }
}

/// Information about a chunk's location and size in a file
///
/// This structure holds metadata about a chunk without containing
/// the actual chunk data, useful for mapping a WAV file structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkInfo {
    pub id: ChunkID,
    /// The offset of the chunk in the file (in bytes)
    pub offset: usize,
    /// The size of the chunk data (in bytes)
    pub size: u32,
}

impl ChunkInfo {
    /// Creates a new chunk info with the given offset and size
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset of the chunk in the file
    /// * `size` - The size of the chunk data
    ///
    /// # Returns
    ///
    /// A new ChunkInfo instance
    pub const fn new(id: ChunkID, offset: usize, size: u32) -> Self {
        Self { id, offset, size }
    }

    /// Gets the end position of the chunk in the file
    ///
    /// # Returns
    ///
    /// The end position (offset + 8 + size)
    pub fn end_position(&self) -> usize {
        self.offset + 8 + self.size as usize
    }

    /// Checks if this chunk overlaps with another chunk
    ///
    /// # Arguments
    ///
    /// * `other` - The other chunk info to check against
    ///
    /// # Returns
    ///
    /// `true` if the chunks overlap, otherwise `false`
    pub fn overlaps_with(&self, other: &ChunkInfo) -> bool {
        let self_start = self.offset;
        let self_end = self.end_position();
        let other_start = other.offset;
        let other_end = other.end_position();

        // Check if either chunk's range overlaps with the other
        (self_start <= other_start && other_start < self_end)
            || (other_start <= self_start && self_start < other_end)
    }
}

impl Display for ChunkInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChunkInfo {{ offset: {}, size: {} bytes }}",
            self.offset, self.size
        )
    }
}

impl From<(ChunkID, usize, u32)> for ChunkInfo {
    fn from((id, offset, size): (ChunkID, usize, u32)) -> Self {
        Self::new(id, offset, size)
    }
}

/// Helper function to check if a chunk is valid
///
/// This validates that a chunk's data size matches the declared size.
///
/// # Arguments
///
/// * `chunk` - The chunk to validate
///
/// # Returns
///
/// `true` if the chunk is valid, otherwise `false`
pub fn is_valid_chunk(chunk: &Chunk) -> bool {
    chunk.size() as usize == chunk.data.len()
}

/// Helper function to align a chunk size to an even number of bytes
///
/// RIFF/WAV chunks should be aligned to even byte boundaries.
/// If a chunk has an odd size, it's padded with a single byte.
///
/// # Arguments
///
/// * `size` - The size to align
///
/// # Returns
///
/// The aligned size
pub fn align_chunk_size(size: u32) -> u32 {
    // If size is odd, add a padding byte
    if size % 2 == 1 { size + 1 } else { size }
}

/// Manager for WAV file chunks
///
/// The ChunkManager provides a container and utilities for working with
/// a collection of chunks in a WAV file.
#[derive(Debug)]
pub struct ChunkManager {
    /// The collection of chunks in the WAV file
    chunks: Vec<ChunkInfo>,
    /// Parsed LIST chunks indexed by their offset
    list_chunks: HashMap<usize, list::ListChunk>,
}

impl ChunkManager {
    /// Creates a new empty ChunkManager
    ///
    /// # Returns
    ///
    /// A new ChunkManager instance with no chunks
    pub fn new() -> Self {
        Self {
            chunks: Vec::with_capacity(4), // Pre-allocate for a small set of wav chunks, RIFF,
                                           // FMT, DATA, EXTRA
            list_chunks: HashMap::new(),
        }
    }

    /// Creates a new ChunkManager with a pre-populated list of chunks
    ///
    /// # Arguments
    ///
    /// * `chunks` - A vector of chunks to initialize the manager with
    ///
    /// # Returns
    ///
    /// A new ChunkManager containing the provided chunks
    pub fn with_chunks(chunks: Vec<ChunkInfo>) -> Self {
        Self { 
            chunks,
            list_chunks: HashMap::new(),
        }
    }

    /// Adds a chunk to the manager
    ///
    /// # Arguments
    ///
    /// * `chunk` - The chunk to add
    pub fn add_chunk_info(&mut self, chunk_info: ChunkInfo) {
        self.chunks.push(chunk_info);
    }

    pub fn get_chunk_info(&self, chunk_id: &ChunkID) -> Option<&ChunkInfo> {
        self.chunks.iter().find(|info| info.id == *chunk_id)
    }

    pub fn contains_id(&self, chunk_id: &ChunkID) -> bool {
        self.get_chunk_info(chunk_id).is_some()
    }

    /// Gets the total number of chunks
    ///
    /// # Returns
    ///
    /// The number of chunks in the manager
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Checks if the manager is empty
    ///
    /// # Returns
    ///
    /// `true` if there are no chunks, otherwise `false`
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Gets an iterator over all chunks
    ///
    /// # Returns
    ///
    /// An iterator yielding references to all chunks
    pub fn iter(&self) -> impl Iterator<Item = &ChunkInfo> {
        self.chunks.iter()
    }

    /// Gets a mutable iterator over all chunks
    ///
    /// # Returns
    ///
    /// An iterator yielding mutable references to all chunks
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ChunkInfo> {
        self.chunks.iter_mut()
    }

    /// Gets a reference to the underlying chunk vector
    ///
    /// # Returns
    ///
    /// A reference to the vector of chunks
    pub fn as_slice(&self) -> &[ChunkInfo] {
        &self.chunks
    }

    /// Clears all chunks from the manager
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.list_chunks.clear();
    }

    /// Adds a parsed LIST chunk to the manager
    ///
    /// # Arguments
    /// * `offset` - The offset of the LIST chunk in the file
    /// * `list_chunk` - The parsed LIST chunk
    pub fn add_list_chunk(&mut self, offset: usize, list_chunk: list::ListChunk) {
        self.list_chunks.insert(offset, list_chunk);
    }

    /// Gets a LIST chunk by its offset
    ///
    /// # Arguments
    /// * `offset` - The offset of the LIST chunk in the file
    ///
    /// # Returns
    /// A reference to the LIST chunk if found
    pub fn get_list_chunk(&self, offset: usize) -> Option<&list::ListChunk> {
        self.list_chunks.get(&offset)
    }

    /// Gets all LIST chunks
    ///
    /// # Returns
    /// An iterator over all LIST chunks with their offsets
    pub fn list_chunks(&self) -> impl Iterator<Item = (usize, &list::ListChunk)> {
        self.list_chunks.iter().map(|(k, v)| (*k, v))
    }

    /// Gets all INFO LIST chunks
    ///
    /// # Returns
    /// An iterator over INFO LIST chunks with their offsets
    pub fn info_chunks(&self) -> impl Iterator<Item = (usize, &list::ListChunk)> {
        self.list_chunks.iter()
            .filter(|(_, chunk)| chunk.is_info())
            .map(|(k, v)| (*k, v))
    }

    /// Extracts metadata from all INFO LIST chunks
    ///
    /// # Returns
    /// A merged metadata structure from all INFO chunks
    pub fn extract_metadata(&self) -> WaversResult<list::WavMetadata> {
        let mut combined_metadata = list::WavMetadata::default();
        
        for (_, list_chunk) in self.info_chunks() {
            let metadata = list_chunk.extract_metadata()?;
            
            // Merge metadata, preferring non-None values
            if combined_metadata.artist.is_none() { combined_metadata.artist = metadata.artist; }
            if combined_metadata.title.is_none() { combined_metadata.title = metadata.title; }
            if combined_metadata.copyright.is_none() { combined_metadata.copyright = metadata.copyright; }
            if combined_metadata.comment.is_none() { combined_metadata.comment = metadata.comment; }
            if combined_metadata.creation_date.is_none() { combined_metadata.creation_date = metadata.creation_date; }
            if combined_metadata.engineer.is_none() { combined_metadata.engineer = metadata.engineer; }
            if combined_metadata.genre.is_none() { combined_metadata.genre = metadata.genre; }
            if combined_metadata.keywords.is_none() { combined_metadata.keywords = metadata.keywords; }
            if combined_metadata.medium.is_none() { combined_metadata.medium = metadata.medium; }
            if combined_metadata.product.is_none() { combined_metadata.product = metadata.product; }
            if combined_metadata.subject.is_none() { combined_metadata.subject = metadata.subject; }
            if combined_metadata.software.is_none() { combined_metadata.software = metadata.software; }
            if combined_metadata.source.is_none() { combined_metadata.source = metadata.source; }
            if combined_metadata.source_form.is_none() { combined_metadata.source_form = metadata.source_form; }
            if combined_metadata.technician.is_none() { combined_metadata.technician = metadata.technician; }
        }
        
        Ok(combined_metadata)
    }

    pub fn get_start(&self) -> WaversResult<usize> {
        if self.is_empty() {
            return Err(WaversErrorKind::Operation(OperationError::InvalidOp {
                reason: "Called get_start on empty chunk manager.".to_string(),
            }));
        }
        Ok(self.get_chunk_info(&RIFF).unwrap().offset)
    }

    pub fn get_end(&self) -> WaversResult<usize> {
        if self.is_empty() {
            return Err(WaversErrorKind::Operation(OperationError::InvalidOp {
                reason: "Called get_end on empty chunk manager.".to_string(),
            }));
        }
        Ok(self
            .chunks
            .iter()
            .max_by(|info_a, info_b| info_a.offset.cmp(&info_b.offset))
            .unwrap()
            .offset)
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for ChunkManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ChunkManager containing {} chunks:", self.chunks.len())?;
        for (idx, chunk) in self.chunks.iter().enumerate() {
            writeln!(f, "  [{}]: {}", idx, chunk)?;
        }
        Ok(())
    }
}

impl IntoIterator for ChunkManager {
    type Item = ChunkInfo;
    type IntoIter = std::vec::IntoIter<ChunkInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.chunks.into_iter()
    }
}

impl<'a> IntoIterator for &'a ChunkManager {
    type Item = &'a ChunkInfo;
    type IntoIter = std::slice::Iter<'a, ChunkInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.chunks.iter()
    }
}

impl<'a> IntoIterator for &'a mut ChunkManager {
    type Item = &'a mut ChunkInfo;
    type IntoIter = std::slice::IterMut<'a, ChunkInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.chunks.iter_mut()
    }
}
