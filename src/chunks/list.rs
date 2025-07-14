//! Implementation of LIST chunk parsing and management.
//!
//! LIST chunks contain sub-chunks organized by type. The most common use is for INFO chunks
//! which contain metadata about the audio file (artist, title, copyright, etc.).
//!
//! ## Structure
//!
//! A LIST chunk has the format:
//! ```text
//! "LIST" (4 bytes) - Chunk ID
//! size   (4 bytes) - Chunk size (excluding the 8-byte header)
//! type   (4 bytes) - List type (e.g., "INFO")
//! data   (variable) - Sub-chunks specific to the list type
//! ```
//!
//! ## Supported List Types
//!
//! - **INFO**: Metadata chunks (IART, INAM, ICOP, etc.)
//! - **adtl**: Associated data list (labels, notes, etc.)
//! - **wavl**: Wave list (for complex wave structures)

use crate::{
    chunks::ChunkID,
    error::{ParseError, ValidationError},
    WaversErrorKind, WaversResult,
};
use std::fmt::{Debug, Display};
use std::io::{Read, Seek};

/// LIST chunk type identifiers
pub const LIST_INFO: ChunkID = ChunkID::new(*b"INFO");
pub const LIST_ADTL: ChunkID = ChunkID::new(*b"adtl");
pub const LIST_WAVL: ChunkID = ChunkID::new(*b"wavl");

/// INFO chunk sub-types for metadata
pub const INFO_IART: ChunkID = ChunkID::new(*b"IART"); // Artist
pub const INFO_INAM: ChunkID = ChunkID::new(*b"INAM"); // Name/Title
pub const INFO_ICOP: ChunkID = ChunkID::new(*b"ICOP"); // Copyright
pub const INFO_ICMT: ChunkID = ChunkID::new(*b"ICMT"); // Comment
pub const INFO_ICRD: ChunkID = ChunkID::new(*b"ICRD"); // Creation date
pub const INFO_IENG: ChunkID = ChunkID::new(*b"IENG"); // Engineer
pub const INFO_IGNR: ChunkID = ChunkID::new(*b"IGNR"); // Genre
pub const INFO_IKEY: ChunkID = ChunkID::new(*b"IKEY"); // Keywords
pub const INFO_IMED: ChunkID = ChunkID::new(*b"IMED"); // Medium
pub const INFO_IPRD: ChunkID = ChunkID::new(*b"IPRD"); // Product/Album
pub const INFO_ISBJ: ChunkID = ChunkID::new(*b"ISBJ"); // Subject
pub const INFO_ISFT: ChunkID = ChunkID::new(*b"ISFT"); // Software
pub const INFO_ISRC: ChunkID = ChunkID::new(*b"ISRC"); // Source
pub const INFO_ISRF: ChunkID = ChunkID::new(*b"ISRF"); // Source form
pub const INFO_ITCH: ChunkID = ChunkID::new(*b"ITCH"); // Technician

/// Represents a parsed LIST chunk with its sub-chunks
#[derive(Debug, Clone)]
pub struct ListChunk {
    /// The type of LIST chunk (INFO, adtl, wavl, etc.)
    pub list_type: ChunkID,
    /// Sub-chunks contained within this LIST chunk
    pub sub_chunks: Vec<SubChunk>,
    /// Raw data for unrecognized sub-chunks
    pub unknown_data: Vec<u8>,
}

/// Represents a sub-chunk within a LIST chunk
#[derive(Debug, Clone)]
pub struct SubChunk {
    /// 4-byte identifier of the sub-chunk
    pub id: ChunkID,
    /// Size of the sub-chunk data
    pub size: u32,
    /// Raw data content of the sub-chunk
    pub data: Vec<u8>,
}

/// Metadata extracted from INFO LIST chunks
#[derive(Debug, Clone, Default)]
pub struct WavMetadata {
    /// Artist name
    pub artist: Option<String>,
    /// Track/song title
    pub title: Option<String>,
    /// Copyright information
    pub copyright: Option<String>,
    /// Comment/description
    pub comment: Option<String>,
    /// Creation date
    pub creation_date: Option<String>,
    /// Engineer name
    pub engineer: Option<String>,
    /// Genre
    pub genre: Option<String>,
    /// Keywords
    pub keywords: Option<String>,
    /// Medium (CD, Tape, etc.)
    pub medium: Option<String>,
    /// Product/Album name
    pub product: Option<String>,
    /// Subject
    pub subject: Option<String>,
    /// Software used for creation
    pub software: Option<String>,
    /// Source
    pub source: Option<String>,
    /// Source form
    pub source_form: Option<String>,
    /// Technician name
    pub technician: Option<String>,
}

impl SubChunk {
    /// Creates a new SubChunk
    pub fn new(id: ChunkID, size: u32, data: Vec<u8>) -> Self {
        Self { id, size, data }
    }

    /// Returns the data as a null-terminated string if it contains valid UTF-8
    pub fn as_string(&self) -> Option<String> {
        // Find the first null byte or use the entire data
        let end = self.data.iter().position(|&b| b == 0).unwrap_or(self.data.len());
        let trimmed = &self.data[..end];
        
        String::from_utf8(trimmed.to_vec()).ok()
    }

    /// Returns the data as raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

impl ListChunk {
    /// Parses a LIST chunk from raw data
    pub fn parse<R: Read + Seek>(reader: &mut R, chunk_size: u32) -> WaversResult<Self> {
        // Read the list type (4 bytes)
        let mut type_buf = [0u8; 4];
        reader.read_exact(&mut type_buf)?;
        let list_type = ChunkID::new(type_buf);

        // Remaining data size after the type field
        let remaining_size = chunk_size.saturating_sub(4);
        let mut remaining_data = vec![0u8; remaining_size as usize];
        reader.read_exact(&mut remaining_data)?;

        // Parse sub-chunks from the remaining data
        let sub_chunks = Self::parse_sub_chunks(&remaining_data)?;

        Ok(ListChunk {
            list_type,
            sub_chunks,
            unknown_data: Vec::new(), // We'll store unparsed data here if needed
        })
    }

    /// Parses sub-chunks from raw data
    fn parse_sub_chunks(data: &[u8]) -> WaversResult<Vec<SubChunk>> {
        let mut sub_chunks = Vec::new();
        let mut offset = 0;

        while offset + 8 <= data.len() {
            // Read sub-chunk ID (4 bytes)
            if offset + 4 > data.len() {
                break;
            }
            let id_bytes: [u8; 4] = data[offset..offset + 4]
                .try_into()
                .map_err(|_| WaversErrorKind::Parse(ParseError::Nom {
                    offset,
                    msg: "Failed to read sub-chunk ID".to_string(),
                }))?;
            let id = ChunkID::new(id_bytes);
            offset += 4;

            // Read sub-chunk size (4 bytes)
            if offset + 4 > data.len() {
                break;
            }
            let size_bytes: [u8; 4] = data[offset..offset + 4]
                .try_into()
                .map_err(|_| WaversErrorKind::Parse(ParseError::Nom {
                    offset,
                    msg: "Failed to read sub-chunk size".to_string(),
                }))?;
            let size = u32::from_le_bytes(size_bytes);
            offset += 4;

            // Read sub-chunk data
            let data_end = offset + size as usize;
            if data_end > data.len() {
                return Err(WaversErrorKind::Validate(ValidationError::InvalidWavFile {
                    reason: format!("Sub-chunk data extends beyond LIST chunk boundaries: {} > {}", data_end, data.len()),
                }));
            }

            let chunk_data = data[offset..data_end].to_vec();
            sub_chunks.push(SubChunk::new(id, size, chunk_data));

            // Move to next sub-chunk (with padding alignment)
            offset = data_end;
            // Handle odd-sized chunks with padding
            if size % 2 == 1 && offset < data.len() {
                offset += 1; // Skip padding byte
            }
        }

        Ok(sub_chunks)
    }

    /// Extracts metadata from an INFO LIST chunk
    pub fn extract_metadata(&self) -> WaversResult<WavMetadata> {
        if self.list_type != LIST_INFO {
            return Err(WaversErrorKind::Operation(
                crate::error::OperationError::InvalidOp {
                    reason: format!("Cannot extract metadata from non-INFO LIST chunk: {}", self.list_type),
                }
            ));
        }

        let mut metadata = WavMetadata::default();

        for sub_chunk in &self.sub_chunks {
            if let Some(text) = sub_chunk.as_string() {
                match sub_chunk.id {
                    INFO_IART => metadata.artist = Some(text),
                    INFO_INAM => metadata.title = Some(text),
                    INFO_ICOP => metadata.copyright = Some(text),
                    INFO_ICMT => metadata.comment = Some(text),
                    INFO_ICRD => metadata.creation_date = Some(text),
                    INFO_IENG => metadata.engineer = Some(text),
                    INFO_IGNR => metadata.genre = Some(text),
                    INFO_IKEY => metadata.keywords = Some(text),
                    INFO_IMED => metadata.medium = Some(text),
                    INFO_IPRD => metadata.product = Some(text),
                    INFO_ISBJ => metadata.subject = Some(text),
                    INFO_ISFT => metadata.software = Some(text),
                    INFO_ISRC => metadata.source = Some(text),
                    INFO_ISRF => metadata.source_form = Some(text),
                    INFO_ITCH => metadata.technician = Some(text),
                    _ => {
                        // Unknown INFO sub-chunk, skip for now
                        #[cfg(feature = "log")]
                        log::debug!("Unknown INFO sub-chunk: {}", sub_chunk.id);
                    }
                }
            }
        }

        Ok(metadata)
    }

    /// Gets a specific sub-chunk by ID
    pub fn get_sub_chunk(&self, id: &ChunkID) -> Option<&SubChunk> {
        self.sub_chunks.iter().find(|chunk| chunk.id == *id)
    }

    /// Gets all sub-chunks with a specific ID
    pub fn get_sub_chunks(&self, id: &ChunkID) -> Vec<&SubChunk> {
        self.sub_chunks.iter().filter(|chunk| chunk.id == *id).collect()
    }

    /// Returns true if this is an INFO LIST chunk
    pub fn is_info(&self) -> bool {
        self.list_type == LIST_INFO
    }

    /// Returns true if this is an adtl LIST chunk
    pub fn is_adtl(&self) -> bool {
        self.list_type == LIST_ADTL
    }

    /// Returns true if this is a wavl LIST chunk
    pub fn is_wavl(&self) -> bool {
        self.list_type == LIST_WAVL
    }
}

impl Display for ListChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LIST Chunk ({})", self.list_type)?;
        writeln!(f, "  Sub-chunks: {}", self.sub_chunks.len())?;
        
        for (i, sub_chunk) in self.sub_chunks.iter().enumerate() {
            writeln!(f, "    [{}] {} ({} bytes)", i, sub_chunk.id, sub_chunk.size)?;
            if let Some(text) = sub_chunk.as_string() {
                // Truncate very long strings for display
                let display_text = if text.len() > 50 {
                    format!("{}...", &text[..47])
                } else {
                    text
                };
                writeln!(f, "        \"{}\"", display_text)?;
            }
        }
        Ok(())
    }
}

impl Display for WavMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "WAV Metadata:")?;
        if let Some(ref artist) = self.artist {
            writeln!(f, "  Artist: {}", artist)?;
        }
        if let Some(ref title) = self.title {
            writeln!(f, "  Title: {}", title)?;
        }
        if let Some(ref copyright) = self.copyright {
            writeln!(f, "  Copyright: {}", copyright)?;
        }
        if let Some(ref comment) = self.comment {
            writeln!(f, "  Comment: {}", comment)?;
        }
        if let Some(ref creation_date) = self.creation_date {
            writeln!(f, "  Creation Date: {}", creation_date)?;
        }
        if let Some(ref engineer) = self.engineer {
            writeln!(f, "  Engineer: {}", engineer)?;
        }
        if let Some(ref genre) = self.genre {
            writeln!(f, "  Genre: {}", genre)?;
        }
        if let Some(ref keywords) = self.keywords {
            writeln!(f, "  Keywords: {}", keywords)?;
        }
        if let Some(ref medium) = self.medium {
            writeln!(f, "  Medium: {}", medium)?;
        }
        if let Some(ref product) = self.product {
            writeln!(f, "  Product: {}", product)?;
        }
        if let Some(ref subject) = self.subject {
            writeln!(f, "  Subject: {}", subject)?;
        }
        if let Some(ref software) = self.software {
            writeln!(f, "  Software: {}", software)?;
        }
        if let Some(ref source) = self.source {
            writeln!(f, "  Source: {}", source)?;
        }
        if let Some(ref source_form) = self.source_form {
            writeln!(f, "  Source Form: {}", source_form)?;
        }
        if let Some(ref technician) = self.technician {
            writeln!(f, "  Technician: {}", technician)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_empty_list_chunk() {
        let data = b"INFO"; // Just the type, no sub-chunks
        let mut cursor = Cursor::new(data);
        let result = ListChunk::parse(&mut cursor, 4);
        assert!(result.is_ok());
        let list_chunk = result.unwrap();
        assert_eq!(list_chunk.list_type, LIST_INFO);
        assert_eq!(list_chunk.sub_chunks.len(), 0);
    }

    #[test]
    fn test_parse_info_chunk_with_metadata() {
        // Create test data: "INFO" + IART chunk + INAM chunk
        let mut data = Vec::new();
        data.extend_from_slice(b"INFO"); // List type
        
        // Add IART (Artist) sub-chunk
        data.extend_from_slice(b"IART"); // Sub-chunk ID
        data.extend_from_slice(&8u32.to_le_bytes()); // Size (including null terminator)
        data.extend_from_slice(b"TestArt\0"); // Data with null terminator
        
        // Add INAM (Title) sub-chunk
        data.extend_from_slice(b"INAM"); // Sub-chunk ID
        data.extend_from_slice(&10u32.to_le_bytes()); // Size
        data.extend_from_slice(b"TestTitle\0"); // Data with null terminator

        let mut cursor = Cursor::new(&data);
        let result = ListChunk::parse(&mut cursor, data.len() as u32);
        assert!(result.is_ok());
        
        let list_chunk = result.unwrap();
        assert_eq!(list_chunk.list_type, LIST_INFO);
        assert_eq!(list_chunk.sub_chunks.len(), 2);
        
        // Check metadata extraction
        let metadata = list_chunk.extract_metadata().unwrap();
        assert_eq!(metadata.artist, Some("TestArt".to_string()));
        assert_eq!(metadata.title, Some("TestTitle".to_string()));
        assert_eq!(metadata.copyright, None);
    }

    #[test]
    fn test_sub_chunk_as_string() {
        let data = b"Hello\0World\0".to_vec();
        let sub_chunk = SubChunk::new(INFO_IART, data.len() as u32, data);
        
        // Should return string up to first null terminator
        assert_eq!(sub_chunk.as_string(), Some("Hello".to_string()));
    }

    #[test]
    fn test_metadata_display() {
        let mut metadata = WavMetadata::default();
        metadata.artist = Some("Test Artist".to_string());
        metadata.title = Some("Test Title".to_string());
        
        let display_str = format!("{}", metadata);
        assert!(display_str.contains("Artist: Test Artist"));
        assert!(display_str.contains("Title: Test Title"));
    }
}