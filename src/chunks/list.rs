//! List Chunk - a chunk that contains a list of other chunks. Each chunk in the list is identified by a 4 byte ID, followed by a 4 byte size, and then the data.
use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

use crate::log;

#[cfg(feature = "colored")]
use colored::Colorize;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::{
    chunks::{Chunk, LIST},
    core::alloc_box_buffer,
    ReadSeek, WaversResult,
};

pub type InfoId = [u8; 4];

/// A List Chunk - a chunk that contains a list of other chunks. Each chunk in the list is identified by a 4 byte ID, followed by a 4 byte size, and then the data.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListChunk {
    list_type_id: [u8; 4],
    data: HashMap<InfoId, String>,
}

impl Chunk for ListChunk {
    /// Returns the ID of the ListChunk - "LIST".
    fn id(&self) -> &[u8; 4] {
        &LIST
    }

    /// Returns the size of the ListChunk in bytes less the size of the ID and size field itself.
    fn size(&self) -> u32 {
        let n_bytes = 4 + self
            .data
            .iter()
            .map(|(id, value)| id.len() as u32 + value.len() as u32 + 4)
            .sum::<u32>();
        n_bytes // 4 bytes for the list_type_id
    }

    /// Returns the full ListChunk in bytes.
    fn as_bytes(&self) -> Box<[u8]> {
        let mut bytes = alloc_box_buffer(8 + self.size() as usize);
        bytes[0..4].copy_from_slice(&LIST); // Chunk ID
        bytes[4..8].copy_from_slice(&(self.size() as u32).to_ne_bytes()); // Chunk Size
        bytes[8..12].copy_from_slice(&self.list_type_id); // List Type ID

        let mut i = 12;

        for (id, value) in &self.data {
            bytes[i..i + 4].copy_from_slice(id);
            i += 4;
            let size = value.len() as u32;
            bytes[i..i + 4].copy_from_slice(&size.to_ne_bytes());
            i += 4;
            bytes[i..i + size as usize].copy_from_slice(value.as_bytes());
            i += size as usize;
        }

        bytes
    }

    /// Reads the ListChunk from a reader.
    fn from_reader(
        reader: &mut Box<dyn ReadSeek>,
        info: &crate::header::HeaderChunkInfo,
    ) -> WaversResult<Self>
    where
        Self: Sized,
    {
        let offset = info.offset as u64;
        reader.seek(std::io::SeekFrom::Start(offset))?;
        let mut buf = [0; 4];

        reader.read_exact(&mut buf)?;
        let chunk_id = buf;
        println!("{:?}", std::str::from_utf8(&chunk_id).unwrap_or("ERROR"));

        reader.read_exact(&mut buf)?; // Read the chunk size
        let _chunk_size = u32::from_ne_bytes(buf);
        log!(log::Level::Debug, "Chunk Size: {}", _chunk_size);

        reader.read_exact(&mut buf)?; // Read the list type id
        let list_type_id = buf;

        let mut data = HashMap::new();

        while (reader.seek(std::io::SeekFrom::Current(0))? as usize)
            < info.offset + info.size as usize + 8
        {
            reader.read_exact(&mut buf)?;
            let id = buf;
            reader.read_exact(&mut buf)?;
            let size = u32::from_ne_bytes(buf) as usize;
            let mut value = alloc_box_buffer(size);
            reader.read_exact(&mut value)?;
            let value = std::str::from_utf8(&value).unwrap_or("[ERROR]").to_string();
            data.insert(id, value);
        }
        log!(log::Level::Debug, "List Chunk Data: {:?}", data);
        Ok(ListChunk::new(list_type_id, data))
    }
}

impl ListChunk {
    /// Creates a new ListChunk.
    pub(crate) fn new(list_type_id: [u8; 4], data: HashMap<InfoId, String>) -> Self {
        Self { list_type_id, data }
    }

    /// Creates a new ListChunk from bytes.
    pub fn from_bytes(bytes: &[u8]) -> WaversResult<Self> {
        let list_type_id = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let mut data = HashMap::new();
        let mut i: usize = 12;
        while i < bytes.len() {
            if i > bytes.len() {
                break;
            }
            let id = [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]];
            i += 4;
            let size =
                u32::from_ne_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
            i += 4;
            let value =
                String::from_utf8(bytes[i..i + size].to_vec()).unwrap_or("[ERROR]".to_string());
            data.insert(id, value);
            i += size;
        }
        Ok(ListChunk::new(list_type_id, data))
    }
}

impl Default for ListChunk {
    fn default() -> Self {
        ListChunk::new(*b"INFO", HashMap::new())
    }
}

#[cfg(feature = "colored")]
impl Display for ListChunk {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match std::str::from_utf8(&self.list_type_id) {
            Ok(id) => {
                write!(
                    f,
                    "{} ({})",
                    "ListChunk:".white().bold(),
                    id.white().underline()
                )?;
            }
            Err(_) => write!(f, "ListChunk: {}", "[ERROR]".red().bold().underline())?,
        }
        for (id, value) in &self.data {
            match std::str::from_utf8(id) {
                Ok(id) => {
                    write!(f, "\n\t{}: ", id.green().bold())?;
                    write!(f, "{}", value.white())?;
                }
                Err(_) => write!(f, "\n\t{}", "[ERROR]".red().bold().underline())?,
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "colored"))]
impl Display for ListChunk {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match std::str::from_utf8(&self.list_type_id) {
            Ok(id) => write!(f, "ListChunk: {}", id)?,
            Err(_) => write!(f, "ListChunk: [ERROR]")?,
        }
        for (id, value) in &self.data {
            match std::str::from_utf8(id) {
                Ok(id) => {
                    write!(f, "\n\t{}: ", id)?;
                    write!(f, "{}", value)?;
                }
                Err(_) => write!(f, "\n\t{}", "[ERROR]")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod list_chunk_tests {

    use super::*;

    #[test]
    fn can_deocde_list_bytes() {
        let list_type_id = *b"INFO";
        let mut data = HashMap::new();

        data.insert(*b"INAM", "Title".to_string());
        data.insert(*b"IART", "Jack Geraghty".to_string());

        let list_chunk = ListChunk::new(list_type_id, data);
        let bytes = list_chunk.as_bytes();
        let decoded = ListChunk::from_bytes(&bytes).unwrap();
        // let bytes = list_chunk.as_bytes();
        // let decoded = ListChunk::from_bytes(&mut bytes.clone(), bytes.len()).unwrap();
        println!("{}", decoded);
        // assert_eq!(decoded, list_chunk);
    }
}
