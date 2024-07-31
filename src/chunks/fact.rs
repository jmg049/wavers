//! Contains the FactChunk struct and its implementation.
use std::fmt::{Display, Formatter};

#[cfg(feature = "colored")]
use colored::Colorize;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::{
    chunks::{Chunk, FACT},
    header::HeaderChunkInfo,
    ReadSeek,
};

/// The fact chunk of a wav file. Contains a single field, ``num_samples``. This field is the number of samples in the wav file per channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct FactChunk {
    pub num_samples: u32,
}

impl FactChunk {
    /// Creates a new FactChunk with the given number of samples.
    pub(crate) fn new(num_samples: u32) -> Self {
        Self { num_samples }
    }
}

impl Chunk for FactChunk {
    /// Returns the ID of the FactChunk - "FACT".
    fn id(&self) -> &[u8; 4] {
        &FACT
    }

    /// Returns the size of the FactChunk in bytes less the size of the ID and size field itself.
    fn size(&self) -> u32 {
        4
    }

    /// Returns the full FactChunk in bytes.
    fn as_bytes(&self) -> Box<[u8]> {
        let mut buf = [0; 12];
        buf[0..4].copy_from_slice(&FACT);
        buf[4..8].copy_from_slice(&4u32.to_ne_bytes());
        buf[8..12].copy_from_slice(&self.num_samples.to_ne_bytes());
        Box::new(buf)
    }

    /// Reads the FactChunk from a reader.
    fn from_reader(
        reader: &mut Box<dyn ReadSeek>,
        info: &HeaderChunkInfo,
    ) -> crate::WaversResult<Self>
    where
        Self: Sized,
    {
        let offset = info.offset as u64 + 8;
        reader.seek(std::io::SeekFrom::Start(offset))?;
        let mut buf = [0; 4];
        reader.read_exact(&mut buf)?;
        let num_samples = u32::from_ne_bytes(buf);
        Ok(FactChunk::new(num_samples))
    }
}

impl From<[u8; 4]> for FactChunk {
    fn from(bytes: [u8; 4]) -> Self {
        FactChunk::new(u32::from_ne_bytes(bytes))
    }
}

impl Default for FactChunk {
    fn default() -> Self {
        FactChunk { num_samples: 0 }
    }
}

#[cfg(feature = "colored")]
impl Display for FactChunk {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\n\t{} {}",
            "FactChunk: ".white().bold().underline(),
            "num_samples:".green().bold(),
            self.num_samples.to_string().white()
        )
    }
}

#[cfg(not(feature = "colored"))]
impl Display for FactChunk {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FactChunk: num_samples: {}", self.num_samples)
    }
}

#[cfg(test)]
mod fact_tests {
    use super::*;

    #[test]
    fn test_as_bytes() {
        let fact = FactChunk::new(10);
        let fact_bytes = fact.as_bytes();

        let mut fact_buf = [0; 12];
        fact_buf[0..4].copy_from_slice(&FACT);
        fact_buf[4..8].copy_from_slice(&4u32.to_ne_bytes());
        fact_buf[8..12].copy_from_slice(&10u32.to_ne_bytes());

        assert_eq!(
            fact_buf.len(),
            fact_bytes.len(),
            "FactChunk as_bytes length is not 12"
        );
        for byte in fact_buf.iter().zip(fact_bytes.iter()) {
            assert_eq!(
                byte.0, byte.1,
                "FactChunk as_bytes is not equal to expected"
            );
        }
    }
}
