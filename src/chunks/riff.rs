use super::{ChunkID, WAVE};
use crate::{WaversErrorKind, WaversResult, error::ValidationError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RIFFChunk {
    pub riff_tag: ChunkID,
    pub file_size: u32,
    pub wave_tag: ChunkID,
}

impl RIFFChunk {
    pub fn parse(bytes: [u8; 12]) -> WaversResult<Self> {
        // Safety for unwrap(). The caller is responsible for checking the RIFF tag first.
        let riff_tag: [u8; 4] = bytes[0..4].try_into().unwrap();
        let riff_tag = ChunkID::new(riff_tag);
        let file_size = match bytes[4..8].try_into() {
            Ok(r) => r,
            Err(_) => {
                return Err(WaversErrorKind::Validate(ValidationError::Header(format!(
                    "Bad RIFF file size parse: {:?}",
                    &bytes[4..8]
                ))));
            }
        };
        let file_size = u32::from_le_bytes(file_size);
        let wave_tag = match bytes[8..12].try_into() {
            Ok(r) => r,
            Err(_) => {
                return Err(WaversErrorKind::Validate(ValidationError::Header(format!(
                    "Bad WAVE tag parse: {:?}",
                    &bytes[8..12]
                ))));
            }
        };
        let wave_tag = ChunkID::new(wave_tag);

        if wave_tag != WAVE {
            return Err(WaversErrorKind::Validate(ValidationError::Header(format!(
                "Invalid WAVE tag: got {} expected {}",
                wave_tag, WAVE
            ))));
        }
        Ok(Self {
            riff_tag,
            file_size,
            wave_tag,
        })
    }
}
