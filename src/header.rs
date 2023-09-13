///
/// Module containing functions and structs for working with Wav file headers.
///
use std::{
    any::TypeId,
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::{
    core::WavEncoding,
    error::{WaversError, WaversResult},
    AudioSample,
};

pub const RIFF: &[u8; 4] = b"RIFF";
pub const DATA: &[u8; 4] = b"data";
pub const WAVE: &[u8; 4] = b"WAVE";
pub const FMT: &[u8; 4] = b"fmt ";

pub const RIFF_SIZE: usize = 12;
const FMT_SIZE: usize = 16;

/// A struct used to store the offset and size of a chunk
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HeaderEntryInfo {
    pub offset: usize,
    pub size: u32,
}

impl Display for HeaderEntryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(offset: {}, size: {})", self.offset, self.size)
    }
}

impl HeaderEntryInfo {
    /// Constructs a new HeaderEntryInfo struct with a given offset and size.
    pub fn new(offset: usize, size: u32) -> Self {
        HeaderEntryInfo { offset, size }
    }
}

#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavHeader {
    header_info: HashMap<ChunkIdentifier, HeaderEntryInfo>,
    pub fmt_chunk: FmtChunk,
    pub current_file_size: usize, // convenience field for keeping track of the current file size
}

/// A struct representing the header of a wav file. It stores the offset and size of each chunk in the header,
/// the format information and the current size of the file in bytes.
#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavHeader {
    pub(crate) header_info: HashMap<ChunkIdentifier, HeaderEntryInfo>,
    pub fmt_chunk: FmtChunk,
    pub current_file_size: usize, // convenience field for keeping track of the current file size
}

impl WavHeader {
    /// Constructs a new WavHeader struct using the provided header information.
    pub fn new(
        header_info: HashMap<ChunkIdentifier, HeaderEntryInfo>,
        fmt_chunk: FmtChunk,
        current_file_size: usize,
    ) -> Self {
        WavHeader {
            header_info,
            fmt_chunk,
            current_file_size,
        }
    }

    /// Creates a new WavHeader with the given sample rate, number of channels and number of samples.
    /// This function inserts the required fields into the WavHeader such as RIFF, WAVE, FMT and DATA.
    pub fn new_header<T>(sample_rate: i32, n_channels: u16, n_samples: usize) -> WaversResult<Self>
    where
        T: AudioSample,
    {
        let encoding = {
            if TypeId::of::<T>() == crate::core::I16 || TypeId::of::<T>() == crate::core::I32 {
                1
            } else if TypeId::of::<T>() == crate::core::F32 || TypeId::of::<T>() == crate::core::F64
            {
                3
            } else {
                return Err(WaversError::InvalidType(format!("{:?}", TypeId::of::<T>())));
            }
        };
        let size_t_bytes = std::mem::size_of::<T>();
        let size_t_bits = size_t_bytes * 8;
        let fmt_chunk = FmtChunk::new(encoding, n_channels, sample_rate, size_t_bits as u16);
        let mut header_info = HashMap::new();

        let data_size_bytes = n_samples * size_t_bytes;
        let file_size_bytes = data_size_bytes + 44; // 4 bytes for RIFF + 4 bytes for size + 4 bytes for WAVE + 4 bytes for FMT  + 4 bytes for fmt size + 16 bytes for fmt chunk + 4 bytes for DATA + 4 bytes for data size + data_size_bytes

        header_info.insert(RIFF.into(), HeaderEntryInfo::new(0, RIFF_SIZE as u32));
        // insert WAVE
        header_info.insert(WAVE.into(), HeaderEntryInfo::new(8, 4));
        // insert fmt
        header_info.insert(FMT.into(), HeaderEntryInfo::new(12, FMT_SIZE as u32));
        // insert data
        header_info.insert(
            DATA.into(),
            HeaderEntryInfo::new(36, data_size_bytes as u32),
        );
        let current_file_size = file_size_bytes;
        Ok(WavHeader {
            header_info,
            fmt_chunk,
            current_file_size,
        })
    }

    /// Returns the current size of the file in bytes.
    pub fn file_size(&self) -> usize {
        self.current_file_size
    }

    /// Converts the WavHeader into an array of bytes.
    /// Since the data chunk is not included at this stage, the size is known.
    pub fn as_bytes(&self) -> [u8; 36] {
        let mut bytes = [0; 36]; // 4 bytes for RIFF + 4 bytes for size + 4 bytes for WAVE + 4 bytes for FMT  + 4 bytes for fmt size + 16 bytes for fmt chunk
        bytes[0..4].copy_from_slice(RIFF);
        let size = self.file_size() as u32;
        bytes[4..8].copy_from_slice(&size.to_ne_bytes());
        bytes[8..12].copy_from_slice(WAVE);
        bytes[12..16].copy_from_slice(FMT);
        bytes[16..20].copy_from_slice(&(FMT_SIZE as u32).to_ne_bytes());
        let fmt_bytes: [u8; FMT_SIZE] = self.fmt_chunk.into();
        bytes[20..36].copy_from_slice(&fmt_bytes);
        bytes
    }

    pub fn get(&self, chunk_identifier: ChunkIdentifier) -> Option<&HeaderEntryInfo> {
        self.header_info.get(&chunk_identifier)
    }
}

/// Wrapper around a 4 byte buffer. Used for storing and displaying/debugging the chunk identifier of a chunk.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ChunkIdentifier {
    identifier: [u8; 4],
}

impl ChunkIdentifier {
    pub fn new(identifier: [u8; 4]) -> Self {
        ChunkIdentifier { identifier }
    }
}

impl From<&[u8; 4]> for ChunkIdentifier {
    fn from(identifier: &[u8; 4]) -> Self {
        ChunkIdentifier {
            identifier: *identifier,
        }
    }
}

impl From<[u8; 4]> for ChunkIdentifier {
    fn from(identifier: [u8; 4]) -> Self {
        ChunkIdentifier {
            identifier: identifier,
        }
    }
}

impl Display for ChunkIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let as_str: &str = match std::str::from_utf8(&self.identifier) {
            Ok(s) => s,
            Err(_) => "Invalid identifier",
        };
        write!(f, "{:?}", as_str)
    }
}

///
/// A struct for storing the necessary format information about a wav file.
///
/// In total the struct is 16 bytes
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FmtChunk {
    /// Format of the audio data. 1 for PCM, 3 for IEEE float.
    pub format: u16,
    /// Number of channels in the audio data.
    pub channels: u16,
    /// Sample rate of the audio data.
    pub sample_rate: i32,
    /// Byte rate of the audio data.
    pub byte_rate: i32,
    /// Block align of the audio data.
    pub block_align: u16,
    /// Bits per sample of the audio data.
    pub bits_per_sample: u16,
}

impl FmtChunk {
    /// Constructs a new FmtChunk using the provided format, number of channels, sample rate and bits per sample.
    /// The remaining fields are calculating using these arguments
    pub fn new(format: u16, channels: u16, sample_rate: i32, bits_per_sample: u16) -> Self {
        let block_align = (channels * bits_per_sample) / 8;
        let byte_rate = sample_rate * (block_align as i32);
        FmtChunk {
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        }
    }
}

impl From<[u8; FMT_SIZE]> for FmtChunk {
    fn from(value: [u8; FMT_SIZE]) -> Self {
        unsafe { std::mem::transmute_copy::<[u8; FMT_SIZE], Self>(&value) }
    }
}

impl Into<[u8; FMT_SIZE]> for FmtChunk {
    fn into(self) -> [u8; FMT_SIZE] {
        unsafe { std::mem::transmute_copy::<Self, [u8; FMT_SIZE]>(&self) }
    }
}

impl FmtChunk {
    /// Function to update a WavHeader to a new encoding, for example i16 to f32. Does this in-place.
    #[inline(always)]
    pub fn update_header(&mut self, new_type_id: TypeId) -> WaversResult<()> {
        let current_type = match (self.format, self.block_align) {
            (1, 2) => crate::core::I16,
            (1, 4) => crate::core::I32,
            (3, 4) => crate::core::F32,
            (3, 8) => crate::core::F64,
            _ => {
                return Err(WaversError::InvalidType(format!(
                    "Unsupported format {:?}",
                    new_type_id
                )))
            }
        };

        if current_type == new_type_id {
            return Ok(());
        }

        // if statement since std::any::TypeId does not implement PartialEq
        let new_format = {
            if new_type_id == crate::core::I16 {
                1
            } else if new_type_id == crate::core::I32 {
                1
            } else if new_type_id == crate::core::F32 {
                3
            } else if new_type_id == crate::core::F64 {
                3
            } else {
                return Err(WaversError::InvalidType(format!(
                    "Unsupported format {:?}",
                    new_type_id
                )));
            }
        };

        let new_block_align = {
            if new_type_id == crate::core::I16 {
                2
            } else if new_type_id == crate::core::I32 {
                4
            } else if new_type_id == crate::core::F32 {
                4
            } else if new_type_id == crate::core::F64 {
                8
            } else {
                return Err(WaversError::InvalidType(format!(
                    "Unsupported format {:?}",
                    new_type_id
                )));
            }
        };

        let new_byte_rate: i32 =
            self.sample_rate * (self.channels as i32) * (new_block_align as i32);

        let new_bits_per_samples = {
            if new_type_id == crate::core::I16 {
                16
            } else if new_type_id == crate::core::I32 {
                32
            } else if new_type_id == crate::core::F32 {
                32
            } else if new_type_id == crate::core::F64 {
                64
            } else {
                return Err(WaversError::InvalidType(format!(
                    "Unsupported format {:?}",
                    new_type_id
                )));
            }
        };

        self.format = new_format;
        self.block_align = new_block_align;
        self.byte_rate = new_byte_rate;
        self.bits_per_sample = new_bits_per_samples;
        Ok(())
    }
}

/// Reads the header of a wav file and returns a tuple containing the header information and the wav encoding.
/// Mostly for convenience, but can also be used to inspect a wav file without reading the data.
pub fn read_header(file: &mut File) -> WaversResult<(WavHeader, WavEncoding)> {
    // reset the buffer reader to the start of the file
    file.seek(SeekFrom::Start(0))?;

    let header_info: HashMap<ChunkIdentifier, HeaderEntryInfo> = discover_all_header_chunks(file)?;

    match header_info.contains_key(&FMT.into()) {
        true => (),
        false => {
            return Err(WaversError::from(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File does not contain a fmt chunk",
            )));
        }
    }

    let fmt_entry = header_info.get(&FMT.into()).unwrap(); // Safe since we just checked that the key exists
    file.seek(SeekFrom::Start(fmt_entry.offset as u64))?; // +4 to move to beyond the chunk identifier

    let mut fmt_buf: [u8; FMT_SIZE] = [0; FMT_SIZE as usize];
    file.read_exact(&mut fmt_buf)?;
    let fmt_chunk: FmtChunk = FmtChunk::from(fmt_buf);

    let wav_encoding = crate::core::WavEncoding::new(fmt_chunk.format, fmt_chunk.bits_per_sample);
    let total_size_in_bytes = header_info
        .get(&DATA.into())
        .expect("File does not contain a data chunk")
        .size
        + 44; // 44 bytes for the header
    let header = WavHeader::new(header_info, fmt_chunk, total_size_in_bytes as usize);

    Ok((header, wav_encoding))
}

// This shouldn't cause too many performance issues. Would wager than there is only ever the core header chunks and maybe a handful more.
// Each iteration is simply just a read of 8 (4+4) bytes.
fn discover_all_header_chunks(
    buf_reader: &mut File,
) -> WaversResult<HashMap<ChunkIdentifier, HeaderEntryInfo>> {
    let mut entries: HashMap<ChunkIdentifier, HeaderEntryInfo> = HashMap::new();

    // create a reusable buffer for reading header chunks
    let mut buf: [u8; 4] = [0; 4];
    // The first 4 bytes of the file should be the RIFF chunk
    buf_reader.read_exact(&mut buf)?;
    match buf_eq(RIFF, &buf) {
        true => (),
        false => {
            return Err(WaversError::from(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File is not a valid RIFF file",
            )));
        }
    }

    buf_reader.read_exact(&mut buf)?; // read the next 4 bytes which should be the size of the file

    entries.insert(RIFF.into(), HeaderEntryInfo::new(0, RIFF_SIZE as u32));

    // The next 4 bytes should be the WAVE chunk
    buf_reader.read_exact(&mut buf)?;
    let _: ChunkIdentifier = buf.into();

    while let Ok(_) = buf_reader.read_exact(&mut buf) {
        let chunk_identifier: ChunkIdentifier = buf.into();

        buf_reader.read_exact(&mut buf)?;
        let chunk_size: u32 =
            buf[0] as u32 | (buf[1] as u32) << 8 | (buf[2] as u32) << 16 | (buf[3] as u32) << 24;

        entries.insert(
            chunk_identifier,
            HeaderEntryInfo::new(buf_reader.stream_position()? as usize, chunk_size),
        );

        buf_reader.seek(SeekFrom::Current(chunk_size as i64))?;
    }

    Ok(entries)
}

#[inline(always)]
fn buf_eq(buf: &[u8; 4], chunk_id: &[u8; 4]) -> bool {
    buf[0] == chunk_id[0] && buf[1] == chunk_id[1] && buf[2] == chunk_id[2] && buf[3] == chunk_id[3]
}

#[cfg(test)]
mod header_tests {
    use super::*;
    use crate::FmtChunk;

    const TEST_FILE: &str = "./test_resources/one_channel_i16.wav";

    const ONE_CHANNEL_FMT_CHUNK: FmtChunk = FmtChunk {
        format: 1,
        channels: 1,
        sample_rate: 16000,
        byte_rate: 16000 * 2 * 1,
        block_align: 2,
        bits_per_sample: 16,
    };

    #[test]
    fn can_read_header() {
        let mut file = File::open(TEST_FILE).unwrap();
        let (header, _) = read_header(&mut file).expect("Failed to read header");
        assert_eq!(
            header.fmt_chunk, ONE_CHANNEL_FMT_CHUNK,
            "Fmt chunk does not match"
        );
    }

    #[test]
    fn can_convert_to_and_from_bytes() {
        let mut file = File::open(TEST_FILE).unwrap();
        let (header, _) = read_header(&mut file).expect("Failed to read header");
        let fmt_bytes: [u8; FMT_SIZE] = header.fmt_chunk.into();

        let new_fmt = FmtChunk::from(fmt_bytes);
        assert_eq!(header.fmt_chunk, new_fmt, "Fmt chunk does not match");
    }
}
