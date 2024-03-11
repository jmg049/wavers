///
/// Module containing functions and structs for working with Wav file headers.
///
use std::{
    any::TypeId,
    collections::HashMap,
    fmt::{Debug, Display},
    io::SeekFrom,
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::{
    chunks::{
        fmt::{CbSize, ExtFmtChunkInfo, FMT_CB_SIZE},
        FmtChunk,
    },
    conversion::AudioSample,
    core::{ReadSeek, WavInfo},
    error::{WaversError, WaversResult},
    wav_type::{FormatCode, WavType},
    Wav,
};

use crate::chunks::fmt::{FMT_SIZE_BASE_SIZE, FMT_SIZE_EXTENDED_SIZE};

const RIFF: [u8; 4] = *b"RIFF";
pub const DATA: [u8; 4] = *b"data";
const WAVE: [u8; 4] = *b"WAVE";
const FMT: [u8; 4] = *b"fmt ";

const RIFF_SIZE: usize = 12;

/// A struct used to store the offset and size of a chunk
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HeaderChunkInfo {
    pub offset: usize,
    pub size: u32,
}

impl Display for HeaderChunkInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(offset: {}, size: {})", self.offset, self.size)
    }
}

impl HeaderChunkInfo {
    /// Constructs a new HeaderEntryInfo struct with a given offset and size.
    pub fn new(offset: usize, size: u32) -> Self {
        HeaderChunkInfo { offset, size }
    }
}

impl Into<(usize, u32)> for HeaderChunkInfo {
    fn into(self) -> (usize, u32) {
        (self.offset, self.size)
    }
}

impl Into<(usize, u32)> for &HeaderChunkInfo {
    fn into(self) -> (usize, u32) {
        (self.offset, self.size)
    }
}

#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavHeader {
    header_info: HashMap<ChunkIdentifier, HeaderChunkInfo>,
    pub fmt_chunk: FmtChunk,
    pub current_file_size: usize, // convenience field for keeping track of the current file size
}

/// A struct representing the header of a wav file. It stores the offset and size of each chunk in the header,
/// the format information and the current size of the file in bytes.
#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavHeader {
    pub header_info: HashMap<ChunkIdentifier, HeaderChunkInfo>,
    pub fmt_chunk: FmtChunk,
    pub current_file_size: usize, // convenience field for keeping track of the current file size
}

impl WavHeader {
    /// Constructs a new WavHeader struct using the provided header information.
    pub fn new(
        header_info: HashMap<ChunkIdentifier, HeaderChunkInfo>,
        fmt_chunk: FmtChunk,
        current_file_size: usize,
    ) -> Self {
        assert!(
            header_info.contains_key(&DATA.into()),
            "Header info must contain a DATA chunk"
        );
        WavHeader {
            header_info,
            fmt_chunk,
            current_file_size,
        }
    }

    pub fn data(&self) -> &HeaderChunkInfo {
        self.header_info.get(&DATA.into()).unwrap() // Safe since a header cannot be created without a DATA chunk
    }

    pub fn fmt(&self) -> &HeaderChunkInfo {
        self.header_info.get(&FMT.into()).unwrap() // Safe since a header cannot be created without a FMT chunk
    }

    /// Creates a new WavHeader with the given sample rate, number of channels and number of samples.
    /// This function inserts the required fields into the WavHeader such as RIFF, WAVE, FMT and DATA.
    pub fn new_header<T>(sample_rate: i32, n_channels: u16, n_samples: usize) -> WaversResult<Self>
    where
        T: AudioSample,
    {
        let wav_type: WavType = TypeId::of::<T>().try_into()?;

        let (main_format, sub_format) = match wav_type {
            WavType::Pcm16 | WavType::Pcm24 | WavType::Pcm32 => {
                (FormatCode::WAV_FORMAT_PCM, FormatCode::WAV_FORMAT_PCM)
            }
            WavType::Float32 | WavType::Float64 => (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
            ),
            WavType::EPcm16 | WavType::EPcm24 | WavType::EPcm32 => (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                FormatCode::WAV_FORMAT_PCM,
            ),
            WavType::EFloat32 | WavType::EFloat64 => (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
            ),
        };

        let (_, bits_per_sample, _) = wav_type.into();

        let ext_fmt_chunk = match (main_format, sub_format) {
            (FormatCode::WAV_FORMAT_PCM, FormatCode::WAV_FORMAT_PCM) => {
                ExtFmtChunkInfo::new(CbSize::Base, bits_per_sample, 0, FormatCode::WAV_FORMAT_PCM)
            }
            (FormatCode::WAVE_FORMAT_EXTENSIBLE, FormatCode::WAV_FORMAT_PCM) => {
                ExtFmtChunkInfo::new(CbSize::Base, bits_per_sample, 0, FormatCode::WAV_FORMAT_PCM)
            }
            (FormatCode::WAVE_FORMAT_EXTENSIBLE, FormatCode::WAV_FORMAT_IEEE_FLOAT) => {
                ExtFmtChunkInfo::new(
                    CbSize::Base,
                    bits_per_sample,
                    0,
                    FormatCode::WAV_FORMAT_IEEE_FLOAT,
                )
            }
            _ => {
                return Err(WaversError::InvalidType(format!(
                    "Unsupported wav type: {:?}",
                    wav_type
                )))
            }
        };

        let fmt_chunk = FmtChunk::new(
            main_format,
            n_channels,
            sample_rate,
            bits_per_sample,
            ext_fmt_chunk,
        );
        let mut header_info = HashMap::new();

        let data_size_bytes = n_samples * (bits_per_sample / 8) as usize;
        let file_size_bytes = data_size_bytes + 44; // 4 bytes for RIFF + 4 bytes for size + 4 bytes for WAVE + 4 bytes for FMT  + 4 bytes for fmt size + 16 bytes for fmt chunk + 4 bytes for DATA + 4 bytes for data size + data_size_bytes

        let header_offset = match main_format {
            FormatCode::WAV_FORMAT_PCM => FMT_SIZE_BASE_SIZE,
            FormatCode::WAV_FORMAT_IEEE_FLOAT | FormatCode::WAVE_FORMAT_EXTENSIBLE => {
                FMT_SIZE_EXTENDED_SIZE
            }
            _ => {
                return Err(WaversError::InvalidType(format!(
                    "Unsupported wav type: {:?}",
                    wav_type
                )))
            }
        };

        header_info.insert(RIFF.into(), HeaderChunkInfo::new(0, RIFF_SIZE as u32));
        // insert fmt
        header_info.insert(FMT.into(), HeaderChunkInfo::new(12, header_offset as u32));
        // insert data
        header_info.insert(
            DATA.into(),
            HeaderChunkInfo::new(12 + header_offset, data_size_bytes as u32),
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

    pub fn as_base_bytes(&self) -> [u8; 36] {
        let mut bytes = [0; 36];
        bytes[0..4].copy_from_slice(&RIFF);
        let size = self.file_size() as u32;
        bytes[4..8].copy_from_slice(&size.to_ne_bytes());
        bytes[8..12].copy_from_slice(&WAVE);
        bytes[12..16].copy_from_slice(&FMT);
        bytes[16..20].copy_from_slice(&(FMT_SIZE_BASE_SIZE as u32).to_ne_bytes());
        let fmt_bytes: [u8; FMT_SIZE_BASE_SIZE] = self.fmt_chunk.into();
        bytes[20..36].copy_from_slice(&fmt_bytes);
        bytes
    }

    pub fn as_cb_bytes(&self) -> [u8; 38] {
        let mut bytes = [0; 38];
        bytes[0..4].copy_from_slice(&RIFF);
        let size = self.file_size() as u32;
        bytes[4..8].copy_from_slice(&size.to_ne_bytes());
        bytes[8..12].copy_from_slice(&WAVE);
        bytes[12..16].copy_from_slice(&FMT);
        bytes[16..20].copy_from_slice(&(FMT_CB_SIZE as u32).to_ne_bytes());
        let fmt_bytes: [u8; FMT_CB_SIZE] = self.fmt_chunk.into();
        bytes[16..38].copy_from_slice(&fmt_bytes);
        bytes
    }

    pub fn as_extended_bytes(&self) -> [u8; 60] {
        let mut bytes = [0; 60];
        bytes[0..4].copy_from_slice(&RIFF);
        let size = self.file_size() as u32;
        bytes[4..8].copy_from_slice(&size.to_ne_bytes());
        bytes[8..12].copy_from_slice(&WAVE);
        bytes[12..16].copy_from_slice(&FMT);
        bytes[16..20].copy_from_slice(&(FMT_SIZE_EXTENDED_SIZE as u32).to_ne_bytes());
        let fmt_bytes: [u8; FMT_SIZE_EXTENDED_SIZE] = self.fmt_chunk.into();
        bytes[20..60].copy_from_slice(&fmt_bytes);
        bytes
    }

    pub fn get_chunk(&self, chunk_identifier: ChunkIdentifier) -> Option<&HeaderChunkInfo> {
        self.header_info.get(&chunk_identifier)
    }
}

/// Wrapper around a 4 byte buffer. Used for storing and displaying/debugging the chunk identifier of a chunk.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ChunkIdentifier {
    identifier: [u8; 4],
}

impl ChunkIdentifier {
    pub fn new(identifier: [u8; 4]) -> Self {
        ChunkIdentifier { identifier }
    }
}

impl Into<[u8; 4]> for ChunkIdentifier {
    fn into(self) -> [u8; 4] {
        self.identifier
    }
}

impl Into<ChunkIdentifier> for [u8; 4] {
    fn into(self) -> ChunkIdentifier {
        ChunkIdentifier::new(self)
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

impl Debug for ChunkIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out_str = match std::str::from_utf8(&self.identifier) {
            Ok(s) => s,
            Err(_) => "Invalid identifier",
        };
        write!(f, "{}", out_str)
    }
}

/// Reads the header of a wav file and returns a tuple containing the header information and the wav encoding.
/// Mostly for convenience, but can also be used to inspect a wav file without reading the data.
pub(crate) fn read_header(readable: &mut Box<dyn ReadSeek>) -> WaversResult<WavInfo> {
    // reset the buffer reader to the start of the file
    readable.seek(SeekFrom::Start(0))?;

    let header_info: HashMap<ChunkIdentifier, HeaderChunkInfo> =
        discover_all_header_chunks(readable)?;

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
    readable.seek(SeekFrom::Start(fmt_entry.offset as u64))?; // +4 to move to beyond the chunk identifier

    let mut fmt_code_buf: [u8; 2] = [0; 2];
    readable.read_exact(&mut fmt_code_buf)?;

    // move the reader back two bytes so that we can decode the fmt format code again
    readable.seek(SeekFrom::Current(-2))?;

    let main_format_code = FormatCode::try_from(u16::from_ne_bytes(fmt_code_buf))?;
    println!("Main format code: {}", main_format_code);

    let fmt_chunk: FmtChunk = match main_format_code {
        FormatCode::WAV_FORMAT_PCM => {
            let mut fmt_buf: [u8; FMT_SIZE_BASE_SIZE] = [0; FMT_SIZE_BASE_SIZE];
            readable.read_exact(&mut fmt_buf)?;
            fmt_buf.into()
        }
        FormatCode::WAV_FORMAT_IEEE_FLOAT => {
            // In theory, the below is the correct approach for non-PCM formats, but so far with testing, it seems that the fmt chunk is always 16 bytes long when the format is set to 3.
            // let mut fmt_buf: [u8; FMT_CB_SIZE] = [0; FMT_CB_SIZE];
            let mut fmt_buf: [u8; FMT_SIZE_BASE_SIZE] = [0; FMT_SIZE_BASE_SIZE];
            readable.read_exact(&mut fmt_buf)?;
            fmt_buf.into()
        }
        FormatCode::WAVE_FORMAT_ALAW => todo!(),
        FormatCode::WAVE_FORMAT_MULAW => todo!(),
        FormatCode::WAVE_FORMAT_EXTENSIBLE => {
            let mut fmt_buf: [u8; FMT_SIZE_EXTENDED_SIZE] = [0; FMT_SIZE_EXTENDED_SIZE];
            readable.read_exact(&mut fmt_buf)?;
            fmt_buf.into()
        }
    };

    // let mut fmt_buf: [u8; FMT_SIZE] = [0; FMT_SIZE as usize];
    // readable.read_exact(&mut fmt_buf)?;
    // let fmt_chunk: FmtChunk = fmt_buf.into();

    let wav_type = WavType::try_from((
        fmt_chunk.format,
        fmt_chunk.bits_per_sample,
        fmt_chunk.format(),
    ))?;

    let total_size_in_bytes = header_info
        .get(&DATA.into())
        .expect("File does not contain a data chunk")
        .size
        + 44; // 44 bytes for the header // TODO! This is not always true!
    let wav_header = WavHeader::new(header_info, fmt_chunk, total_size_in_bytes as usize);

    Ok(WavInfo {
        wav_type,
        wav_header,
    })
}

// This shouldn't cause too many performance issues. Would wager than there is only ever the core header chunks and maybe a handful more.
// Each iteration is simply just a read of 8 (4+4) bytes.
fn discover_all_header_chunks(
    reader: &mut Box<dyn ReadSeek>,
) -> WaversResult<HashMap<ChunkIdentifier, HeaderChunkInfo>> {
    let mut entries: HashMap<ChunkIdentifier, HeaderChunkInfo> = HashMap::new();

    // create a reusable buffer for reading header chunks
    let mut buf: [u8; 4] = [0; 4];
    // The first 4 bytes of the file should be the RIFF chunk
    reader.read_exact(&mut buf)?;
    match buf_eq(&RIFF, &buf) {
        true => (),
        false => {
            return Err(WaversError::from(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File is not a valid RIFF file",
            )));
        }
    }

    reader.read_exact(&mut buf)?; // read the next 4 bytes which should be the size of the file

    entries.insert(RIFF.into(), HeaderChunkInfo::new(0, RIFF_SIZE as u32));

    // The next 4 bytes should be the WAVE chunk
    reader.read_exact(&mut buf)?;
    let _: ChunkIdentifier = buf.into();

    while let Ok(_) = reader.read_exact(&mut buf) {
        let chunk_identifier: ChunkIdentifier = buf.into();

        reader.read_exact(&mut buf)?;
        let chunk_size: u32 =
            buf[0] as u32 | (buf[1] as u32) << 8 | (buf[2] as u32) << 16 | (buf[3] as u32) << 24;

        entries.insert(
            chunk_identifier,
            HeaderChunkInfo::new(reader.stream_position()? as usize, chunk_size),
        );

        reader.seek(SeekFrom::Current(chunk_size as i64))?;
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
    use crate::chunks::fmt::{CbSize, ExtFmtChunkInfo};
    use std::fs::File;

    const TEST_FILE: &str = "./test_resources/one_channel_i16.wav";

    const ONE_CHANNEL_FMT_CHUNK: FmtChunk = FmtChunk {
        format: FormatCode::WAV_FORMAT_PCM,
        channels: 1,
        sample_rate: 16000,
        byte_rate: 16000 * 2 * 1,
        block_align: 2,
        bits_per_sample: 16,
        ext_fmt_chunk: ExtFmtChunkInfo::new(CbSize::Base, 16, 0, FormatCode::WAV_FORMAT_PCM),
    };

    #[test]
    fn can_read_header() {
        let file = File::open(TEST_FILE).unwrap();
        let mut file = Box::new(file) as Box<dyn ReadSeek>;
        let wav_info = read_header(&mut file).expect("Failed to read header");
        assert_eq!(
            wav_info.wav_header.fmt_chunk, ONE_CHANNEL_FMT_CHUNK,
            "Fmt chunk does not match"
        );
    }

    #[test]
    fn can_convert_to_and_from_bytes() {
        let file = File::open(TEST_FILE).unwrap();
        let mut file = Box::new(file) as Box<dyn ReadSeek>;
        let wav_info = read_header(&mut file).expect("Failed to read header");
        let fmt_bytes: [u8; FMT_SIZE_BASE_SIZE] = wav_info.wav_header.fmt_chunk.into();

        let new_fmt = fmt_bytes.into();
        assert_eq!(
            wav_info.wav_header.fmt_chunk, new_fmt,
            "Fmt chunk does not match"
        );
    }
}
