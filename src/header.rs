//! Module containing functions and structs for working with Wav file headers.
use std::{
    any::TypeId,
    collections::HashMap,
    fmt::{Debug, Display},
    io::SeekFrom,
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "colored")]
use colored::Colorize;

use crate::{
    chunks::{
        fmt::{CbSize, ExtFmtChunkInfo, FMT_CB_SIZE, FMT_SIZE_BASE_SIZE, FMT_SIZE_EXTENDED_SIZE},
        read_chunk, Chunk, FmtChunk, DATA, FMT, RIFF, WAVE,
    },
    conversion::AudioSample,
    core::{ReadSeek, WavInfo},
    error::{WaversError, WaversResult},
    log,
    wav_type::{format_info_to_wav_type, FormatCode, WavType},
};

const _RIFF_SIZE: usize = 4; // do not count the size or RIFF id, only the WAVE id. The other chunks will be used for the remaining size
const HEADER_FMT_BASE_SIZE: usize = 36;
const HEADER_FMT_CB_SIZE: usize = 38;
const HEADER_FMT_EXTENDED_SIZE: usize = 60;
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

impl PartialOrd for HeaderChunkInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.offset.cmp(&other.offset))
    }
}

impl Ord for HeaderChunkInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.offset.cmp(&other.offset)
    }
}

#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavHeader {
    header_info: HashMap<ChunkIdentifier, HeaderChunkInfo>,
    #[pyo3(get)]
    pub fmt_chunk: FmtChunk,
    #[pyo3(get)]
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
        log!(
            log::Level::Debug,
            "Creating new WavHeader with fmt chunk: {:?}",
            fmt_chunk
        );
        WavHeader {
            header_info,
            fmt_chunk,
            current_file_size,
        }
    }

    /// Returns the information relating to the DATA chunk.
    pub fn data(&self) -> &HeaderChunkInfo {
        self.header_info.get(&DATA.into()).unwrap() // Safe since a header cannot be created without a DATA chunk
    }

    /// Returns the information relating to the FMT chunk.
    pub fn fmt(&self) -> &HeaderChunkInfo {
        self.header_info.get(&FMT.into()).unwrap() // Safe since a header cannot be created without a FMT chunk
    }

    /// Creates a new WavHeader with the given sample rate, number of channels and number of samples.
    /// This function inserts the required fields into the WavHeader such as RIFF, WAVE, FMT and DATA.
    pub fn new_header<T>(sample_rate: i32, n_channels: u16, n_samples: usize) -> WaversResult<Self>
    where
        T: AudioSample,
    {
        log!(
            log::Level::Debug,
            "Creating new header with sample rate: {}, channels: {}, samples: {}",
            sample_rate,
            n_channels,
            n_samples
        );
        let wav_type: WavType = TypeId::of::<T>().try_into()?;

        // Calculate sizes of data chunks
        let bits_per_sample = wav_type.n_bits();
        let data_size_bytes = n_samples * (bits_per_sample / 8) as usize;
        let fmt_data_size = match wav_type {
            WavType::Pcm16 | WavType::Pcm24 | WavType::Pcm32 | WavType::Float32
            | WavType::Float64=> FMT_SIZE_BASE_SIZE,
            WavType::EPcm16
            | WavType::EPcm24
            | WavType::EPcm32
            | WavType::EFloat32
            | WavType::EFloat64 => FMT_SIZE_EXTENDED_SIZE,
        };

        let mut header_info: HashMap<ChunkIdentifier, HeaderChunkInfo> = HashMap::new();

        // The RIFF chunk size should be:
        // sizeof("WAVE") + sizeof(fmt_chunk) + sizeof(data_chunk)
        // where each chunk includes its header (identifier + size field)
        let riff_chunk_size = 4 +                     // "WAVE"
        (8 + fmt_data_size) +  // "fmt " + size + data
        (8 + data_size_bytes); // "data" + size + data

        header_info.insert(
            RIFF.into(),
            HeaderChunkInfo::new(0, (riff_chunk_size - 8) as u32),
        );
        header_info.insert(FMT.into(), HeaderChunkInfo::new(12, fmt_data_size as u32));
        header_info.insert(
            DATA.into(),
            HeaderChunkInfo::new(20 + fmt_data_size, data_size_bytes as u32),
        );

        // Total file size includes everything
        let current_file_size = 8 + riff_chunk_size - 8; // RIFF + size + (chunk_size - 8)

        // Create fmt chunk
        let (main_format, sub_format) = match wav_type {
            WavType::Pcm16 | WavType::Pcm24 | WavType::Pcm32 => {
                (FormatCode::WAV_FORMAT_PCM, FormatCode::WAV_FORMAT_PCM)
            }
            WavType::Float32 | WavType::Float64 => (
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
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

        let ext_fmt_chunk = ExtFmtChunkInfo::new(CbSize::Base, bits_per_sample, 0, sub_format);

        let fmt_chunk = FmtChunk::new(
            main_format,
            n_channels,
            sample_rate,
            bits_per_sample,
            ext_fmt_chunk,
        );

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

    /// Returns the header in bytes, assuming that the FmtChunk is in the base format.
    pub fn as_base_bytes(&self) -> [u8; HEADER_FMT_BASE_SIZE] {
        let mut bytes = [0; HEADER_FMT_BASE_SIZE];
        bytes[0..4].copy_from_slice(&RIFF);
        let size = self.file_size() as u32;
        bytes[4..8].copy_from_slice(&size.to_ne_bytes());
        bytes[8..12].copy_from_slice(&WAVE);

        let fmt_bytes = self.fmt_chunk.as_bytes();
        debug_assert!(
            fmt_bytes.len() == FMT_SIZE_BASE_SIZE + 8,
            "Fmt bytes length is not equal to the expected length: {} vs {}",
            fmt_bytes.len(),
            FMT_SIZE_BASE_SIZE + 8
        );
        bytes[12..12 + fmt_bytes.len()].copy_from_slice(&fmt_bytes);
        bytes
    }

    /// Returns the header in bytes, assuming that the FmtChunk is in the cb (has cb field but no other extensible format fields) format.
    pub fn as_cb_bytes(&self) -> [u8; HEADER_FMT_CB_SIZE] {
        let mut bytes = [0; HEADER_FMT_CB_SIZE];
        bytes[0..4].copy_from_slice(&RIFF);
        let size = self.file_size() as u32;
        bytes[4..8].copy_from_slice(&size.to_ne_bytes());
        bytes[8..12].copy_from_slice(&WAVE);

        let fmt_bytes = self.fmt_chunk.as_bytes();
        debug_assert!(
            fmt_bytes.len() == FMT_CB_SIZE + 8,
            "Fmt bytes length is not equal to the expected length: {} vs {}",
            fmt_bytes.len(),
            FMT_CB_SIZE + 8
        );
        bytes[12..12 + fmt_bytes.len()].copy_from_slice(&fmt_bytes);

        bytes
    }

    /// Returns the header in bytes, assuming that the FmtChunk is in the extensible format.
    pub fn as_extended_bytes(&self) -> [u8; HEADER_FMT_EXTENDED_SIZE] {
        let mut bytes = [0; HEADER_FMT_EXTENDED_SIZE];
        bytes[0..4].copy_from_slice(&RIFF);
        let size = self.file_size() as u32;
        bytes[4..8].copy_from_slice(&size.to_ne_bytes());
        bytes[8..12].copy_from_slice(&WAVE);
        bytes[12..16].copy_from_slice(&FMT);
        bytes[16..20].copy_from_slice(&(FMT_SIZE_EXTENDED_SIZE as u32).to_ne_bytes());
        let fmt_bytes: [u8; FMT_SIZE_EXTENDED_SIZE] = self.fmt_chunk.extended_bytes();
        bytes[20..60].copy_from_slice(&fmt_bytes);
        bytes
    }

    /// Attempt to get some chunk information from the header. Returns None if the chunk is not found.
    pub fn get_chunk_info(&self, chunk_identifier: ChunkIdentifier) -> Option<&HeaderChunkInfo> {
        self.header_info.get(&chunk_identifier)
    }
}

#[cfg(feature = "colored")]
impl Display for WavHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut header_info_string = String::new();
        header_info_string.push_str(
            "Header Info:\n"
                .white()
                .bold()
                .underline()
                .to_string()
                .as_str(),
        );
        let mut sorted_info: Vec<(&ChunkIdentifier, &HeaderChunkInfo)> =
            self.header_info.iter().collect();

        sorted_info.sort_by(|a, b| a.1.cmp(b.1));

        for (chunk_id, chunk_info) in sorted_info {
            let k = format!("{:?}", chunk_id).green().bold();
            let v = format!("{:?}", chunk_info).white();
            header_info_string.push_str(&format!("\t{}: {}\n", k, v));
        }

        let current_file_size = format!("Current file size: ").white().bold().underline();
        let current_size = format!("{}", self.current_file_size).white();
        write!(
            f,
            "{}\n{}\n{}{}",
            header_info_string, self.fmt_chunk, current_file_size, current_size
        )
    }
}

#[cfg(not(feature = "colored"))]
impl Display for WavHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut header_info_string = String::new();
        header_info_string.push_str("Header Info:\n");

        let mut sorted_info: Vec<(&ChunkIdentifier, &HeaderChunkInfo)> =
            self.header_info.iter().collect();

        sorted_info.sort_by(|a, b| a.1.cmp(b.1));

        for (chunk_id, chunk_info) in sorted_info {
            let k = format!("{:?}", chunk_id);
            let v = format!("{:?}", chunk_info);
            header_info_string.push_str(&format!("\t{}: {}\n", k, v));
        }

        let current_file_size = format!("Current file size: ");
        let current_size = format!("{}", self.current_file_size);

        write!(
            f,
            "{}\n{}\n{}{}",
            header_info_string, self.fmt_chunk, current_file_size, current_size
        )
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

impl AsRef<[u8; 4]> for ChunkIdentifier {
    fn as_ref(&self) -> &[u8; 4] {
        &self.identifier
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
    let fmt_chunk: FmtChunk = read_chunk::<FmtChunk>(readable, fmt_entry)?;

    let wav_type = format_info_to_wav_type((
        fmt_chunk.format,
        fmt_chunk.bits_per_sample,
        fmt_chunk.format(),
    ))?;

    let total_size = header_info.get(&RIFF.into()).unwrap().size as usize + 8;
    let wav_header = WavHeader::new(header_info, fmt_chunk, total_size);

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

    let file_size: u32 =
        buf[0] as u32 | (buf[1] as u32) << 8 | (buf[2] as u32) << 16 | (buf[3] as u32) << 24;
    entries.insert(RIFF.into(), HeaderChunkInfo::new(0, file_size as u32));

    // The next 4 bytes should be the RIFF type id
    reader.read_exact(&mut buf)?;
    let _: ChunkIdentifier = buf.into();

    while let Ok(_) = reader.read_exact(&mut buf) {
        let chunk_identifier: ChunkIdentifier = buf.into();
        reader.read_exact(&mut buf)?;
        let chunk_size: u32 =
            buf[0] as u32 | (buf[1] as u32) << 8 | (buf[2] as u32) << 16 | (buf[3] as u32) << 24;
        entries.insert(
            chunk_identifier,
            HeaderChunkInfo::new(reader.stream_position()? as usize - 8, chunk_size),
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
    use std::fs::File;

    const TEST_FILE: &str = "./test_resources/one_channel_i16.wav";
    const TWO_CHANNEL_TEST_FILE: &str = "./test_resources/two_channel_i16.wav";

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
        let fmt_bytes = wav_info.wav_header.fmt_chunk.base_bytes();
        let new_fmt = FmtChunk::from_bytes(&fmt_bytes);
        assert_eq!(
            wav_info.wav_header.fmt_chunk, new_fmt,
            "Fmt chunk does not match"
        );
    }

    #[test]
    fn test_printing() {
        let file = File::open(TEST_FILE).unwrap();
        let mut file = Box::new(file) as Box<dyn ReadSeek>;
        let wav_info = read_header(&mut file).expect("Failed to read header");
        println!("{}", wav_info.wav_header);
    }

    #[test]
    fn test_size() {
        let file = File::open(TEST_FILE).unwrap();
        let mut file = Box::new(file) as Box<dyn ReadSeek>;
        let wav_info = read_header(&mut file).expect("Failed to read header");
        assert_eq!(
            wav_info.wav_header.file_size(),
            320044,
            "File size does not match"
        );

        let file = File::open(TWO_CHANNEL_TEST_FILE).unwrap();
        let mut file = Box::new(file) as Box<dyn ReadSeek>;
        let wav_info = read_header(&mut file).expect("Failed to read header");
        assert_eq!(
            wav_info.wav_header.file_size(),
            640044,
            "File size does not match"
        );
    }
}
