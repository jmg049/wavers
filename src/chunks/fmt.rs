//! The fmt chunk of a wav file. This chunk contains information about the format of the audio data.
use std::{fmt::Display, io::SeekFrom};

use crate::{
    chunks::{Chunk, FACT, FMT},
    error::FormatError,
    wav_type::{format_info_to_wav_type, wav_type_to_format_info, FormatCode, WavType},
    ReadSeek, WaversError, WaversResult,
};

#[cfg(feature = "colored")]
use colored::Colorize;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub const FMT_SIZE_BASE_SIZE: usize = 16; // Standard wav file format size
pub const FMT_CB_SIZE: usize = 18; // An extended Format chunk is used for non-PCM data. The cbSize field gives the size of the extension. (0 or 22)
pub const FMT_SIZE_EXTENDED_SIZE: usize = 40; // CB_SIZE + 22 (2 bytes valid_bits_per_sample, 4 byte channel_mask, 16(2+14) byte sub_format)

const EXTENDED_FMT_GUID: [u8; 14] = *b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71";
pub const EXT_FORMAT_CODE: u16 = 0xFFFE;

/// The format chunk of a wav file. This chunk contains information about the format of the audio data.
/// This chunk must be present in a wav file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct FmtChunk {
    /// Format of the audio data. 1 for PCM, 3 for IEEE float.
    pub format: FormatCode,
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
    pub ext_fmt_chunk: ExtFmtChunkInfo,
}

impl FmtChunk {
    /// Constructs a new FmtChunk using the provided format, number of channels, sample rate and bits per sample.
    /// The remaining fields are calculating using these arguments
    pub fn new(
        format: FormatCode,
        channels: u16,
        sample_rate: i32,
        bits_per_sample: u16,
        ext_fmt_chunk: ExtFmtChunkInfo,
    ) -> Self {
        let block_align = (channels * bits_per_sample) / 8;
        let byte_rate = sample_rate * (block_align as i32);
        FmtChunk {
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            ext_fmt_chunk,
        }
    }

    /// Function to update a WavHeader to a new encoding, for example i16 to f32. Does this in-place.
    #[inline(always)]
    pub fn update_fmt_chunk(&mut self, new_type: WavType) -> WaversResult<()> {
        let current_type =
            format_info_to_wav_type((self.format, self.bits_per_sample, self.format()))?;

        if current_type == new_type {
            return Ok(());
        }

        let new_type_info: (FormatCode, u16, FormatCode) = wav_type_to_format_info(new_type);
        let (new_format, new_bits_per_sample, _sub_format_code) = new_type_info;
        let new_block_align = new_bits_per_sample * (self.channels as u16) / 8;
        let new_byte_rate: i32 =
            self.sample_rate * (self.channels as i32) * (new_block_align as i32);

        self.format = new_format;
        self.block_align = new_block_align;
        self.byte_rate = new_byte_rate;
        self.bits_per_sample = new_bits_per_sample;

        Ok(())
    }

    pub fn is_extended_format(&self) -> bool {
        self.format as u16 == EXT_FORMAT_CODE
    }

    pub fn format(&self) -> FormatCode {
        match self.is_extended_format() {
            true => self.ext_fmt_chunk.sub_format,
            false => self.format,
        }
    }

    pub fn base_bytes(&self) -> [u8; FMT_SIZE_BASE_SIZE] {
        let mut bytes = [0; FMT_SIZE_BASE_SIZE];
        bytes[0..2].copy_from_slice(&self.format.to_ne_bytes());
        bytes[2..4].copy_from_slice(&self.channels.to_ne_bytes());
        bytes[4..8].copy_from_slice(&self.sample_rate.to_ne_bytes());
        bytes[8..12].copy_from_slice(&self.byte_rate.to_ne_bytes());
        bytes[12..14].copy_from_slice(&self.block_align.to_ne_bytes());
        bytes[14..16].copy_from_slice(&self.bits_per_sample.to_ne_bytes());
        bytes
    }

    pub fn cb_bytes(&self) -> [u8; FMT_CB_SIZE] {
        let mut bytes = [0; FMT_CB_SIZE];
        bytes[0..16].copy_from_slice(&self.base_bytes());
        let cb: u16 = self.ext_fmt_chunk.cb_size as u16;
        bytes[16..18].copy_from_slice(&cb.to_ne_bytes());
        bytes
    }

    pub fn extended_bytes(&self) -> [u8; FMT_SIZE_EXTENDED_SIZE] {
        let mut bytes = [0; FMT_SIZE_EXTENDED_SIZE];
        bytes[0..16].copy_from_slice(&self.base_bytes());
        bytes[16..40].copy_from_slice(&self.ext_fmt_chunk.as_bytes());
        bytes
    }

    fn from_base_bytes(bytes: [u8; FMT_SIZE_BASE_SIZE]) -> Self {
        let mut buf: [u8; FMT_SIZE_EXTENDED_SIZE] = [0; FMT_SIZE_EXTENDED_SIZE];

        let mut info_buf: [u8; 2] = [0; 2];
        info_buf.copy_from_slice(&bytes[0..2]);
        let format = FormatCode::try_from(u16::from_ne_bytes(info_buf)).unwrap();

        info_buf.copy_from_slice(&bytes[14..16]);
        let bits_per_sample = u16::from_ne_bytes(info_buf);

        buf[0..FMT_SIZE_BASE_SIZE].copy_from_slice(&bytes);
        buf[FMT_SIZE_BASE_SIZE..FMT_SIZE_EXTENDED_SIZE]
            .copy_from_slice(&ExtFmtChunkInfo::default_bytes(format, bits_per_sample));
        FmtChunk::from_extended_bytes(buf)
    }

    fn from_cb_bytes(bytes: [u8; FMT_CB_SIZE]) -> Self {
        let mut buf: [u8; FMT_SIZE_EXTENDED_SIZE] = [0; FMT_SIZE_EXTENDED_SIZE];
        buf[0..FMT_CB_SIZE].copy_from_slice(&bytes);
        let mut info_buf: [u8; 2] = [0; 2];

        info_buf.copy_from_slice(&bytes[0..2]);

        let format = FormatCode::try_from(u16::from_ne_bytes(info_buf)).unwrap();

        info_buf.copy_from_slice(&bytes[14..16]);
        let bits_per_sample = u16::from_ne_bytes(info_buf);
        buf[FMT_CB_SIZE - 2..FMT_SIZE_EXTENDED_SIZE]
            .copy_from_slice(&ExtFmtChunkInfo::default_bytes(format, bits_per_sample));
        FmtChunk::from_extended_bytes(buf)
    }

    fn from_extended_bytes(bytes: [u8; FMT_SIZE_EXTENDED_SIZE]) -> Self {
        unsafe { std::mem::transmute_copy::<[u8; FMT_SIZE_EXTENDED_SIZE], FmtChunk>(&bytes) }
    }

    #[allow(unused)]
    // This function is only used in the tests
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        match bytes.len() {
            FMT_SIZE_BASE_SIZE => FmtChunk::from_base_bytes(bytes.try_into().unwrap()),
            FMT_CB_SIZE => FmtChunk::from_cb_bytes(bytes.try_into().unwrap()),
            FMT_SIZE_EXTENDED_SIZE => FmtChunk::from_extended_bytes(bytes.try_into().unwrap()),
            _ => panic!("Invalid fmt chunk size: {}", bytes.len()),
        }
    }
}

impl Chunk for FmtChunk {
    fn id(&self) -> &[u8; 4] {
        &FMT
    }

    fn size(&self) -> u32 {
        match self.is_extended_format() {
            true => match self.ext_fmt_chunk.cb_size {
                CbSize::Base => FMT_CB_SIZE as u32,
                CbSize::Extended => FMT_SIZE_EXTENDED_SIZE as u32,
            },
            false => FMT_SIZE_BASE_SIZE as u32,
        }
    }

    fn as_bytes(&self) -> Box<[u8]> {
        match self.is_extended_format() {
            true => match self.ext_fmt_chunk.cb_size {
                CbSize::Base => {
                    let mut bytes: [u8; FMT_CB_SIZE + 8] = [0; FMT_CB_SIZE + 8];
                    bytes[0..4].copy_from_slice(&FACT);
                    bytes[4..8].copy_from_slice(&self.size().to_ne_bytes());
                    bytes[8..FMT_CB_SIZE + 8].copy_from_slice(&self.cb_bytes());
                    Box::new(bytes)
                }
                CbSize::Extended => {
                    let mut bytes: [u8; FMT_SIZE_EXTENDED_SIZE + 8] =
                        [0; FMT_SIZE_EXTENDED_SIZE + 8];
                    bytes[0..4].copy_from_slice(&FACT);
                    bytes[4..8].copy_from_slice(&self.size().to_ne_bytes());
                    bytes[8..FMT_SIZE_EXTENDED_SIZE + 8].copy_from_slice(&self.extended_bytes());
                    Box::new(bytes)
                }
            },
            false => {
                let mut bytes: [u8; FMT_SIZE_BASE_SIZE + 8] = [0; FMT_SIZE_BASE_SIZE + 8];
                bytes[0..4].copy_from_slice(&FMT);
                bytes[4..8].copy_from_slice(&self.size().to_ne_bytes());
                bytes[8..FMT_SIZE_BASE_SIZE + 8].copy_from_slice(&self.base_bytes());
                Box::new(bytes)
            }
        }
    }

    fn from_reader(
        reader: &mut Box<dyn ReadSeek>,
        info: &crate::header::HeaderChunkInfo,
    ) -> WaversResult<Self>
    where
        Self: Sized,
    {
        let offset = info.offset as u64 + 8;
        reader.seek(SeekFrom::Start(offset))?;
        let mut fmt_code_buf: [u8; 2] = [0; 2];
        reader.read_exact(&mut fmt_code_buf)?;

        // move the reader back two bytes so that we can decode the fmt format code again
        reader.seek(SeekFrom::Current(-2))?;

        let main_format_code = FormatCode::try_from(u16::from_ne_bytes(fmt_code_buf))?;
        let total_size_in_bytes = info.size as usize;
        let fmt_chunk: FmtChunk = match main_format_code {
            FormatCode::WAV_FORMAT_PCM => {
                let mut fmt_buf: [u8; FMT_SIZE_BASE_SIZE] = [0; FMT_SIZE_BASE_SIZE];
                reader.read_exact(&mut fmt_buf)?;
                FmtChunk::from_base_bytes(fmt_buf)
            }
            FormatCode::WAV_FORMAT_IEEE_FLOAT => {
                // In theory, the below is the correct approach for non-PCM formats, but so far with testing, it seems that the fmt chunk is always 16 bytes long when the format is set to 3.
                // let mut fmt_buf: [u8; FMT_CB_SIZE] = [0; FMT_CB_SIZE];
                let mut fmt_buf: [u8; FMT_SIZE_BASE_SIZE] = [0; FMT_SIZE_BASE_SIZE];
                reader.read_exact(&mut fmt_buf)?;
                FmtChunk::from_base_bytes(fmt_buf)
            }
            FormatCode::WAVE_FORMAT_ALAW => todo!(),
            FormatCode::WAVE_FORMAT_MULAW => todo!(),
            FormatCode::WAVE_FORMAT_EXTENSIBLE => match total_size_in_bytes {
                FMT_CB_SIZE => {
                    let mut fmt_buf: [u8; FMT_CB_SIZE] = [0; FMT_CB_SIZE];
                    reader.read_exact(&mut fmt_buf)?;
                    FmtChunk::from_cb_bytes(fmt_buf)
                }
                FMT_SIZE_EXTENDED_SIZE => {
                    let mut fmt_buf: [u8; FMT_SIZE_EXTENDED_SIZE] = [0; FMT_SIZE_EXTENDED_SIZE];
                    reader.read_exact(&mut fmt_buf)?;
                    FmtChunk::from_extended_bytes(fmt_buf)
                }
                _ => return Err(FormatError::InvalidFmtChunkSize(total_size_in_bytes).into()),
            },
        };
        Ok(fmt_chunk)
    }
}

#[cfg(feature = "colored")]
impl Display for FmtChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fmt = &"FmtChunk\n".white().bold().underline();
        let format = &"\tFormat: ".green().bold();
        let format_value = &format!("{},\n", self.format).white();

        let channels = &"\tChannels: ".green().bold();
        let channels_value = &format!("{},\n", self.channels).white();

        let sample_rate = &"\tSample Rate: ".green().bold();
        let sample_rate_value = &format!("{},\n", self.sample_rate).white();

        let byte_rate = &"\tByte Rate: ".green().bold();
        let byte_rate_value = &format!("{},\n", self.byte_rate).white();

        let block_align = &"\tBlock Align: ".green().bold();
        let block_align_value = &format!("{},\n", self.block_align).white();

        let bits_per_sample = &"\tBits Per Sample: ".green().bold();
        let bits_per_sample_value = &format!("{},\n", self.bits_per_sample).white();

        write!(
            f,
            "{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
            fmt,
            format,
            format_value,
            channels,
            channels_value,
            sample_rate,
            sample_rate_value,
            byte_rate,
            byte_rate_value,
            block_align,
            block_align_value,
            bits_per_sample,
            bits_per_sample_value,
            self.ext_fmt_chunk
        )
    }
}

#[cfg(not(feature = "colored"))]
impl Display for FmtChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FmtChunk\n")?;
        write!(f, "\tFormat: {}\n", self.format)?;
        write!(f, "\tChannels: {}\n", self.channels)?;
        write!(f, "\tSample Rate: {}\n", self.sample_rate)?;
        write!(f, "\tByte Rate: {}\n", self.byte_rate)?;
        write!(f, "\tBlock Align: {}\n", self.block_align)?;
        write!(f, "\tBits Per Sample: {}\n", self.bits_per_sample)?;
        write!(f, "{}", self.ext_fmt_chunk)
    }
}

/// An enum used to represent the two possible sizes of the extensible format chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub enum CbSize {
    Base = 0,
    Extended = 22,
}

impl Default for CbSize {
    fn default() -> Self {
        CbSize::Base
    }
}

impl Display for CbSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CbSize::Base => write!(f, "Base (0)"),
            CbSize::Extended => write!(f, "Extended (22)"),
        }
    }
}

/// The extended format chunk of a wav file. This chunk contains additional information about the format of the audio data.
/// This chunk is present when the format code is set to WAVE_FORMAT_EXTENSIBLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct ExtFmtChunkInfo {
    cb_size: CbSize,
    valid_bits_per_sample: u16,
    channel_mask: u32,
    sub_format: FormatCode,
    guid: [u8; 14],
}

impl ExtFmtChunkInfo {
    pub const fn new(
        cb_size: CbSize,
        valid_bits_per_sample: u16,
        channel_mask: u32,
        sub_format: FormatCode,
    ) -> Self {
        ExtFmtChunkInfo {
            cb_size,
            valid_bits_per_sample,
            channel_mask,
            sub_format,
            guid: EXTENDED_FMT_GUID,
        }
    }

    pub fn cb_size(&self) -> CbSize {
        self.cb_size
    }

    pub fn valid_bits_per_sample(&self) -> u16 {
        self.valid_bits_per_sample
    }

    pub fn channel_mask(&self) -> u32 {
        self.channel_mask
    }

    pub fn sub_format(&self) -> FormatCode {
        self.sub_format
    }

    pub fn as_bytes(&self) -> [u8; 24] {
        let mut bytes = [0; 24];
        let cb_size = self.cb_size as u16;

        bytes[0..2].copy_from_slice(&cb_size.to_ne_bytes());
        bytes[2..4].copy_from_slice(&self.valid_bits_per_sample.to_ne_bytes());
        bytes[4..8].copy_from_slice(&self.channel_mask.to_ne_bytes());
        bytes[8..10].copy_from_slice(&self.sub_format.to_ne_bytes());
        bytes[10..24].copy_from_slice(&self.guid);
        bytes
    }

    pub fn to_bytes(self) -> [u8; 24] {
        let mut bytes = [0; 24];
        let cb_size = self.cb_size as u16;
        bytes[0..2].copy_from_slice(&cb_size.to_ne_bytes());
        bytes[2..4].copy_from_slice(&self.valid_bits_per_sample.to_ne_bytes());
        bytes[4..8].copy_from_slice(&self.channel_mask.to_ne_bytes());
        bytes[8..10].copy_from_slice(&self.sub_format.to_ne_bytes());
        bytes[10..24].copy_from_slice(&self.guid);
        bytes
    }

    pub const fn default_bytes(
        format_code: FormatCode,
        bits_per_sample: u16,
    ) -> [u8; std::mem::size_of::<Self>()] {
        let mut bytes = [0; std::mem::size_of::<Self>()];
        let cb_size = CbSize::Base as u16;
        let channel_mask: i32 = 0;

        let sub_format = format_code;
        let guid = EXTENDED_FMT_GUID;

        let cb_size_bytes = cb_size.to_ne_bytes();
        let valid_bits_per_sample_bytes = bits_per_sample.to_ne_bytes();
        let channel_mask_bytes = channel_mask.to_ne_bytes();
        let sub_format_bytes = sub_format.to_ne_bytes();

        let mut i = 0;

        while i < cb_size_bytes.len() {
            bytes[i] = cb_size_bytes[i];
            i += 1;
        }
        let mut i = 0;

        while i < valid_bits_per_sample_bytes.len() {
            bytes[i + 2] = valid_bits_per_sample_bytes[i];
            i += 1;
        }
        let mut i = 0;

        while i < channel_mask_bytes.len() {
            bytes[i + 4] = channel_mask_bytes[i];
            i += 1;
        }
        let mut i = 0;

        while i < sub_format_bytes.len() {
            bytes[i + 8] = sub_format_bytes[i];
            i += 1;
        }
        let mut i = 0;

        while i < guid.len() {
            bytes[i + 10] = guid[i];
            i += 1;
        }

        bytes
    }
}

impl Default for ExtFmtChunkInfo {
    fn default() -> Self {
        Self {
            cb_size: CbSize::Base,
            valid_bits_per_sample: (std::mem::size_of::<i16>() * 8) as u16,
            channel_mask: 0,
            sub_format: FormatCode::WAV_FORMAT_PCM,
            guid: EXTENDED_FMT_GUID,
        }
    }
}

#[cfg(feature = "colored")]
impl Display for ExtFmtChunkInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ext_fmt = &"\tExtended Format Chunk\n".white().bold().underline();

        let cb_size = &"\t\tCB Size: ".green().bold();
        let cb_size_value = &format!("{},\n", self.cb_size).white();

        let valid_bits_per_sample = &"\t\tValid Bits Per Sample: ".green().bold();
        let valid_bits_per_sample_value = &format!("{},\n", self.valid_bits_per_sample).white();

        let channel_mask = &"\t\tChannel Mask: ".green().bold();
        let channel_mask_value = &format!("{},\n", self.channel_mask).white();

        let sub_format = &"\t\tSub Format: ".green().bold();
        let sub_format_value = &format!("{},\n", self.sub_format).white();

        let guid = &"\t\tGUID: ".green().bold();
        let guid_value = &format!("{:?},\n", self.guid).white();

        write!(
            f,
            "{}{}{}{}{}{}{}{}{}{}{}",
            ext_fmt,
            cb_size,
            cb_size_value,
            valid_bits_per_sample,
            valid_bits_per_sample_value,
            channel_mask,
            channel_mask_value,
            sub_format,
            sub_format_value,
            guid,
            guid_value
        )
    }
}

#[cfg(not(feature = "colored"))]
impl Display for ExtFmtChunkInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\tExtended Format Chunk\n")?;
        write!(f, "\t\tCB Size: {}\n", self.cb_size)?;
        write!(
            f,
            "\t\tValid Bits Per Sample: {}\n",
            self.valid_bits_per_sample
        )?;
        write!(f, "\t\tChannel Mask: {}\n", self.channel_mask)?;
        write!(f, "\t\tSub Format: {}\n", self.sub_format)?;
        write!(f, "\t\tGUID: {:?}\n", self.guid)
    }
}
