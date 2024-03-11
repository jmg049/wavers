use num_traits::ToBytes;

use crate::{
    wav_type::{FormatCode, WavType},
    WaversResult,
};

pub const FMT_SIZE_BASE_SIZE: usize = 16; // Standard wav file format size
pub const FMT_CB_SIZE: usize = 18; // An extended Format chunk is used for non-PCM data. The cbSize field gives the size of the extension. (0 or 22)
pub const FMT_SIZE_EXTENDED_SIZE: usize = 40; // CB_SIZE + 22 (2 bytes valid_bits_per_sample, 4 byte channel_mask, 16(2+14) byte sub_format)

const EXTENDED_FMT_GUID: [u8; 14] = *b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71";
pub const EXT_FORMAT_CODE: u16 = 0xFFFE;

#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
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

#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
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
        let current_type = WavType::try_from((self.format, self.bits_per_sample, self.format()))?;

        if current_type == new_type {
            return Ok(());
        }

        let new_type_info: (FormatCode, u16, FormatCode) = new_type.into();
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
}

impl From<[u8; FMT_SIZE_BASE_SIZE]> for FmtChunk {
    fn from(value: [u8; FMT_SIZE_BASE_SIZE]) -> Self {
        let mut ext_buf: [u8; 40] = [0; 40];
        ext_buf[0..FMT_SIZE_BASE_SIZE].copy_from_slice(&value); // standard fmt
        ext_buf[FMT_SIZE_BASE_SIZE..FMT_SIZE_EXTENDED_SIZE]
            .copy_from_slice(&ExtFmtChunkInfo::default_bytes());

        unsafe { std::mem::transmute_copy::<[u8; FMT_SIZE_EXTENDED_SIZE], FmtChunk>(&ext_buf) }
    }
}

impl From<[u8; FMT_CB_SIZE]> for FmtChunk {
    fn from(value: [u8; FMT_CB_SIZE]) -> Self {
        let mut ext_buf: [u8; 40] = [0; 40];
        ext_buf[0..FMT_CB_SIZE].copy_from_slice(&value);
        ext_buf[FMT_SIZE_BASE_SIZE..FMT_SIZE_EXTENDED_SIZE]
            .copy_from_slice(&ExtFmtChunkInfo::default_bytes());

        unsafe { std::mem::transmute_copy::<[u8; FMT_SIZE_EXTENDED_SIZE], FmtChunk>(&ext_buf) }
    }
}

impl From<[u8; FMT_SIZE_EXTENDED_SIZE]> for FmtChunk {
    fn from(value: [u8; FMT_SIZE_EXTENDED_SIZE]) -> Self {
        unsafe { std::mem::transmute_copy::<[u8; FMT_SIZE_EXTENDED_SIZE], FmtChunk>(&value) }
    }
}

impl Into<[u8; FMT_SIZE_EXTENDED_SIZE]> for FmtChunk {
    fn into(self) -> [u8; FMT_SIZE_EXTENDED_SIZE] {
        unsafe { std::mem::transmute_copy::<FmtChunk, [u8; FMT_SIZE_EXTENDED_SIZE]>(&self) }
    }
}

impl Into<[u8; FMT_SIZE_BASE_SIZE]> for FmtChunk {
    fn into(self) -> [u8; FMT_SIZE_BASE_SIZE] {
        let full_bytes =
            unsafe { std::mem::transmute_copy::<FmtChunk, [u8; FMT_SIZE_EXTENDED_SIZE]>(&self) };
        let mut bytes = [0; FMT_SIZE_BASE_SIZE];
        bytes.copy_from_slice(&full_bytes[0..FMT_SIZE_BASE_SIZE]);
        bytes
    }
}

impl Into<[u8; FMT_CB_SIZE]> for FmtChunk {
    fn into(self) -> [u8; FMT_CB_SIZE] {
        let full_bytes =
            unsafe { std::mem::transmute_copy::<FmtChunk, [u8; FMT_SIZE_EXTENDED_SIZE]>(&self) };
        let mut bytes = [0; FMT_CB_SIZE];
        bytes.copy_from_slice(&full_bytes[0..FMT_CB_SIZE]);
        bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CbSize {
    Base = 0,
    Extended = 22,
}

impl Default for CbSize {
    fn default() -> Self {
        CbSize::Base
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]

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

    pub const fn default_bytes() -> [u8; 24] {
        let mut bytes = [0; 24];
        let cb_size = CbSize::Base as u16;
        let valid_bits_per_sample: i16 = 16;
        let channel_mask: i32 = 0;
        let sub_format = FormatCode::WAV_FORMAT_PCM;
        let guid = EXTENDED_FMT_GUID;

        let cb_size_bytes = cb_size.to_ne_bytes();
        let valid_bits_per_sample_bytes = valid_bits_per_sample.to_ne_bytes();
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
