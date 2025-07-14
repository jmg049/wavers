//! Provides types and utilities for parsing and validating the `fmt ` chunk of a WAV file.
//!
//! The `fmt ` chunk is a required component of the WAV format, specifying how the audio data is
//! encoded. It includes parameters like sample rate, number of channels, bit depth, and format code
//! (e.g., PCM, IEEE float, or extensible).
//!
//! This module supports both standard and extensible WAV format chunks, including:
//!
//! - [`FmtChunk`]: Core struct representing the audio format specification.
//! - [`ExtFmtChunkInfo`] and [`ExtendedFormatData`]: Structures for handling `WAVE_FORMAT_EXTENSIBLE`
//!   (format code 0xFFFE), allowing detailed descriptions of speaker layout, bit depth, and sub-format.
//! - Constants for standard speaker positions and channel configurations.
//! - Helpers for parsing, serializing, and validating format chunk data.
//!
//! This module assumes that chunks are well-formed when parsed from validated WAV input, but all
//! constructors and parsing logic still include robust validation to ensure correctness.

use crate::error::{ParseError, ValidationError};
use crate::wav_type::ValidBitSize;
use crate::{FormatCode, WaversErrorKind, WaversResult};

use std::fmt::Display;

use nom::IResult;
use nom::number::complete::{i32, u16, u32};

/// GUID used for the extended format chunk
const EXTENDED_FMT_GUID: [u8; 14] = *b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71";

/// Standard sizes for the format chunk
#[allow(dead_code)]
pub mod chunk_sizes {
    /// The standard size of a basic (non-extensible) format chunk: 16 bytes
    pub const STANDARD_SIZE: usize = 16;
    /// The size of a format chunk with cb_size field but no extended data: 18 bytes
    pub const WITH_CB_SIZE: usize = 18;
    /// The size of a format chunk with full extended data: 40 bytes
    pub const EXTENDED_SIZE: usize = 40;
}

/// The format chunk (`fmt `) of a WAV file.
///
/// This chunk contains essential information about the audio format,
/// including the number of channels, sample rate, and bits per sample.
/// For more complex formats, this chunk may also contain extended format information.
///
/// The format chunk is one of the required chunks in a WAV file and appears
/// immediately after the RIFF header. It defines the audio data format and
/// how it should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmtChunk {
    /// Format of the audio data. 1 for PCM, 3 for IEEE float, etc.
    pub format: FormatCode,
    /// Number of channels in the audio data (1 for mono, 2 for stereo, etc.).
    pub channels: u16,
    /// Sample rate of the audio data in Hz (e.g., 44100, 48000).
    pub sample_rate: i32,
    /// Byte rate of the audio data (sample_rate * block_align).
    pub byte_rate: i32,
    /// Block align of the audio data (channels * bits_per_sample / 8).
    /// This value represents the number of bytes for one sample including all channels.
    pub block_align: u16,
    /// Bits per sample of the audio data (e.g., 8, 16, 24, 32).
    pub bits_per_sample: u16,
    /// Optional extended format information, present for WaveFormatExtensible.
    pub ext_fmt_chunk: Option<ExtFmtChunkInfo>,
}

// Constants for channel position mask bits according to Microsoft specifications
/// Front left speaker
pub const SPEAKER_FRONT_LEFT: u32 = 0x1;
/// Front right speaker
pub const SPEAKER_FRONT_RIGHT: u32 = 0x2;
/// Front center speaker
pub const SPEAKER_FRONT_CENTER: u32 = 0x4;
/// Low frequency effects speaker (subwoofer)
pub const SPEAKER_LOW_FREQUENCY: u32 = 0x8;
/// Back left speaker
pub const SPEAKER_BACK_LEFT: u32 = 0x10;
/// Back right speaker
pub const SPEAKER_BACK_RIGHT: u32 = 0x20;
/// Front left of center speaker
pub const SPEAKER_FRONT_LEFT_OF_CENTER: u32 = 0x40;
/// Front right of center speaker
pub const SPEAKER_FRONT_RIGHT_OF_CENTER: u32 = 0x80;
/// Back center speaker
pub const SPEAKER_BACK_CENTER: u32 = 0x100;
/// Side left speaker
pub const SPEAKER_SIDE_LEFT: u32 = 0x200;
/// Side right speaker
pub const SPEAKER_SIDE_RIGHT: u32 = 0x400;
/// Top center speaker
pub const SPEAKER_TOP_CENTER: u32 = 0x800;
/// Top front left speaker
pub const SPEAKER_TOP_FRONT_LEFT: u32 = 0x1000;
/// Top front center speaker
pub const SPEAKER_TOP_FRONT_CENTER: u32 = 0x2000;
/// Top front right speaker
pub const SPEAKER_TOP_FRONT_RIGHT: u32 = 0x4000;
/// Top back left speaker
pub const SPEAKER_TOP_BACK_LEFT: u32 = 0x8000;
/// Top back center speaker
pub const SPEAKER_TOP_BACK_CENTER: u32 = 0x10000;
/// Top back right speaker
pub const SPEAKER_TOP_BACK_RIGHT: u32 = 0x20000;

/// Common channel configurations
pub mod channel_configs {
    #[allow(dead_code)]
    use super::*;

    /// Mono configuration (1 channel)
    pub const MONO: u32 = SPEAKER_FRONT_CENTER;

    /// Stereo configuration (2 channels)
    pub const STEREO: u32 = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;

    /// 2.1 configuration (3 channels)
    pub const CONFIG_2_1: u32 = STEREO | SPEAKER_LOW_FREQUENCY;

    /// 3.0 configuration (3 channels)
    pub const CONFIG_3_0: u32 = STEREO | SPEAKER_FRONT_CENTER;

    /// 3.1 configuration (4 channels)
    pub const CONFIG_3_1: u32 = CONFIG_3_0 | SPEAKER_LOW_FREQUENCY;

    /// Quad configuration (4 channels)
    pub const QUAD: u32 = STEREO | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT;

    /// 4.1 configuration (5 channels)
    pub const CONFIG_4_1: u32 = QUAD | SPEAKER_LOW_FREQUENCY;

    /// 5.0 configuration (5 channels)
    pub const CONFIG_5_0: u32 = CONFIG_3_0 | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT;

    /// 5.1 configuration (6 channels) - most common surround format
    pub const CONFIG_5_1: u32 = CONFIG_5_0 | SPEAKER_LOW_FREQUENCY;

    /// 6.1 configuration (7 channels)
    pub const CONFIG_6_1: u32 = CONFIG_5_1 | SPEAKER_BACK_CENTER;

    /// 7.1 configuration (8 channels)
    pub const CONFIG_7_1: u32 = CONFIG_5_1 | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT;
}

// Helper functions for working with channel masks
/// Functions for working with channel masks
#[allow(dead_code)]
pub mod channel_mask {
    use super::*;

    /// Get a human-readable description of a channel mask
    ///
    /// Returns a string describing which speaker positions are present in the mask
    ///
    /// # Arguments
    ///
    /// * `mask` - The channel mask to describe
    ///
    /// # Returns
    ///
    /// A string describing the channel configuration
    pub fn describe(mask: u32) -> String {
        let mut positions = Vec::new();

        if mask & SPEAKER_FRONT_LEFT != 0 {
            positions.push("Front Left");
        }
        if mask & SPEAKER_FRONT_RIGHT != 0 {
            positions.push("Front Right");
        }
        if mask & SPEAKER_FRONT_CENTER != 0 {
            positions.push("Front Center");
        }
        if mask & SPEAKER_LOW_FREQUENCY != 0 {
            positions.push("LFE");
        }
        if mask & SPEAKER_BACK_LEFT != 0 {
            positions.push("Back Left");
        }
        if mask & SPEAKER_BACK_RIGHT != 0 {
            positions.push("Back Right");
        }
        if mask & SPEAKER_FRONT_LEFT_OF_CENTER != 0 {
            positions.push("Front Left of Center");
        }
        if mask & SPEAKER_FRONT_RIGHT_OF_CENTER != 0 {
            positions.push("Front Right of Center");
        }
        if mask & SPEAKER_BACK_CENTER != 0 {
            positions.push("Back Center");
        }
        if mask & SPEAKER_SIDE_LEFT != 0 {
            positions.push("Side Left");
        }
        if mask & SPEAKER_SIDE_RIGHT != 0 {
            positions.push("Side Right");
        }
        if mask & SPEAKER_TOP_CENTER != 0 {
            positions.push("Top Center");
        }
        if mask & SPEAKER_TOP_FRONT_LEFT != 0 {
            positions.push("Top Front Left");
        }
        if mask & SPEAKER_TOP_FRONT_CENTER != 0 {
            positions.push("Top Front Center");
        }
        if mask & SPEAKER_TOP_FRONT_RIGHT != 0 {
            positions.push("Top Front Right");
        }
        if mask & SPEAKER_TOP_BACK_LEFT != 0 {
            positions.push("Top Back Left");
        }
        if mask & SPEAKER_TOP_BACK_CENTER != 0 {
            positions.push("Top Back Center");
        }
        if mask & SPEAKER_TOP_BACK_RIGHT != 0 {
            positions.push("Top Back Right");
        }

        if positions.is_empty() {
            return "No channels specified".to_string();
        }

        positions.join(", ")
    }

    /// Determine the number of channels from a channel mask
    ///
    /// Counts the number of set bits in the mask
    ///
    /// # Arguments
    ///
    /// * `mask` - The channel mask
    ///
    /// # Returns
    ///
    /// The number of channels represented by the mask
    pub fn channel_count(mask: u32) -> u16 {
        mask.count_ones() as u16
    }

    /// Get the common name for a channel configuration if it matches a standard layout
    ///
    /// # Arguments
    ///
    /// * `mask` - The channel mask
    ///
    /// # Returns
    ///
    /// A string with the common name of the configuration, or None if it doesn't match a standard
    pub fn get_config_name(mask: u32) -> Option<&'static str> {
        use super::channel_configs::*;

        match mask {
            MONO => Some("Mono"),
            STEREO => Some("Stereo"),
            CONFIG_2_1 => Some("2.1"),
            CONFIG_3_0 => Some("3.0"),
            CONFIG_3_1 => Some("3.1"),
            QUAD => Some("Quad"),
            CONFIG_4_1 => Some("4.1"),
            CONFIG_5_0 => Some("5.0"),
            CONFIG_5_1 => Some("5.1"),
            CONFIG_6_1 => Some("6.1"),
            CONFIG_7_1 => Some("7.1"),
            _ => None,
        }
    }

    /// Create a channel mask for a given number of channels using default speaker positions
    ///
    /// # Arguments
    ///
    /// * `channels` - The number of channels
    ///
    /// # Returns
    ///
    /// A channel mask with appropriate speaker positions for the channel count
    pub fn from_channel_count(channels: u16) -> u32 {
        match channels {
            1 => channel_configs::MONO,
            2 => channel_configs::STEREO,
            3 => channel_configs::CONFIG_3_0,
            4 => channel_configs::QUAD,
            5 => channel_configs::CONFIG_5_0,
            6 => channel_configs::CONFIG_5_1,
            7 => channel_configs::CONFIG_6_1,
            8 => channel_configs::CONFIG_7_1,
            _ => {
                // For non-standard channel counts, just set the first n bits
                (1u32 << channels) - 1
            }
        }
    }

    /// Validates that a channel mask is consistent with a channel count
    ///
    /// # Arguments
    ///
    /// * `mask` - The channel mask to validate
    /// * `channels` - The expected number of channels
    ///
    /// # Returns
    ///
    /// Ok(()) if valid, or an error if inconsistent
    pub fn validate(mask: u32, channels: u16) -> Result<(), WaversErrorKind> {
        let mask_channels = channel_count(mask);

        if mask_channels != channels {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!(
                    "Channel mask specifies {} channels but format specifies {} channels",
                    mask_channels, channels
                ),
            }));
        }

        Ok(())
    }
}

impl FmtChunk {
    /// Creates a new FmtChunk with the given parameters.
    ///
    /// This function automatically calculates the byte_rate and block_align fields
    /// based on the provided parameters.
    ///
    /// # Arguments
    ///
    /// * `format` - The audio format code
    /// * `channels` - Number of audio channels
    /// * `sample_rate` - Sample rate in Hz
    /// * `bits_per_sample` - Bits per sample
    /// * `ext_fmt_chunk` - Optional extended format information
    ///
    /// # Returns
    ///
    /// A new FmtChunk instance
    pub fn new(
        format: FormatCode,
        channels: u16,
        sample_rate: i32,
        bits_per_sample: u16,
        ext_fmt_chunk: Option<ExtFmtChunkInfo>,
    ) -> Self {
        let block_align = (channels * bits_per_sample) / 8;
        let byte_rate = sample_rate * (block_align as i32);

        Self {
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            ext_fmt_chunk,
        }
    }

    /// Creates a standard PCM format chunk.
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of audio channels (1 for mono, 2 for stereo, etc.)
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100, 48000)
    /// * `bits_per_sample` - Bits per sample (8, 16, 24, 32)
    ///
    /// # Returns
    ///
    /// A new FmtChunk configured for PCM audio
    pub fn new_pcm(channels: u16, sample_rate: i32, bits_per_sample: u16) -> Self {
        Self::new(
            FormatCode::WavFormatPcm,
            channels,
            sample_rate,
            bits_per_sample,
            None,
        )
    }

    /// Creates a standard IEEE float format chunk.
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of audio channels (1 for mono, 2 for stereo, etc.)
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100, 48000)
    /// * `bits_per_sample` - Bits per sample (32 or 64 for float)
    ///
    /// # Returns
    ///
    /// A new FmtChunk configured for IEEE float audio
    pub fn new_float(channels: u16, sample_rate: i32, bits_per_sample: u16) -> Self {
        Self::new(
            FormatCode::WavFormatIeeeFloat,
            channels,
            sample_rate,
            bits_per_sample,
            None,
        )
    }

    /// Creates a WaveFormatExtensible format chunk.
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of audio channels
    /// * `sample_rate` - Sample rate in Hz
    /// * `bits_per_sample` - Bits per sample
    /// * `valid_bits_per_sample` - Valid bits per sample (can be less than bits_per_sample)
    /// * `channel_mask` - Channel mask (bit flags indicating speaker positions)
    /// * `sub_format` - The actual audio format code
    ///
    /// # Returns
    ///
    /// A new FmtChunk configured for extensible format audio
    pub fn new_extensible(
        channels: u16,
        sample_rate: i32,
        bits_per_sample: u16,
        valid_bits_per_sample: u16,
        channel_mask: u32,
        sub_format: FormatCode,
    ) -> Self {
        // Create extended format data
        let extended_data = ExtendedFormatData {
            valid_bits_per_sample,
            channel_mask,
            sub_format,
            guid: EXTENDED_FMT_GUID,
        };

        // Create extended format chunk info
        let ext_fmt_chunk = ExtFmtChunkInfo {
            cb_size: CbSize::Extended,
            extended_data: Some(extended_data),
        };

        Self::new(
            FormatCode::WaveFormatExtensible,
            channels,
            sample_rate,
            bits_per_sample,
            Some(ext_fmt_chunk),
        )
    }

    /// Checks if this format is using the extensible format.
    ///
    /// # Returns
    ///
    /// `true` if this is a WaveFormatExtensible format, otherwise `false`
    pub fn is_extensible(&self) -> bool {
        self.format == FormatCode::WaveFormatExtensible
    }

    /// Gets the effective format code, taking into account extensible formats.
    ///
    /// For normal formats, this returns the format field.
    /// For extensible formats, this returns the sub_format from the extended info.
    ///
    /// # Returns
    ///
    /// The effective format code
    pub fn effective_format(&self) -> FormatCode {
        match self.is_extensible() {
            true => {
                // For extensible formats, use the sub_format if available
                if let Some(ext_info) = &self.ext_fmt_chunk {
                    if let Some(sub_format) = ext_info.sub_format() {
                        return sub_format;
                    }
                }
                // Fallback to the main format if sub_format is not available
                self.format
            }
            false => self.format,
        }
    }

    /// Gets the effective number of valid bits per sample.
    ///
    /// For normal formats, this returns bits_per_sample.
    /// For extensible formats, this returns valid_bits_per_sample if available,
    /// otherwise falls back to bits_per_sample.
    ///
    /// # Returns
    ///
    /// The effective number of valid bits per sample
    pub fn effective_bits_per_sample(&self) -> WaversResult<ValidBitSize> {
        match self.is_extensible() {
            true => {
                // For extensible formats, use valid_bits_per_sample if available
                if let Some(ext_info) = &self.ext_fmt_chunk {
                    if let Some(valid_bits) = ext_info.valid_bits_per_sample() {
                        return ValidBitSize::try_from(valid_bits);
                    }
                }
                // Fallback to the main bits_per_sample
                ValidBitSize::try_from(self.bits_per_sample)
            }
            false => ValidBitSize::try_from(self.bits_per_sample),
        }
    }

    /// Gets the channel mask for extensible formats.
    ///
    /// # Returns
    ///
    /// The channel mask if available, otherwise None
    pub fn channel_mask(&self) -> Option<u32> {
        if self.is_extensible() {
            if let Some(ext_info) = &self.ext_fmt_chunk {
                return ext_info.channel_mask();
            }
        }
        None
    }

    /// Calculates the number of bytes per sample for all channels.
    ///
    /// # Returns
    ///
    /// The number of bytes for a complete frame (all channels)
    pub fn bytes_per_frame(&self) -> u16 {
        self.block_align
    }

    /// Calculates the number of bytes per sample for a single channel.
    ///
    /// # Returns
    ///
    /// The number of bytes for a single channel sample
    pub fn bytes_per_sample(&self) -> u16 {
        self.bits_per_sample / 8
    }

    /// Validates the internal consistency of the format chunk.
    ///
    /// This checks that:
    /// - The block_align value is correctly calculated from channels and bits_per_sample
    /// - The byte_rate value is correctly calculated from sample_rate and block_align
    /// - Extended format fields are consistent if present
    ///
    /// # Returns
    ///
    /// Ok(()) if valid, or an error describing the inconsistency
    pub fn validate(&self) -> Result<(), WaversErrorKind> {
        // Validate block_align
        let expected_block_align = (self.channels * self.bits_per_sample) / 8;
        if self.block_align != expected_block_align {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!(
                    "Block align value inconsistent: expected {} (channels {} * bits_per_sample {} / 8), got {}",
                    expected_block_align, self.channels, self.bits_per_sample, self.block_align
                ),
            }));
        }

        // Validate byte_rate
        let expected_byte_rate = self.sample_rate * (self.block_align as i32);
        if self.byte_rate != expected_byte_rate {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!(
                    "Byte rate value inconsistent: expected {} (sample_rate {} * block_align {}), got {}",
                    expected_byte_rate, self.sample_rate, self.block_align, self.byte_rate
                ),
            }));
        }

        // Validate extended format info if present
        if self.is_extensible() {
            if let Some(ext_info) = &self.ext_fmt_chunk {
                ext_info.validate(self.format, self.bits_per_sample)?;
            } else {
                return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                    chunk_id: FMT,
                    reason: "Format is WaveFormatExtensible but extended format info is missing"
                        .to_string(),
                }));
            }
        }

        Ok(())
    }

    /// Converts the format chunk to its binary representation.
    ///
    /// # Returns
    ///
    /// A byte vector containing the serialized format chunk
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(40); // Max possible size

        // Format code
        result.extend_from_slice(&(self.format as u16).to_le_bytes());

        // Channels
        result.extend_from_slice(&self.channels.to_le_bytes());

        // Sample rate
        result.extend_from_slice(&self.sample_rate.to_le_bytes());

        // Byte rate
        result.extend_from_slice(&self.byte_rate.to_le_bytes());

        // Block align
        result.extend_from_slice(&self.block_align.to_le_bytes());

        // Bits per sample
        result.extend_from_slice(&self.bits_per_sample.to_le_bytes());

        // Extended format info if present
        if self.is_extensible() {
            if let Some(ext_info) = &self.ext_fmt_chunk {
                result.extend_from_slice(&ext_info.as_bytes());
            }
        }

        result
    }

    /// Returns the size of this chunk in bytes.
    ///
    /// # Returns
    ///
    /// The size of the serialized chunk in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self.is_extensible() {
            true => {
                if let Some(ext_info) = &self.ext_fmt_chunk {
                    match ext_info.cb_size {
                        CbSize::Base => 16 + 2,          // Standard size + cb_size field
                        CbSize::Extended => 16 + 2 + 22, // Standard size + cb_size field + extended data
                    }
                } else {
                    16 // Fallback to standard size
                }
            }
            false => 16, // Standard size for non-extensible formats
        }
    }
}

#[cfg(feature = "colored_output")]
use colored::*;

use super::FMT;

impl Display for FmtChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(feature = "colored_output")]
        {
            // Define color theme with colored crate
            let title_style = "Format Chunk".bright_blue().bold();
            let label_style = |s: &str| s.yellow();
            let value_style = |s: String| s.green();
            writeln!(f, "{}", title_style)?;
            writeln!(
                f,
                "  {}: {} ({})",
                label_style("Format"),
                value_style(self.format.to_string()),
                value_style(format!("{}", self.format as u16))
            )?;
            writeln!(
                f,
                "  {}: {}",
                label_style("Channels"),
                value_style(self.channels.to_string())
            )?;
            writeln!(
                f,
                "  {}: {} Hz",
                label_style("Sample Rate"),
                value_style(self.sample_rate.to_string())
            )?;
            writeln!(
                f,
                "  {}: {} bytes/sec",
                label_style("Byte Rate"),
                value_style(self.byte_rate.to_string())
            )?;
            writeln!(
                f,
                "  {}: {} bytes",
                label_style("Block Align"),
                value_style(self.block_align.to_string())
            )?;
            writeln!(
                f,
                "  {}: {} bits",
                label_style("Bits Per Sample"),
                value_style(self.bits_per_sample.to_string())
            )?;
        }

        #[cfg(not(feature = "colored_output"))]
        {
            writeln!(f, "Format Chunk")?;
            writeln!(f, "  Format: {} ({})", self.format, self.format as u16)?;
            writeln!(f, "  Channels: {}", self.channels)?;
            writeln!(f, "  Sample Rate: {} Hz", self.sample_rate)?;
            writeln!(f, "  Byte Rate: {} bytes/sec", self.byte_rate)?;
            writeln!(f, "  Block Align: {} bytes", self.block_align)?;
            writeln!(f, "  Bits Per Sample: {} bits", self.bits_per_sample)?;
        }

        // Display extended format information if present
        if self.is_extensible() {
            if let Some(ext_info) = &self.ext_fmt_chunk {
                #[cfg(feature = "colored_output")]
                {
                    let label_style = |s: &str| s.yellow();
                    let value_style = |s: String| s.green();
                    let section_style = |s: &str| s.magenta().underline();
                    let hex_style = |s: String| s.bright_blue();
                    writeln!(f, "  {}", section_style("Extended Format Information:"))?;
                    writeln!(
                        f,
                        "    {}: {}",
                        label_style("CB Size"),
                        value_style(ext_info.cb_size.to_string())
                    )?;

                    if let Some(ext_data) = &ext_info.extended_data {
                        writeln!(
                            f,
                            "    {}: {}",
                            label_style("Valid Bits Per Sample"),
                            value_style(ext_data.valid_bits_per_sample.to_string())
                        )?;

                        // Include channel mask information
                        let mask_hex = format!("0x{:08X}", ext_data.channel_mask);
                        writeln!(
                            f,
                            "    {}: {}",
                            label_style("Channel Mask"),
                            hex_style(mask_hex)
                        )?;

                        // Try to get the configuration name
                        let config_name = channel_mask::get_config_name(ext_data.channel_mask);
                        if let Some(name) = config_name {
                            writeln!(
                                f,
                                "    {}: {}",
                                label_style("Channel Configuration"),
                                value_style(name.to_string())
                            )?;
                        }

                        // Show channel positions
                        writeln!(
                            f,
                            "    {}: {}",
                            label_style("Channel Positions"),
                            value_style(channel_mask::describe(ext_data.channel_mask))
                        )?;

                        writeln!(
                            f,
                            "    {}: {} ({})",
                            label_style("Sub Format"),
                            value_style(ext_data.sub_format.to_string()),
                            value_style(format!("{}", ext_data.sub_format as u16))
                        )?;

                        // Format GUID with custom coloring
                        let guid_formatted = format!("{:02X?}", ext_data.guid);
                        writeln!(
                            f,
                            "    {}: {}",
                            label_style("GUID"),
                            hex_style(guid_formatted)
                        )?;
                    }
                }

                #[cfg(not(feature = "colored_output"))]
                {
                    writeln!(f, "  Extended Format Information:")?;
                    writeln!(f, "    CB Size: {}", ext_info.cb_size)?;

                    if let Some(ext_data) = &ext_info.extended_data {
                        writeln!(
                            f,
                            "    Valid Bits Per Sample: {}",
                            ext_data.valid_bits_per_sample
                        )?;

                        // Include channel mask information
                        writeln!(f, "    Channel Mask: 0x{:08X}", ext_data.channel_mask)?;

                        // Try to get the configuration name
                        let config_name = channel_mask::get_config_name(ext_data.channel_mask);
                        if let Some(name) = config_name {
                            writeln!(f, "    Channel Configuration: {}", name)?;
                        }

                        // Show channel positions
                        writeln!(
                            f,
                            "    Channel Positions: {}",
                            channel_mask::describe(ext_data.channel_mask)
                        )?;

                        writeln!(
                            f,
                            "    Sub Format: {} ({})",
                            ext_data.sub_format, ext_data.sub_format as u16
                        )?;
                        writeln!(f, "    GUID: {:02X?}", ext_data.guid)?;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Helper function to parse a little-endian u16 value from a byte slice.
fn take_u16(b: &[u8]) -> IResult<&[u8], u16> {
    u16(nom::number::Endianness::Little)(b)
}

/// Helper function to parse a little-endian u32 value from a byte slice.
fn take_u32(b: &[u8]) -> IResult<&[u8], u32> {
    u32(nom::number::Endianness::Little)(b)
}

/// Helper function to parse a little-endian i32 value from a byte slice.
fn take_i32(b: &[u8]) -> IResult<&[u8], i32> {
    i32(nom::number::Endianness::Little)(b)
}

/// Generic function to parse different types of values from a byte slice.
///
/// This function wraps the various `take_*` functions and provides consistent error handling.
///
/// # Arguments
///
/// * `data` - The byte slice to parse from
/// * `parser` - The parsing function to use
/// * `field_name` - The name of the field being parsed (used for error messages)
///
/// # Returns
///
/// A result containing either the parsed value and remaining data, or an error.
fn parse_field<'a, T, F>(
    data: &'a [u8],
    parser: F,
    field_name: &'a str,
) -> Result<(&'a [u8], T), WaversErrorKind>
where
    F: Fn(&'a [u8]) -> IResult<&'a [u8], T>,
{
    let parse_result: Result<(&[u8], T), nom::Err<nom::error::Error<&[u8]>>> = parser(data);

    match parse_result {
        Ok((remaining, value)) => Ok((remaining, value)),
        Err(e) => {
            // Handle parsing error
            let error_message = format!("Error parsing {}: {}", field_name, e);
            Err(WaversErrorKind::Parse(ParseError::Nom {
                offset: data.len() - std::mem::size_of::<T>(),
                msg: error_message,
            }))
        }
    }

    // .map_err(|e| WaversErrorKind::Parse(ParseError::Nom{offset: e.. , msg: format!("Error parsing {}: {}", field_name, e)}))
}

impl TryFrom<&[u8]> for FmtChunk {
    type Error = WaversErrorKind;

    fn try_from(chunk: &[u8]) -> Result<Self, Self::Error> {
        // Verify minimum chunk size
        if chunk.len() < 16 {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!(
                    "Format chunk too small: expected at least 16 bytes, got {} bytes",
                    chunk.len()
                ),
            }));
        }

        let data: &[u8] = chunk;

        // Parse format code
        let (remaining, format_code) = parse_field(data, take_u16, "format code")?;
        let format = FormatCode::from(format_code);

        // Parse channels
        let (remaining, channels) = parse_field(remaining, take_u16, "channels")?;

        // Validate channels
        if channels == 0 {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: "Number of channels cannot be 0".to_string(),
            }));
        }

        // Parse sample rate
        let (remaining, sample_rate) = parse_field(remaining, take_i32, "sample rate")?;

        // Validate sample rate
        if sample_rate <= 0 {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!("Invalid sample rate: {}", sample_rate),
            }));
        }

        // Parse byte rate
        let (remaining, byte_rate) = parse_field(remaining, take_i32, "byte rate")?;

        // Parse block align
        let (remaining, block_align) = parse_field(remaining, take_u16, "block align")?;

        // Parse bits per sample
        let (remaining, bits_per_sample) = parse_field(remaining, take_u16, "bits per sample")?;

        // Validate bits per sample
        if bits_per_sample == 0 {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: "Bits per sample cannot be 0".to_string(),
            }));
        }

        // Verify byte_rate and block_align for consistency
        let expected_block_align = (channels * bits_per_sample) / 8;
        if block_align != expected_block_align {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!(
                    "Block align value inconsistent: expected {} (channels {} * bits_per_sample {} / 8), got {}",
                    expected_block_align, channels, bits_per_sample, block_align
                ),
            }));
        }

        let expected_byte_rate = sample_rate * (block_align as i32);
        if byte_rate != expected_byte_rate {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!(
                    "Byte rate value inconsistent: expected {} (sample_rate {} * block_align {}), got {}",
                    expected_byte_rate, sample_rate, block_align, byte_rate
                ),
            }));
        }

        // Parse extended format chunk if format is WaveFormatExtensible
        // Also verify that the chunk contains enough data for the extended format
        let ext_fmt_chunk = match format {
            FormatCode::WaveFormatExtensible => {
                // Check if we have enough data for at least the cb_size field
                if remaining.len() < 2 {
                    return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                        chunk_id: FMT,
                        reason: format!(
                            "Extended format chunk data too small: expected at least 2 bytes for cb_size, got {} bytes",
                            remaining.len()
                        ),
                    }));
                }

                // Peek at the cb_size value to determine expected data size
                let (_, cb_size_raw) = parse_field(remaining, take_u16, "cb_size peek")?;
                let expected_size = match cb_size_raw {
                    0 => 2,   // Just the cb_size field for Base
                    22 => 24, // CB_Size + Extended data fields (2+2+4+2+14)
                    _ => {
                        return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                            chunk_id: FMT,
                            reason: format!(
                                "Invalid cb_size value in extended format chunk: {}",
                                cb_size_raw
                            ),
                        }));
                    }
                };

                // Verify remaining data size against expected size
                if remaining.len() < expected_size {
                    return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                        chunk_id: FMT,
                        reason: format!(
                            "Extended format chunk data too small: expected {} bytes for cb_size {}, got {} bytes",
                            expected_size,
                            cb_size_raw,
                            remaining.len()
                        ),
                    }));
                }

                Some(ExtFmtChunkInfo::try_from(remaining)?)
            }
            _ => None,
        };

        // Verify that the chunk size from RIFF matches the format and data size
        let expected_chunk_size = match format {
            FormatCode::WaveFormatExtensible => {
                if let Some(ext_info) = &ext_fmt_chunk {
                    match ext_info.cb_size {
                        CbSize::Base => 16 + 2,          // Base size + cb_size field
                        CbSize::Extended => 16 + 2 + 22, // Base size + cb_size field + extended data
                    }
                } else {
                    return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                        chunk_id: FMT,
                        reason:
                            "Format is WaveFormatExtensible but extended format chunk is missing"
                                .to_string(),
                    }));
                }
            }
            _ => 16, // Base size only for non-extensible formats
        };

        if chunk.len() as usize != expected_chunk_size {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!(
                    "Format chunk size mismatch: RIFF chunk indicates {} bytes, but format type requires {} bytes",
                    chunk.len(),
                    expected_chunk_size
                ),
            }));
        }

        // Create and return the FmtChunk
        let fmt_chunk = Self {
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            ext_fmt_chunk,
        };

        Ok(fmt_chunk)
    }
}

/// An enum used to represent the two possible sizes of the extensible format chunk.
///
/// The WAV format specification defines two possible values for the cbSize field:
/// - Base (0): No extended data is present
/// - Extended (22): Full extended data structure is present (22 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CbSize {
    /// Base size (0 bytes of extra data)
    Base = 0,
    /// Extended size (22 bytes of extra data)
    Extended = 22,
}

impl TryFrom<u16> for CbSize {
    type Error = WaversErrorKind;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(CbSize::Base),
            22 => Ok(CbSize::Extended),
            _ => Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!("Error parsing CB Size, invalid value : {}", value),
            })),
        }
    }
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

/// Additional data for extended WAV formats.
///
/// Contains detailed information about the audio format when using
/// the WAVE_FORMAT_EXTENSIBLE format (format code 0xFFFE).
///
/// This extended format allows for:
/// - More precise bit depth specification (valid_bits_per_sample)
/// - Speaker position mapping (channel_mask)
/// - Support for additional format types (sub_format)
/// - Consistent GUID-based format identification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtendedFormatData {
    /// Valid bits per sample. May be less than the container size specified in bits_per_sample.
    /// For example, 20-bit audio in 24-bit containers would have bits_per_sample=24 and valid_bits_per_sample=20.
    pub valid_bits_per_sample: u16,
    /// Bit mask indicating which channels are present in the stream.
    /// Each bit corresponds to a specific speaker position according to the Microsoft
    /// channel position specifications (e.g., bit 0 = front left, bit 1 = front right).
    pub channel_mask: u32,
    /// The actual format of the audio data (when format is WAVE_FORMAT_EXTENSIBLE).
    /// This is equivalent to the format field in the main chunk for non-extensible formats.
    pub sub_format: FormatCode,
    /// GUID identifying the format. For standard formats, this is always EXTENDED_FMT_GUID.
    /// The GUID uniquely identifies the format type and ensures compatibility across systems.
    pub guid: [u8; 14],
}

impl ExtendedFormatData {
    /// Creates a new ExtendedFormatData instance with standard values.
    ///
    /// # Arguments
    ///
    /// * `valid_bits_per_sample` - The number of valid bits per sample
    /// * `channel_mask` - The channel configuration mask
    /// * `sub_format` - The actual format code
    ///
    /// # Returns
    ///
    /// A new ExtendedFormatData instance with the standard GUID
    pub fn new(valid_bits_per_sample: u16, channel_mask: u32, sub_format: FormatCode) -> Self {
        Self {
            valid_bits_per_sample,
            channel_mask,
            sub_format,
            guid: EXTENDED_FMT_GUID,
        }
    }

    /// Creates a new ExtendedFormatData instance with a custom GUID.
    ///
    /// # Arguments
    ///
    /// * `valid_bits_per_sample` - The number of valid bits per sample
    /// * `channel_mask` - The channel configuration mask
    /// * `sub_format` - The actual format code
    /// * `guid` - A custom GUID to use
    ///
    /// # Returns
    ///
    /// A new ExtendedFormatData instance with the provided GUID
    pub fn with_custom_guid(
        valid_bits_per_sample: u16,
        channel_mask: u32,
        sub_format: FormatCode,
        guid: [u8; 14],
    ) -> Self {
        Self {
            valid_bits_per_sample,
            channel_mask,
            sub_format,
            guid,
        }
    }

    /// Convert to bytes representation.
    ///
    /// # Returns
    ///
    /// A byte array containing the serialized data
    pub fn as_bytes(&self) -> [u8; 22] {
        let mut bytes = [0u8; 22];

        // Write valid_bits_per_sample (2 bytes)
        bytes[0..2].copy_from_slice(&self.valid_bits_per_sample.to_le_bytes());

        // Write channel_mask (4 bytes)
        bytes[2..6].copy_from_slice(&self.channel_mask.to_le_bytes());

        // Write sub_format (2 bytes)
        bytes[6..8].copy_from_slice(&(self.sub_format as u16).to_le_bytes());

        // Write GUID (14 bytes)
        bytes[8..22].copy_from_slice(&self.guid);

        bytes
    }

    /// Checks if this extended format data is using the standard GUID.
    ///
    /// # Returns
    ///
    /// true if the GUID matches the standard GUID, false otherwise
    pub fn is_standard_guid(&self) -> bool {
        self.guid == EXTENDED_FMT_GUID
    }
}

impl Display for ExtendedFormatData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Extended Format Data:")?;
        writeln!(f, "  Valid Bits Per Sample: {}", self.valid_bits_per_sample)?;
        writeln!(f, "  Channel Mask: 0x{:08X}", self.channel_mask)?;

        // Show channel positions
        writeln!(
            f,
            "  Channel Positions: {}",
            channel_mask::describe(self.channel_mask)
        )?;

        writeln!(
            f,
            "  Sub Format: {} ({})",
            self.sub_format, self.sub_format as u16
        )?;

        // Note if it's using a non-standard GUID
        if self.is_standard_guid() {
            writeln!(f, "  GUID: Standard")?;
        } else {
            writeln!(f, "  GUID: Custom - {:02X?}", self.guid)?;
        }

        Ok(())
    }
}

/// The extended format chunk of a WAV file.
///
/// This chunk contains additional information about the format of the audio data.
/// It is present when the format code is set to WAVE_FORMAT_EXTENSIBLE (0xFFFE).
/// The extended information allows for more precise specification of the audio format,
/// including valid bits per sample, channel mask, and sub-format.
///
/// The extensible format addresses limitations in the original WAV format:
/// - Support for more than 2 channels with defined speaker positions
/// - Support for bit depths that don't align to byte boundaries
/// - Clearer distinction between container size and actual bit depth
/// - More robust format identification via GUIDs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtFmtChunkInfo {
    /// Size of the extended format chunk in bytes.
    /// This must be either CbSize::Base (0) or CbSize::Extended (22).
    pub cb_size: CbSize,
    /// Optional extended format data, present when cb_size is Extended.
    /// Contains all the additional format information for extensible formats.
    pub extended_data: Option<ExtendedFormatData>,
}

impl ExtFmtChunkInfo {
    /// Creates a new ExtFmtChunkInfo instance.
    ///
    /// # Arguments
    ///
    /// * `cb_size` - Size of the extended format chunk
    /// * `extended_data` - Optional extended format data
    pub fn new(cb_size: CbSize, extended_data: Option<ExtendedFormatData>) -> Self {
        Self {
            cb_size,
            extended_data,
        }
    }

    /// Creates a new ExtFmtChunkInfo with default values for the given format and bits per sample.
    ///
    /// # Arguments
    ///
    /// * `format` - The format code to use
    /// * `bits_per_sample` - The bits per sample value to use
    /// * `channels` - The number of channels (used for channel mask calculation)
    ///
    /// # Returns
    ///
    /// A new ExtFmtChunkInfo instance with appropriate default values
    pub fn create_default(format: FormatCode, bits_per_sample: u16, channels: u16) -> Self {
        match format {
            FormatCode::WaveFormatExtensible => {
                // For extended format, provide reasonable defaults
                let channel_mask = Self::calculate_default_channel_mask(channels);

                Self {
                    cb_size: CbSize::Extended,
                    extended_data: Some(ExtendedFormatData {
                        valid_bits_per_sample: bits_per_sample,
                        channel_mask,
                        sub_format: format,
                        guid: EXTENDED_FMT_GUID,
                    }),
                }
            }
            _ => {
                // For non-extended formats, use the basic structure
                Self {
                    cb_size: CbSize::Base,
                    extended_data: None,
                }
            }
        }
    }

    /// Calculate a default channel mask based on the number of channels.
    ///
    /// This follows standard Windows channel configurations:
    /// - 1 channel: front center
    /// - 2 channels: front left, front right
    /// - 3 channels: front left, front right, front center
    /// - 4 channels: quad configuration
    /// - 5 channels: 5.0 surround
    /// - 6 channels: 5.1 surround
    /// etc.
    ///
    /// # Arguments
    ///
    /// * `channels` - The number of channels
    ///
    /// # Returns
    ///
    /// A channel mask appropriate for the given number of channels
    fn calculate_default_channel_mask(channels: u16) -> u32 {
        match channels {
            1 => 0x4,                 // SPEAKER_FRONT_CENTER
            2 => 0x3,                 // SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT
            3 => 0x7,   // SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER
            4 => 0x33, // SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT
            5 => 0x37, // 5.0 surround
            6 => 0x3F, // 5.1 surround
            7 => 0x13F, // 6.1 surround
            8 => 0x63F, // 7.1 surround
            _ => (1 << channels) - 1, // Generic bit mask with all channels set
        }
    }

    /// Convenience method to access valid_bits_per_sample from extended_data if present.
    pub fn valid_bits_per_sample(&self) -> Option<u16> {
        self.extended_data.map(|data| data.valid_bits_per_sample)
    }

    /// Convenience method to access channel_mask from extended_data if present.
    pub fn channel_mask(&self) -> Option<u32> {
        self.extended_data.map(|data| data.channel_mask)
    }

    /// Convenience method to access sub_format from extended_data if present.
    pub fn sub_format(&self) -> Option<FormatCode> {
        self.extended_data.map(|data| data.sub_format)
    }

    /// Convenience method to access guid from extended_data if present.
    pub fn guid(&self) -> Option<[u8; 14]> {
        self.extended_data.map(|data| data.guid)
    }

    /// Convert to bytes representation.
    ///
    /// This creates a byte array containing the serialized form of the ExtFmtChunkInfo.
    ///
    /// # Returns
    ///
    /// A byte array containing the serialized data
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // Add the cb_size field
        let cb_size_value = self.cb_size as u16;
        result.extend_from_slice(&cb_size_value.to_le_bytes());

        // Add extended data if present
        if let Some(data) = &self.extended_data {
            // Add valid_bits_per_sample
            result.extend_from_slice(&data.valid_bits_per_sample.to_le_bytes());

            // Add channel_mask
            result.extend_from_slice(&data.channel_mask.to_le_bytes());

            // Add sub_format
            let sub_format_value = data.sub_format as u16;
            result.extend_from_slice(&sub_format_value.to_le_bytes());

            // Add GUID
            result.extend_from_slice(&data.guid);
        }

        result
    }

    /// Validates that the extended format information is consistent with the given format and bits_per_sample.
    ///
    /// # Arguments
    ///
    /// * `format` - The format code to validate against
    /// * `bits_per_sample` - The bits per sample value to validate against
    ///
    /// # Returns
    ///
    /// Ok(()) if valid, or an error describing the inconsistency
    pub fn validate(
        &self,
        format: FormatCode,
        bits_per_sample: u16,
    ) -> Result<(), WaversErrorKind> {
        // If format is not extended, ext_fmt_chunk should have cb_size Base
        if format != FormatCode::WaveFormatExtensible && self.cb_size != CbSize::Base {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: format!(
                    "Format code is {:?} but extended format info has cb_size {:?}",
                    format, self.cb_size
                ),
            }));
        }

        // If format is extended, check that we have extended data
        if format == FormatCode::WaveFormatExtensible {
            if self.cb_size != CbSize::Extended {
                return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                    chunk_id: FMT,
                    reason: format!(
                        "Format code is WaveFormatExtensible but cb_size is {:?}",
                        self.cb_size
                    ),
                }));
            }

            if self.extended_data.is_none() {
                return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                    chunk_id: FMT,
                    reason: "Format code is WaveFormatExtensible but extended data is missing"
                        .to_string(),
                }));
            }

            // Validate valid_bits_per_sample
            if let Some(data) = &self.extended_data {
                if data.valid_bits_per_sample > bits_per_sample {
                    return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                        chunk_id: FMT,
                        reason: format!(
                            "valid_bits_per_sample ({}) cannot be greater than bits_per_sample ({})",
                            data.valid_bits_per_sample, bits_per_sample
                        ),
                    }));
                }
            }
        }

        Ok(())
    }
}

impl Display for ExtFmtChunkInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Extended Format Info:")?;
        writeln!(f, "  CB Size: {}", self.cb_size)?;

        if let Some(data) = &self.extended_data {
            writeln!(f, "  Valid Bits Per Sample: {}", data.valid_bits_per_sample)?;
            writeln!(f, "  Channel Mask: 0x{:08X}", data.channel_mask)?;

            // Show channel configuration if it matches a standard layout
            if let Some(config_name) = channel_mask::get_config_name(data.channel_mask) {
                writeln!(f, "  Channel Configuration: {}", config_name)?;
            }

            writeln!(
                f,
                "  Channel Positions: {}",
                channel_mask::describe(data.channel_mask)
            )?;
            writeln!(
                f,
                "  Sub Format: {} ({})",
                data.sub_format, data.sub_format as u16
            )?;

            // Format GUID in a readable way
            let guid_str = data
                .guid
                .iter()
                .map(|b| format!("{:02X}", b))
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(f, "  GUID: {}", guid_str)?;
        } else {
            writeln!(f, "  No extended data (Base format)")?;
        }

        Ok(())
    }
}

impl TryFrom<&[u8]> for ExtFmtChunkInfo {
    type Error = WaversErrorKind;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        // Validate minimum data size
        if data.is_empty() {
            return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                chunk_id: FMT,
                reason: "Extended format chunk data is empty".to_string(),
            }));
        }

        // Parse CB size
        let (data, cb_size_raw) = parse_field(data, take_u16, "cb size")?;
        let cb_size = match CbSize::try_from(cb_size_raw) {
            Ok(size) => size,
            Err(e) => {
                // Provide more context about the error
                return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                    chunk_id: FMT,
                    reason: format!(
                        "Invalid cb_size value {}: {}. Expected values are 0 (Base) or 22 (Extended)",
                        cb_size_raw, e
                    ),
                }));
            }
        };

        // Process remaining fields based on CB size
        let extended_data = match cb_size {
            CbSize::Base => None,
            CbSize::Extended => {
                // Validate we have enough data for the extended fields
                if data.len() < 22 {
                    return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                        chunk_id: FMT,
                        reason: format!(
                            "Extended format chunk data too small: expected 22 bytes for extended data, got {} bytes",
                            data.len()
                        ),
                    }));
                }

                // Parse valid bits per sample
                let (data, valid_bits_per_sample) =
                    parse_field(data, take_u16, "valid bits per sample")?;

                // Validate valid_bits_per_sample
                if valid_bits_per_sample == 0 {
                    return Err(WaversErrorKind::Validate(ValidationError::BitDepth {
                        bits: 0,
                        format: FormatCode::WaveFormatExtensible,
                    }));
                }

                // Parse channel mask
                let (data, channel_mask) = parse_field(data, take_u32, "channel mask")?;

                // Parse sub-format
                let (remaining, sub_format_raw) = parse_field(data, take_u16, "sub-format code")?;
                let sub_format = FormatCode::from(sub_format_raw);

                // Verify we have GUID data
                if remaining.len() < 14 {
                    return Err(WaversErrorKind::Validate(ValidationError::Chunk {
                        chunk_id: FMT,
                        reason: format!(
                            "Missing GUID data in extended format chunk: expected 14 bytes, got {}",
                            remaining.len()
                        ),
                    }));
                }

                // Check if the GUID matches the standard
                let guid = if remaining[0..14] == EXTENDED_FMT_GUID {
                    EXTENDED_FMT_GUID
                } else {
                    // Copy the custom GUID
                    let mut custom_guid = [0u8; 14];
                    custom_guid.copy_from_slice(&remaining[0..14]);

                    // Log a warning that we're using a non-standard GUID
                    // (Assuming there's a logging mechanism in the crate)
                    #[cfg(feature = "log")]
                    log::warn!("Non-standard GUID found in extended format chunk");

                    custom_guid
                };

                Some(ExtendedFormatData {
                    valid_bits_per_sample,
                    channel_mask,
                    sub_format,
                    guid,
                })
            }
        };

        Ok(Self {
            cb_size,
            extended_data,
        })
    }
}

#[cfg(test)]
mod fmt_chunk_tests {
    use super::*;
    use crate::wav_type::ValidBitSize;

    // Basic construction tests
    #[test]
    fn test_new_pcm_fmt_chunk() {
        let fmt = FmtChunk::new_pcm(2, 44100, 16);

        assert_eq!(fmt.format, FormatCode::WavFormatPcm);
        assert_eq!(fmt.channels, 2);
        assert_eq!(fmt.sample_rate, 44100);
        assert_eq!(fmt.bits_per_sample, 16);
        assert_eq!(fmt.block_align, 4); // 2 channels * 16 bits / 8
        assert_eq!(fmt.byte_rate, 176400); // 44100 * 4
        assert!(fmt.ext_fmt_chunk.is_none());
    }

    #[test]
    fn test_new_float_fmt_chunk() {
        let fmt = FmtChunk::new_float(1, 48000, 32);

        assert_eq!(fmt.format, FormatCode::WavFormatIeeeFloat);
        assert_eq!(fmt.channels, 1);
        assert_eq!(fmt.sample_rate, 48000);
        assert_eq!(fmt.bits_per_sample, 32);
        assert_eq!(fmt.block_align, 4); // 1 channel * 32 bits / 8
        assert_eq!(fmt.byte_rate, 192000); // 48000 * 4
        assert!(fmt.ext_fmt_chunk.is_none());
    }

    #[test]
    fn test_new_extensible_fmt_chunk() {
        let fmt = FmtChunk::new_extensible(
            2,
            44100,
            24,
            20,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
        );

        assert_eq!(fmt.format, FormatCode::WaveFormatExtensible);
        assert_eq!(fmt.channels, 2);
        assert_eq!(fmt.sample_rate, 44100);
        assert_eq!(fmt.bits_per_sample, 24);
        assert_eq!(fmt.block_align, 6); // 2 channels * 24 bits / 8
        assert_eq!(fmt.byte_rate, 264600); // 44100 * 6

        assert!(fmt.ext_fmt_chunk.is_some());
        let ext_fmt = fmt.ext_fmt_chunk.unwrap();
        assert_eq!(ext_fmt.cb_size, CbSize::Extended);

        assert!(ext_fmt.extended_data.is_some());
        let ext_data = ext_fmt.extended_data.unwrap();
        assert_eq!(ext_data.valid_bits_per_sample, 20);
        assert_eq!(ext_data.channel_mask, channel_configs::STEREO);
        assert_eq!(ext_data.sub_format, FormatCode::WavFormatPcm);
    }

    // Helper methods tests
    #[test]
    fn test_is_extensible() {
        let fmt_pcm = FmtChunk::new_pcm(2, 44100, 16);
        let fmt_extensible = FmtChunk::new_extensible(
            2,
            44100,
            24,
            20,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
        );

        assert!(!fmt_pcm.is_extensible());
        assert!(fmt_extensible.is_extensible());
    }

    #[test]
    fn test_effective_format() {
        let fmt_pcm = FmtChunk::new_pcm(2, 44100, 16);
        assert_eq!(fmt_pcm.effective_format(), FormatCode::WavFormatPcm);

        let fmt_extensible = FmtChunk::new_extensible(
            2,
            44100,
            24,
            20,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
        );
        assert_eq!(fmt_extensible.effective_format(), FormatCode::WavFormatPcm);

        // Test case with custom sub-format
        let fmt_extensible_float = FmtChunk::new_extensible(
            2,
            44100,
            32,
            32,
            channel_configs::STEREO,
            FormatCode::WavFormatIeeeFloat,
        );
        assert_eq!(
            fmt_extensible_float.effective_format(),
            FormatCode::WavFormatIeeeFloat
        );
    }

    #[test]
    fn test_effective_bits_per_sample() -> WaversResult<()> {
        let fmt_pcm = FmtChunk::new_pcm(2, 44100, 16);
        assert_eq!(fmt_pcm.effective_bits_per_sample()?, ValidBitSize::Bits16);

        let fmt_extensible = FmtChunk::new_extensible(
            2,
            44100,
            24,
            24,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
        );
        assert_eq!(
            fmt_extensible.effective_bits_per_sample()?,
            ValidBitSize::Bits24
        );

        Ok(())
    }

    #[test]
    fn test_channel_mask() {
        let fmt_pcm = FmtChunk::new_pcm(2, 44100, 16);
        assert_eq!(fmt_pcm.channel_mask(), None);

        let fmt_extensible = FmtChunk::new_extensible(
            2,
            44100,
            24,
            20,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
        );
        assert_eq!(fmt_extensible.channel_mask(), Some(channel_configs::STEREO));
    }

    #[test]
    fn test_bytes_per_frame() {
        let fmt = FmtChunk::new_pcm(2, 44100, 16);
        assert_eq!(fmt.bytes_per_frame(), 4); // 2 channels * 16 bits / 8

        let fmt_24bit = FmtChunk::new_pcm(2, 44100, 24);
        assert_eq!(fmt_24bit.bytes_per_frame(), 6); // 2 channels * 24 bits / 8
    }

    #[test]
    fn test_bytes_per_sample() {
        let fmt = FmtChunk::new_pcm(2, 44100, 16);
        assert_eq!(fmt.bytes_per_sample(), 2); // 16 bits / 8

        let fmt_24bit = FmtChunk::new_pcm(2, 44100, 24);
        assert_eq!(fmt_24bit.bytes_per_sample(), 3); // 24 bits / 8
    }

    // Validation tests
    #[test]
    fn test_validate_valid_fmt_chunk() {
        let fmt = FmtChunk::new_pcm(2, 44100, 16);
        assert!(fmt.validate().is_ok());

        let fmt_extensible = FmtChunk::new_extensible(
            2,
            44100,
            24,
            20,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
        );
        assert!(fmt_extensible.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_block_align() {
        let mut fmt = FmtChunk::new_pcm(2, 44100, 16);
        fmt.block_align = 5; // Should be 4

        let result = fmt.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_byte_rate() {
        let mut fmt = FmtChunk::new_pcm(2, 44100, 16);
        fmt.byte_rate = 200000; // Should be 176400

        let result = fmt.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_missing_ext_fmt_chunk() {
        let mut fmt = FmtChunk::new_pcm(2, 44100, 16);
        fmt.format = FormatCode::WaveFormatExtensible;

        let result = fmt.validate();
        assert!(result.is_err());
    }

    // Serialization and deserialization tests
    #[test]
    fn test_to_bytes_and_from_bytes_pcm() {
        let fmt_original = FmtChunk::new_pcm(2, 44100, 16);
        let bytes = fmt_original.to_bytes();

        let fmt_decoded = FmtChunk::try_from(bytes.as_slice()).expect("Failed to decode FmtChunk");

        assert_eq!(fmt_original.format, fmt_decoded.format);
        assert_eq!(fmt_original.channels, fmt_decoded.channels);
        assert_eq!(fmt_original.sample_rate, fmt_decoded.sample_rate);
        assert_eq!(fmt_original.bits_per_sample, fmt_decoded.bits_per_sample);
        assert_eq!(fmt_original.block_align, fmt_decoded.block_align);
        assert_eq!(fmt_original.byte_rate, fmt_decoded.byte_rate);
        assert_eq!(fmt_original.ext_fmt_chunk, fmt_decoded.ext_fmt_chunk);
    }

    #[test]
    fn test_to_bytes_and_from_bytes_extensible() {
        let fmt_original = FmtChunk::new_extensible(
            2,
            44100,
            24,
            20,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
        );
        let bytes = fmt_original.to_bytes();

        let fmt_decoded = FmtChunk::try_from(bytes.as_slice()).expect("Failed to decode FmtChunk");

        assert_eq!(fmt_original.format, fmt_decoded.format);
        assert_eq!(fmt_original.channels, fmt_decoded.channels);
        assert_eq!(fmt_original.sample_rate, fmt_decoded.sample_rate);
        assert_eq!(fmt_original.bits_per_sample, fmt_decoded.bits_per_sample);
        assert_eq!(fmt_original.block_align, fmt_decoded.block_align);
        assert_eq!(fmt_original.byte_rate, fmt_decoded.byte_rate);

        // Check extended format info
        assert!(fmt_decoded.ext_fmt_chunk.is_some());

        let original_ext = fmt_original.ext_fmt_chunk.unwrap();
        let decoded_ext = fmt_decoded.ext_fmt_chunk.unwrap();

        assert_eq!(original_ext.cb_size, decoded_ext.cb_size);

        let original_data = original_ext.extended_data.unwrap();
        let decoded_data = decoded_ext.extended_data.unwrap();

        assert_eq!(
            original_data.valid_bits_per_sample,
            decoded_data.valid_bits_per_sample
        );
        assert_eq!(original_data.channel_mask, decoded_data.channel_mask);
        assert_eq!(original_data.sub_format, decoded_data.sub_format);
        assert_eq!(original_data.guid, decoded_data.guid);
    }

    #[test]
    fn test_size_in_bytes() {
        let fmt_pcm = FmtChunk::new_pcm(2, 44100, 16);
        assert_eq!(fmt_pcm.size_in_bytes(), 16);

        let fmt_extensible = FmtChunk::new_extensible(
            2,
            44100,
            24,
            20,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
        );
        assert_eq!(fmt_extensible.size_in_bytes(), 40); // 16 + 2 + 22
    }

    // Channel mask tests
    #[test]
    fn test_channel_mask_describe() {
        let stereo_desc = channel_mask::describe(channel_configs::STEREO);
        assert!(stereo_desc.contains("Front Left"));
        assert!(stereo_desc.contains("Front Right"));

        let surround_desc = channel_mask::describe(channel_configs::CONFIG_5_1);
        assert!(surround_desc.contains("Front Left"));
        assert!(surround_desc.contains("Front Right"));
        assert!(surround_desc.contains("Front Center"));
        assert!(surround_desc.contains("LFE"));
        assert!(surround_desc.contains("Back Left"));
        assert!(surround_desc.contains("Back Right"));
    }

    #[test]
    fn test_channel_mask_channel_count() {
        assert_eq!(channel_mask::channel_count(channel_configs::MONO), 1);
        assert_eq!(channel_mask::channel_count(channel_configs::STEREO), 2);
        assert_eq!(channel_mask::channel_count(channel_configs::CONFIG_5_1), 6);
        assert_eq!(channel_mask::channel_count(channel_configs::CONFIG_7_1), 8);
    }

    #[test]
    fn test_channel_mask_get_config_name() {
        assert_eq!(
            channel_mask::get_config_name(channel_configs::MONO),
            Some("Mono")
        );
        assert_eq!(
            channel_mask::get_config_name(channel_configs::STEREO),
            Some("Stereo")
        );
        assert_eq!(
            channel_mask::get_config_name(channel_configs::CONFIG_5_1),
            Some("5.1")
        );
        assert_eq!(channel_mask::get_config_name(0x12345), None); // Non-standard configuration
    }

    #[test]
    fn test_channel_mask_from_channel_count() {
        assert_eq!(channel_mask::from_channel_count(1), channel_configs::MONO);
        assert_eq!(channel_mask::from_channel_count(2), channel_configs::STEREO);
        assert_eq!(
            channel_mask::from_channel_count(6),
            channel_configs::CONFIG_5_1
        );

        // Non-standard channel count
        let mask_for_10 = channel_mask::from_channel_count(10);
        assert_eq!(channel_mask::channel_count(mask_for_10), 10);
    }

    #[test]
    fn test_channel_mask_validate() {
        // Valid cases
        assert!(channel_mask::validate(channel_configs::STEREO, 2).is_ok());
        assert!(channel_mask::validate(channel_configs::CONFIG_5_1, 6).is_ok());

        // Invalid cases
        let result = channel_mask::validate(channel_configs::STEREO, 3);
        assert!(result.is_err());

        let result = channel_mask::validate(channel_configs::CONFIG_5_1, 5);
        assert!(result.is_err());
    }

    // ExtendedFormatData tests
    #[test]
    fn test_extended_format_data_new() {
        let data = ExtendedFormatData::new(16, channel_configs::STEREO, FormatCode::WavFormatPcm);

        assert_eq!(data.valid_bits_per_sample, 16);
        assert_eq!(data.channel_mask, channel_configs::STEREO);
        assert_eq!(data.sub_format, FormatCode::WavFormatPcm);
        assert_eq!(data.guid, EXTENDED_FMT_GUID);
    }

    #[test]
    fn test_extended_format_data_with_custom_guid() {
        let custom_guid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let data = ExtendedFormatData::with_custom_guid(
            16,
            channel_configs::STEREO,
            FormatCode::WavFormatPcm,
            custom_guid,
        );

        assert_eq!(data.valid_bits_per_sample, 16);
        assert_eq!(data.channel_mask, channel_configs::STEREO);
        assert_eq!(data.sub_format, FormatCode::WavFormatPcm);
        assert_eq!(data.guid, custom_guid);

        assert!(!data.is_standard_guid());
    }

    #[test]
    fn test_extended_format_data_as_bytes() {
        let data = ExtendedFormatData::new(16, channel_configs::STEREO, FormatCode::WavFormatPcm);
        let bytes = data.as_bytes();

        assert_eq!(bytes.len(), 22);

        // Check valid_bits_per_sample (first 2 bytes)
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 16);

        // Check channel_mask (next 4 bytes)
        assert_eq!(
            u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]),
            channel_configs::STEREO
        );

        // Check sub_format (next 2 bytes)
        assert_eq!(
            u16::from_le_bytes([bytes[6], bytes[7]]),
            FormatCode::WavFormatPcm as u16
        );

        // Check GUID (remaining 14 bytes)
        assert_eq!(&bytes[8..22], &EXTENDED_FMT_GUID);
    }

    // CbSize tests
    #[test]
    fn test_cb_size_try_from() {
        assert_eq!(CbSize::try_from(0).unwrap(), CbSize::Base);
        assert_eq!(CbSize::try_from(22).unwrap(), CbSize::Extended);

        let result = CbSize::try_from(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_cb_size_default() {
        assert_eq!(CbSize::default(), CbSize::Base);
    }

    // ExtFmtChunkInfo tests
    #[test]
    fn test_ext_fmt_chunk_info_new() {
        let info = ExtFmtChunkInfo::new(CbSize::Base, None);

        assert_eq!(info.cb_size, CbSize::Base);
        assert!(info.extended_data.is_none());

        let ext_data =
            ExtendedFormatData::new(16, channel_configs::STEREO, FormatCode::WavFormatPcm);
        let info = ExtFmtChunkInfo::new(CbSize::Extended, Some(ext_data));

        assert_eq!(info.cb_size, CbSize::Extended);
        assert!(info.extended_data.is_some());
        assert_eq!(info.extended_data.unwrap().valid_bits_per_sample, 16);
    }

    #[test]
    fn test_ext_fmt_chunk_info_create_default() {
        // For non-extensible format
        let info = ExtFmtChunkInfo::create_default(FormatCode::WavFormatPcm, 16, 2);

        assert_eq!(info.cb_size, CbSize::Base);
        assert!(info.extended_data.is_none());

        // For extensible format
        let info = ExtFmtChunkInfo::create_default(FormatCode::WaveFormatExtensible, 16, 2);

        assert_eq!(info.cb_size, CbSize::Extended);
        assert!(info.extended_data.is_some());

        let data = info.extended_data.unwrap();
        assert_eq!(data.valid_bits_per_sample, 16);
        assert_eq!(data.channel_mask, channel_configs::STEREO);
        assert_eq!(data.sub_format, FormatCode::WaveFormatExtensible);
        assert_eq!(data.guid, EXTENDED_FMT_GUID);
    }

    #[test]
    fn test_ext_fmt_chunk_info_accessors() {
        let ext_data =
            ExtendedFormatData::new(16, channel_configs::STEREO, FormatCode::WavFormatPcm);
        let info = ExtFmtChunkInfo::new(CbSize::Extended, Some(ext_data));

        assert_eq!(info.valid_bits_per_sample(), Some(16));
        assert_eq!(info.channel_mask(), Some(channel_configs::STEREO));
        assert_eq!(info.sub_format(), Some(FormatCode::WavFormatPcm));
        assert_eq!(info.guid(), Some(EXTENDED_FMT_GUID));

        // Test with no extended data
        let info = ExtFmtChunkInfo::new(CbSize::Base, None);

        assert_eq!(info.valid_bits_per_sample(), None);
        assert_eq!(info.channel_mask(), None);
        assert_eq!(info.sub_format(), None);
        assert_eq!(info.guid(), None);
    }

    #[test]
    fn test_ext_fmt_chunk_info_as_bytes() {
        let ext_data =
            ExtendedFormatData::new(16, channel_configs::STEREO, FormatCode::WavFormatPcm);
        let info = ExtFmtChunkInfo::new(CbSize::Extended, Some(ext_data));

        let bytes = info.as_bytes();

        // Verify bytes
        assert_eq!(bytes.len(), 24); // 2 + 22
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 22); // CbSize::Extended value

        // Check valid_bits_per_sample (next 2 bytes)
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 16);

        // No extended data
        let info = ExtFmtChunkInfo::new(CbSize::Base, None);
        let bytes = info.as_bytes();

        assert_eq!(bytes.len(), 2); // Just cb_size
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 0); // CbSize::Base value
    }

    #[test]
    fn test_ext_fmt_chunk_info_validate() {
        // Valid cases
        let ext_data =
            ExtendedFormatData::new(16, channel_configs::STEREO, FormatCode::WavFormatPcm);
        let info = ExtFmtChunkInfo::new(CbSize::Extended, Some(ext_data));
        assert!(info.validate(FormatCode::WaveFormatExtensible, 16).is_ok());

        // Non-extensible format should have CbSize::Base
        let info = ExtFmtChunkInfo::new(CbSize::Base, None);
        assert!(info.validate(FormatCode::WavFormatPcm, 16).is_ok());

        // Invalid cases

        // Non-extensible format with extended data
        let info = ExtFmtChunkInfo::new(CbSize::Extended, Some(ext_data));
        let result = info.validate(FormatCode::WavFormatPcm, 16);
        assert!(result.is_err());

        // Extensible format with base size
        let info = ExtFmtChunkInfo::new(CbSize::Base, None);
        let result = info.validate(FormatCode::WaveFormatExtensible, 16);
        assert!(result.is_err());

        // Extensible format with missing extended data
        let info = ExtFmtChunkInfo::new(CbSize::Extended, None);
        let result = info.validate(FormatCode::WaveFormatExtensible, 16);
        assert!(result.is_err());

        // Valid bits greater than container bits
        let ext_data =
            ExtendedFormatData::new(24, channel_configs::STEREO, FormatCode::WavFormatPcm);
        let info = ExtFmtChunkInfo::new(CbSize::Extended, Some(ext_data));
        let result = info.validate(FormatCode::WaveFormatExtensible, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_ext_fmt_chunk_info_try_from() {
        // Valid data
        let ext_data =
            ExtendedFormatData::new(16, channel_configs::STEREO, FormatCode::WavFormatPcm);
        let original_info = ExtFmtChunkInfo::new(CbSize::Extended, Some(ext_data));
        let bytes = original_info.as_bytes();

        let decoded_info = ExtFmtChunkInfo::try_from(bytes.as_slice()).expect("Failed to decode");

        assert_eq!(decoded_info.cb_size, original_info.cb_size);
        assert!(decoded_info.extended_data.is_some());

        let decoded_data = decoded_info.extended_data.unwrap();
        let original_data = original_info.extended_data.unwrap();

        assert_eq!(
            decoded_data.valid_bits_per_sample,
            original_data.valid_bits_per_sample
        );
        assert_eq!(decoded_data.channel_mask, original_data.channel_mask);
        assert_eq!(decoded_data.sub_format, original_data.sub_format);

        // Invalid CB size
        let mut corrupt_bytes = bytes.clone();
        corrupt_bytes[0] = 10; // Invalid cb_size value
        corrupt_bytes[1] = 0;

        let result = ExtFmtChunkInfo::try_from(corrupt_bytes.as_slice());
        assert!(result.is_err());

        // Extended size but truncated data
        let mut truncated_bytes = vec![22, 0]; // cb_size = 22 (Extended)
        truncated_bytes.extend_from_slice(&[0, 0, 0, 0]); // Only 6 bytes total, not enough for extended data

        let result = ExtFmtChunkInfo::try_from(truncated_bytes.as_slice());
        assert!(result.is_err());
    }

    // Edge cases and special conditions
    #[test]
    fn test_mono_and_multichannel_configurations() {
        // Mono
        let fmt_mono = FmtChunk::new_pcm(1, 44100, 16);
        assert_eq!(fmt_mono.channels, 1);
        assert_eq!(fmt_mono.block_align, 2); // 1 channel * 16 bits / 8

        // 8 channels
        let fmt_multi = FmtChunk::new_pcm(8, 44100, 16);
        assert_eq!(fmt_multi.channels, 8);
        assert_eq!(fmt_multi.block_align, 16); // 8 channels * 16 bits / 8
    }

    #[test]
    fn test_unusual_bit_depths() {
        // 8-bit PCM
        let fmt_8bit = FmtChunk::new_pcm(2, 44100, 8);
        assert_eq!(fmt_8bit.bits_per_sample, 8);
        assert_eq!(fmt_8bit.block_align, 2); // 2 channels * 8 bits / 8

        // 24-bit PCM
        let fmt_24bit = FmtChunk::new_pcm(2, 44100, 24);
        assert_eq!(fmt_24bit.bits_per_sample, 24);
        assert_eq!(fmt_24bit.block_align, 6); // 2 channels * 24 bits / 8

        // 32-bit float
        let fmt_float = FmtChunk::new_float(2, 44100, 32);
        assert_eq!(fmt_float.bits_per_sample, 32);
        assert_eq!(fmt_float.block_align, 8); // 2 channels * 32 bits / 8
    }
}
