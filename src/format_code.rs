//! Definitions and utilities for WAV format tags.

use std::fmt::Display;

/// An enumeration of known format codes used in the WAV file specification.
///
/// Format codes identify the encoding used for audio data in the `fmt` chunk.
/// These include both legacy and modern extensible representations.
///
/// Unknown or unsupported codes will be mapped to [`FormatCode::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum FormatCode {
    /// PCM (Pulse-code modulation), standard uncompressed integer format.
    WavFormatPcm = 0x0001,

    /// IEEE 32-bit or 64-bit floating point audio.
    WavFormatIeeeFloat = 0x0003,

    /// A-law encoding (companded 8-bit), rare in WAV files.
    WaveFormatAlaw = 0x0006,

    /// μ-law encoding (companded 8-bit), also rare.
    WaveFormatMulaw = 0x0007,

    /// Extended format which defers actual encoding to a subformat GUID.
    ///
    /// Used in conjunction with a subformat field inside the `fmt` chunk.
    WaveFormatExtensible = 0xFFFE,

    /// Placeholder for unrecognised or unsupported format codes.
    Unknown = 0x0000,
}

impl FormatCode {
    /// Returns the format code as native-endian bytes.
    ///
    /// Useful when writing binary WAV headers using the system’s endian layout.
    pub const fn to_ne_bytes(self) -> [u8; 2] {
        (self as u16).to_ne_bytes()
    }

    /// Returns the format code as little-endian bytes.
    ///
    /// WAV files are usually encoded in little-endian format regardless of platform.
    pub const fn to_le_bytes(self) -> [u8; 2] {
        (self as u16).to_le_bytes()
    }

    /// Returns the format code as big-endian bytes.
    ///
    /// This is only necessary in rare cases involving custom formats or testing.
    pub const fn to_be_bytes(self) -> [u8; 2] {
        (self as u16).to_be_bytes()
    }
}

impl Display for FormatCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatCode::WavFormatPcm => write!(f, "WAV_FORMAT_PCM"),
            FormatCode::WavFormatIeeeFloat => write!(f, "WAV_FORMAT_IEEE_FLOAT"),
            FormatCode::WaveFormatAlaw => write!(f, "WAVE_FORMAT_ALAW"),
            FormatCode::WaveFormatMulaw => write!(f, "WAVE_FORMAT_MULAW"),
            FormatCode::WaveFormatExtensible => write!(f, "WAV_EXTENSIBLE_FORMAT"),
            FormatCode::Unknown => write!(f, "Unknown"),
        }
    }
}

impl From<u16> for FormatCode {
    fn from(value: u16) -> Self {
        match value {
            0x0001 => FormatCode::WavFormatPcm,
            0x0003 => FormatCode::WavFormatIeeeFloat,
            0x0006 => FormatCode::WaveFormatAlaw,
            0x0007 => FormatCode::WaveFormatMulaw,
            0xFFFE => FormatCode::WaveFormatExtensible,
            _ => FormatCode::Unknown,
        }
    }
}
