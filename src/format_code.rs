use std::fmt::Display;

/// An enum representing some of the format codes in the wav file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum FormatCode {
    WavFormatPcm = 0x0001,
    WavFormatIeeeFloat = 0x0003,
    WaveFormatAlaw = 0x0006,
    WaveFormatMulaw = 0x0007,
    WaveFormatExtensible = 0xFFFE,
    Unknown = 0x0000,
}

impl FormatCode {
    /// Convert the format code to a native endian byte array.
    pub const fn to_ne_bytes(self) -> [u8; 2] {
        (self as u16).to_ne_bytes()
    }

    /// Convert the format code to a little endian byte array
    pub const fn to_le_bytes(self) -> [u8; 2] {
        (self as u16).to_le_bytes()
    }

    /// Convert the format code to a big endian byte array.
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
