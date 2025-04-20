use crate::{FormatCode, WaversErrorKind};
use audio_sample::I24;
use std::any::TypeId;
use std::fmt::{Display, Formatter};

pub(crate) const PCM_16_BYTES: usize = std::mem::size_of::<i16>();
pub(crate) const PCM_24_BYTES: usize = std::mem::size_of::<I24>() - 1; // -1 since i24 is 3 bytes, but internally it is an i32
pub(crate) const PCM_32_BYTES: usize = std::mem::size_of::<i32>();
pub(crate) const FLOAT_32_BYTES: usize = std::mem::size_of::<f32>();
pub(crate) const FLOAT_64_BYTES: usize = std::mem::size_of::<f64>();

#[repr(u16)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum ValidBitSize {
    Bits16 = (std::mem::size_of::<i16>() * 8) as u16,
    Bits24 = ((std::mem::size_of::<I24>() - 1) * 8) as u16,
    Bits32 = (std::mem::size_of::<f32>() * 8) as u16,
    Bits64 = (std::mem::size_of::<f64>() * 8) as u16,
}

impl TryFrom<u16> for ValidBitSize {
    type Error = WaversErrorKind;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            16 => Ok(ValidBitSize::Bits16),
            24 => Ok(ValidBitSize::Bits24),
            32 => Ok(ValidBitSize::Bits32),
            64 => Ok(ValidBitSize::Bits64),
            _ => Err(WaversErrorKind::InvalidBitSize(value)),
        }
    }
}

/// Enum representing the encoding of a WAV file.
///
/// This enum distinguishes between standard WAV formats and extended formats
/// (those using the WAVE_FORMAT_EXTENSIBLE format code). It covers the common
/// PCM integer formats (16, 24, and 32 bit) as well as IEEE floating point
/// formats (32 and 64 bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WavType {
    /// 16-bit signed PCM format (standard)
    Pcm16,
    /// 24-bit signed PCM format (standard)
    Pcm24,
    /// 32-bit signed PCM format (standard)
    Pcm32,
    /// 32-bit IEEE floating point format (standard)
    Float32,
    /// 64-bit IEEE floating point format (standard)
    Float64,
    /// 16-bit signed PCM format (extensible)
    EPcm16,
    /// 24-bit signed PCM format (extensible)
    EPcm24,
    /// 32-bit signed PCM format (extensible)
    EPcm32,
    /// 32-bit IEEE floating point format (extensible)
    EFloat32,
    /// 64-bit IEEE floating point format (extensible)
    EFloat64,
    Unknown,
}

impl WavType {
    /// Converts a tuple of format codes and bits per sample to a WavType.
    pub const fn format_info_to_wav_type(info: (FormatCode, ValidBitSize, FormatCode)) -> WavType {
        match info {
            (FormatCode::WavFormatPcm, ValidBitSize::Bits16, FormatCode::WavFormatPcm) => {
                WavType::Pcm16
            }
            (FormatCode::WavFormatPcm, ValidBitSize::Bits24, FormatCode::WavFormatPcm) => {
                WavType::Pcm24
            }
            (FormatCode::WavFormatPcm, ValidBitSize::Bits32, FormatCode::WavFormatPcm) => {
                WavType::Pcm32
            }
            (
                FormatCode::WavFormatIeeeFloat,
                ValidBitSize::Bits32,
                FormatCode::WavFormatIeeeFloat,
            ) => WavType::Float32,
            (
                FormatCode::WavFormatIeeeFloat,
                ValidBitSize::Bits64,
                FormatCode::WavFormatIeeeFloat,
            ) => WavType::Float64,
            (FormatCode::WaveFormatExtensible, ValidBitSize::Bits16, FormatCode::WavFormatPcm) => {
                WavType::EPcm16
            }
            (FormatCode::WaveFormatExtensible, ValidBitSize::Bits24, FormatCode::WavFormatPcm) => {
                WavType::EPcm24
            }
            (FormatCode::WaveFormatExtensible, ValidBitSize::Bits32, FormatCode::WavFormatPcm) => {
                WavType::EPcm32
            }
            (
                FormatCode::WaveFormatExtensible,
                ValidBitSize::Bits32,
                FormatCode::WavFormatIeeeFloat,
            ) => WavType::EFloat32,
            (
                FormatCode::WaveFormatExtensible,
                ValidBitSize::Bits64,
                FormatCode::WavFormatIeeeFloat,
            ) => WavType::EFloat64,
            _ => WavType::Unknown,
        }
    }

    /// Converts the WavType to the number of bytes per sample.
    ///
    /// # Returns
    ///
    /// The number of bytes used by each sample of this format
    pub const fn n_bytes(&self) -> usize {
        match self {
            WavType::Pcm16 | WavType::EPcm16 => PCM_16_BYTES,
            WavType::Pcm24 | WavType::EPcm24 => PCM_24_BYTES,
            WavType::Pcm32 | WavType::EPcm32 => PCM_32_BYTES,
            WavType::Float32 | WavType::EFloat32 => FLOAT_32_BYTES,
            WavType::Float64 | WavType::EFloat64 => FLOAT_64_BYTES,
            WavType::Unknown => 0,
        }
    }

    /// Converts the WavType to the number of bits per sample.
    ///
    /// # Returns
    ///
    /// The number of bits used by each sample of this format
    pub const fn n_bits(&self) -> usize {
        self.n_bytes() * 8
    }

    /// Checks if this format is PCM (integer) audio.
    ///
    /// # Returns
    ///
    /// `true` if this is a PCM format, `false` for floating point
    pub const fn is_pcm(&self) -> bool {
        matches!(
            self,
            WavType::Pcm16
                | WavType::Pcm24
                | WavType::Pcm32
                | WavType::EPcm16
                | WavType::EPcm24
                | WavType::EPcm32
        )
    }

    /// Checks if this format is IEEE floating point audio.
    ///
    /// # Returns
    ///
    /// `true` if this is a floating point format, `false` for PCM
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            WavType::Float32 | WavType::Float64 | WavType::EFloat32 | WavType::EFloat64
        )
    }

    /// Checks if this format uses the extensible wave format.
    ///
    /// # Returns
    ///
    /// `true` if this is an extensible format, `false` for standard format
    pub const fn is_extensible(&self) -> bool {
        matches!(
            self,
            WavType::EPcm16
                | WavType::EPcm24
                | WavType::EPcm32
                | WavType::EFloat32
                | WavType::EFloat64
        )
    }

    /// Gets the corresponding standard format for an extensible format.
    ///
    /// If this is already a standard format, it returns itself.
    ///
    /// # Returns
    ///
    /// The equivalent standard format
    pub const fn to_standard(&self) -> Self {
        match self {
            WavType::EPcm16 => WavType::Pcm16,
            WavType::EPcm24 => WavType::Pcm24,
            WavType::EPcm32 => WavType::Pcm32,
            WavType::EFloat32 => WavType::Float32,
            WavType::EFloat64 => WavType::Float64,
            _ => *self, // Already a standard format
        }
    }

    /// Gets the corresponding extensible format for a standard format.
    ///
    /// If this is already an extensible format, it returns itself.
    ///
    /// # Returns
    ///
    /// The equivalent extensible format
    pub const fn to_extensible(&self) -> Self {
        match self {
            WavType::Pcm16 => WavType::EPcm16,
            WavType::Pcm24 => WavType::EPcm24,
            WavType::Pcm32 => WavType::EPcm32,
            WavType::Float32 => WavType::EFloat32,
            WavType::Float64 => WavType::EFloat64,
            _ => *self, // Already an extensible format
        }
    }

    /// Creates a WavType from its components.
    ///
    /// # Arguments
    ///
    /// * `is_pcm` - Whether this is PCM (true) or floating point (false)
    /// * `bit_depth` - The bit depth (16, 24, 32 for PCM; 32, 64 for float)
    /// * `is_extensible` - Whether this uses the extensible format
    ///
    /// # Returns
    ///
    /// The corresponding WavType, or None if the combination is invalid
    pub fn from_components(is_pcm: bool, bit_depth: usize, is_extensible: bool) -> Self {
        match (is_pcm, bit_depth, is_extensible) {
            (true, 16, false) => WavType::Pcm16,
            (true, 24, false) => WavType::Pcm24,
            (true, 32, false) => WavType::Pcm32,
            (false, 32, false) => WavType::Float32,
            (false, 64, false) => WavType::Float64,
            (true, 16, true) => WavType::EPcm16,
            (true, 24, true) => WavType::EPcm24,
            (true, 32, true) => WavType::EPcm32,
            (false, 32, true) => WavType::EFloat32,
            (false, 64, true) => WavType::EFloat64,
            _ => WavType::Unknown, // Invalid combination
        }
    }

    /// Gets the maximum valid sample value for PCM formats.
    ///
    /// # Returns
    ///
    /// The maximum sample value as f64, or 1.0 for floating point formats
    pub fn max_value(&self) -> f64 {
        match self {
            WavType::Pcm16 | WavType::EPcm16 => 32767.0, // 2^15 - 1
            WavType::Pcm24 | WavType::EPcm24 => 8_388_607.0, // 2^23 - 1
            WavType::Pcm32 | WavType::EPcm32 => 2_147_483_647.0, // 2^31 - 1
            WavType::Float32 | WavType::EFloat32 => 1.0,
            WavType::Float64 | WavType::EFloat64 => 1.0,
            WavType::Unknown => 0.0,
        }
    }

    /// Gets the minimum valid sample value for PCM formats.
    ///
    /// # Returns
    ///
    /// The minimum sample value as f64, or -1.0 for floating point formats
    pub fn min_value(&self) -> f64 {
        match self {
            WavType::Pcm16 | WavType::EPcm16 => -32768.0, // -2^15
            WavType::Pcm24 | WavType::EPcm24 => -8_388_608.0, // -2^23
            WavType::Pcm32 | WavType::EPcm32 => -2_147_483_648.0, // -2^31
            WavType::Float32 | WavType::EFloat32 => -1.0,
            WavType::Float64 | WavType::EFloat64 => -1.0,
            WavType::Unknown => 0.0,
        }
    }
}

impl Display for WavType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WavType::Pcm16 => writeln!(f, "PCM_16"),
            WavType::Pcm24 => writeln!(f, "PCM_24"),
            WavType::Pcm32 => writeln!(f, "PCM_32"),
            WavType::Float32 => writeln!(f, "IEEE_FLOAT_32"),
            WavType::Float64 => writeln!(f, "IEEE_FLOAT_64"),
            WavType::EPcm16 => writeln!(f, "EXTENSIBLE_PCM_16"),
            WavType::EPcm24 => writeln!(f, "EXTENSIBLE_PCM_24"),
            WavType::EPcm32 => writeln!(f, "EXTENSIBLE_PCM_32"),
            WavType::EFloat32 => writeln!(f, "EXTENSIBLE_IEEE_FLOAT_32"),
            WavType::EFloat64 => writeln!(f, "EXTENSIBLE_IEEE_FLOAT_64"),
            WavType::Unknown => writeln!(f, "Unknown"),
        }
    }
}
impl From<TypeId> for WavType {
    fn from(value: TypeId) -> Self {
        match value {
            x if x == TypeId::of::<i16>() => WavType::Pcm16,
            x if x == TypeId::of::<I24>() => WavType::Pcm24,
            x if x == TypeId::of::<i32>() => WavType::Pcm32,
            x if x == TypeId::of::<f32>() => WavType::Float32, // Stuck with this for now instead of EFloat since a lot of players do no support the extensible format
            x if x == TypeId::of::<f64>() => WavType::Float64,
            _ => WavType::Unknown,
        }
    }
}
