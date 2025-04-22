//! Types and conversions related to WAV sample types.

use crate::{FormatCode, WaversErrorKind};
use audio_sample::I24;
use std::any::TypeId;
use std::fmt::{Display, Formatter};

/// Number of bytes in a 16‑bit PCM sample.
pub(crate) const PCM_16_BYTES: usize = std::mem::size_of::<i16>();
/// Number of bytes in a 24‑bit PCM sample.
pub(crate) const PCM_24_BYTES: usize = std::mem::size_of::<I24>() - 1; // I24 is stored as i32 but only uses 3 bytes.
/// Number of bytes in a 32‑bit PCM sample.
pub(crate) const PCM_32_BYTES: usize = std::mem::size_of::<i32>();
/// Number of bytes in a 32‑bit float sample.
pub(crate) const FLOAT_32_BYTES: usize = std::mem::size_of::<f32>();
/// Number of bytes in a 64‑bit float sample.
pub(crate) const FLOAT_64_BYTES: usize = std::mem::size_of::<f64>();

/// Set of valid sample bit depths for supported formats.
///
/// Used for validating WAV headers and mapping bit depths to types.
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

    /// Attempts to convert a raw bit depth into a supported [`ValidBitSize`].
    ///
    /// # Errors
    ///
    /// Returns [`WaversErrorKind::Validate`] if the value does not map
    /// to a supported bit depth.
    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            16 => Ok(ValidBitSize::Bits16),
            24 => Ok(ValidBitSize::Bits24),
            32 => Ok(ValidBitSize::Bits32),
            64 => Ok(ValidBitSize::Bits64),
            _ => Err(WaversErrorKind::Validate(
                crate::error::ValidationError::BitDepth {
                    bits: value,
                    format: FormatCode::Unknown, // FormatCode not known in this context
                },
            )),
        }
    }
}

/// Enum representing all supported WAV encodings.
///
/// Includes standard and extensible format variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WavType {
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
    Float64,
    EPcm16,
    EPcm24,
    EPcm32,
    EFloat32,
    EFloat64,
    Unknown,
}

impl WavType {
    /// Maps format triplets (format tag, bit size, subformat) into a [`WavType`].
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

    /// Returns the number of bytes per sample.
    pub const fn n_bytes(&self) -> usize {
        match self {
            Self::Pcm16 | Self::EPcm16 => PCM_16_BYTES,
            Self::Pcm24 | Self::EPcm24 => PCM_24_BYTES,
            Self::Pcm32 | Self::EPcm32 => PCM_32_BYTES,
            Self::Float32 | Self::EFloat32 => FLOAT_32_BYTES,
            Self::Float64 | Self::EFloat64 => FLOAT_64_BYTES,
            Self::Unknown => 0,
        }
    }

    /// Returns the number of bits per sample.
    pub const fn n_bits(&self) -> usize {
        self.n_bytes() * 8
    }

    /// Returns true if this format is integer PCM.
    pub const fn is_pcm(&self) -> bool {
        matches!(
            self,
            Self::Pcm16 | Self::Pcm24 | Self::Pcm32 | Self::EPcm16 | Self::EPcm24 | Self::EPcm32
        )
    }

    /// Returns true if this format is IEEE floating point.
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::Float32 | Self::Float64 | Self::EFloat32 | Self::EFloat64
        )
    }

    /// Returns true if this format is based on the extensible WAV format.
    pub const fn is_extensible(&self) -> bool {
        matches!(
            self,
            Self::EPcm16 | Self::EPcm24 | Self::EPcm32 | Self::EFloat32 | Self::EFloat64
        )
    }

    /// Converts an extensible format into its standard equivalent.
    ///
    /// # Returns
    ///
    /// The standard variant if one exists, else the original value.
    pub const fn to_standard(&self) -> Self {
        match self {
            Self::EPcm16 => Self::Pcm16,
            Self::EPcm24 => Self::Pcm24,
            Self::EPcm32 => Self::Pcm32,
            Self::EFloat32 => Self::Float32,
            Self::EFloat64 => Self::Float64,
            _ => *self,
        }
    }

    /// Converts a standard format into its extensible equivalent.
    ///
    /// # Returns
    ///
    /// The extensible variant if one exists, else the original value.
    pub const fn to_extensible(&self) -> Self {
        match self {
            Self::Pcm16 => Self::EPcm16,
            Self::Pcm24 => Self::EPcm24,
            Self::Pcm32 => Self::EPcm32,
            Self::Float32 => Self::EFloat32,
            Self::Float64 => Self::EFloat64,
            _ => *self,
        }
    }

    /// Creates a WavType from audio characteristics.
    ///
    /// # Arguments
    ///
    /// * `is_pcm` – whether the format is PCM
    /// * `bit_depth` – expected bit depth
    /// * `is_extensible` – whether the format uses extensible encoding
    pub fn from_components(is_pcm: bool, bit_depth: usize, is_extensible: bool) -> Self {
        match (is_pcm, bit_depth, is_extensible) {
            (true, 16, false) => Self::Pcm16,
            (true, 24, false) => Self::Pcm24,
            (true, 32, false) => Self::Pcm32,
            (false, 32, false) => Self::Float32,
            (false, 64, false) => Self::Float64,
            (true, 16, true) => Self::EPcm16,
            (true, 24, true) => Self::EPcm24,
            (true, 32, true) => Self::EPcm32,
            (false, 32, true) => Self::EFloat32,
            (false, 64, true) => Self::EFloat64,
            _ => Self::Unknown,
        }
    }

    /// Returns the maximum valid sample value for the format.
    ///
    /// For float formats, returns `1.0`.
    pub fn max_value(&self) -> f64 {
        match self {
            Self::Pcm16 | Self::EPcm16 => 32767.0,
            Self::Pcm24 | Self::EPcm24 => 8_388_607.0,
            Self::Pcm32 | Self::EPcm32 => 2_147_483_647.0,
            Self::Float32 | Self::EFloat32 => 1.0,
            Self::Float64 | Self::EFloat64 => 1.0,
            Self::Unknown => 0.0,
        }
    }

    /// Returns the minimum valid sample value for the format.
    ///
    /// For float formats, returns `-1.0`.
    pub fn min_value(&self) -> f64 {
        match self {
            Self::Pcm16 | Self::EPcm16 => -32768.0,
            Self::Pcm24 | Self::EPcm24 => -8_388_608.0,
            Self::Pcm32 | Self::EPcm32 => -2_147_483_648.0,
            Self::Float32 | Self::EFloat32 => -1.0,
            Self::Float64 | Self::EFloat64 => -1.0,
            Self::Unknown => 0.0,
        }
    }
}

impl Display for WavType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use WavType::*;
        write!(
            f,
            "{}",
            match self {
                Pcm16 => "PCM_16",
                Pcm24 => "PCM_24",
                Pcm32 => "PCM_32",
                Float32 => "IEEE_FLOAT_32",
                Float64 => "IEEE_FLOAT_64",
                EPcm16 => "EXTENSIBLE_PCM_16",
                EPcm24 => "EXTENSIBLE_PCM_24",
                EPcm32 => "EXTENSIBLE_PCM_32",
                EFloat32 => "EXTENSIBLE_IEEE_FLOAT_32",
                EFloat64 => "EXTENSIBLE_IEEE_FLOAT_64",
                Unknown => "Unknown",
            }
        )
    }
}

impl From<TypeId> for WavType {
    /// Converts from a Rust [`TypeId`] to a corresponding [`WavType`].
    ///
    /// Used primarily in conversions and type-driven dispatch.
    fn from(value: TypeId) -> Self {
        if value == TypeId::of::<i16>() {
            Self::Pcm16
        } else if value == TypeId::of::<I24>() {
            Self::Pcm24
        } else if value == TypeId::of::<i32>() {
            Self::Pcm32
        } else if value == TypeId::of::<f32>() {
            Self::Float32
        } else if value == TypeId::of::<f64>() {
            Self::Float64
        } else {
            Self::Unknown
        }
    }
}
