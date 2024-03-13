use std::{any::TypeId, fmt::Display};

use crate::{i24, WaversError};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// Verified as taking up 2 bytes in memory
pub enum FormatCode {
    WAV_FORMAT_PCM = 1,
    WAV_FORMAT_IEEE_FLOAT = 3,
    WAVE_FORMAT_ALAW = 6,
    WAVE_FORMAT_MULAW = 7,
    WAVE_FORMAT_EXTENSIBLE = 0xFFFE,
}

impl FormatCode {
    pub const fn to_ne_bytes(self) -> [u8; 2] {
        (self as u16).to_ne_bytes()
    }
}

impl Display for FormatCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatCode::WAV_FORMAT_PCM => write!(f, "WAV_FORMAT_PCM"),
            FormatCode::WAV_FORMAT_IEEE_FLOAT => write!(f, "WAV_FORMAT_IEEE_FLOAT"),
            FormatCode::WAVE_FORMAT_ALAW => write!(f, "WAVE_FORMAT_ALAW"),
            FormatCode::WAVE_FORMAT_MULAW => write!(f, "WAVE_FORMAT_MULAW"),
            FormatCode::WAVE_FORMAT_EXTENSIBLE => write!(f, "EXT_FORMAT_CODE"),
        }
    }
}

impl TryFrom<u16> for FormatCode {
    type Error = WaversError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(FormatCode::WAV_FORMAT_PCM),
            3 => Ok(FormatCode::WAV_FORMAT_IEEE_FLOAT),
            6 => Ok(FormatCode::WAVE_FORMAT_ALAW),
            7 => Ok(FormatCode::WAVE_FORMAT_MULAW),
            0xFFFE => Ok(FormatCode::WAVE_FORMAT_EXTENSIBLE),
            _ => Err(WaversError::InvalidType(
                format!("Invalid format code: {}", value).into(),
            )),
        }
    }
}

/// Enum representing the encoding of a wav file.
#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

/// Enum representing the encoding of a wav file.
/// This enum is used when the pyo3 feature is enabled.
#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavType {
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
    Float64,
}

impl WavType {
    pub fn n_bytes(&self) -> usize {
        match self {
            WavType::Pcm16 | WavType::EPcm16 => 2,
            WavType::Pcm24 | WavType::EPcm24 => 3,
            WavType::Pcm32 | WavType::EPcm32 => 4,
            WavType::Float32 | WavType::EFloat32 => 4,
            WavType::Float64 | WavType::EFloat64 => 8,
        }
    }

    pub fn n_bits(&self) -> u16 {
        match self {
            WavType::Pcm16 | WavType::EPcm16 => 16,
            WavType::Pcm24 | WavType::EPcm24 => 24,
            WavType::Pcm32 | WavType::EPcm32 => 32,
            WavType::Float32 | WavType::EFloat32 => 32,
            WavType::Float64 | WavType::EFloat64 => 64,
        }
    }
}

impl std::fmt::Display for WavType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WavType::Pcm16 => write!(f, "PCM_16"),
            WavType::Pcm24 => write!(f, "PCM_24"),
            WavType::Pcm32 => write!(f, "PCM_32"),
            WavType::Float32 => write!(f, "IEEE_FLOAT_32"),
            WavType::Float64 => write!(f, "IEEE_FLOAT_64"),
            WavType::EPcm16 => write!(f, "EXTENSIBLE_PCM_16"),
            WavType::EPcm24 => write!(f, "EXTENSIBLE_PCM_24"),
            WavType::EPcm32 => write!(f, "EXTENSIBLE_PCM_32"),
            WavType::EFloat32 => write!(f, "EXTENSIBLE_IEEE_FLOAT_32"),
            WavType::EFloat64 => write!(f, "EXTENSIBLE_IEEE_FLOAT_64"),
        }
    }
}

const PCM_16_BITS: u16 = (std::mem::size_of::<i16>() * 8) as u16;
const PCM_24_BITS: u16 = 24;
const PCM_32_BITS: u16 = (std::mem::size_of::<i32>() * 8) as u16;
const FLOAT_32_BITS: u16 = (std::mem::size_of::<f32>() * 8) as u16;
const FLOAT_64_BITS: u16 = (std::mem::size_of::<f64>() * 8) as u16;

impl TryFrom<(FormatCode, u16, FormatCode)> for WavType {
    type Error = WaversError;

    fn try_from(value: (FormatCode, u16, FormatCode)) -> Result<Self, Self::Error> {
        Ok(match value {
            (FormatCode::WAV_FORMAT_PCM, PCM_16_BITS, _) => WavType::Pcm16,
            (FormatCode::WAV_FORMAT_PCM, PCM_24_BITS, _) => WavType::Pcm24,
            (FormatCode::WAV_FORMAT_PCM, PCM_32_BITS, _) => WavType::Pcm32,
            (FormatCode::WAV_FORMAT_IEEE_FLOAT, FLOAT_32_BITS, _) => WavType::Float32,
            (FormatCode::WAV_FORMAT_IEEE_FLOAT, FLOAT_64_BITS, _) => WavType::Float64,
            (FormatCode::WAVE_FORMAT_EXTENSIBLE, PCM_16_BITS, FormatCode::WAV_FORMAT_PCM) => {
                WavType::EPcm16
            }
            (FormatCode::WAVE_FORMAT_EXTENSIBLE, PCM_24_BITS, FormatCode::WAV_FORMAT_PCM) => {
                WavType::EPcm24
            }
            (FormatCode::WAVE_FORMAT_EXTENSIBLE, PCM_32_BITS, FormatCode::WAV_FORMAT_PCM) => {
                WavType::EPcm32
            }
            (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                FLOAT_32_BITS,
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
            ) => WavType::EFloat32,
            (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                FLOAT_64_BITS,
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
            ) => WavType::EFloat64,
            _ => {
                return Err(WaversError::InvalidType(
                    format!(
                        "Invalid wav type. Unsupported type {}, and number of bytes per samples {}",
                        value.0, value.1
                    )
                    .into(),
                ))
            }
        })
    }
}

impl From<WavType> for (FormatCode, u16, FormatCode) {
    fn from(value: WavType) -> Self {
        match value {
            WavType::Pcm16 => (
                FormatCode::WAV_FORMAT_PCM,
                PCM_16_BITS,
                FormatCode::WAV_FORMAT_PCM,
            ),
            WavType::Pcm24 => (
                FormatCode::WAV_FORMAT_PCM,
                PCM_24_BITS,
                FormatCode::WAV_FORMAT_PCM,
            ),
            WavType::Pcm32 => (
                FormatCode::WAV_FORMAT_PCM,
                PCM_32_BITS,
                FormatCode::WAV_FORMAT_PCM,
            ),
            WavType::Float32 => (
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
                FLOAT_32_BITS,
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
            ),
            WavType::Float64 => (
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
                FLOAT_64_BITS,
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
            ),
            WavType::EPcm16 => (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                PCM_16_BITS,
                FormatCode::WAV_FORMAT_PCM,
            ),
            WavType::EPcm24 => (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                PCM_24_BITS,
                FormatCode::WAV_FORMAT_PCM,
            ),
            WavType::EPcm32 => (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                PCM_32_BITS,
                FormatCode::WAV_FORMAT_PCM,
            ),
            WavType::EFloat32 => (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                FLOAT_32_BITS,
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
            ),
            WavType::EFloat64 => (
                FormatCode::WAVE_FORMAT_EXTENSIBLE,
                FLOAT_64_BITS,
                FormatCode::WAV_FORMAT_IEEE_FLOAT,
            ),
        }
    }
}

impl From<WavType> for TypeId {
    fn from(value: WavType) -> Self {
        match value {
            WavType::Pcm16 | WavType::EPcm16 => TypeId::of::<i16>(),
            WavType::Pcm24 | WavType::EPcm24 => TypeId::of::<i24>(),
            WavType::Pcm32 | WavType::EPcm32 => TypeId::of::<i32>(),
            WavType::Float32 | WavType::EFloat32 => TypeId::of::<f32>(),
            WavType::Float64 | WavType::EFloat64 => TypeId::of::<f64>(),
        }
    }
}

impl TryFrom<TypeId> for WavType {
    type Error = WaversError;

    fn try_from(value: TypeId) -> Result<Self, Self::Error> {
        match value {
            x if x == TypeId::of::<i16>() => Ok(WavType::Pcm16),
            x if x == TypeId::of::<i24>() => Ok(WavType::Pcm24),
            x if x == TypeId::of::<i32>() => Ok(WavType::Pcm32),
            x if x == TypeId::of::<f32>() => Ok(WavType::Float32),
            x if x == TypeId::of::<f64>() => Ok(WavType::Float64),
            _ => Err(WaversError::InvalidType(
                format!("Invalid type id: {:?}", value).into(),
            )),
        }
    }
}
