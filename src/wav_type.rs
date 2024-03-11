use std::{
    any::TypeId,
    fmt::{self, Display, Formatter},
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::{i24, WaversError, WaversResult};

const PCM_16_BITS: u16 = (std::mem::size_of::<i16>() * 8) as u16;
const PCM_24_BITS: u16 = (std::mem::size_of::<i24>() * 8) as u16;
const PCM_32_BITS: u16 = (std::mem::size_of::<i32>() * 8) as u16;
const FLOAT_32_BITS: u16 = (std::mem::size_of::<f32>() * 8) as u16;
const FLOAT_64_BITS: u16 = (std::mem::size_of::<f64>() * 8) as u16;

/// An enum representing some of the format codes in the wav file format.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(not(feature = "pyo3"))]
pub enum FormatCode {
    WAV_FORMAT_PCM = 0x0001,
    WAV_FORMAT_IEEE_FLOAT = 0x0003,
    WAVE_FORMAT_ALAW = 0x0006,
    WAVE_FORMAT_MULAW = 0x0007,
    WAVE_FORMAT_EXTENSIBLE = 0xFFFE,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "pyo3")]
#[pyclass]
pub enum FormatCode {
    WAV_FORMAT_PCM = 0x0001,
    WAV_FORMAT_IEEE_FLOAT = 0x0003,
    WAVE_FORMAT_ALAW = 0x0006,
    WAVE_FORMAT_MULAW = 0x0007,
    WAVE_FORMAT_EXTENSIBLE = 0xFFFE,
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
            FormatCode::WAV_FORMAT_PCM => write!(f, "WAV_FORMAT_PCM"),
            FormatCode::WAV_FORMAT_IEEE_FLOAT => write!(f, "WAV_FORMAT_IEEE_FLOAT"),
            FormatCode::WAVE_FORMAT_ALAW => write!(f, "WAVE_FORMAT_ALAW"),
            FormatCode::WAVE_FORMAT_MULAW => write!(f, "WAVE_FORMAT_MULAW"),
            FormatCode::WAVE_FORMAT_EXTENSIBLE => write!(f, "WAV_EXTENSIBLE_FORMAT"),
        }
    }
}

impl TryFrom<u16> for FormatCode {
    type Error = WaversError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0x0001 => Ok(FormatCode::WAV_FORMAT_PCM),
            0x0003 => Ok(FormatCode::WAV_FORMAT_IEEE_FLOAT),
            0x0006 => Ok(FormatCode::WAVE_FORMAT_ALAW),
            0x0007 => Ok(FormatCode::WAVE_FORMAT_MULAW),
            0xFFFE => Ok(FormatCode::WAVE_FORMAT_EXTENSIBLE),
            _ => Err(WaversError::InvalidBitsPerSample(value)),
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
    EPcm16,
    EPcm24,
    EPcm32,
    EFloat32,
    EFloat64,
}

impl WavType {
    /// Converts the WavType to the number of bytes per sample.
    pub const fn n_bytes(&self) -> usize {
        match self {
            WavType::Pcm16 | WavType::EPcm16 => std::mem::size_of::<i16>(),
            WavType::Pcm24 | WavType::EPcm24 => std::mem::size_of::<i24>(),
            WavType::Pcm32 | WavType::EPcm32 => std::mem::size_of::<i32>(),
            WavType::Float32 | WavType::EFloat32 => std::mem::size_of::<f32>(),
            WavType::Float64 | WavType::EFloat64 => std::mem::size_of::<f64>(),
        }
    }

    /// Converts the WavType to the number of bits per sample.
    pub const fn n_bits(&self) -> u16 {
        match self {
            WavType::Pcm16 | WavType::EPcm16 => PCM_16_BITS,
            WavType::Pcm24 | WavType::EPcm24 => PCM_24_BITS,
            WavType::Pcm32 | WavType::EPcm32 => PCM_32_BITS,
            WavType::Float32 | WavType::EFloat32 => FLOAT_32_BITS,
            WavType::Float64 | WavType::EFloat64 => FLOAT_64_BITS,
        }
    }
}

impl Display for WavType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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

pub(crate) const fn format_info_to_wav_type(
    info: (FormatCode, u16, FormatCode),
) -> WaversResult<WavType> {
    Ok(match info {
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
        (FormatCode::WAVE_FORMAT_EXTENSIBLE, FLOAT_32_BITS, FormatCode::WAV_FORMAT_IEEE_FLOAT) => {
            WavType::EFloat32
        }
        (FormatCode::WAVE_FORMAT_EXTENSIBLE, FLOAT_64_BITS, FormatCode::WAV_FORMAT_IEEE_FLOAT) => {
            WavType::EFloat64
        }
        _ => return Err(WaversError::InvalidType(info.0, info.1, info.2)),
    })
}

pub(crate) const fn wav_type_to_format_info(wav_type: WavType) -> (FormatCode, u16, FormatCode) {
    match wav_type {
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
            x if x == TypeId::of::<f32>() => Ok(WavType::EFloat32),
            x if x == TypeId::of::<f64>() => Ok(WavType::EFloat64),
            _ => Err(WaversError::InvalidTypeId(format!("{:?}", value))),
        }
    }
}
