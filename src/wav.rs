
use std::path::Path;

use bytemuck::cast_slice;

use crate::fmt_chunk::FmtChunk;


trait ConvertSamples {
    fn to_i16(self) -> WavData<i16>;
    fn to_i32(self) -> WavData<i32>;
    fn to_f32(self) -> WavData<f32>;
    fn to_f64(self) -> WavData<f64>;
}

trait ConvertSample {
    fn to_i16(self) -> i16;
    fn to_i32(self) -> i32;
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
}

pub struct Wav<T> {
    pub header: WavHeader,
    pub data: WavData<T>,
}

pub struct WavHeader {}

pub struct WavData<T> {
    samples: Box<[T]>
}

impl WavData<u8> {
    pub fn new(samples: &[u8]) -> WavData<u8> {
        WavData {
            samples: Box::from(samples)
        }
    }
}

impl Into<WavData<i16>> for WavData<u8> {
    fn into(self) -> WavData<i16> {
        let data = &(*self.samples);
        WavData { samples: Box::from(cast_slice::<u8, i16>(data)) }
    }
}

impl Into<WavData<i32>> for WavData<u8> {
    fn into(self) -> WavData<i32> {
        let data = &(*self.samples);
        WavData { samples: Box::from(cast_slice::<u8, i32>(data)) }
    }
}

impl Into<WavData<f32>> for WavData<u8> {
    fn into(self) -> WavData<f32> {
        let data = &(*self.samples);
        WavData { samples: Box::from(cast_slice::<u8, f32>(data)) }
    }
}

impl Into<WavData<f64>> for WavData<u8> {
    fn into(self) -> WavData<f64> {
        let data = &(*self.samples);
        WavData { samples: Box::from(cast_slice::<u8, f64>(data)) }
    }
}

impl ConvertSample for i16 {
    ///
    /// Converts an i16 to an i16.
    /// 
    fn to_i16(self) -> i16 {
        self
    }


    ///
    /// Converts an i16 to an i32.
    /// 
    fn to_i32(self) -> i32 {
        self as i32
    }

    ///
    /// Converts an i16 to an f32.
    /// 
    fn to_f32(self) -> f32 {
        (self as f32 / 32768.0).clamp(-1.0, 1.0)
    }

    ///
    /// Converts an i16 to an f64.
    /// 
    fn to_f64(self) -> f64 {
        (self as f64 / 32768.0).clamp(-1.0, 1.0)
    }
}

impl ConvertSample for i32 {
    
    ///
    /// Converts an i32 to an i16.
    /// 
    fn to_i16(self) -> i16 {
        (self >> 16) as i16
    }

    ///
    /// Converts an i32 to an i32.
    /// 
    fn to_i32(self) -> i32 {
        self
    }


    ///
    /// Converts an i32 to an f32.
    /// 
    fn to_f32(self) -> f32 {
        (self as f32 / 2147483648.0).clamp(-1.0, 1.0)
    }

    ///
    /// Converts an i32 to an f64.
    /// 
    fn to_f64(self) -> f64 {
        (self as f64 / 2147483648.0).clamp(-1.0, 1.0)
    }
}

impl ConvertSample for f32 {

    ///
    /// Converts an f32 to an i16.
    /// 
    fn to_i16(self) -> i16 {
        (self * 32768.0).clamp(-32768.0, 32767.0) as i16
    }

    ///
    /// Converts an f32 to an i32.
    /// 
    fn to_i32(self) -> i32 {
        (self * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32
    }

    ///
    /// Converts an f32 to an f32.
    /// 
    fn to_f32(self) -> f32 {
        self
    }

    ///
    /// Converts an f32 to an f64.
    /// 
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ConvertSample for f64 {

    ///
    /// Converts an f64 to an i16.
    /// 
    fn to_i16(self) -> i16 {
        (self * 32768.0).clamp(-32768.0, 32767.0) as i16
    }

    ///
    /// Converts an f64 to an i32.
    /// 
    fn to_i32(self) -> i32 {
        (self * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32
    }

    ///
    /// Converts an f64 to an f32.
    /// 
    fn to_f32(self) -> f32 {
        self as f32
    }

    ///
    /// Converts an f64 to an f64.
    /// 
    fn to_f64(self) -> f64 {
        self
    }
}

impl WavData<i16> {
    pub fn new(samples: &[u8]) -> WavData<i16> {
        WavData {
            samples: Box::from(cast_slice::<u8, i16>(samples))
        }
    }

    pub fn from_file(path: &Path) -> WavData<i16> {
        let file = File::open(fp)?;
        let mut buf_reader = BufReader::new(file);
        let fmt_chunk = FmtChunk::from_buf_reader(&mut buf_reader)?;
    }
}

impl ConvertSamples for WavData<i16> {
    fn to_i16(self) -> WavData<i16> {
        self
    }

    fn to_i32(self) -> WavData<i32> {
        WavData {samples: self.samples.iter().map(|&x| x.to_i32()).collect()}
    }

    fn to_f32(self) -> WavData<f32> {
        WavData {samples: self.samples.iter().map(|&x| x.to_f32()).collect()}
    }

    fn to_f64(self) -> WavData<f64> {
        WavData {samples: self.samples.iter().map(|&x| x.to_f64()).collect()}
    }
}

impl WavData<i32> {
    pub fn new(samples: &[u8]) -> WavData<i32> {
        WavData {
            samples: Box::from(cast_slice::<u8, i32>(samples))
        }
    }
}

impl ConvertSamples for WavData<i32> {
    fn to_i16(self) -> WavData<i16> {
        WavData {samples: self.samples.iter().map(|&x| x.to_i16()).collect()}
    }

    fn to_i32(self) -> WavData<i32> {
        self
    }

    fn to_f32(self) -> WavData<f32> {
        WavData {samples: self.samples.iter().map(|&x| x.to_f32()).collect()}
    }

    fn to_f64(self) -> WavData<f64> {
        WavData {samples: self.samples.iter().map(|&x| x.to_f64()).collect()}
    }
}

impl WavData<f32> {
    pub fn new(samples: &[u8]) -> WavData<f32> {
        WavData {
            samples: Box::from(cast_slice::<u8, f32>(samples))
        }
    }
}

impl ConvertSamples for WavData<f32> {
    fn to_i16(self) -> WavData<i16> {
        WavData {samples: self.samples.iter().map(|&x| x.to_i16()).collect()}
    }

    fn to_i32(self) -> WavData<i32> {
        WavData {samples: self.samples.iter().map(|&x| x.to_i32()).collect()}
    }

    fn to_f32(self) -> WavData<f32> {
        self
    }

    fn to_f64(self) -> WavData<f64> {
        WavData {samples: self.samples.iter().map(|&x| x.to_f64()).collect()}
    }
}

impl WavData<f64> {
    pub fn new(samples: &[u8]) -> WavData<f64> {
        WavData {
            samples: Box::from(cast_slice::<u8, f64>(samples))
        }
    }
}

impl ConvertSamples for WavData<f64> {
    fn to_i16(self) -> WavData<i16> {
        WavData {samples: self.samples.iter().map(|&x| x.to_i16()).collect()}
    }

    fn to_i32(self) -> WavData<i32> {
        WavData {samples: self.samples.iter().map(|&x| x.to_i32()).collect()}
    }

    fn to_f32(self) -> WavData<f32> {
        WavData {samples: self.samples.iter().map(|&x| x.to_f32()).collect()}
    }

    fn to_f64(self) -> WavData<f64> {
        self
    }
}