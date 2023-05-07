

///
/// A SampleType is a type that can be used to represent a single sample. Allows for the passing of samples in a generic way,
/// while also allowing for the conversion between different sample types.
///
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleType {
    I16(i16),
    I32(i32),
    F32(f32),
    F64(f64),
}

impl SampleType {
    #[inline(always)]
    pub fn to_i16(&self) -> i16 {
        match self {
            SampleType::I16(sample) => *sample,
            SampleType::I32(sample) => sample.convert_to_i16(),
            SampleType::F32(sample) => sample.convert_to_i16(),
            SampleType::F64(sample) => sample.convert_to_i16(),
        }
    }
    #[inline(always)]
    pub fn to_i32(&self) -> i32 {
        match self {
            SampleType::I16(sample) => sample.convert_to_i32(),
            SampleType::I32(sample) => *sample,
            SampleType::F32(sample) => sample.convert_to_i32(),
            SampleType::F64(sample) => sample.convert_to_i32(),
        }
    }
    #[inline(always)]
    pub fn to_f32(&self) -> f32 {
        match self {
            SampleType::I16(sample) => sample.convert_to_f32(),
            SampleType::I32(sample) => sample.convert_to_f32(),
            SampleType::F32(sample) => *sample,
            SampleType::F64(sample) => sample.convert_to_f32(),
        }
    }
    #[inline(always)]
    pub fn to_f64(&self) -> f64 {
        match self {
            SampleType::I16(sample) => sample.convert_to_f64(),
            SampleType::I32(sample) => sample.convert_to_f64(),
            SampleType::F32(sample) => sample.convert_to_f64(),
            SampleType::F64(sample) => *sample,
        }
    }

    #[inline(always)]
    pub fn size_of_underlying(&self) -> usize {
        match self {
            SampleType::I16(_) => std::mem::size_of::<i16>(),
            SampleType::I32(_) => std::mem::size_of::<i32>(),
            SampleType::F32(_) => std::mem::size_of::<f32>(),
            SampleType::F64(_) => std::mem::size_of::<f64>(),
        }
    }
}

impl std::fmt::Display for SampleType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SampleType::I16(sample) => write!(f, "{}", sample),
            SampleType::I32(sample) => write!(f, "{}", sample),
            SampleType::F32(sample) => write!(f, "{}", sample),
            SampleType::F64(sample) => write!(f, "{}", sample),
        }
    }
}

impl PartialEq<i16> for SampleType {
    fn eq(&self, other: &i16) -> bool {
        match self {
            SampleType::I16(sample) => sample == other,
            SampleType::I32(sample) => sample.convert_to_i16() == *other,
            SampleType::F32(sample) => sample.convert_to_i16() == *other,
            SampleType::F64(sample) => sample.convert_to_i16() == *other,
        }
    }
}

impl PartialEq<i32> for SampleType {
    fn eq(&self, other: &i32) -> bool {
        match self {
            SampleType::I16(sample) => sample.convert_to_i32() == *other,
            SampleType::I32(sample) => sample == other,
            SampleType::F32(sample) => sample.convert_to_i32() == *other,
            SampleType::F64(sample) => sample.convert_to_i32() == *other,
        }
    }
}

impl PartialEq<f32> for SampleType {
    fn eq(&self, other: &f32) -> bool {
        match self {
            SampleType::I16(sample) => sample.convert_to_f32() == *other,
            SampleType::I32(sample) => sample.convert_to_f32() == *other,
            SampleType::F32(sample) => sample == other,
            SampleType::F64(sample) => sample.convert_to_f32() == *other,
        }
    }
}

impl PartialEq<f64> for SampleType {
    fn eq(&self, other: &f64) -> bool {
        match self {
            SampleType::I16(sample) => sample.convert_to_f64() == *other,
            SampleType::I32(sample) => sample.convert_to_f64() == *other,
            SampleType::F32(sample) => sample.convert_to_f64() == *other,
            SampleType::F64(sample) => sample == other,
        }
    }
}

impl SampleType {
    
    pub fn to_le_bytes(self) -> Vec<u8> {
        match self {
            SampleType::I16(sample) => sample.to_le_bytes().to_vec(),
            SampleType::I32(sample) => sample.to_le_bytes().to_vec(),
            SampleType::F32(sample) => sample.to_le_bytes().to_vec(),
            SampleType::F64(sample) => sample.to_le_bytes().to_vec(),
        }
    }

    pub fn to_ne_bytes(self) -> Vec<u8> {
        match self {
            SampleType::I16(sample) => sample.to_ne_bytes().to_vec(),
            SampleType::I32(sample) => sample.to_ne_bytes().to_vec(),
            SampleType::F32(sample) => sample.to_ne_bytes().to_vec(),
            SampleType::F64(sample) => sample.to_ne_bytes().to_vec(),
        }
    }

    #[inline(always)]
    pub fn convert_to(self, as_type: SampleType) -> SampleType {
        match as_type {
            SampleType::I16(_) => SampleType::I16(self.convert_to_i16()),
            SampleType::I32(_) => SampleType::I32(self.convert_to_i32()),
            SampleType::F32(_) => SampleType::F32(self.convert_to_f32()),
            SampleType::F64(_) => SampleType::F64(self.convert_to_f64()),
        }
    }
}

impl From<i16> for SampleType {
    fn from(sample: i16) -> Self {
        SampleType::I16(sample)
    }
}

impl From<i32> for SampleType {
    fn from(sample: i32) -> Self {
        SampleType::I32(sample)
    }
}

impl From<f32> for SampleType {
    fn from(sample: f32) -> Self {
        SampleType::F32(sample)
    }
}

impl From<f64> for SampleType {
    fn from(sample: f64) -> Self {
        SampleType::F64(sample)
    }
}
///
/// Trait to convert between ``SampleType`` variants
///
pub trait Sample {
    fn convert_to_i16(self) -> i16;
    fn convert_to_i32(self) -> i32;
    fn convert_to_f32(self) -> f32;
    fn convert_to_f64(self) -> f64;
}

impl Sample for SampleType {
    #[inline(always)]
    fn convert_to_i16(self) -> i16 {
        match self {
            SampleType::I16(sample) => sample,
            SampleType::I32(sample) => sample.convert_to_i16(),
            SampleType::F32(sample) => sample.convert_to_i16(),
            SampleType::F64(sample) => sample.convert_to_i16(),
        }
    }
    #[inline(always)]
    fn convert_to_i32(self) -> i32 {
        match self {
            SampleType::I16(sample) => sample.convert_to_i32(),
            SampleType::I32(sample) => sample,
            SampleType::F32(sample) => sample.convert_to_i32(),
            SampleType::F64(sample) => sample.convert_to_i32(),
        }
    }
    #[inline(always)]
    fn convert_to_f32(self) -> f32 {
        match self {
            SampleType::I16(sample) => sample.convert_to_f32(),
            SampleType::I32(sample) => sample.convert_to_f32(),
            SampleType::F32(sample) => sample,
            SampleType::F64(sample) => sample.convert_to_f32(),
        }
    }
    #[inline(always)]
    fn convert_to_f64(self) -> f64 {
        match self {
            SampleType::I16(sample) => sample.convert_to_f64(),
            SampleType::I32(sample) => sample.convert_to_f64(),
            SampleType::F32(sample) => sample.convert_to_f64(),
            SampleType::F64(sample) => sample,
        }
    }
}

impl Sample for i16 {
    #[inline(always)]
    fn convert_to_i16(self) -> i16 {
        self
    }
    #[inline(always)]
    fn convert_to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn convert_to_f32(self) -> f32 {
        (self as f32 / 32768.0).clamp(-1.0, 1.0)
    }
    #[inline(always)]
    fn convert_to_f64(self) -> f64 {
        (self as f64 / 32768.0).clamp(-1.0, 1.0)
    }
}

impl Sample for i32 {
    #[inline(always)]
    fn convert_to_i16(self) -> i16 {
        (self >> 16) as i16
    }
    #[inline(always)]
    fn convert_to_i32(self) -> i32 {
        self
    }
    #[inline(always)]
    fn convert_to_f32(self) -> f32 {
        (self as f32 / 2147483648.0).clamp(-1.0, 1.0)
    }
    #[inline(always)]
    fn convert_to_f64(self) -> f64 {
        (self as f64 / 2147483648.0).clamp(-1.0, 1.0)
    }
}

impl Sample for f32 {
    fn convert_to_i16(self) -> i16 {
        (self * 32768.0).clamp(-32768.0, 32767.0) as i16
    }
    #[inline(always)]
    fn convert_to_i32(self) -> i32 {
        (self * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32
    }
    #[inline(always)]
    fn convert_to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn convert_to_f64(self) -> f64 {
        self as f64
    }
}

impl Sample for f64 {
    #[inline(always)]
    fn convert_to_i16(self) -> i16 {
        (self * 32768.0).clamp(-32768.0, 32767.0) as i16
    }
    #[inline(always)]
    fn convert_to_i32(self) -> i32 {
        (self * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32
    }
    #[inline(always)]
    fn convert_to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn convert_to_f64(self) -> f64 {
        self
    }
}
