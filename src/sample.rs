use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sample {
    I16(i16),
    I32(i32),
    F32(f32),
    F64(f64),
}

impl Sample {
    pub fn to_le_bytes(self) -> Vec<u8> {
        match self {
            Sample::I16(sample) => sample.to_le_bytes().to_vec(),
            Sample::I32(sample) => sample.to_le_bytes().to_vec(),
            Sample::F32(sample) => sample.to_le_bytes().to_vec(),
            Sample::F64(sample) => sample.to_le_bytes().to_vec(),
        }
    }

    pub fn to_ne_bytes(self) -> Vec<u8> {
        match self {
            Sample::I16(sample) => sample.to_ne_bytes().to_vec(),
            Sample::I32(sample) => sample.to_ne_bytes().to_vec(),
            Sample::F32(sample) => sample.to_ne_bytes().to_vec(),
            Sample::F64(sample) => sample.to_ne_bytes().to_vec(),
        }
    }

    #[inline(always)]
    pub fn size_of_underlying(&self) -> usize {
        match self {
            Sample::I16(_) => std::mem::size_of::<i16>(),
            Sample::I32(_) => std::mem::size_of::<i32>(),
            Sample::F32(_) => std::mem::size_of::<f32>(),
            Sample::F64(_) => std::mem::size_of::<f64>(),
        }
    }
}

impl Add for Sample {
    type Output = Sample;

    fn add(self, rhs: Self) -> Self::Output {
        match self {
            Sample::I16(x) => Sample::I16(x + rhs.as_i16()),
            Sample::I32(x) => Sample::I32(x + rhs.as_i32()),
            Sample::F32(x) => Sample::F32(x + rhs.as_f32()),
            Sample::F64(x) => Sample::F64(x + rhs.as_f64()),
        }
    }
}

impl Sub for Sample {
    type Output = Sample;

    fn sub(self, rhs: Self) -> Self::Output {
        match self {
            Sample::I16(x) => Sample::I16(x - rhs.as_i16()),
            Sample::I32(x) => Sample::I32(x - rhs.as_i32()),
            Sample::F32(x) => Sample::F32(x - rhs.as_f32()),
            Sample::F64(x) => Sample::F64(x - rhs.as_f64()),
        }
    }
}

impl Mul for Sample {
    type Output = Sample;

    fn mul(self, rhs: Self) -> Self::Output {
        match self {
            Sample::I16(x) => Sample::I16(x * rhs.as_i16()),
            Sample::I32(x) => Sample::I32(x * rhs.as_i32()),
            Sample::F32(x) => Sample::F32(x * rhs.as_f32()),
            Sample::F64(x) => Sample::F64(x * rhs.as_f64()),
        }
    }
}

impl Div for Sample {
    type Output = Sample;

    fn div(self, rhs: Self) -> Self::Output {
        match self {
            Sample::I16(x) => Sample::I16(x / rhs.as_i16()),
            Sample::I32(x) => Sample::I32(x / rhs.as_i32()),
            Sample::F32(x) => Sample::F32(x / rhs.as_f32()),
            Sample::F64(x) => Sample::F64(x / rhs.as_f64()),
        }
    }
}

impl Neg for Sample {
    type Output = Sample;

    fn neg(self) -> Self::Output {
        match self {
            Sample::I16(x) => Sample::I16(-x),
            Sample::I32(x) => Sample::I32(-x),
            Sample::F32(x) => Sample::F32(-x),
            Sample::F64(x) => Sample::F64(-x),
        }
    }
}

pub trait AudioConversion {
    fn as_i16(self) -> i16;
    fn as_i32(self) -> i32;
    fn as_f32(self) -> f32;
    fn as_f64(self) -> f64;
    fn as_type(self, as_type: Sample) -> Sample
    where
        Self: Sized,
    {
        match as_type {
            Sample::I16(_) => Sample::I16(self.as_i16()),
            Sample::I32(_) => Sample::I32(self.as_i32()),
            Sample::F32(_) => Sample::F32(self.as_f32()),
            Sample::F64(_) => Sample::F64(self.as_f64()),
        }
    }
}

pub trait IterAudioConversion {
    fn as_i16(&mut self) -> Vec<i16>;
    fn as_i32(&mut self) -> Vec<i32>;
    fn as_f32(&mut self) -> Vec<f32>;
    fn as_f64(&mut self) -> Vec<f64>;
    fn as_i16_samples(&mut self) -> Vec<Sample> {
        self.as_i16()
            .iter()
            .map(|sample| Sample::I16(*sample))
            .collect::<Vec<Sample>>()
    }
    fn as_i32_samples(&mut self) -> Vec<Sample> {
        self.as_i32()
            .iter()
            .map(|sample| Sample::I32(*sample))
            .collect::<Vec<Sample>>()
    }

    fn as_f32_samples(&mut self) -> Vec<Sample> {
        self.as_f32()
            .iter()
            .map(|sample| Sample::F32(*sample))
            .collect::<Vec<Sample>>()
    }

    fn as_f64_samples(&mut self) -> Vec<Sample> {
        self.as_f64()
            .iter()
            .map(|sample| Sample::F64(*sample))
            .collect::<Vec<Sample>>()
    }

    fn as_sample_type(&mut self, as_type: Sample) -> Vec<Sample> {
        match as_type {
            Sample::I16(_) => self.as_i16_samples(),
            Sample::I32(_) => self.as_i32_samples(),
            Sample::F32(_) => self.as_f32_samples(),
            Sample::F64(_) => self.as_f64_samples(),
        }
    }
}

impl AudioConversion for Sample {
    fn as_i16(self) -> i16 {
        match self {
            Sample::I16(sample) => sample,
            Sample::I32(sample) => sample.as_i16(),
            Sample::F32(sample) => sample.as_i16(),
            Sample::F64(sample) => sample.as_i16(),
        }
    }

    fn as_i32(self) -> i32 {
        match self {
            Sample::I16(sample) => sample.as_i32(),
            Sample::I32(sample) => sample,
            Sample::F32(sample) => sample.as_i32(),
            Sample::F64(sample) => sample.as_i32(),
        }
    }

    fn as_f32(self) -> f32 {
        match self {
            Sample::I16(sample) => sample.as_f32(),
            Sample::I32(sample) => sample.as_f32(),
            Sample::F32(sample) => sample,
            Sample::F64(sample) => sample.as_f32(),
        }
    }

    fn as_f64(self) -> f64 {
        match self {
            Sample::I16(sample) => sample.as_f64(),
            Sample::I32(sample) => sample.as_f64(),
            Sample::F32(sample) => sample.as_f64(),
            Sample::F64(sample) => sample,
        }
    }

    fn as_type(self, as_type: Sample) -> Sample {
        match as_type {
            Sample::I16(_) => Sample::I16(self.as_i16()),
            Sample::I32(_) => Sample::I32(self.as_i32()),
            Sample::F32(_) => Sample::F32(self.as_f32()),
            Sample::F64(_) => Sample::F64(self.as_f64()),
        }
    }
}

impl IterAudioConversion for Vec<Sample> {
    fn as_i16(&mut self) -> Vec<i16> {
        self.iter_mut()
            .map(|sample| sample.as_i16())
            .collect::<Vec<i16>>()
    }

    fn as_i32(&mut self) -> Vec<i32> {
        self.iter_mut()
            .map(|sample| sample.as_i32())
            .collect::<Vec<i32>>()
    }

    fn as_f32(&mut self) -> Vec<f32> {
        self.iter_mut()
            .map(|sample| sample.as_f32())
            .collect::<Vec<f32>>()
    }

    fn as_f64(&mut self) -> Vec<f64> {
        self.iter_mut()
            .map(|sample| sample.as_f64())
            .collect::<Vec<f64>>()
    }
}

impl AudioConversion for i16 {
    fn as_i16(self) -> i16 {
        self
    }

    fn as_i32(self) -> i32 {
        self as i32
    }

    fn as_f32(self) -> f32 {
        (self as f32 / 32768.0).clamp(-1.0, 1.0)
    }

    fn as_f64(self) -> f64 {
        (self as f64 / 32768.0).clamp(-1.0, 1.0)
    }
}

impl AudioConversion for i32 {
    fn as_i16(self) -> i16 {
        (self >> 16) as i16
    }

    fn as_i32(self) -> i32 {
        self
    }

    fn as_f32(self) -> f32 {
        (self as f32 / 2147483648.0).clamp(-1.0, 1.0)
    }

    fn as_f64(self) -> f64 {
        (self as f64 / 2147483648.0).clamp(-1.0, 1.0)
    }
}

impl AudioConversion for f32 {
    fn as_i16(self) -> i16 {
        (self * 32768.0).clamp(-32768.0, 32767.0) as i16
    }

    fn as_i32(self) -> i32 {
        (self * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32
    }

    fn as_f32(self) -> f32 {
        self
    }

    fn as_f64(self) -> f64 {
        self as f64
    }
}

impl AudioConversion for f64 {
    fn as_i16(self) -> i16 {
        (self * 32768.0).clamp(-32768.0, 32767.0) as i16
    }

    fn as_i32(self) -> i32 {
        (self * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32
    }

    fn as_f32(self) -> f32 {
        self as f32
    }

    fn as_f64(self) -> f64 {
        self
    }
}

impl std::fmt::Display for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Sample::I16(sample) => write!(f, "{}", *sample),
            Sample::I32(sample) => write!(f, "{}", *sample),
            Sample::F32(sample) => write!(f, "{}", *sample),
            Sample::F64(sample) => write!(f, "{}", *sample),
        }
    }
}

impl PartialEq<i16> for Sample {
    fn eq(&self, other: &i16) -> bool {
        match self {
            Sample::I16(sample) => sample == other,
            Sample::I32(sample) => sample.as_i16() == *other,
            Sample::F32(sample) => sample.as_i16() == *other,
            Sample::F64(sample) => sample.as_i16() == *other,
        }
    }
}

impl PartialEq<i32> for Sample {
    fn eq(&self, other: &i32) -> bool {
        match self {
            Sample::I16(sample) => sample.as_i32() == *other,
            Sample::I32(sample) => sample == other,
            Sample::F32(sample) => sample.as_i32() == *other,
            Sample::F64(sample) => sample.as_i32() == *other,
        }
    }
}

impl PartialEq<f32> for Sample {
    fn eq(&self, other: &f32) -> bool {
        match self {
            Sample::I16(sample) => sample.as_f32() == *other,
            Sample::I32(sample) => sample.as_f32() == *other,
            Sample::F32(sample) => sample == other,
            Sample::F64(sample) => sample.as_f32() == *other,
        }
    }
}

impl PartialEq<f64> for Sample {
    fn eq(&self, other: &f64) -> bool {
        match self {
            Sample::I16(sample) => sample.as_f64() == *other,
            Sample::I32(sample) => sample.as_f64() == *other,
            Sample::F32(sample) => sample.as_f64() == *other,
            Sample::F64(sample) => sample == other,
        }
    }
}

#[cfg(test)]
pub mod sample_test {
    use super::Sample;

    #[test]
    fn test_sample_can_add() {
        let sample1 = Sample::I16(1);
        let sample2 = Sample::I16(2);
        let sample3 = Sample::I16(3);
        assert_eq!(sample1 + sample2, sample3, "I16 Sample addition failed");

        let sample1 = Sample::I32(1);
        let sample2 = Sample::I32(2);
        let sample3 = Sample::I32(3);
        assert_eq!(sample1 + sample2, sample3, "I32 Sample addition failed");

        let sample1 = Sample::F32(1.0);
        let sample2 = Sample::F32(2.0);
        let sample3 = Sample::F32(3.0);
        assert_eq!(sample1 + sample2, sample3, "F32 Sample addition failed");

        let sample1 = Sample::F64(1.0);
        let sample2 = Sample::F64(2.0);
        let sample3 = Sample::F64(3.0);
        assert_eq!(sample1 + sample2, sample3, "F64 Sample addition failed");
    }
}
