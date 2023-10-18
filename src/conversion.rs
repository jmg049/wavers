use std::fmt::Debug;

/// Module containing the functionality for converting between the supported audio sample types
use bytemuck::Pod;
use num_traits::Num;

use crate::core::alloc_sample_buffer;

/// Trait used to indicate that a type is an audio sample and can be treated as such.
pub trait AudioSample:
    Copy
    + Pod
    + Num
    + ConvertTo<i16>
    + ConvertTo<i32>
    + ConvertTo<f32>
    + ConvertTo<f64>
    + Sync
    + Send
    + Debug
{
}

impl AudioSample for i16 {}
impl AudioSample for i32 {}
impl AudioSample for f32 {}
impl AudioSample for f64 {}

/// Trait for converting between audio sample types
/// The type ``T`` must implement the ``AudioSample`` trait
pub trait ConvertTo<T: AudioSample> {
    fn convert_to(&self) -> T
    where
        Self: Sized + AudioSample;
}

impl ConvertTo<i16> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        *self
    }
}

impl ConvertTo<i32> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        (*self as i32) << 16
    }
}

impl ConvertTo<f32> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        ((*self as f32) / (i16::MAX as f32)).clamp(-1.0, 1.0)
    }
}

impl ConvertTo<f64> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        ((*self as f64) / (i16::MAX as f64)).clamp(-1.0, 1.0)
    }
}

impl ConvertTo<i16> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        (*self >> 16) as i16
    }
}

impl ConvertTo<i32> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        *self
    }
}

impl ConvertTo<f32> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        ((*self as f32) / (i32::MAX as f32)).clamp(-1.0, 1.0)
    }
}

impl ConvertTo<f64> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        ((*self as f64) / (i32::MAX as f64)).clamp(-1.0, 1.0)
    }
}

impl ConvertTo<i16> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        ((*self * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16
    }
}

impl ConvertTo<i32> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        ((*self * (i32::MAX as f32)).clamp(i32::MIN as f32, i32::MAX as f32)).round() as i32
    }
}

impl ConvertTo<f32> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        *self
    }
}

impl ConvertTo<f64> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        *self as f64
    }
}

impl ConvertTo<i16> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        ((*self * (i16::MAX as f64)).clamp(i16::MIN as f64, i16::MAX as f64)).round() as i16
    }
}

impl ConvertTo<i32> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        ((*self * (i32::MAX as f64)).clamp(i32::MIN as f64, i32::MAX as f64)).round() as i32
    }
}

impl ConvertTo<f32> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        *self as f32
    }
}

impl ConvertTo<f64> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        *self
    }
}

#[cfg(test)]
mod conversion_tests {

    use super::*;
    use std::fs::File;
    use std::io::BufRead;
    use std::path::Path;
    use std::str::FromStr;

    use approx_eq::assert_approx_eq;

    #[test]
    fn i16_to_f32() {
        let i16_samples: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let i16_samples: &[i16] = &i16_samples;

        let f32_samples: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();
        let f32_samples: &[f32] = &f32_samples;
        for (expected_sample, actual_sample) in f32_samples.iter().zip(i16_samples) {
            let actual_sample: f32 = actual_sample.convert_to();
            assert_approx_eq!(*expected_sample as f64, actual_sample as f64, 1e-4);
        }
    }

    #[test]
    fn i16_to_f32_slice() {
        let i16_samples: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let i16_samples: Box<[i16]> = i16_samples.into_boxed_slice();
        let f32_samples: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();

        let f32_samples: &[f32] = &f32_samples;
        let converted_i16_samples: Box<[f32]> = i16_samples.convert_slice();

        for (idx, (expected_sample, actual_sample)) in
            converted_i16_samples.iter().zip(f32_samples).enumerate()
        {
            assert_approx_eq!(*expected_sample as f64, *actual_sample as f64, 1e-4);
        }
    }

    #[test]
    fn f32_to_i16() {
        let i16_samples: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let i16_samples: &[i16] = &i16_samples;

        let f32_samples: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();

        let f32_samples: &[f32] = &f32_samples;
        for (expected_sample, actual_sample) in i16_samples.iter().zip(f32_samples) {
            let converted_sample: i16 = actual_sample.convert_to();
            assert_eq!(
                *expected_sample, converted_sample,
                "Failed to convert sample {} to i16",
                actual_sample
            );
        }
    }

    #[cfg(test)]
    fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(std::io::BufReader::new(file).lines())
    }

    #[cfg(test)]
    fn read_text_to_vec<T: FromStr>(fp: &Path) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        <T as FromStr>::Err: std::error::Error + 'static,
    {
        let mut data = Vec::new();
        let lines = read_lines(fp)?;
        for line in lines {
            let line = line?;
            for sample in line.split(" ") {
                let parsed_sample: T = match sample.trim().parse::<T>() {
                    Ok(num) => num,
                    Err(err) => {
                        eprintln!("Failed to parse {}", sample);
                        panic!("{}", err)
                    }
                };
                data.push(parsed_sample);
            }
        }
        Ok(data)
    }
}

#[cfg(feature = "ndarray")]
pub mod ndarray_conversion {
    use crate::{conversion::AudioSample, error::WaversResult};
    use ndarray::{Array2, CowArray, Ix2, ShapeError};
    pub trait IntoNdarray {
        type Target: AudioSample;
        fn into_ndarray(self) -> Result<Array2<Self::Target>, ShapeError>;
    }

    pub trait AsNdarray {
        type Target: AudioSample;
        fn as_ndarray(&self) -> Result<CowArray<Self::Target, Ix2>, ShapeError>;
    }

    pub trait IntoWav {
        type Target: AudioSample;
        fn into(self, sample_rate: i32) -> WaversResult<crate::core::Wav<Self::Target>>;
    }
}

pub trait ConvertSlice<T: AudioSample> {
    fn convert_slice(self) -> Box<[T]>;
}

#[cfg(feature = "simd")]
use std::simd::{f32x64, f64x64, i16x32, i16x64, i32x64, Simd, SimdInt};

const F32_I16: f32 = 1.0 / i16::MAX as f32;
const F64_I16: f64 = 1.0 / i16::MAX as f64;

const FOUR_BYTE_LANES: usize = 64; // for floats and i32
const EIGHT_BYTE_LANES: usize = 32; // for doubles

#[cfg(feature = "simd")]
impl ConvertSlice<i16> for Box<[i16]> {
    fn convert_slice(self) -> Box<[i16]> {
        Box::from(self)
    }
}

impl ConvertSlice<i32> for Box<[i16]> {
    #[cfg(feature = "simd")]
    fn convert_slice(self) -> Box<[i32]> {
        let mut out = Vec::with_capacity(self.len());
        unsafe {
            out.set_len(self.len());
        }
        let chunks = self.chunks_exact(FOUR_BYTE_LANES);
        let remainder = chunks.remainder();

        let mut i = 0;
        let mut shift = i32x64::splat(16);

        for chunk in chunks {
            let mut s: Simd<i32, FOUR_BYTE_LANES> = i16x64::from_slice(chunk).cast();
            s <<= shift;
            out[i..i + FOUR_BYTE_LANES].copy_from_slice(s.as_array());
            i += FOUR_BYTE_LANES;
        }
        for sample in remainder.iter() {
            out[i] = (*sample as i32) << 16;
            i += 1;
        }
        out.into_boxed_slice()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[i32]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<f32> for Box<[i16]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[f32]> {
        let mut out_buf = vec![F32_I16; self.len()];

        let chunks = self.chunks_exact(FOUR_BYTE_LANES);
        let remainder = chunks.remainder();

        let f32_splat = f32x64::splat(F32_I16);

        let mut i = 0;
        for chunk in chunks {
            let mut chunk: Simd<f32, FOUR_BYTE_LANES> = i16x64::from_slice(chunk).cast();
            chunk *= f32_splat;
            out_buf[i..i + FOUR_BYTE_LANES].copy_from_slice(chunk.as_array());
            i += FOUR_BYTE_LANES;
        }

        for sample in remainder.iter() {
            out_buf[i] = (*sample as f32) * F32_I16;
            i += 1;
        }

        Box::from(out_buf)
    }
    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[f32]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<f64> for Box<[i16]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[f64]> {
        let mut out_buf = vec![F64_I16; self.len()];

        let chunks = self.chunks_exact(FOUR_BYTE_LANES);
        let remainder = chunks.remainder();

        let f32_splat = f64x64::splat(F64_I16);

        let mut i = 0;
        for chunk in chunks {
            let mut chunk: Simd<f64, FOUR_BYTE_LANES> = i16x64::from_slice(chunk).cast();
            chunk *= f32_splat;
            out_buf[i..i + FOUR_BYTE_LANES].copy_from_slice(chunk.as_array());
            i += FOUR_BYTE_LANES;
        }

        for sample in remainder.iter() {
            out_buf[i] = (*sample as f64) * F64_I16;
            i += 1;
        }

        Box::from(out_buf)
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[f64]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<i16> for Box<[i32]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[i16]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[i16]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<i32> for Box<[i32]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[i32]> {
        Box::from(self)
    }
    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[i32]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}
impl ConvertSlice<f32> for Box<[i32]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[f32]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[f32]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<f64> for Box<[i32]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[f64]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[f64]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<i16> for Box<[f32]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[i16]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[i16]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<i32> for Box<[f32]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[i32]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[i32]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<f32> for Box<[f32]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[f32]> {
        Box::from(self)
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[f32]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<f64> for Box<[f32]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[f64]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[f64]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}

impl ConvertSlice<i16> for Box<[f64]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[i16]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[i16]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}
impl ConvertSlice<i32> for Box<[f64]> {
    #[cfg(feature = "simd")]
    fn convert_slice(self) -> Box<[i32]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[i32]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}
impl ConvertSlice<f32> for Box<[f64]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[f32]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[f32]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}
impl ConvertSlice<f64> for Box<[f64]> {
    #[cfg(feature = "simd")]

    fn convert_slice(self) -> Box<[f64]> {
        todo!()
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice(self) -> Box<[f64]> {
        let mut out = alloc_sample_buffer(self.len());
        for (idx, sample) in self.iter().enumerate() {
            out[idx] = sample.convert_to();
        }
        out
    }
}
