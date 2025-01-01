//! Module containing the functionality for converting between the supported audio sample types
use std::fmt::Debug;

use bytemuck::Pod;
use i24::i24;
use num_traits::Num;

use crate::core::alloc_sample_buffer;

/// Trait used to indicate that a type is an audio sample and can be treated as such.
pub trait AudioSample:
    Copy
    + Pod
    + Num
    + ConvertTo<i16>
    + ConvertTo<i32>
    + ConvertTo<i24>
    + ConvertTo<f32>
    + ConvertTo<f64>
    + Sync
    + Send
    + Debug
    + Default
{
}

impl AudioSample for i16 {}
impl AudioSample for i24 {}
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

/// Trait for converting between audio sample types in a slice
/// The type ``T`` must implement the ``AudioSample`` trait
pub trait ConvertSlice<T: AudioSample> {
    fn convert_slice(self) -> Box<[T]>;
}

impl<T: AudioSample, F> ConvertSlice<T> for Box<[F]>
where
    F: AudioSample + ConvertTo<T>,
{
    fn convert_slice(self) -> Box<[T]> {
        let mut out: Box<[T]> = alloc_sample_buffer(self.len());
        for i in 0..self.len() {
            out[i] = self[i].convert_to();
        }
        out
    }
}

// i16 //
impl ConvertTo<i16> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        *self
    }
}

impl ConvertTo<i24> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        i24::from_i32((*self as i32) << 8)
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

// i24 //
impl ConvertTo<i16> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        (self.to_i32() >> 8) as i16
    }
}

impl ConvertTo<i24> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        *self
    }
}

impl ConvertTo<i32> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        self.to_i32() << 8
    }
}

impl ConvertTo<f32> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        (self.to_i32() as f32) / (i32::MAX as f32)
    }
}

impl ConvertTo<f64> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        (self.to_i32() as f64) / (i32::MAX as f64)
    }
}

// i32 //
impl ConvertTo<i16> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        (*self >> 16) as i16
    }
}

impl ConvertTo<i24> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        i24::from_i32(*self >> 8)
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

// f32 //
impl ConvertTo<i16> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        ((*self * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16
    }
}

impl ConvertTo<i24> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        i24::from_i32(
            ((*self * (i32::MAX as f32)).clamp(i32::MIN as f32, i32::MAX as f32)).round() as i32,
        )
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

// f64 //
impl ConvertTo<i16> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        ((*self * (i16::MAX as f64)).clamp(i16::MIN as f64, i16::MAX as f64)).round() as i16
    }
}

impl ConvertTo<i24> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        i24::from_i32(
            ((*self * (i32::MAX as f64)).clamp(i32::MIN as f64, i32::MAX as f64)).round() as i32,
        )
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

        for (_, (expected_sample, actual_sample)) in
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
pub trait IntoNdarray {
    type Target: AudioSample;
    fn into_ndarray(self) -> crate::WaversResult<(ndarray::Array2<Self::Target>, i32)>;
}

#[cfg(feature = "ndarray")]
pub trait AsNdarray {
    type Target: AudioSample;
    fn as_ndarray(&mut self) -> crate::WaversResult<(ndarray::Array2<Self::Target>, i32)>;
}
