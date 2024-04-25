//!
//! # Wavers
//! WaveRs is a fast and lightweight library for reading and writing ``wav`` files.
//! Currently, it supports reading and writing of ``i16``, ``i32``, ``f32``, and ``f64`` audio samples.
//!
//! **Experimental** support for ``i24`` audio samples is also available. The ``i24`` type supports conversion to and from the other sample types.
//! The ``i24`` type supports Add, Sub, Mul, Div, Rem, Neg, BitXor, BitOr, BitAnd, Shl, Shr, Not, and their assignment counterparts.
//! Feedback and bugs welcome!
//!
//! ## Highlights
//! * Fast and lightweight
//! * Simple API, read a wav file with ``read`` and write a wav file with ``write``
//! * Easy and efficient conversion between different types of audio samples.
//! * Support for the Extensible format.
//! * Increasing support for different chunks in the wav file.
//! * Support for the ``ndarray`` crate.
//! * Support for iteration over the frames and channels of the wav file.
//!
//! ## Crate Status
//! * This crate is currently in development. Changes to the core API will either not happen or they will be kept to a minimum. Any planned additions to the API will be built on top of the existing API.
//! * Documentation is currently in progress, it is mostly complete but will be updated as necessary.
//! * The API is tested, but there can always be more tests.
//! * The crate has been benchmarked, but there can always be more benchmarks.
//! * Some examples of planned features:
//!     * Investigate the performance of the ``write`` function.
//!     * Any suggestions or requests are welcome.
//!
//! ## Examples
//! The following examples show how to read and write a wav file, as well as retrieving information from the header.
//!
//!
//! ## Reading
//!
//! ```no_run
//! use wavers::{Wav, read};
//! use std::path::Path;
//!
//! fn main() {
//! 	let fp = "path/to/wav.wav";
//!     // creates a Wav file struct, does not read the audio data. Just the header information.
//!     let wav: Wav<i16> = Wav::from_path(fp).unwrap();
//!     // or to read the audio data directly
//!     let (samples, sample_rate): (Samples<i16>, i32) = read::<i16, _>(fp).unwrap();
//!     // samples can be derefed to a slice of samples
//!     let samples: &[i16] = &samples;
//! }
//! ```
//!
//! ## Conversion
//! ```no_run
//! use wavers::{Wav, read, ConvertTo};
//! use std::path::Path;
//!
//! fn main() {
//!     // Two ways of converted a wav file
//!     let fp: "./path/to/i16_encoded_wav.wav";
//!     let wav: Wav<f32> = Wav::from_path(fp).unwrap();
//!     // conversion happens automatically when you read
//!     let samples: &[f32] = &wav.read().unwrap();
//!
//!     // or read and then call the convert function on the samples.
//!     let (samples, sample_rate): (Samples<i16>, i32) = read::<i16, _>(fp).unwrap();
//!     let samples: &[f32] = &samples.convert();
//! }
//! ```
//!
//! ## Writing
//! ```no_run
//! use wavers::Wav;
//! use std::path::Path;
//!
//! fn main() {
//! 	let fp: &Path = &Path::new("path/to/wav.wav");
//! 	let out_fp: &Path = &Path::new("out/path/to/wav.wav");
//!
//!     // two main ways, read and write as the type when reading
//!     let wav: Wav<i16> = Wav::from_path(fp).unwrap();
//!     wav.write(out_fp).unwrap();
//!
//!     // or read, convert, and write
//!     let (samples, sample_rate): (Samples<i16>,i32) = read::<i16, _>(fp).unwrap();
//!     let sample_rate = wav.sample_rate();
//!     let n_channels = wav.n_channels();
//!
//!     let samples: &[f32] = &samples.convert();
//!     write(out_fp, samples, sample_rate, n_channels).unwrap();
//! }
//! ```
//! ## Iteration
//! ``WaveRs`` provides two primary methods of iteration: Frame-wise and Channel-wise. These can be performed using the ``Wav::frames`` and ``Wav::channels`` functions respectively. Both methods return an iterator over the samples in the wav file. The ``frames`` method returns an iterator over the frames of the wav file, where a frame is a single sample from each channel. The ``channels`` method returns an iterator over the channels of the wav file, where a channel is all the samples for a single channel.
//!
//! ```no_run
//! use wavers::Wav;
//!
//! fn main() {
//!     let wav = Wav::from_path("path/to/two_channel.wav").unwrap();
//!     for frame in wav.frames() {
//!        assert_eq!(frame.len(), 2, "The frame should have two samples since the wav file has two channels");
//!        // do something with the frame
//!     }
//!
//!     for channel in wav.channels() {
//!         // do something with the channel
//!         assert_eq!(channel.len(), wav.n_samples() / wav.n_channels(), "The channel should have the same number of samples as the wav file divided by the number of channels");
//!     }
//! }
//! ````
//!
//!
//! ## Wav Utilities
//! ```no_run
//! use wavers::wav_spec;
//! fn main() {
//!	    let fp = "path/to/wav.wav";
//!     let wav: Wav<i16> = Wav::from_path(fp).unwrap();
//!     let wav_spec = wav.wav_spec(); // returns the duration and the header
//!     println!("{:?}", wav_spec);
//! }
//! ```
//! Check out [wav_inspect](https://crates.io/crates/wav_inspect) for a simnple command line tool to inspect the headers of wav files.
//! ## Features
//! The following section describes the features available in the WaveRs crate.
//! ### Ndarray
//!
//! The ``ndarray`` feature is used to provide functions that allow wav files to be read as ``ndarray`` 2-D arrays (samples x channels). There are two functions provided, ``into_ndarray`` and ``as_ndarray``. ``into_ndarray`` consumes the samples and ``as_ndarray`` creates a ``Array2`` from the samples.
//!
//! ```no_run
//! use wavers::{read, Wav, AsNdarray, IntoNdarray, Samples};
//! use ndarray::{Array2, CowArray2};
//!
//! fn main() {
//! 	let fp = "path/to/wav.wav";
//!     let wav: Wav<i16> = Wav::from_path(fp).unwrap();
//!
//!     // does not consume the wav file struct
//! 	let (i16_array, sample_rate): (Array2<i16>, i32) = wav.as_ndarray().unwrap();
//!     
//!     // consumes the wav file struct
//! 	let (i16_array, sample_rate): (Array2<i16>, i32) = wav.into_ndarray().unwrap();
//!
//!     // convert the array to samples.
//!     let samples: Samples<i16> = Samples::from(i16_array);
//! }
//! ```
//!
//! ## Benchmarks
//! To check out the benchmarks head on over to the benchmarks wiki page on the WaveRs <a href=https://github.com/jmg049/wavers/wiki/Benchmarks>GitHub</a>.
//! Benchmarks were conducted on the reading and writing functionality of WaveRs and compared to the ``hound`` crate.
//!
mod chunks;
mod conversion;
mod core;
mod error;
mod header;
mod iter;
mod wav_type;

use std::fmt::{self, Display};
use std::fs;
use std::io::Write;
use std::path::Path;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub use crate::conversion::{AudioSample, ConvertSlice, ConvertTo};

pub use crate::chunks::{FactChunk, FmtChunk, ListChunk, DATA, FACT, LIST, RIFF, WAVE};
pub use crate::core::{wav_spec, ReadSeek, Samples, Wav};
pub use crate::error::{WaversError, WaversResult};
pub use crate::header::WavHeader;
pub use crate::wav_type::{FormatCode, WavType};

/// Reads a wav file and returns the samples and the sample rate.
///
/// Throws an error if the file cannot be opened.
///
/// # Examples
///
/// ```no_run
/// use wavers::{read, Wav, Samples};
///
/// fn main() {
///     let fp = "path/to/wav.wav";
///     // reads the audio data as i16 samples
///     let (samples, sample_rate): (Samples<i16>, i32) = read::<i16, _>(fp).unwrap();
///     // or read the same file as f32 samples
///     let (samples, sample_rate): (Samples<f32>, i32) = read::<f32, _>(fp).unwrap();
/// }
///
#[inline(always)]
pub fn read<T: AudioSample, P: AsRef<Path>>(path: P) -> WaversResult<(Samples<T>, i32)>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i24]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
{
    let mut wav: Wav<T> = Wav::from_path(path)?;
    let samples = wav.read()?;
    Ok((samples, wav.sample_rate()))
}

/// Writes wav samples to disk.
///
/// # Examples
///
/// The code below will generate a wav file from a 10 second, 1-channel sine wave and write it to disk.
/// ```
/// use wavers::{read,write, Samples, AudioSample, ConvertTo, ConvertSlice};
///
/// fn main() {
///     let fp = "./wav.wav";
///     let sr: i32 = 16000;
///     let duration = 10;
///     let mut samples: Vec<f32> = (0..sr * duration).map(|x| (x as f32 / sr as f32)).collect();
///     for sample in samples.iter_mut() {
///         *sample *= 440.0 * 2.0 * std::f32::consts::PI;
///         *sample = sample.sin();
///         *sample *= i16::MAX as f32;
///     }
///     let samples: Samples<f32> = Samples::from(samples.into_boxed_slice()).convert();
///     assert!(write(fp, &samples, sr, 1).is_ok());
///     std::fs::remove_file(fp).unwrap();
/// }
///
#[inline(always)]
pub fn write<T: AudioSample, P: AsRef<Path>>(
    fp: P,
    samples: &[T],
    sample_rate: i32,
    n_channels: u16,
) -> WaversResult<()>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i24]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
{
    let s = Samples::from(samples);
    let samples_bytes = s.as_bytes();

    let new_header = WavHeader::new_header::<T>(sample_rate, n_channels, s.len())?;
    let mut f = fs::File::create(fp)?;

    match new_header.fmt_chunk.format {
        FormatCode::WAV_FORMAT_PCM => {
            let header_bytes = new_header.as_base_bytes();
            f.write_all(&header_bytes)?;
        }
        FormatCode::WAV_FORMAT_IEEE_FLOAT | FormatCode::WAVE_FORMAT_EXTENSIBLE => {
            let header_bytes = new_header.as_extended_bytes();
            f.write_all(&header_bytes)?;
        }
        _ => {
            return Err(WaversError::InvalidType(
                new_header.fmt_chunk.format,
                new_header.fmt_chunk.bits_per_sample,
                new_header.fmt_chunk.ext_fmt_chunk.sub_format(),
            ))
        }
    }

    f.write_all(&DATA)?;
    let data_size_bytes = samples_bytes.len() as u32; // write up to the data size
    f.write_all(&data_size_bytes.to_ne_bytes())?; // write the data size
    f.write_all(&samples_bytes)?; // write the data
    Ok(())
}

use std::{
    ops::{Neg, Not},
    str::FromStr,
};

use bytemuck::{Pod, Zeroable};
use num_traits::{Num, One, Zero};

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg(not(feature = "pyo3"))]
/// An experimental 24-bit unsigned integer type.
///
/// This type is a wrapper around ``[u8; 3]`` and is used to represent 24-bit audio samples.
/// It should not be used anywhere important. It is still unverified and experimental.
///
/// The type is not yet fully implemented and is not guaranteed to work.
/// Supports basic arithmetic operations and conversions to and from ``i32``.
/// The [AudioSample](wavers::core::AudioSample) trait is implemented for this type and so are the [ConvertTo](wavers::core::ConvertTo) and [ConvertSlice](wavers::core::ConvertSlice) traits.
///
pub struct i24 {
    pub data: [u8; 3],
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct i24 {
    pub data: [u8; 3],
}

unsafe impl Zeroable for i24 {}
unsafe impl Pod for i24 {}

impl One for i24 {
    fn one() -> Self {
        i24::from_i32(1)
    }
}

impl Zero for i24 {
    fn zero() -> Self {
        i24::from_i32(0)
    }

    fn is_zero(&self) -> bool {
        i24::from_i32(0) == *self
    }
}

impl Num for i24 {
    type FromStrRadixErr = std::num::ParseIntError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let i32_result = i32::from_str_radix(str, radix)?;
        Ok(i24::from_i32(i32_result))
    }
}

impl Display for i24 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_i32())
    }
}

impl FromStr for i24 {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let i32_result = i32::from_str(s)?;
        Ok(i24::from_i32(i32_result))
    }
}

impl i24 {
    /// Returns the 24-bit integer as an i32.
    pub const fn to_i32(self) -> i32 {
        let [a, b, c] = self.data;
        i32::from_ne_bytes([a, b, c, 0])
    }

    /// Returns the i32 as a 24-bit integer.
    pub const fn from_i32(n: i32) -> Self {
        let [a, b, c, _d] = i32::to_ne_bytes(n);
        Self { data: [a, b, c] }
    }

    /// Creates a 24-bit integer from a array of 3 bytes in native endian format.
    pub const fn from_ne_bytes(bytes: [u8; 3]) -> Self {
        let i32_result = i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], 0]);
        i24::from_i32(i32_result)
    }

    /// Creates a 24-bit integer from a array of 3 bytes in little endian format.
    pub const fn from_le_bytes(bytes: [u8; 3]) -> Self {
        let i32_result = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
        i24::from_i32(i32_result)
    }

    /// Creates a 24-bit integer from a array of 3 bytes in big endian format.
    pub const fn from_be_bytes(bytes: [u8; 3]) -> Self {
        let i32_result = i32::from_be_bytes([bytes[0], bytes[1], bytes[2], 0]);
        i24::from_i32(i32_result)
    }
}

impl Neg for i24 {
    type Output = Self;

    fn neg(self) -> Self {
        let i32_result = self.to_i32().wrapping_neg();
        i24::from_i32(i32_result)
    }
}

impl Not for i24 {
    type Output = Self;

    fn not(self) -> Self {
        let i32_result = !self.to_i32();
        i24::from_i32(i32_result)
    }
}
#[cfg(feature = "pyo3")]
use numpy::Element;

#[cfg(feature = "pyo3")]
unsafe impl Element for i24 {
    const IS_COPY: bool = true;

    fn get_dtype<'py>(py: Python<'py>) -> &'py numpy::PyArrayDescr {
        numpy::dtype::<i24>(py)
    }
}

macro_rules! implement_ops {
    ($($trait_path:path { $($function_name:ident),* }),*) => {
        $(
            impl $trait_path for i24 {
                $(
                    type Output = Self;

                    fn $function_name(self, other: Self) -> Self {
                        let self_i32: i32 = self.to_i32();
                        let other_i32: i32 = other.to_i32();
                        let result = self_i32.$function_name(other_i32);
                        Self::from_i32(result)
                    }
                )*
            }
        )*
    };
}

macro_rules! implement_ops_assign {
    ($($trait_path:path { $($function_name:ident),* }),*) => {
        $(
            impl $trait_path for i24 {
                $(
                    fn $function_name(&mut self, other: Self){
                        let mut self_i32: i32 = self.to_i32();
                        let other_i32: i32 = other.to_i32();
                        self_i32.$function_name(other_i32);
                    }
                )*
            }
        )*
    };
}

macro_rules! implement_ops_assign_ref {
    ($($trait_path:path { $($function_name:ident),* }),*) => {
        $(
            impl $trait_path for &i24 {
                $(
                    fn $function_name(&mut self, other: Self){
                        let mut self_i32: i32 = self.to_i32();
                        let other_i32: i32 = other.to_i32();
                        self_i32.$function_name(other_i32);
                    }
                )*
            }
        )*
    };
}

implement_ops!(
    std::ops::Add { add },
    std::ops::Sub { sub },
    std::ops::Mul { mul },
    std::ops::Div { div },
    std::ops::Rem { rem },
    std::ops::BitAnd { bitand },
    std::ops::BitOr { bitor },
    std::ops::BitXor { bitxor },
    std::ops::Shl { shl },
    std::ops::Shr { shr }
);

implement_ops_assign!(
    std::ops::AddAssign { add_assign },
    std::ops::SubAssign { sub_assign },
    std::ops::MulAssign { mul_assign },
    std::ops::DivAssign { div_assign },
    std::ops::RemAssign { rem_assign },
    std::ops::BitAndAssign { bitand_assign },
    std::ops::BitOrAssign { bitor_assign },
    std::ops::BitXorAssign { bitxor_assign },
    std::ops::ShlAssign { shl_assign },
    std::ops::ShrAssign { shr_assign }
);

implement_ops_assign_ref!(
    std::ops::AddAssign { add_assign },
    std::ops::SubAssign { sub_assign },
    std::ops::MulAssign { mul_assign },
    std::ops::DivAssign { div_assign },
    std::ops::RemAssign { rem_assign },
    std::ops::BitAndAssign { bitand_assign },
    std::ops::BitOrAssign { bitor_assign },
    std::ops::BitXorAssign { bitxor_assign },
    std::ops::ShlAssign { shl_assign },
    std::ops::ShrAssign { shr_assign }
);

#[cfg(test)]
mod lib_tests {
    use approx_eq::assert_approx_eq;
    use std::io::BufRead;
    use std::{fs::File, path::Path, str::FromStr};

    use super::{read, write, Samples, Wav};

    const TEST_OUTPUT: &str = "./test_resources/tmp/";

    #[test]
    fn test_write() {
        let expected_path = "./test_resources/one_channel_f32.txt";
        let out_path = "./test_resources/tmp/one_channel_f32_tmp.wav";

        let mut wav: Wav<f32> = Wav::from_path("./test_resources/one_channel_i16.wav")
            .expect("Failed to open file wav file");

        let samples = wav.read().expect("Failed to read data");

        write(out_path, &samples, wav.sample_rate(), wav.n_channels())
            .expect("Failed to write data");

        let mut wav: Wav<f32> = Wav::from_path(out_path).expect("Failed to open file wav file");
        let samples: Samples<f32> = wav.read().expect("Failed to read data");

        let expected: Vec<f32> = read_text_to_vec(expected_path).expect("failed to load from txt");

        for (exp, act) in expected.iter().zip(samples.as_ref()) {
            assert_approx_eq!(*exp as f64, *act as f64, 1e-4);
        }
        std::fs::remove_file(Path::new(&out_path)).unwrap();
    }

    #[test]
    fn test_read() {
        let input_path = "./test_resources/one_channel_i16.wav";
        let expected_path = "./test_resources/one_channel_i16.txt";
        let expected_sr = 16000;

        let mut wav: Wav<i16> = Wav::from_path(input_path).expect("Failed to open file wav file");
        let samples: Samples<i16> = wav.read().expect("Failed to read data");
        let actual_sr = wav.sample_rate();
        let expected: Vec<i16> = read_text_to_vec(expected_path).expect("failed to load from txt");

        assert_eq!(expected_sr, actual_sr, "Sample rates do not match");
        for (exp, act) in expected.iter().zip(samples.as_ref()) {
            assert_eq!(*exp, *act, "Samples do not match");
        }
    }

    use std::stringify;
    macro_rules! read_tests {
        ($($T:ident), *) => {
            $(
                paste::item! {
                    #[test]
                    fn [<read_$T>]() {
                        let t_string: &str = stringify!($T);

                        let wav_str = format!("./test_resources/one_channel_{}.wav", t_string);
                        let expected_str = format!("./test_resources/one_channel_{}.txt", t_string);

                        let (sample_data, _): (Samples<$T>, i32) = match read::<$T, _>(&wav_str) {
                            Ok((s, sr)) => (s, sr),
                            Err(e) => {eprintln!("{}\n{}", wav_str, e); panic!("Failed to read wav file")}
                        };

                        let expected_data: Vec<$T> = match read_text_to_vec(Path::new(&expected_str)) {
                            Ok(w) => w,
                            Err(e) => {eprintln!("{}\n{}", wav_str, e); panic!("Failed to read txt file")}
                        };

                        for (expected, actual) in expected_data.iter().zip(sample_data.iter()) {
                            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
                        }
                    }
                }
            )*
        }
    }
    read_tests!(i16, i32, f32, f64);

    // No tests for i24 as it it requires a different approach to testing.
    // It is tested in crate::core::core_tests.
    macro_rules! write_tests {
        ($($T:ident), *) => {
            $(
                paste::item! {
                    #[test]
                    fn [<write_$T>]() {
                        if !Path::new(TEST_OUTPUT).exists() {
                            std::fs::create_dir(TEST_OUTPUT).unwrap();
                        }
                        let t_string: &str = stringify!($T);

                        let wav_str = format!("./test_resources/one_channel_{}.wav", t_string);
                        let expected_str = format!("./test_resources/one_channel_{}.txt", t_string);

                        let mut wav: Wav<$T> =
                            Wav::from_path(wav_str).expect("Failed to create wav file");
                        let expected_samples: Samples<$T> = Samples::from(
                            read_text_to_vec(&Path::new(&expected_str)).expect("Failed to read to vec"),
                        );


                        let out = format!("{}_one_channel_{}.wav", TEST_OUTPUT, t_string);
                        let out_path = Path::new(&out);

                        wav.write::<$T, _>(out_path)
                            .expect("Failed to write file");

                        let mut new_wav: Wav<$T> = Wav::<$T>::from_path(out_path).unwrap();

                        for (expected, actual) in expected_samples
                            .iter()
                            .zip(new_wav.read().unwrap().iter())
                        {
                            assert_eq!(expected, actual, "{} != {}", expected, actual);
                        }
                        std::fs::remove_file(Path::new(&out_path)).unwrap();
                    }
                }
            )*
        };
    }

    // No tests for i24 as it it requires a different approach to testing.
    // It is tested in the crate::core::core_tests.
    write_tests!(i16, i32, f32, f64);

    use crate::ConvertSlice;
    #[test]
    fn write_sin_wav() {
        let fp = "./wav.wav";
        let sr: i32 = 16000;
        let duration = 10;
        let mut samples: Vec<f32> = (0..sr * duration).map(|x| (x as f32 / sr as f32)).collect();
        for sample in samples.iter_mut() {
            *sample *= 440.0 * 2.0 * std::f32::consts::PI;
            *sample = sample.sin();
            *sample *= i16::MAX as f32;
        }
        let samples: Samples<f32> = Samples::from(samples.into_boxed_slice().convert_slice());

        write(fp, &samples, sr, 1).unwrap();
        std::fs::remove_file(fp).unwrap();
    }

    fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(std::io::BufReader::new(file).lines())
    }

    fn read_text_to_vec<T: FromStr, P: AsRef<Path>>(
        fp: P,
    ) -> Result<Vec<T>, Box<dyn std::error::Error>>
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
pub use conversion::{AsNdarray, IntoNdarray};
