//!
//! # Wavers
//! WaveRs is a fast and lightweight library for reading and writing ``wav`` files.
//! Currently, it supports reading and writing of ``i16``, ``i32``, ``f32``, and ``f64`` audio samples.
//!
//! ## Highlights
//! * Fast and lightweight
//! * Simple API, read a wav file with ``read`` and write a wav file with ``write``
//! * Easy and efficient conversion between different types of audio samples.
//! * Support for the ``ndarray`` crate.
//!
//! ## Crate Status
//! * This crate is currently in development. Changes to the core API will either not happen or they will be kept to a minimum. Any planned additions to the API will be built on top of the existing API.
//! * Documentation is currently in progress, it is mostly complete but will be updated as necessary.
//! * The API is tested, but there can always be more tests.
//! * The crate has been benchmarked, but there can always be more benchmarks.
//! * Some examples of planned features:
//!     * Support for reading and writing of ``i24`` audio samples.
//!     * Support iteration over samples in a wav file beyond calling ``iter()`` on the samples. Will providing windowing and other useful features.
//!     * Investigate the performance of the ``write`` function.
//!     * Channel wise iteration over samples in a wav file.
//!     * Resampling algorithms
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
//! 	let fp: &Path = &Path::new("path/to/wav.wav");
//!
//! 	// read a wav file as PCM_16 (may perform a conversion)
//! 	let i16_wav: Wav<i16> = read::<i16>(fp).unwrap();
//! 	let i16_samples: &[i16] = i16_wav.as_ref();
//!
//! 	// Alternatively, may also perform a conversion
//! 	let f32_wav: Wav<f32> = Wav::<f32>::read(fp).unwrap();
//! 	let f32_samples: &[f32] = f32_wav.as_ref();
//! }
//! ```
//!
//! ## Conversion
//! ```no_run
//! use wavers::{Wav, read, ConvertTo};
//! use std::path::Path;
//!
//! fn main() {
//! 	let fp = "path/to/wav.wav";
//!
//! 	let i16_wav: Wav<i16> = Wav::<i16>::from_path(fp).unwrap();
//!     let f32_wav: Wav<f32> = i16_wav.convert();
//! }
//! ```
//!
//! ## Writing
//! ```no_run
//! use wavers::Wav;
//! use std::path::Path;
//!
//!
//! fn main() {
//! 	let fp: &Path = &Path::new("path/to/wav.wav");
//! 	let out_fp: &Path = &Path::new("out/path/to/wav.wav");
//!
//! 	let mut i16_wav: Wav<i16> = Wav::<i16>::read(fp).unwrap();
//! 	// Some code that modifies the wav data
//!
//! 	i16_wav.write(&out_fp).unwrap();
//! 	// or we can convert and write
//! 	i16_wav.as_::<f32>().unwrap().write(&out_fp).unwrap();
//! }
//!
//! ```
//! ## Wav Utilities
//! ```no_run
//! use wavers::wav_spec;
//! fn main() {
//!	    let fp = "path/to/wav.wav";
//!     let wav: Wav<i16> = Wav::from_path(fp).unwrap();
//!     let sample_rate = wav.sample_rate();
//!     let n_channels = wav.n_channels();
//!     let duration = wav.duration();
//!     let encoding = wav.encoding();
//!     let (sample_rate, n_channels, duration, encoing) = wav_spec(fp).unwrap();
//! }
//! ```
//!
//! ## Features
//! The following section describes the features available in the WaveRs crate.
//! ### Ndarray
//!
//! The ``ndarray`` feature is used to provide functions that allow wav files to be read as ``ndarray`` 2-D arrays (samples x channels). There are two functions provided, ``into_ndarray`` and ``as_ndarray``. ``into_ndarray`` consumes the samples and ``as_ndarray`` creates a ``Array2`` from the samples.
//!
//! ```no_run
//! use wavers::{read, Wav, AsNdarray, IntoNdarray};
//! use ndarray::{Array2, CowArray2};
//!
//! fn main() {
//! 	let fp = "path/to/wav.wav";
//! 	let i16_wav: Wav<i16> = read::<i16>(fp).unwrap();
//! 	let i16_array: Array2<i16> = i16_wav.into_ndarray().unwrap();
//!
//! 	let i16_wav: Wav<i16> = read::<i16>(fp).unwrap();
//! 	let i16_array: Array2<i16> = i16_wav.as_ndarray().unwrap();
//! }
//!
//! ```
//!
//! ## Benchmarks
//! To check out the benchmarks head on over to the benchmarks wiki page on the WaveRs <a href=https://github.com/jmg049/wavers/wiki/Benchmarks>GitHub</a>.
//! Benchmarks were conducted on the reading and writing functionality of WaveRs and compared to the ``hound`` crate.
//!

mod conversion;
mod core;
pub mod error;
mod header;

use std::fs;
use std::io::Write;
use std::path::Path;

pub use crate::conversion::{AudioSample, ConvertTo};

pub use crate::conversion::ConvertSlice;

pub use crate::core::ReadSeek;
pub use crate::core::{wav_spec, Samples, Wav, WavType};
pub use crate::error::WaversResult;
pub use crate::header::{FmtChunk, WavHeader};

#[inline(always)]
pub fn read<T: AudioSample, P: AsRef<Path>>(path: P) -> WaversResult<Samples<T>>
where
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
{
    Wav::<T>::from_path(path)?.read()
}

#[inline(always)]
pub fn write<T: AudioSample, P: AsRef<Path>>(
    fp: P,
    samples: &[T],
    sample_rate: i32,
    n_channels: u16,
) -> WaversResult<()>
where
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
{
    let s = Samples::from(samples);
    let samples_bytes = s.as_bytes();
    let header_bytes =
        WavHeader::new_header::<T>(sample_rate, n_channels, n_channels as usize)?.as_bytes();

    let mut f = fs::File::create(fp)?;
    f.write_all(&header_bytes)?;

    let data_size_bytes = samples_bytes.len() as u32; // write up to the data size
    f.write_all(&data_size_bytes.to_ne_bytes())?; // write the data size
    f.write_all(&samples_bytes)?; // write the data
    Ok(())
}

mod tests {

    use approx_eq::assert_approx_eq;
    use std::io::BufRead;
    use std::{fs::File, path::Path, str::FromStr};

    use super::{write, Samples, Wav};
    use crate::core::alloc_sample_buffer;
    #[test]
    fn test_write() {
        let expected_path = "./test_resources/one_channel_f32.txt";
        let out_path = "./test_resources/tmp/one_channel_f32_tmp.wav";
        let sr: i32 = 16000;

        let samples: Box<[f32]> = alloc_sample_buffer(sr as usize * 10); // 10s @ 16Khz
        let samples = Samples::from(samples);

        let n_channels = 1;

        write(out_path, &samples, sr, n_channels).expect("Failed to write data");
        let mut wav: Wav<f32> = Wav::from_path(out_path).expect("Failed to open file wav file");
        let samples: Samples<f32> = wav.read().expect("Failed to read data");

        let expected: Vec<f32> = read_text_to_vec(expected_path).expect("failed to load from txt");

        for (exp, act) in expected.iter().zip(samples.as_ref()) {
            assert_approx_eq!(*exp as f64, *act as f64);
        }
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
