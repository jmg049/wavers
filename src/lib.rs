//!
//! # Wavers
//! WaveRs is a fast and lightweight library for reading and writing ``wav`` files.
//! Currently, it supports reading and writing of ``i16``, ``i24``, ``i32``, ``f32``, and ``f64`` audio samples.
//!
//! Feedback and bugs welcome!
//!
//! ## Highlights
//! * Fast and lightweight
//! * Simple API, read a wav file with ``read`` and write a wav file with ``write``
//! * Easy and efficient conversion between different types of audio samples (**should** compile down to simd instructions provided you build with the appropriate SIMD instruction set for your architecture).
//! * Support for the Extensible format (Happy to try and support anything else that pops up, just ask or open a PR).
//! * Increasing support for different chunks in the wav file.
//! * Support for iteration over the frames, channels and overlapping blocks of the wav file.
//! * Support for the ``ndarray`` crate. Enable the ``ndarray`` feature to enable ndarray support.
//! * Support for the ``pyo3`` crate. Enable the ``pyo3`` feature to enable pyo3 support. This is mostly for [PyWavers](https://github.com/jmg049/Pywavers).
//! * Supports logging through the ``log`` crate. Enable the ``logging`` feature to enable logging.
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
pub mod chunks;
pub mod conversion;
pub mod core;

pub mod error;
pub mod header;

pub mod iter;
pub mod wav_type;
use error::FormatError;
use i24::i24;
use std::fs;
use std::io::Write;
use std::path::Path;

pub use crate::conversion::{AudioSample, ConvertSlice, ConvertTo};

pub use crate::chunks::{FactChunk, FmtChunk, ListChunk, DATA, FACT, LIST, RIFF, WAVE};
pub use crate::core::{wav_spec, ReadSeek, Samples, Wav};
pub use crate::error::{WaversError, WaversResult};
pub use crate::header::WavHeader;
pub use crate::wav_type::{format_info_to_wav_type, wav_type_to_format_info, FormatCode, WavType};

/// A macro for logging messages if the logging feature is enabled.
#[macro_export]
macro_rules! log {
    ($level:expr, $($arg:tt)+) => {
        #[cfg(feature = "logging")]
        log::log!($level, $($arg)+);
    };
}

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
    let mut wav: Wav<T> = Wav::from_path(&path)?;
    let samples = wav.read()?;
    log!(
        log::Level::Debug,
        "Read wav file from {}\n{}",
        path.as_ref().display(),
        wav,
    );
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

    let mut f = fs::File::create(&fp)?;

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
            return Err(FormatError::InvalidTypeId("Invalid type ID").into());
        }
    }

    f.write_all(&DATA)?;
    let data_size_bytes = samples_bytes.len() as u32; // write up to the data size
    f.write_all(&data_size_bytes.to_ne_bytes())?; // write the data size
    f.write_all(&samples_bytes)?; // write the data
    log!(
        log::Level::Debug,
        "Wrote wav file to {}",
        fp.as_ref().display()
    );
    Ok(())
}

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
