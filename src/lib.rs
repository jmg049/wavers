//! # WaveRs
//!
//! WaveRs is a high-performance library for reading, writing, and manipulating WAV audio files in Rust.
//! It focuses on simplicity, type safety, performance, and ergonomic APIs while supporting many advanced WAV format features.
//!
//! ## Core Features
//!
//! - **Type-Safe Audio Handling**: First-class support for common audio formats:
//!   - Integer PCM: `i16`, `i24`, `i32`
//!   - Floating point: `f32`, `f64`
//!   - Zero-cost conversions between formats
//!
//! - **Flexible Processing**: Multiple ways to interact with audio data:
//!   - Frame-wise iteration (all channels at one time point)
//!   - Channel-wise iteration (process channels independently)
//!   - Block-wise processing with overlap support
//!   - Direct buffer access when needed
//!
//! - **Format Support**:
//!   - Standard PCM and IEEE float formats
//!   - Extensible WAV format support
//!   - Multi-channel audio
//!   - Basic metadata through FACT chunks
//!   - Preliminary LIST-INFO chunk support
//!
//! - **Optional Features**:
//!   - `ndarray`: Matrix operations via ndarray integration
//!   - `resampling`: High-quality sample rate conversion
//!   - `simd`: SIMD acceleration for resampling operations
//!   - `colored`: Enhanced debug output
//!   - `logging`: Detailed operation logging
//!
//! ## Quick Examples
//!
//! ### Reading Audio
//!
//! ```no_run
//! use wavers::{Wav, read};
//!
//! // Quick read with automatic conversion to f32
//! let (samples, sample_rate) = read::<f32, _>("input.wav")?;
//!
//! // More control over the process
//! let mut wav = Wav::<f32>::from_path("input.wav")?;
//! println!("Channels: {}", wav.n_channels());
//! println!("Sample Rate: {}", wav.sample_rate());
//! println!("Duration: {}s", wav.duration());
//!
//! let samples = wav.read()?;
//! ```
//!
//! ### Writing Audio
//!
//! ```no_run
//! use wavers::write;
//!
//! // Write a simple sine wave
//! let sample_rate = 44100;
//! let freq = 440.0; // Hz
//! let samples: Vec<f32> = (0..sample_rate)
//!     .map(|i| {
//!         let t = i as f32 / sample_rate as f32;
//!         (t * freq * 2.0 * std::f32::consts::PI).sin()
//!     })
//!     .collect();
//!
//! write("sine.wav", &samples, sample_rate as i32, 1)?;
//! ```
//!
//! ### Processing Audio
//!
//! ```no_run
//! use wavers::Wav;
//!
//! let mut wav = Wav::<f32>::from_path("stereo.wav")?;
//!
//! // Frame-wise processing (all channels together)
//! for frame in wav.frames() {
//!     let left = frame[0];
//!     let right = frame[1];
//!     // Process stereo frame...
//! }
//!
//! // Channel-wise processing
//! for channel in wav.channels() {
//!     // Process entire channel...
//! }
//!
//! // Block processing with overlap
//! for block in wav.blocks(1024, 512) {
//!     // Process 1024-sample blocks with 512 samples overlap
//! }
//! ```
//!
//! ### Sample Type Conversion
//!
//! ```no_run
//! use wavers::{Wav, ConvertTo};
//!
//! // Automatic conversion during read
//! let wav = Wav::<f32>::from_path("pcm16.wav")?;
//! let samples = wav.read()?; // Automatically converts i16 to f32
//!
//! // Explicit conversion
//! let wav = Wav::<i16>::from_path("pcm16.wav")?;
//! let samples = wav.read()?.convert::<f32>();
//! ```
//!
//! ## Optional Features
//!
//! ### NDArray Integration
//!
//! Enable with `cargo add wavers --features ndarray`
//!
//! ```no_run
//! # #[cfg(feature = "ndarray")]
//! use wavers::{Wav, AsNdarray};
//! # #[cfg(feature = "ndarray")]
//! use ndarray::Array2;
//!
//! # #[cfg(feature = "ndarray")]
//! fn example() -> wavers::WaversResult<()> {
//!     let wav = Wav::<f32>::from_path("audio.wav")?;
//!     let (audio_matrix, sample_rate): (Array2<f32>, i32) = wav.as_ndarray()?;
//!     // audio_matrix shape: (samples, channels)
//!     Ok(())
//! }
//! ```
//!
//! ### Resampling
//!
//! Enable with `cargo add wavers --features resampling`
//!
//! ```no_run
//! # #[cfg(feature = "resampling")]
//! use wavers::{Wav, resample};
//!
//! # #[cfg(feature = "resampling")]
//! fn example() -> wavers::WaversResult<()> {
//!     let mut wav = Wav::<f32>::from_path("input.wav")?;
//!     let target_fs = 48000;
//!     let resampled = resample(&mut wav, target_fs)?;
//!     Ok(())
//! }
//! ```
//!
//! ## Implementation Details
//!
//! WaveRs uses a combination of techniques to achieve good performance:
//!
//! - Zero-copy operations where possible
//! - Efficient memory allocation strategies
//! - SIMD optimizations for resampling (when enabled)
//! - Stream-based processing for large files
//!
//! ## Error Handling
//!
//! WaveRs uses the `WaversResult<T>` type alias for operations that can fail:
//!
//! ```no_run
//! pub type WaversResult<T> = Result<T, WaversError>;
//! ```
//!
//! Common error cases include:
//! - Invalid file format or corrupted files
//! - Unsupported format variations
//! - I/O errors during reading/writing
//! - Invalid operations (e.g., incorrect channel counts)
//!
//! ## Development Status
//!
//! WaveRs is stable and functional but under periodic development. While the core
//! API is stable, new features and optimizations are added periodically. Current areas
//! of development include:
//!
//! - Enhanced metadata support (LIST-INFO chunks)
//! - Additional optimization opportunities
//! - Expanded format support
//!
//! Contributions are welcome! See the repository README for details.
//!
#![cfg_attr(
    feature = "simd",
    doc = r##"
## SIMD
WaveRs uses the `portable_simd` feature to enable SIMD instructions for the resampling.
"##
)]
#![feature(portable_simd)]

pub mod chunks;
pub mod conversion;
pub mod core;

#[cfg(feature = "resampling")]
pub mod resampling;

pub mod error;
pub mod header;

pub mod iter;
pub mod wav_type;
use error::FormatError;
use i24::i24;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub use crate::chunks::{FactChunk, FmtChunk, ListChunk, DATA, FACT, LIST, RIFF, WAVE};
pub use crate::conversion::{AudioSample, ConvertSlice, ConvertTo};
pub use crate::core::{wav_spec, ReadSeek, Samples, Wav};
pub use crate::error::{WaversError, WaversResult};
pub use crate::header::WavHeader;
use crate::wav_type::{FormatCode, WavType};

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

    let f = File::create(&fp)?;
    let mut buf_writer = BufWriter::new(f);

    match new_header.fmt_chunk.format {
        FormatCode::WAV_FORMAT_PCM | FormatCode::WAV_FORMAT_IEEE_FLOAT => {
            let header_bytes = new_header.as_base_bytes();
            buf_writer.write_all(&header_bytes)?;
        }
        FormatCode::WAVE_FORMAT_EXTENSIBLE => {
            let header_bytes = new_header.as_extended_bytes();
            buf_writer.write_all(&header_bytes)?;
        }
        _ => {
            return Err(FormatError::InvalidTypeId("Invalid type ID").into());
        }
    }

    buf_writer.write_all(&DATA)?;
    let data_size_bytes = samples_bytes.len() as u32; // write up to the data size
    buf_writer.write_all(&data_size_bytes.to_ne_bytes())?; // write the data size
    buf_writer.write_all(&samples_bytes)?; // write the data
    log!(
        log::Level::Debug,
        "Wrote wav file to {}",
        fp.as_ref().display()
    );
    Ok(())
}

/// Writes wav samples to disk using the supplied header supplied.
#[inline(always)]
pub fn write_with_header<T: AudioSample, P: AsRef<Path>>(
    fp: P,
    samples: &[T],
    wav_header: &WavHeader,
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
    let sample_rate = wav_header.fmt_chunk.sample_rate;
    let n_channels = wav_header.fmt_chunk.channels;

    write(fp, samples, sample_rate, n_channels)
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
