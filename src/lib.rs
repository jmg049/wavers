#![feature(slice_as_chunks)]

//!
//! The ``wavers`` crate provides a simple interface, with a powerful backend for reading and writing ``.wav`` files.
//!
//! ## Highlights:
//!
//! * Easy to use interface for reading and writing ``.wav`` files.
//! * Benchmarking shows it is faster than ``hound`` for reading ``.wav`` files. Hound is still better for writing.
//! * Currently supports reading and writing of ``i16``, ``i32``, ``f32`` and ``f64`` wav files.
//! * Supports easy conversion between different types of ``.wav`` files.
//! * Supports reading and writing of multi-channel ``.wav`` files.
//! * Has optional support reading ``.wav`` files as ``ndarray`` arrays.
//!
//! # Examples
//! The following examples show how to read and write ``.wav`` files in wavers. For more fine-grained control over the reading and writing of ``.wav`` files see the ``wavers::WavFile`` struct.
//! ## Reading a ``.wav`` file
//! To do this we can simple use the ``wavers::read`` read function and an Option ``Sample`` specifying the desired format. ``None`` will indicate to use the same format as the file.
//! A ``wavers::sample::Sample`` is used to represent all of the different types of samples that can be read from a ``.wav`` file.
//!
//!
//! ```rust
//! use std::path::Path;
//! use wavers::{read, Sample};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let fp = Path::new("path/to/file.wav");
//!
//!     // if the file is already in the format we want to work with
//!     let signal: Vec<Sample> = read(fp, None).expect("Failed to read wav file");
//!
//!     // or if we want to convert to something else
//!     let f32_signal: Vec<Sample> = read(fp, Some(Sample::F32(0.0))).expect("Failed to read wav file as f32");
//!     Ok(())
//! }
//! ```
//!
//! ## Writing a ``.wav`` file
//! Writing a wav file can be done by using the ``wavers::write`` function. This function takes a path to write to, a vector of samples, an optional sample type, a sample rate and the number of channels.
//!
//!
//! ```rust
//! use std::path::Path;
//! use wavers::{read, write, Sample};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let fp = Path::new("path/to/file.wav");
//!
//!     let mut signal: Vec<Sample> = read(fp, Some(Sample::I16(0))).expect("Failed to read wav file");
//!     let new_fp = Path::new("path/to/new_file.wav");
//!     let sample_rate = 16000;
//!     let n_channels = 1;
//!     write(new_fp, &mut signal, Some(Sample::F32(0.0)), sample_rate, n_channels).expect("Failed to write wav file");
//!     Ok(())
//! }
//! ```
//!
//! ## Crate Status
//! * Still very much in development.
//! * The API is not stable and is subject to change.
//! * The API is not fully documented yet.
//! * The API is not fully tested yet.
//! * The API is not fully benchmarked yet.
//! * Targeting a 1.0 release in the next few months.
//! * 1.0 release will be fully documented, tested and benchmarked.
//! * 1.0 release will have a stable API.
//! * 1.0 release will have the following features:
//!    * Reading and writing of ``.wav`` files.
//!    * Conversion between different types of ``.wav`` files.
//!    * Reading and writing of ``.wav`` files as ``ndarray`` arrays (maybe in parallel in future if it is worth it).
//!    * Iteration over ``.wav`` files in windows which are lazy loaded (i.e. only loaded into memory when needed).
//!    * Iteration over each cahnnel of a ``.wav`` file.
//!
//! ## Crate Feature Flags
//! The following feature flags are available:
//! * ``ndarray``: Enables reading and writing of ``.wav`` files as ``ndarray`` arrays by using the ``wavers::IntoArray`` trait.
//!
//! ## Crate Benchmarks
//! The following benchmarks are available:
//! * ``wavers::read`` vs ``hound::WavReader``: This benchmark compares the performance of ``wavers::read`` to ``hound::WavReader``. The benchmark is run on both mono and stereo ``.wav`` files. The results are shown below:
//!     * Mono ``.wav`` file:
//!
//!     * Stereo ``.wav`` file:
//!
<<<<<<< Updated upstream
=======
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
mod mma_reader;
>>>>>>> Stashed changes

pub mod sample;
pub mod wave;
// pub mod iter;
// pub mod iter;

<<<<<<< Updated upstream
pub use sample::{AudioConversion, IterAudioConversion, Sample};
pub use wave::{
    read, signal_channels, signal_duration, signal_info, signal_sample_rate, write, SignalInfo,
    WavFile,
=======
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub use crate::conversion::{AudioSample, ConvertSlice, ConvertTo};

pub use crate::chunks::{FactChunk, FmtChunk, ListChunk, DATA, FACT, LIST, RIFF, WAVE};
pub use crate::core::{wav_spec, ReadSeek, Samples, Wav};
pub use crate::error::{WaversError, WaversResult};
pub use crate::header::WavHeader;
pub use crate::wav_type::{format_info_to_wav_type, wav_type_to_format_info, FormatCode, WavType};
pub use crate::mma_reader::MMapReader;
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
>>>>>>> Stashed changes
};

#[cfg(test)]
mod tests {
    use approx_eq::assert_approx_eq;
    use std::{fs::File, io::BufRead, path::Path, str::FromStr};

    use super::*;
    use crate::{
        sample::Sample, signal_channels, signal_duration, signal_info, signal_sample_rate, write,
    };

    #[test]
    fn can_read_one_channel_i16() {
        let fp = Path::new("./test_resources/one_channel.wav");
        let signal = read(fp, Some(Sample::I16(0))).expect("Failed to read wav file");

        let expected_signal =
            read_text_to_vec::<i16>(Path::new("./test_resources/one_channel_i16.txt"))
                .expect("Failed to read expected signal")
                .iter()
                .map(|s| Sample::I16(*s))
                .collect::<Vec<Sample>>();
        for (expected, actual) in std::iter::zip(expected_signal, signal) {
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn can_read_two_channel_i16() {
        let fp = Path::new("./test_resources/two_channel.wav");
        let signal = read(fp, Some(Sample::I16(0))).expect("Failed to read wav file");

        let expected_signal =
            read_text_to_vec::<i16>(Path::new("./test_resources/two_channel_i16.txt"))
                .expect("Failed to read expected signal")
                .iter()
                .map(|s| Sample::I16(*s))
                .collect::<Vec<Sample>>();
        for (expected, actual) in std::iter::zip(expected_signal, signal) {
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn can_read_one_channel_i16_as_f32() {
        let fp = Path::new("./test_resources/one_channel.wav");
        let signal = read(fp, Some(Sample::F32(0.0))).expect("Failed to read wav file");

        let expected_signal =
            read_text_to_vec::<f32>(Path::new("./test_resources/one_channel_f32.txt"))
                .expect("Failed to read expected signal")
                .iter()
                .map(|s| Sample::F32(*s))
                .collect::<Vec<Sample>>();
        for (expected, actual) in std::iter::zip(expected_signal, signal) {
            assert_approx_eq!(expected.as_f64(), actual.as_f64(), 1e-4);
        }
    }

    #[test]
    fn can_read_two_channel_i16_as_f32() {
        let fp = Path::new("./test_resources/two_channel.wav");
        let signal = read(fp, Some(Sample::F32(0.0))).expect("Failed to read wav file");

        let expected_signal =
            read_text_to_vec::<f32>(Path::new("./test_resources/two_channel_f32.txt"))
                .expect("Failed to read expected signal")
                .iter()
                .map(|s| Sample::F32(*s))
                .collect::<Vec<Sample>>();
        for (expected, actual) in std::iter::zip(expected_signal, signal) {
            assert_approx_eq!(expected.as_f64(), actual.as_f64(), 1e-4);
        }
    }

    #[test]
    fn can_write_one_channel_i16() {
        let fp = Path::new("./test_resources/one_channel.wav");
        let mut signal = read(fp, Some(Sample::I16(0))).expect("Failed to read wav file");
        use uuid::Uuid;

        // create tmp_out dir if it doesn't exist
        let tmp_out_dir = Path::new("./test_resources/tmp_out");
        if !tmp_out_dir.exists() {
            std::fs::create_dir(tmp_out_dir).expect("Failed to create tmp_out dir");
        }

        let id = Uuid::new_v4();
        let out_id = format!("./test_resources/tmp_out/one_channel_out_{}.wav", id);
        let mut output_fp = Path::new(&out_id);

        write(&mut output_fp, &mut signal, Some(Sample::I16(0)), 1, 16000)
            .expect("Failed to write wav file");

        let output_signal =
            read(output_fp, Some(Sample::I16(0))).expect("Failed to read output wav file");
        for (expected, actual) in std::iter::zip(signal, output_signal) {
            assert_eq!(
                expected, actual,
                "Expected: {}, Actual: {}",
                expected, actual
            );
        }

        std::fs::remove_file(output_fp).expect("Failed to remove output file");
    }

    #[test]
    fn can_write_one_channel_f32() {
        let fp = Path::new("./test_resources/one_channel.wav");
        let mut signal = read(fp, Some(Sample::I16(0))).expect("Failed to read wav file");

        use uuid::Uuid;

        // create tmp_out dir if it doesn't exist
        let tmp_out_dir = Path::new("./test_resources/tmp_out");
        if !tmp_out_dir.exists() {
            std::fs::create_dir(tmp_out_dir).expect("Failed to create tmp_out dir");
        }

        let id = Uuid::new_v4();
        let out_id = format!("./test_resources/tmp_out/one_channel_out_{}.wav", id);
        let mut output_fp = Path::new(&out_id);
        write(
            &mut output_fp,
            &mut signal,
            Some(Sample::F32(0.0)),
            1,
            16000,
        )
        .expect("Failed to write wav file");

        let output_signal = match read(output_fp, Some(Sample::F32(0.0))) {
            Ok(s) => s,
            Err(e) => panic!("Failed to read output wav file: {}", e),
        };
        for (expected, actual) in std::iter::zip(signal.as_f32(), output_signal) {
            assert_approx_eq!(expected.as_f64(), actual.as_f64(), 1e-4);
        }
        std::fs::remove_file(output_fp).expect("Failed to remove output file");
    }

    #[test]
    fn can_write_two_channel_i16() {
        let fp = Path::new("./test_resources/two_channel.wav");
        let mut signal = read(fp, Some(Sample::I16(0))).expect("Failed to read wav file");

        use uuid::Uuid;

        // create tmp_out dir if it doesn't exist
        let tmp_out_dir = Path::new("./test_resources/tmp_out");
        if !tmp_out_dir.exists() {
            std::fs::create_dir(tmp_out_dir).expect("Failed to create tmp_out dir");
        }

        let id = Uuid::new_v4();
        let out_id = format!("./test_resources/tmp_out/one_channel_out_{}.wav", id);
        let mut output_fp = Path::new(&out_id);
        write(&mut output_fp, &mut signal, Some(Sample::I16(0)), 2, 16000)
            .expect("Failed to write wav file");

        let output_signal = match read(output_fp, Some(Sample::I16(0))) {
            Ok(s) => s,
            Err(e) => panic!("Failed to read output wav file: {}", e),
        };
        for (expected, actual) in std::iter::zip(signal, output_signal) {
            assert_eq!(
                expected, actual,
                "Expected: {}, Actual: {}",
                expected, actual
            );
        }
        std::fs::remove_file(output_fp).expect("Failed to remove output file");
    }

    #[test]
    fn can_write_two_channel_f32() {
        let fp = Path::new("./test_resources/two_channel.wav");
        let mut signal = read(fp, Some(Sample::I16(0))).expect("Failed to read wav file");

        use uuid::Uuid;

        // create tmp_out dir if it doesn't exist
        let tmp_out_dir = Path::new("./test_resources/tmp_out");
        if !tmp_out_dir.exists() {
            std::fs::create_dir(tmp_out_dir).expect("Failed to create tmp_out dir");
        }

        let id = Uuid::new_v4();
        let out_id = format!("./test_resources/tmp_out/one_channel_out_{}.wav", id);
        let mut output_fp = Path::new(&out_id);
        write(
            &mut output_fp,
            &mut signal,
            Some(Sample::F32(0.0)),
            2,
            16000,
        )
        .expect("Failed to write wav file");

        let output_signal = match read(output_fp, Some(Sample::F32(0.0))) {
            Ok(s) => s,
            Err(e) => panic!("Failed to read output wav file: {}", e),
        };
        for (expected, actual) in std::iter::zip(signal.as_f32(), output_signal) {
            assert_approx_eq!(expected.as_f64(), actual.as_f64(), 1e-4);
        }
        std::fs::remove_file(output_fp).expect("Failed to remove output file");
    }

    #[test]
    fn test_signal_duration() {
        let signal_fp = Path::new("./test_resources/one_channel.wav");
        let duration = signal_duration(signal_fp).unwrap();
        assert_eq!(duration, 10);
    }

    #[test]
    fn test_signal_duration_invalid_file() {
        let signal_fp = Path::new("./test_resources/invalid.wav");
        let duration = signal_duration(signal_fp);
        assert!(duration.is_err());
    }

    #[test]
    fn test_signal_sample_rate() {
        let signal_fp = Path::new("./test_resources/one_channel.wav");
        let sample_rate = signal_sample_rate(signal_fp).unwrap();
        assert_eq!(sample_rate, 16000);
    }

    #[test]
    fn test_signal_sample_rate_invalid_file() {
        let signal_fp = Path::new("./test_resources/invalid.wav");
        let sample_rate = signal_sample_rate(signal_fp);
        assert!(sample_rate.is_err());
    }

    #[test]
    fn test_n_channels() {
        let signal_fp = Path::new("./test_resources/one_channel.wav");
        let n_channels = signal_channels(signal_fp).unwrap();
        assert_eq!(n_channels, 1);

        let signal_fp = Path::new("./test_resources/two_channel.wav");
        let n_channels = signal_channels(signal_fp).unwrap();
        assert_eq!(n_channels, 2);
    }

    #[test]
    fn test_n_channels_invalid_file() {
        let signal_fp = Path::new("./test_resources/invalid.wav");
        let n_channels = signal_channels(signal_fp);
        assert!(n_channels.is_err());
    }

    #[test]
    fn test_signal_info() {
        let signal_fp = Path::new("./test_resources/one_channel.wav");
        let info = signal_info(signal_fp).unwrap();
        assert_eq!(info.duration, 10);
        assert_eq!(info.sample_rate, 16000);
        assert_eq!(info.channels, 1);

        let signal_fp = Path::new("./test_resources/two_channel.wav");
        let info = signal_info(signal_fp).unwrap();
        assert_eq!(info.duration, 10);
        assert_eq!(info.sample_rate, 16000);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_signal_info_invalid_file() {
        let signal_fp = Path::new("./test_resources/invalid.wav");
        let info = signal_info(signal_fp);
        assert!(info.is_err());
    }

    // create a test for reading the fmt chunk of a wav file
    #[test]
    fn test_read_fmt_chunk() {
        let signal_fp = Path::new("./test_resources/one_channel.wav");
        println!("{:?}", wave::signal_duration(signal_fp).unwrap());
        let fmt_chunk = wave::FmtChunk::from_path(signal_fp).unwrap();
        assert_eq!(fmt_chunk.format, 1);
        assert_eq!(fmt_chunk.channels, 1);
        assert_eq!(fmt_chunk.sample_rate, 16000);
        assert_eq!(fmt_chunk.byte_rate, 32000);
        assert_eq!(fmt_chunk.block_align, 2);
        assert_eq!(fmt_chunk.bits_per_sample, 16);
    }

    #[cfg(feature = "ndarray")]
    use crate::wave::IntoArray;

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_signal_to_ndarray_one_channel() {
        let signal_fp = Path::new("./test_resources/one_channel.wav");
        let signal: Vec<Sample> = read(signal_fp, Some(Sample::I16(0))).unwrap();
        let ndarray = signal.clone().into_array(1).unwrap(); // need to clone since normally the into_array function will consume the vector
        let mut idx = 0;
        for (expected, actual) in std::iter::zip(signal, ndarray) {
            assert_eq!(
                actual, expected,
                "Expected {} - Actual {}\nFailed at index {}",
                expected, actual, idx
            );
            idx += 1;
        }
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_signal_to_ndarray_two_channel() {
        let signal_fp = Path::new("./test_resources/two_channel.wav");
        let signal: Vec<Sample> = read(signal_fp, Some(Sample::I16(0))).unwrap();
        let ndarray = signal.clone().into_array(2).unwrap(); // need to clone since normally the into_array function will consume the vector
        let mut idx = 0;
        for (expected, actual) in std::iter::zip(signal, ndarray) {
            assert_eq!(
                actual, expected,
                "Expected {} - Actual {}\nFailed at index {}",
                expected, actual, idx
            );
            idx += 1;
        }
    }

    fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(std::io::BufReader::new(file).lines())
    }

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
