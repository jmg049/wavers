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

pub mod sample;
pub mod wave;
// pub mod iter;
// pub mod iter;

pub use sample::{AudioConversion, IterAudioConversion, Sample};
pub use wave::{
    read, signal_channels, signal_duration, signal_info, signal_sample_rate, write, SignalInfo,
    WavFile,
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
