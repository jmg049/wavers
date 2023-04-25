#![feature(slice_as_chunks)]
mod sample;
mod signal;
pub mod wave;

// pub use signal::;
pub use sample::SampleType;
pub use wave::{
    signal_channels, signal_duration, signal_info, signal_sample_rate, write_wav_as, SignalInfo,
    WavFile,
};

#[cfg(test)]
mod tests {
    use approx_eq::assert_approx_eq;
    use ndarray::Array2;
    use std::{fs::File, io::BufRead, path::Path, str::FromStr};

    use crate::{
        sample::SampleType, signal_channels, signal_duration, signal_sample_rate, wave::WavFile,
        write_wav_as, SignalInfo, signal_info
    };

    #[test]
    fn read_one_channel_i16() {
        let test_file = Path::new("./test_resources/one_channel.wav");
        let wav_file = match WavFile::from_file(test_file) {
            Ok(wav_file) => wav_file,
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };
        let data_vec: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let expected_output: Array2<i16> =
            Array2::from_shape_vec((data_vec.len(), 1), data_vec).unwrap();
        let i16_data = wav_file.read(None);

        let mut idx = 0;
        for (expected, actual) in std::iter::zip(expected_output, i16_data) {
            assert_eq!(
                actual, expected,
                "Expected {} - Actual {}\nFailed at index {}",
                expected, actual, idx
            );
            idx += 1;
        }
    }

    #[test]
    fn read_invalid_file() {
        let signal_fp = Path::new("./test_resources/invalid.wav");
        let wav_file = WavFile::from_file(signal_fp);
        assert!(wav_file.is_err());
    }

    #[test]
    fn read_write_one_channel_i16() {
        let test_file = Path::new("./test_resources/one_channel.wav");
        let wav_file = match WavFile::from_file(test_file) {
            Ok(wav_file) => wav_file,
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };
        let data_vec: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let _expected_output: Array2<i16> =
            Array2::from_shape_vec((data_vec.len(), 1), data_vec).unwrap();
        let i16_data = wav_file.read(None);
        write_wav_as(
            Path::new("./test_resources/write_one_channel_i16.wav"),
            &i16_data,
            None,
            16000,
        )
        .unwrap();

        let test_wav_file =
            match WavFile::from_file(Path::new("./test_resources/write_one_channel_i16.wav")) {
                Ok(wav_file) => wav_file,
                Err(err) => {
                    eprintln!("{}", err);
                    return;
                }
            };

        let test_i16_data = test_wav_file.read(None);

        let mut idx = 0;
        for (expected, actual) in std::iter::zip(test_i16_data, i16_data) {
            assert_eq!(
                actual, expected,
                "Expected {} - Actual {}\nFailed at index {}",
                expected, actual, idx
            );
            idx += 1;
        }
    }

    #[test]
    fn read_one_channel_f32() {
        let test_file = Path::new("./test_resources/one_channel.wav");
        let wav_file = match WavFile::from_file(test_file) {
            Ok(wav_file) => wav_file,
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };
        let data_vec: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();
        let expected_output: Array2<f32> =
            Array2::from_shape_vec((data_vec.len(), 1), data_vec).unwrap();
        let f32_data = wav_file.read(Some(SampleType::F32(0.0)));

        let mut idx = 0;
        for (expected, actual) in std::iter::zip(expected_output, f32_data) {
            let sample: f64 = match actual {
                SampleType::F32(f) => f as f64,
                _ => panic!("Expected F32"),
            };
            assert_approx_eq!(expected as f64, sample, 0.0001);
            idx += 1;
        }
    }

    #[test]
    fn read_write_one_channel_f32() {
        let test_file = Path::new("./test_resources/one_channel.wav");
        let wav_file = match WavFile::from_file(test_file) {
            Ok(wav_file) => wav_file,
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };
        let f32_data = wav_file.read(Some(SampleType::F32(0.0)));
        write_wav_as(
            Path::new("./test_resources/write_one_channel_f32.wav"),
            &f32_data,
            Some(SampleType::F32(0.0)),
            16000,
        )
        .unwrap();

        let test_wav_file =
            match WavFile::from_file(Path::new("./test_resources/write_one_channel_f32.wav")) {
                Ok(wav_file) => wav_file,
                Err(err) => {
                    eprintln!("{}", err);
                    return;
                }
            };
        assert_eq!(
            10,
            signal_duration(Path::new("./test_resources/write_one_channel_f32.wav")).unwrap(),
            "Duration of written file does not match, expected 10 seconds"
        );
        assert_eq!(
            1,
            signal_channels(Path::new("./test_resources/write_one_channel_f32.wav")).unwrap(),
            "Number of channels of written file does not match, expected 1"
        );
        assert_eq!(
            16000,
            signal_sample_rate(Path::new("./test_resources/write_one_channel_f32.wav")).unwrap(),
            "Sample rate of written file does not match, expected 16000"
        );
        let test_f32_data = test_wav_file.read(Some(SampleType::F32(0.0)));

        let mut idx = 0;
        for (_expected, actual) in std::iter::zip(test_f32_data, f32_data) {
            let expected_sample: f64 = match actual {
                SampleType::F32(f) => f as f64,
                _ => panic!("Expected F32"),
            };

            let actual_sample: f64 = match actual {
                SampleType::F32(f) => f as f64,
                _ => panic!("Expected F32"),
            };
            assert_approx_eq!(expected_sample, actual_sample, 0.0001);
            idx += 1;
        }
    }

    #[test]
    fn read_two_channel_i16() {
        let test_file = Path::new("./test_resources/two_channel.wav");
        let wav_file = match WavFile::from_file(test_file) {
            Ok(wav_file) => wav_file,
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };
        let data_vec: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/two_channel_i16.txt")).unwrap();
        let expected_output: Array2<i16> =
            Array2::from_shape_vec((data_vec.len() / 2, 2), data_vec).unwrap();
        let i16_data = wav_file.read(None);

        let mut idx = 0;
        for (expected, actual) in std::iter::zip(expected_output, i16_data) {
            assert_eq!(
                actual, expected,
                "Expected {} - Actual {}\nFailed at index {}",
                expected, actual, idx
            );
            idx += 1;
        }
    }

    #[test]
    fn read_write_two_channel_i16() {
        let test_file = Path::new("./test_resources/two_channel.wav");
        let wav_file = match WavFile::from_file(test_file) {
            Ok(wav_file) => wav_file,
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };
        let i16_data = wav_file.read(None);
        write_wav_as(
            Path::new("./test_resources/write_two_channel_i16.wav"),
            &i16_data,
            None,
            16000,
        )
        .unwrap();

        let test_wav_file =
            match WavFile::from_file(Path::new("./test_resources/write_two_channel_i16.wav")) {
                Ok(wav_file) => wav_file,
                Err(err) => {
                    eprintln!("{}", err);
                    return;
                }
            };
        assert_eq!(
            10,
            signal_duration(Path::new("./test_resources/write_two_channel_i16.wav")).unwrap(),
            "Duration of written file does not match, expected 10 seconds"
        );
        assert_eq!(
            2,
            signal_channels(Path::new("./test_resources/write_two_channel_i16.wav")).unwrap(),
            "Number of channels of written file does not match, expected 2"
        );
        assert_eq!(
            16000,
            signal_sample_rate(Path::new("./test_resources/write_two_channel_i16.wav")).unwrap(),
            "Sample rate of written file does not match, expected 16000"
        );
        let test_i16_data = test_wav_file.read(None);

        let mut idx = 0;
        for (expected, actual) in std::iter::zip(test_i16_data, i16_data) {
            assert_eq!(
                actual, expected,
                "Expected {} - Actual {}\nFailed at index {}",
                expected, actual, idx
            );
            idx += 1;
        }
    }

    #[test]
    fn read_two_channel_f32() {
        let test_file = Path::new("./test_resources/two_channel.wav");
        let wav_file = match WavFile::from_file(test_file) {
            Ok(wav_file) => wav_file,
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };
        let data_vec: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/two_channel_f32.txt")).unwrap();
        let expected_output: Array2<f32> =
            Array2::from_shape_vec((data_vec.len() / 2, 2), data_vec).unwrap();
        let f32_data = wav_file.read(Some(SampleType::F32(0.0)));

        let mut idx = 0;
        for (expected, actual) in std::iter::zip(expected_output, f32_data) {
            let sample: f64 = match actual {
                SampleType::F32(f) => f as f64,
                _ => panic!("Expected F32"),
            };
            assert_approx_eq!(expected as f64, sample, 0.0001);
            idx += 1;
        }
    }

    #[test]
    fn read_write_two_channel_f32() {
        let test_file = Path::new("./test_resources/two_channel.wav");
        let wav_file = match WavFile::from_file(test_file) {
            Ok(wav_file) => wav_file,
            Err(err) => {
                eprintln!("{}", err);
                return;
            }
        };
        let f32_data = wav_file.read(Some(SampleType::F32(0.0)));
        write_wav_as(
            Path::new("./test_resources/write_two_channel_f32.wav"),
            &f32_data,
            Some(SampleType::F32(0.0)),
            16000,
        )
        .unwrap();

        let test_wav_file =
            match WavFile::from_file(Path::new("./test_resources/write_two_channel_f32.wav")) {
                Ok(wav_file) => wav_file,
                Err(err) => {
                    eprintln!("{}", err);
                    return;
                }
            };
        assert_eq!(
            10,
            signal_duration(Path::new("./test_resources/write_two_channel_f32.wav")).unwrap(),
            "Duration of written file does not match, expected 10 seconds"
        );
        assert_eq!(
            2,
            signal_channels(Path::new("./test_resources/write_two_channel_f32.wav")).unwrap(),
            "Number of channels of written file does not match, expected 2"
        );
        assert_eq!(
            16000,
            signal_sample_rate(Path::new("./test_resources/write_two_channel_f32.wav")).unwrap(),
            "Sample rate of written file does not match, expected 16000"
        );
        let test_f32_data = test_wav_file.read(Some(SampleType::F32(0.0)));

        let mut idx = 0;
        for (expected, actual) in std::iter::zip(test_f32_data, f32_data) {
            let expected_sample: f64 = match expected {
                SampleType::F32(f) => f as f64,
                _ => panic!("Expected F32"),
            };

            let actual_sample: f64 = match actual {
                SampleType::F32(f) => f as f64,
                _ => panic!("Expected F32"),
            };
            assert_approx_eq!(expected_sample, actual_sample, 0.0001);
            idx += 1;
        }
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
