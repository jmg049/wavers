pub mod wave;

pub use wave::WavFile;

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
        path::Path,
        str::FromStr,
    };

    use ndarray::Array1;

    use crate::wave::{signal_duration, signal_sample_rate, WavFile};

    #[test]
    fn reads_pcm_i16_correctly() {
        let test_file = Path::new("./test.wav");
        let expected_length: usize = 60 * 16000; // Duration * sample rate
        let expected_output: Array1<i16> =
            match read_text_to_vec::<i16>(Path::new("./out.txt"), Some(expected_length)) {
                Ok(v) => Array1::from_vec(v),
                Err(err) => panic!("{:?}", err),
            };

        let wav_data = match WavFile::new(test_file) {
            Ok(data) => data.read_pcm_i16(),
            Err(err) => panic!("{:?}", err),
        };

        let mut idx = 0;
        for (expected, actual) in std::iter::zip(expected_output, wav_data) {
            assert_eq!(
                expected, actual,
                "Expected {} - Actual {}\nFailed at index {}",
                expected, actual, idx
            );
            idx += 1;
        }
    }

    fn read_text_to_vec<T: FromStr>(
        fp: &Path,
        duration: Option<usize>,
    ) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        <T as FromStr>::Err: std::error::Error + 'static,
    {
        let d = match duration {
            Some(dur) => dur,
            None => 16000,
        };
        let mut data = Vec::with_capacity(d);
        let f_in: File = File::open(fp)?;
        let mut line = String::new();
        let mut buf_rdr = BufReader::new(f_in);
        buf_rdr.read_line(&mut line)?;
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
        Ok(data)
    }

    #[test]
    fn test_signal_duration() {
        let signal_fp = Path::new("./test.wav");
        let duration = signal_duration(signal_fp).unwrap();
        assert_eq!(duration, 60);
    }

    // write a test for the signal_sample_rate function
    #[test]
    fn test_signal_sample_rate() {
        let signal_fp = Path::new("./test.wav");
        let sample_rate = signal_sample_rate(signal_fp).unwrap();
        assert_eq!(sample_rate, 16000);
    }
}
