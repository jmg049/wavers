// Correctness and logic
#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)] // Duplicate match arms
#![warn(clippy::unreachable)] // Detects unreachable code

// Performance-focused
#![warn(clippy::inefficient_to_string)] // `format!("{}", x)` vs `x.to_string()`
#![warn(clippy::map_clone)] // Cloning inside `map()` unnecessarily
#![warn(clippy::unnecessary_to_owned)] // Detects redundant `.to_owned()` or `.clone()`
#![warn(clippy::large_stack_arrays)] // Helps avoid stack overflows
#![warn(clippy::box_collection)] // Warns on boxed `Vec`, `String`, etc.
#![warn(clippy::vec_box)] // Avoids using `Vec<Box<T>>` when unnecessary
#![warn(clippy::needless_collect)] // Avoids `.collect().iter()` chains

// Style and idiomatic Rust
#![warn(clippy::redundant_clone)] // Detects unnecessary `.clone()`
#![warn(clippy::identity_op)] // e.g., `x + 0`, `x * 1`
#![warn(clippy::needless_return)] // Avoids `return` at the end of functions
#![warn(clippy::let_unit_value)] // Avoids binding `()` to variables
#![warn(clippy::manual_map)] // Use `.map()` instead of manual `match`

// Maintainability
#![warn(clippy::missing_panics_doc)] // Docs for functions that might panic
#![warn(clippy::missing_safety_doc)] // Docs for `unsafe` functions
#![warn(clippy::missing_const_for_fn)] // Suggests making eligible functions `const`
#![warn(clippy::missing_errors_doc)] // Docs for functions that return `Result`

//!
//!
//!
//!
//!

pub mod chunks;
pub mod error;
pub mod format_code;
pub mod header;
pub mod iter;
pub mod mmap_reader;
pub mod wav;
pub mod wav_type;
pub use crate::chunks::{DATA, FACT, FMT, INFO, LIST, RIFF, WAVE};

use audio_sample::{AudioSample, ConvertSequence, ConvertTo, I24, Samples};
use chunks::fmt::FmtChunk;
pub use error::{WaversErrorKind, WaversResult};
pub use format_code::FormatCode;
use std::any::type_name;

use std::{
    io::{BufReader, Read, Seek, Write},
    path::Path,
};
pub use wav::{Wav, WavInfo};
pub use wav_type::WavType;

use std::any::TypeId;
use std::fs::File;
use std::io::BufWriter;

pub trait ReadSeek: Read + Seek {}

impl<T: Read + Seek> ReadSeek for T {}

pub fn read_with<T: AudioSample, R: ReadSeek>(reader: R) -> WaversResult<(Samples<T>, u32, u16)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let mut wav = Wav::from_reader(reader)?;
    let fmt_chunk = wav.fmt_chunk();
    let channels = fmt_chunk.channels;
    let sample_rate = fmt_chunk.sample_rate as u32;

    let samples = wav.read()?;
    Ok((samples, sample_rate, channels))
}

pub fn read<P: AsRef<Path>, T: AudioSample>(path: P) -> WaversResult<(Samples<T>, u32, u16)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let file = std::fs::File::open(path)?;
    let buf_reader = BufReader::new(file);
    read_with::<T, _>(buf_reader)
}

pub fn write_with<W: Write, T: AudioSample>(
    mut writer: W,
    samples: &Samples<T>,
    sr: u32,
    channels: Option<u16>,
) -> WaversResult<()> {
    let channels = channels.unwrap_or(1);

    // ———————————————————————————————————————————————————————————
    // 1) Turn Samples<T> into a Vec<u8> *and* pick the final WavType
    // ———————————————————————————————————————————————————————————
    let (wav_type, data_bytes): (WavType, Vec<u8>) = {
        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<i16>() {
            let s: Samples<i16> = samples.as_sequence();
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    s.as_ptr() as *const u8,
                    s.len() * std::mem::size_of::<i16>(),
                )
            };
            (WavType::Pcm16, byte_slice.to_vec())
        } else if tid == TypeId::of::<I24>() {
            let s: Samples<I24> = samples.as_sequence();
            let mut buf = Vec::with_capacity(s.len() * 3);
            for &sample in s.as_ref() {
                let b = sample.to_le_bytes(); // [u8;4], little‑endian
                buf.extend_from_slice(&b[0..3]); // drop the sign‑extension byte
            }
            (WavType::Pcm24, buf)
        } else if tid == TypeId::of::<i32>() {
            let s: Samples<i32> = samples.as_sequence();
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    s.as_ptr() as *const u8,
                    s.len() * std::mem::size_of::<i32>(),
                )
            };
            (WavType::Pcm32, byte_slice.to_vec())
        } else if tid == TypeId::of::<f32>() {
            let s: Samples<f32> = samples.as_sequence();
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    s.as_ptr() as *const u8,
                    s.len() * std::mem::size_of::<f32>(),
                )
            };
            (WavType::Float32, byte_slice.to_vec())
        } else if tid == TypeId::of::<f64>() {
            let s: Samples<f64> = samples.as_sequence();
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    s.as_ptr() as *const u8,
                    s.len() * std::mem::size_of::<f64>(),
                )
            };
            (WavType::Float64, byte_slice.to_vec())
        } else {
            return Err(WaversErrorKind::InvalidTargetType(
                WavType::Unknown,
                type_name::<T>().to_string(),
            ));
        }
    };

    // ———————————————————————————————————————————————————————————
    // 2) Now that we know exactly what WavType we’re writing, build the fmt chunk
    // ———————————————————————————————————————————————————————————
    let fmt_chunk = match wav_type {
        WavType::Pcm16 => FmtChunk::new_pcm(channels, sr as i32, 16),
        WavType::Pcm24 => FmtChunk::new_pcm(channels, sr as i32, 24),
        WavType::Pcm32 => FmtChunk::new_pcm(channels, sr as i32, 32),
        WavType::Float32 => FmtChunk::new_float(channels, sr as i32, 32),
        WavType::Float64 => FmtChunk::new_float(channels, sr as i32, 64),
        _ => unreachable!("all other cases handled above"),
    };

    let fmt_bytes = fmt_chunk.to_bytes();
    let fmt_size = fmt_bytes.len() as u32;
    let data_size = data_bytes.len() as u32;
    // chunk_size = 4 ("WAVE") + (8 + fmt_size) + (8 + data_size)
    let riff_size = 4 + 8 + fmt_size + 8 + data_size;

    // ———————————————————————————————————————————————————————————
    // 3) Emit RIFF header, fmt chunk, data chunk, then raw bytes
    // ———————————————————————————————————————————————————————————
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    writer.write_all(b"fmt ")?;
    writer.write_all(&fmt_size.to_le_bytes())?;
    writer.write_all(&fmt_bytes)?;

    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;
    writer.write_all(&data_bytes)?;

    writer.flush()?;
    Ok(())
}

pub fn write<P: AsRef<Path>, T: AudioSample>(
    path: P,
    samples: &Samples<T>,
    sr: u32,
    channels: Option<u16>,
) -> WaversResult<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    crate::write_with(writer, samples, sr, channels)
}

pub fn info<P: AsRef<Path>>(path: P) -> WaversResult<WavInfo> {
    let file = std::fs::File::open(path)?;
    let buf_reader = BufReader::new(file);
    let mut wav = Wav::from_reader(buf_reader)?;
    Ok(wav.info())

}

#[cfg(test)]
pub(crate) mod wav_api_tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;
    const ONE_CHANNEL_WAV_I16: &str = "./test_resources/one_channel_i16.wav";
    const ONE_CHANNEL_EXPECTED_I16: &str = "./test_resources/one_channel_i16.txt";
    // helper fn available to all sub‑modules
    pub(crate) fn read_text_to_vec<T: std::str::FromStr>(
        path: &Path,
    ) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        <T as std::str::FromStr>::Err: std::error::Error + 'static,
    {
        use std::io::{BufRead, BufReader};
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut data = Vec::new();
        for line in reader.lines() {
            for sample in line?.split_whitespace() {
                data.push(sample.parse()?);
            }
        }
        Ok(data)
    }
    /// Expectation helper: load expected i16 samples from our text file
    fn expected_i16() -> Vec<i16> {
        read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16))
            .expect("failed to load expected samples")
    }

    #[test]
    fn test_info() {
        // exercise the free-standing `info` function
        let info = info(ONE_CHANNEL_WAV_I16).expect("info failed");
        assert_eq!(info.sample_rate, 16000, "sample rate");
        assert_eq!(info.channels, 1, "channels");
        assert_eq!(info.bits_per_sample, 16, "bits_per_sample");
        assert_eq!(info.bytes_per_sample, 2, "bytes_per_sample");

        // duration = n_samples / sample_rate
        let exp_count = expected_i16().len() as u32;
        assert_eq!(info.n_samples, exp_count, "n_samples");
        let exp_dur = exp_count as f32 / info.sample_rate as f32;
        assert!((info.duration - exp_dur).abs() < 1e-6, "duration");
    }

    #[test]
    fn test_read_with() {
        // exercise the free `read_with(reader)` taking a BufReader
        let f = File::open(ONE_CHANNEL_WAV_I16).expect("open file");
        let reader = BufReader::new(f);
        let (samples, sample_rate, channels) =
            read_with::<i16, _>(reader).expect("read_with failed");
        let expected = expected_i16();
        assert_eq!(samples.len(), expected.len(), "length mismatch");
        // check last sample matches
        let last = samples.len() - 1;
        assert_eq!(samples[last], expected[last], "last sample");
        assert_eq!(sample_rate, 16000, "sample rate");
        assert_eq!(channels, 1, "channels");
    }
}
