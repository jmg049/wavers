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

//! # Wavers

//! **Wavers** is a fast, lightweight, and easy-to-use library for reading and writing WAV audio files in Rust.
//! It provides comprehensive support for audio sample types, efficient conversions, and convenient iterators over audio data.
//!
//! Feedback, feature requests, and bug reports are welcome!
//!
//! ---
//!
//! ## Highlights
//!
//! - **Efficient and Lightweight**: Designed for performance-critical applications with minimal overhead.
//! - **Intuitive API**: Read, write, and manipulate WAV files with clear and concise functions.
//! - **Wide Audio Format Support**: Handles various sample formats: `i16`, `i24`, `i32`, `f32`, and `f64`.
//! - **Automatic Sample Conversion**: Built-in conversion between audio sample types via [audio_sample].
//! - **Extended Format Support**: Supports the `WAVE_FORMAT_EXTENSIBLE` format, with detailed channel configurations.
//! - **Flexible Iteration**: Efficient iteration over frames, channels, or blocks of audio data.
//! - **Chunk Management**: Advanced support for managing various WAV file chunks.
//! - **Optional Integrations**:
//!   - `ndarray`: Enables easy interoperability with the [`ndarray`](https://crates.io/crates/ndarray) crate.
//!   - `pyo3`: Allows seamless use within Python via [`PyWavers`](https://github.com/jmg049/Pywavers).
//!   - `log`: Integrated logging support with the [`log`](https://crates.io/crates/log) crate.
//!
//! ---
//!
//! ## Crate Status
//!
//! - **Stable Core API**: The core functionality is stable, with minimal breaking changes anticipated.
//! - **Planned Enhancements**:
//!   - Further optimization of the [crate::read] and [crate::write] functions. Benchmarks have shown there is room for improvement.
//!     Particularly for reading i24 samples and writing in general.
//!  - Additional features and utilities based on user feedback.
//!
//! ---
//!
//! ## Usage Examples
//!
//! ### Reading a WAV File
//!
//! Read WAV file samples easily:
//!
//! ```no_run
//! use wavers::read;
//! use audio_sample::Samples;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let path = "audio/input.wav";
//!     let (samples, sample_rate, channels): (Samples<i16>, u32, u16) = read(path)?;
//!     println!("Sample Rate: {}, Channels: {}", sample_rate, channels);
//!     println!("First sample: {:?}", samples.get(0));
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! ### Converting Audio Samples
//!
//! Convert audio samples effortlessly between formats:
//!
//! ```no_run
//! use wavers::read;
//! use audio_sample::{Samples, ConvertSequence};
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let path = "audio/input_i16.wav";
//!     let (samples_i16, sample_rate, channels): (Samples<i16>, u32, u16) = read(path)?;
//!     // Convert i16 samples to f32
//!     let samples_f32: Samples<f32> = samples_i16.convert_sequence();
//!     println!("Converted first sample: {:?}", samples_f32.get(0));
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! ### Writing WAV Files
//!
//! Quickly write samples to a WAV file:
//!
//! ```no_run
//! use wavers::{write, read};
//! use audio_sample::{Samples, ConvertSequence};
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let input_path = "audio/input.wav";
//!     let output_path = "audio/output.wav";
//!
//!     let (samples, sample_rate, channels): (Samples<f32>, u32, u16) = read(input_path)?;
//!     // Optionally convert samples to a different type
//!     let samples_i16: Samples<i16> = samples.convert_sequence();
//!
//!     // Write to file
//!     write(output_path, &samples_i16, sample_rate, Some(channels))?;
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! ### Iterating Over WAV Data
//!
//! Conveniently iterate over frames and channels:
//!
//! ```no_run
//! use wavers::{Wav, read, WavIterators};
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let path = "audio/stereo.wav";
//!     let mut wav = Wav::<BufReader<File>>::from_path(path)?;
//!
//!     // Iterate over frames
//!     for frame_result in wav.frames::<i16>() {
//!         let frame = frame_result?;
//!         println!("Frame has {} samples (one per channel)", frame.len());
//!     }
//!
//!     // Iterate over channels separately
//!     for channel_result in wav.channels::<i16>() {
//!         let channel = channel_result?;
//!         println!("Channel has {} samples", channel.len());
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! ### Getting WAV File Information
//!
//! Retrieve metadata quickly:
//!
//! ```no_run
//! use wavers::info;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let path = "audio/file.wav";
//!     let wav_info = info(path)?;
//!     println!("Sample Rate: {} Hz", wav_info.sample_rate);
//!     println!("Channels: {}", wav_info.channels);
//!     println!("Duration: {:.2} seconds", wav_info.duration);
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! ### Ndarray Integration
//! *(Enable with feature flag `ndarray`)*
//!
//! ```no_run
//!
//! use wavers::{Wav, AsNdarray};
//! use ndarray::Array2;
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let wav = Wav::<BufReader<File>::from_path("audio/input.wav")?;
//!     let (samples_array, sample_rate): (Array2<i16>, u32) = wav.as_ndarray()?;
//!     println!("Array shape: {:?}", samples_array.shape());
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! # Benchmarks & Performance
//!
//! Full benchmarking results and comparisons against other crates are available on the project [GitHub](https://github.com/jmg049/wavers).
//!
//! ## WAV Library Read and Write Benchmarks - libsndfile vs. Wavers vs. hound

//! ## 1 Methodology in Brief

//! The benchmark exercises 180 parameter combinations (5 sample formats × 3 sample‑rates × 3 durations × 2 channel counts, each for read + write).  

//! * A warm‑up phase removes cold‑start artefacts.  
//! * Up to 100 timed iterations per configuration supply robust mean/median figures.  
//! * Open → process → close is timed as a single operation, matching typical “whole‑file” workflows.  

//! ---

//! ## 2 Reading Performance  

//! | Scenario | Fastest library | Relative observations |
//! |----------|-----------------|-----------------------|
//! |Mono, 1 s, 8 kHz, PCM‑16|`wavers`|2.79 µs (Wavers) vs 5.02 µs (libsndfile), 52.3 µs (hound)|
//! |Mono, 1 s, 44.1 kHz, PCM‑16|`wavers`|5.19 µs (Wavers) vs 6.45 µs (libsndfile) vs 294 µs (hound) |
//! |Mono, 10 s, 8 kHz, PCM‑16|`libsndfile`|7.95 µs (libsndfile) vs 13.3 µs (wavers) vs 514 µs (hound) |

//! * `wavers` records the lowest latency in every short‑file test and maintains that lead for most longer files.  
//! * `libsndfile` narrows—or reverses—the gap as file duration and size increase, suggesting its internal buffering becomes advantageous.  
//! * `hound` incurs higher per‑sample cost across the board, most evident at higher sample rates.

//! ---

//! ## 3 Writing Performance  

//! | Scenario | Fastest library | Relative observations |
//! |----------|-----------------|-----------------------|
//! |Mono, 1 s, 44.1 kHz, PCM‑16|`hound`|302 µs (hound) vs 960 µs (Wavers) vs 5599 µs (libsndfile)|
//! |Mono, 10 s, 8 kHz, PCM‑16|`libsndfile`|714 µs (libsndfile) vs 982 µs (hound) vs 1121 µs (wavers) |
//! |Stereo, 1 s, 8 kHz, PCM‑32|`wavers`|181 µs (Wavers) vs 192 µs (libsndfile) vs 235 µs (hound)|

//! * For very small files at high sample rates, `hound` shows the shortest write time; its minimal header‑generation path dominates the timing.  
//! * On longer writes (≥ 10 s) and lower rates, `libsndfile` generally delivers the best throughput.  
//! * `wavers`’ write times remain close to libsndfile in most integer‑format tests and lead when 24‑ or 32‑bit PCM is involved.

//! ---

//! ## 4 Effect of Individual Parameters  

//! | Parameter | Observed scaling |
//! |-----------|------------------|
//! |**Sample format**|Integer → float adds ≤ 10 % in `wavers` and libsndfile; ~40 – 50 % in `hound`.|
//! |**Sample rate**|Per‑frame time increases almost linearly for `wavers` and libsndfile; non‑linear growth (×5–10) in `hound`.|
//! |**Duration**|Time grows sub‑linearly with duration for all libraries; fixed overheads are amortised quickly.|
//! |**Channels (1 → 2)**|Timing doubles in libsndfile and wavers (ideal scaling); increase is >2× in `hound`, implying extra per‑channel overhead.|

//! ---

//! ## 5 Practical Guidance  

//! | Requirement | Candidate libraries |
//! |-------------|--------------------|
//! |Fastest reads, broad format set (incl. 24‑/64‑bit float)|`wavers` or `libsndfile` (both close; `wavers` leads in most short‑file cases)|
//! |Sustained high‑throughput writes (minutes‑long exports)|`libsndfile`|
//! |Tiny telemetry logs, limited to PCM‑16/32/F32|`hound` (small binary, quickest small‑file writes)|
//! |Archival in IEEE F64|`wavers` or `libsndfile` (not supported by `hound`)|

//! ---

//! ## 6 Limitations and Next Steps

//! * **Format and channel scope.** Only 8–44.1 kHz, mono/stereo cases were profiled. Extending to 96 kHz+ and >2 channels would provide a fuller picture.  
//! * **Platform features.** The benchmark is single‑threaded; libraries offering asynchronous or multi‑threaded paths were not exercised.  
//! * **External resources.** Memory consumption, CPU events and low‑level cache behaviour were not recorded; a complementary study is planned.  

//! Within its defined scope, the data indicate that no single library dominates every workload:  

//! * `wavers` tends to minimise read latency and maintains competitive write times across all supported formats.  
//! * `libsndfile` achieves the highest sustained write throughput for larger files and offers the widest feature set.  
//! * `hound` is efficient for short PCM‑16/32/F32 writes and has the lightest dependency footprint.

//! The results show no single library is universally “best”; each one excels along different axes that matter in practice:
//!
//! - **`wavers` (Rust)**  
//!   Delivers the **lowest mean latency** across all short-file tests and many longer runs, with exceptionally **consistent standard deviation** (sub-microsecond jitter). While its write performance is typically within **10–15% of the fastest implementation** for integer PCM formats, it leads for **24-/32-bit writes**.  
//!   - **Format support**: PCM‑16/24/32, IEEE‑F32/64  
//!   - **Ecosystem & integration**: Memory-safe, pure Rust, feature-rich and well-documented, with benchmarks and a Python API for cross-language integration.

//! - **`libsndfile` (C)**  
//!   Ranks as the **second-fastest reader overall**, occasionally taking the lead on large files where its internal buffering optimizes transfer speed. Delivers the **highest sustained write throughput** for files longer than 10 seconds, scaling efficiently with channel count.  
//!   - **Format support**: Same as wavers, plus RF64 and non-standard chunks  
//!   - **Ecosystem & integration**: Mature C ABI with bindings in Python, Rust, Go, and more; widely adopted in DAWs and CLI tools.

//! - **`hound` (Rust)**  
//!   Incurs **higher per-sample read overhead**, making it more suitable when read speed isn't critical. Excels at **writing very short (<1s) PCM‑16/32 and IEEE‑F32 files**, thanks to its minimalist header emission.  
//!   - **Format support**: PCM‑16/32, IEEE‑F32 only  
//!   - **Ecosystem & integration**: Single-crate with zero FFI dependencies; ideal for embedded or audit-friendly builds, with a small storage footprint (38.5 KiB vs. 199 KiB for wavers and ~535 KiB on Kubuntu).

//! Key take‑aways for users:

//! * Latency‑sensitive readers (DSP front‑ends, packetised streaming) benefit most from wavers thanks to its minimal per‑frame cost and broad sample‑format support.
//! * Bulk exporters and long offline renders are still best served by libsndfile, whose mature buffering yields the greatest bytes‑per‑second once header overhead is amortised.
//! * Resource‑constrained or audit‑heavy projects that touch only PCM‑16/32/F32 can pick hound for its tiny footprint and excellent small‑file write speeds.
//! * Feature completeness matters. If 24‑bit integer, 64‑bit float, or RF64 is required, wavers or libsndfile are the only candidates in this set. Conversely, if eliminating native dependencies is a priority, hound is the only pure‑Rust option.

//! Future benchmarks should extend the matrix (96 kHz+, multichannel, threaded I/O) to validate whether these performance hierarchies persist under higher data rates and parallel workloads, but within the present scope the data provide a clear decision table rather than a single winner.
//!
//! ---
//!
//! # Additional Utilities
//!
//! For quick command-line inspection of WAV files, see [`wav_inspect`](https://crates.io/crates/wav_inspect).
//!
//! ---

pub mod chunks;
pub mod error;
pub mod format_code;
pub mod iter;
pub mod wav;
pub mod wav_type;

#[cfg(feature = "pyo3")]
pub mod py;

/// Struct representing the `fmt` chunk and related funtionality.
pub use crate::chunks::fmt::FmtChunk;
/// Commonly used chunk identifiers.
pub use crate::chunks::{DATA, FACT, FMT, INFO, LIST, RIFF, WAVE};
/// General-purpose result and error kinds for the library.
pub use crate::error::{
    ConversionError, InternalError, OperationError, ParseError, ValidationError, WaversErrorKind,
    WaversResult,
};
/// Enum of recognized WAV format codes.
pub use crate::format_code::FormatCode;
/// Iterator utilities for WAV samples.
pub use crate::iter::WavIterators;
/// Main representation of a WAV file and associated metadata.
pub use crate::wav::{Wav, WavInfo};
/// Enum for representing WAV sample type (bit depth + encoding).
pub use crate::wav_type::WavType;

use std::{
    any::TypeId,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::Path,
};

use audio_sample::{AudioSample, ConvertTo, I24, Samples};

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(not(feature = "ndarray"))]
use std::any::TypeId;


/// Trait alias for types that implement both [Read] and [Seek].
pub trait ReadSeek: Read + Seek {}

impl<T: Read + Seek> ReadSeek for T {}

/// Reads a WAV file from a reader and returns raw audio samples,
/// sample rate (Hz), and number of channels (for non-`ndarray` mode).
///
/// # Type Parameters
/// - `T`: Output sample type implementing [AudioSample].
/// - `R`: Input reader implementing [Read] + [Seek].
///
/// # Returns
/// Tuple containing:
/// - `Box<[T]>`: Decoded audio samples.
/// - `u32`: Sample rate in Hz.
/// - `u16`: Number of channels.
#[cfg(not(feature = "ndarray"))]
pub fn read_with<T: AudioSample, R: ReadSeek>(reader: R) -> WaversResult<(Box<[T]>, u32, u16)>
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

/// Reads a WAV file from the file system and returns raw audio samples,
/// sample rate, and number of channels (non-`ndarray` mode).
///
/// # Arguments
/// - `path`: Path to the WAV file.
///
/// # Returns
/// Same as [`read_with`].
#[cfg(not(feature = "ndarray"))]
pub fn read<P: AsRef<Path>, T: AudioSample>(path: P) -> WaversResult<(Box<[T]>, u32, u16)>
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

/// Reads a WAV file from a reader and returns audio as an [Array2<T>] and sample rate (Hz).
///
/// Requires the `ndarray` feature.
///
/// # Returns
/// - [Array2<T>]: Channels × Samples matrix of decoded audio.
/// - `u32`: Sample rate.
#[cfg(feature = "ndarray")]
pub fn read_with<T: AudioSample, R: ReadSeek>(reader: R) -> WaversResult<(Array2<T>, u32)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let mut wav = Wav::from_reader(reader)?;
    let (samples, sr) = wav.read_ndarray::<T>()?;
    Ok((samples, sr as u32))
}

/// Reads a WAV file from the file system into an `Array2<T>` with sample rate.
///
/// Requires the `ndarray` feature.
#[cfg(feature = "ndarray")]
pub fn read<P: AsRef<Path>, T: AudioSample>(path: P) -> WaversResult<(Array2<T>, u32)>
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

/// Writes a slice of samples to a WAV writer with a given sample rate and optional channel count.
///
/// # Type Parameters
/// - `F`: Input sample type.
/// - `T`: Target WAV storage sample type.
///
/// # Arguments
/// - `writer`: Destination implementing `Write`.
/// - `samples`: Slice of samples.
/// - `sr`: Sample rate (Hz).
/// - `channels`: Optional number of channels (defaults to 1).
pub fn write_with<W: Write, To: AudioSample>(
    mut writer: W,
    samples: &[u8], // must pass an owned reference
    wav_type: WavType,
    sr: u32,
    channels: Option<u16>,
) -> WaversResult<()>
where
    i16: ConvertTo<To>,
    I24: ConvertTo<To>,
    i32: ConvertTo<To>,
    f32: ConvertTo<To>,
    f64: ConvertTo<To>,
{
    let channels = channels.unwrap_or(1);

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
    let data_size = samples.len() as u32;
    let riff_size = 4 + 8 + fmt_size + 8 + data_size;

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    writer.write_all(b"fmt ")?;
    writer.write_all(&fmt_size.to_le_bytes())?;
    writer.write_all(&fmt_bytes)?;

    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;
    writer.write_all(&samples)?;

    writer.flush()?;
    Ok(())
}

/// Writes audio samples to a WAV file from a boxed slice (non-`ndarray` mode).
///
/// # Arguments
/// - `path`: File path to write to.
/// - `samples`: Boxed slice of input samples.
/// - `sr`: Sample rate (Hz).
/// - `channels`: Optional number of channels.
#[cfg(not(feature = "ndarray"))]
pub fn write<P: AsRef<Path>, T: AudioSample>(
    path: P,
    samples: &Box<[T]>,
    sr: u32,
    channels: Option<u16>,
) -> WaversResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{

    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    let bytes = converted_samples.into_bytes();

    let wav_type = WavType::from(TypeId::of::<T>());

    crate::write_with(writer, &bytes, wav_type, sr, channels)
}

/// Writes an `ndarray::Array2` of audio samples to a WAV file.
///
/// Requires the `ndarray` feature.
#[cfg(feature = "ndarray")]
pub fn write<P: AsRef<Path>, T: AudioSample>(
    path: P,
    samples: &Array2<T>,
    sr: u32,
    channels: Option<u16>,
) -> WaversResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    let samples = samples.as_bytes();

    let wav_type = WavType::from(TypeId::of::<T>());
    crate::write_with::<BufWriter<File>, T>(writer, &samples, wav_type, sr, channels)
}

/// Returns WAV file metadata without loading the full sample data.
///
/// # Returns
/// A [crate::wav::WavInfo] struct containing header and format information.
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

    #[cfg(not(feature = "ndarray"))]
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

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_read_with() {
        // exercise the free `read_with(reader)` taking a BufReader
        let f = File::open(ONE_CHANNEL_WAV_I16).expect("open file");
        let reader = BufReader::new(f);
        let (samples, sample_rate) = read_with::<i16, _>(reader).expect("read_with failed");
        let expected = expected_i16();
        assert_eq!(samples.len(), expected.len(), "length mismatch");
        // check last sample matches
        let last = samples.len() - 1;
        assert_eq!(*samples.last().unwrap(), expected[last], "last sample");
        assert_eq!(sample_rate, 16000, "sample rate");
        assert_eq!(samples.shape()[0], 1, "channels");
    }
}
