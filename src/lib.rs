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
//! ### Reading a WAV File (Recommended)
//!
//! Read WAV file as AudioSamples with rich metadata and processing capabilities:
//!
//! ```no_run
//! use wavers::read_audio;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let audio = read_audio::<f32>("audio/input.wav")?;
//!     
//!     println!("Sample Rate: {} Hz", audio.sample_rate());
//!     println!("Channels: {}", audio.channels());
//!     println!("Duration: {:.2} seconds", audio.duration_seconds());
//!     
//!     // Rich audio analysis
//!     let rms = audio.rms();
//!     let peak = audio.peak();
//!     println!("RMS: {:.4}, Peak: {:.4}", rms, peak);
//!     Ok(())
//! }
//! ```
//!
//! ### Reading as Raw Samples (Legacy)
//!
//! For compatibility with existing code:
//!
//! ```no_run
//! use wavers::read;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let (samples, sample_rate, channels): (Box<[i16]>, u32, u16) = read("audio/input.wav")?;
//!     println!("Sample Rate: {}, Channels: {}", sample_rate, channels);
//!     println!("First sample: {:?}", samples.first());
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! ### Converting and Processing Audio
//!
//! AudioSamples supports rich type conversions and processing:
//!
//! ```no_run
//! use wavers::read_audio;
//! use audio_samples::AudioTypeConversion;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     // Read as 16-bit integer
//!     let audio_i16 = read_audio::<i16>("audio/input.wav")?;
//!     
//!     // Convert to 32-bit float for processing
//!     let audio_f32 = audio_i16.convert_to_sample_type::<f32>()?;
//!     
//!     // Apply audio processing
//!     let processed = audio_f32
//!         .remove_dc_offset()?
//!         .normalize(-1.0, 1.0, audio_samples::NormalizationMethod::Peak)?;
//!     
//!     println!("Processed audio: {} samples", processed.total_samples());
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! ### Writing WAV Files
//!
//! Write AudioSamples directly to WAV files:
//!
//! ```no_run
//! use wavers::{read_audio, write_ndarray};
//! use audio_samples::AudioTypeConversion;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     // Read and process audio
//!     let mut audio = read_audio::<f32>("audio/input.wav")?;
//!     
//!     // Apply processing (modifies in place)
//!     audio.normalize(-1.0, 1.0, audio_samples::NormalizationMethod::Peak)?;
//!     audio.low_pass_filter(8000.0)?;
//!     
//!     // Convert to desired output format
//!     let audio_i16 = audio.convert_to_sample_type::<i16>()?;
//!     
//!     // Extract ndarray for writing (current write API)
//!     let array = if audio_i16.is_mono() {
//!         audio_i16.as_mono().unwrap().insert_axis(ndarray::Axis(0))
//!     } else {
//!         audio_i16.as_multi_channel().unwrap().clone()
//!     };
//!     
//!     write_ndarray("audio/output.wav", &array, audio_i16.sample_rate(), None)?;
//!     Ok(())
//! }
//! ```
//!
//! ---
//!
//! ### Advanced Audio Processing
//!
//! AudioSamples provides comprehensive audio analysis capabilities:
//!
//! ```no_run
//! use wavers::read_audio;
//! use audio_samples::{AudioStatistics, AudioTransforms};
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let audio = read_audio::<f32>("audio/input.wav")?;
//!     
//!     // Statistical analysis
//!     println!("RMS: {:.4}", audio.rms());
//!     println!("Peak: {:.4}", audio.peak());
//!     println!("Zero crossings: {}", audio.zero_crossings());
//!     
//!     // Spectral analysis
//!     let centroid = audio.spectral_centroid()?;
//!     println!("Spectral centroid: {:.2} Hz", centroid);
//!     
//!     // Advanced transforms
//!     let mfcc = audio.mfcc(13, 128, 0.0, audio.sample_rate() as f64 / 2.0)?;
//!     println!("MFCC shape: {:?}", mfcc.dim());
//!     Ok(())
//! }
//! ```
//!
//! ### Iterating Over WAV Data (Legacy)
//!
//! For frame and channel iteration:
//!
//! ```no_run
//! use wavers::{Wav, WavIterators};
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! fn main() -> wavers::WaversResult<()> {
//!     let mut wav = Wav::<BufReader<File>>::from_path("audio/stereo.wav")?;
//!
//!     // Iterate over frames
//!     for frame_result in wav.frames::<i16>() {
//!         let frame = frame_result?;
//!         println!("Frame has {} samples (one per channel)", frame.len());
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

/// Struct representing the `fmt` chunk and related funtionality.
pub use crate::chunks::fmt::FmtChunk;
/// Commonly used chunk identifiers.
pub use crate::chunks::{DATA, FACT, FMT, INFO, LIST, RIFF, WAVE};
/// LIST chunk types and metadata structures.
pub use crate::chunks::{ListChunk, SubChunk, WavMetadata};
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

use audio_samples::{AudioSample, AudioSamples, ConvertTo, I24};
use ndarray::Array2;

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

/// Reads a WAV file from a reader and returns an ndarray, sample rate, and number of channels.
///
/// # Type Parameters  
/// - `T`: Output sample type implementing [AudioSample].
/// - `R`: Input reader implementing [Read] + [Seek].
///
/// # Returns
/// Tuple containing:
/// - `Array2<T>`: 2D array with shape (channels, samples).
/// - `u32`: Sample rate in Hz.
pub fn read_with_ndarray<T: AudioSample, R: ReadSeek>(reader: R) -> WaversResult<(Array2<T>, u32)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    use ndarray::Array2;
    let (samples, sample_rate, channels) = read_with::<T, R>(reader)?;
    let num_samples = samples.len() / channels as usize;
    let array = Array2::from_shape_vec((channels as usize, num_samples), samples.into_vec())
        .map_err(|e| {
            WaversErrorKind::Internal(InternalError::Bug(format!(
                "Failed to create ndarray: {}",
                e
            )))
        })?;
    Ok((array, sample_rate))
}

/// Reads a WAV file from the file system and returns an ndarray, sample rate.
///
/// # Arguments
/// - `path`: Path to the WAV file.
///
/// # Returns
/// Same as [`read_with_ndarray`].
pub fn read_ndarray<P: AsRef<Path>, T: AudioSample>(path: P) -> WaversResult<(Array2<T>, u32)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let file = std::fs::File::open(path)?;
    let buf_reader = BufReader::new(file);
    read_with_ndarray::<T, _>(buf_reader)
}

/// Reads a WAV file from a reader and returns AudioSamples with metadata.
///
/// This is the recommended way to read audio data as it provides rich metadata
/// and access to audio processing operations from the audio_samples crate.
///
/// # Type Parameters  
/// - `T`: Output sample type implementing [AudioSample].
/// - `R`: Input reader implementing [Read] + [Seek].
///
/// # Returns
/// [AudioSamples] instance containing the audio data with metadata.
///
/// # Example
/// ```rust,no_run
/// use wavers::read_audio_with;
/// use std::fs::File;
/// use std::io::BufReader;
///
/// fn main() -> wavers::WaversResult<()> {
///     let file = File::open("audio.wav")?;
///     let reader = BufReader::new(file);
///     let audio = read_audio_with::<f32, _>(reader)?;
///     
///     println!("Sample rate: {} Hz", audio.sample_rate());
///     println!("Channels: {}", audio.channels());
///     println!("Duration: {:.2} seconds", audio.duration_seconds());
///     Ok(())
/// }
/// ```
pub fn read_audio_with<T: AudioSample, R: ReadSeek>(reader: R) -> WaversResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let mut wav = Wav::from_reader(reader)?;
    wav.to_audio_samples()
}

/// Reads a WAV file from the file system and returns AudioSamples with metadata.
///
/// This is the recommended way to read audio data as it provides rich metadata
/// and access to audio processing operations from the audio_samples crate.
///
/// # Arguments
/// - `path`: Path to the WAV file.
///
/// # Returns
/// [AudioSamples] instance containing the audio data with metadata.
///
/// # Example
/// ```rust,no_run
/// use wavers::read_audio;
///
/// fn main() -> wavers::WaversResult<()> {
///     let audio = read_audio::<f32>("audio.wav")?;
///     
///     println!("Sample rate: {} Hz", audio.sample_rate());
///     println!("Channels: {}", audio.channels());
///     println!("Duration: {:.2} seconds", audio.duration_seconds());
///     
///     // Access rich audio processing capabilities
///     let rms = audio.rms();
///     let peak = audio.peak();
///     println!("RMS: {:.4}, Peak: {:.4}", rms, peak);
///     Ok(())
/// }
/// ```
pub fn read_audio<T: AudioSample, P: AsRef<Path>>(path: P) -> WaversResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let file = std::fs::File::open(path)?;
    let buf_reader = BufReader::new(file);
    read_audio_with::<T, _>(buf_reader)
}

/// Reads a WAV file from a byte slice and returns the raw sample data.
///
/// This function allows reading WAV data that's already in memory (e.g., from
/// network requests, embedded data, etc.) without needing to write to disk first.
///
/// # Arguments
/// - `bytes`: Byte slice containing WAV file data.
///
/// # Returns
/// A tuple containing (samples, sample_rate, channels).
///
/// # Example
/// ```rust,no_run
/// use wavers::read_from_bytes;
///
/// fn main() -> wavers::WaversResult<()> {
///     let wav_data: &[u8] = &[/* WAV file bytes */];
///     let (samples, sample_rate, channels): (Box<[f32]>, u32, u16) = read_from_bytes(wav_data)?;
///     
///     println!("Sample rate: {} Hz", sample_rate);
///     println!("Channels: {}", channels);
///     println!("Samples: {}", samples.len());
///     Ok(())
/// }
/// ```
pub fn read_from_bytes<T: AudioSample>(bytes: &[u8]) -> WaversResult<(Box<[T]>, u32, u16)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let cursor = std::io::Cursor::new(bytes);
    let mut wav = Wav::from_reader(cursor)?;
    let samples = wav.read::<T>()?;
    let sample_rate = wav.fmt_chunk().sample_rate as u32;
    let channels = wav.fmt_chunk().channels;
    
    Ok((samples, sample_rate, channels))
}

/// Reads a WAV file from a byte slice and returns AudioSamples with metadata.
///
/// This function allows reading WAV data that's already in memory (e.g., from
/// network requests, embedded data, etc.) and returns the rich AudioSamples type
/// with metadata and processing capabilities.
///
/// # Arguments
/// - `bytes`: Byte slice containing WAV file data.
///
/// # Returns
/// [AudioSamples] instance containing the audio data with metadata.
///
/// # Example
/// ```rust,no_run
/// use wavers::read_audio_from_bytes;
///
/// fn main() -> wavers::WaversResult<()> {
///     let wav_data: &[u8] = &[/* WAV file bytes */];
///     let audio = read_audio_from_bytes::<f32>(wav_data)?;
///     
///     println!("Sample rate: {} Hz", audio.sample_rate());
///     println!("Channels: {}", audio.channels());
///     println!("Duration: {:.2} seconds", audio.duration_seconds());
///     
///     // Access rich audio processing capabilities
///     let rms = audio.rms();
///     let peak = audio.peak();
///     println!("RMS: {:.4}, Peak: {:.4}", rms, peak);
///     Ok(())
/// }
/// ```
pub fn read_audio_from_bytes<T: AudioSample>(bytes: &[u8]) -> WaversResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let cursor = std::io::Cursor::new(bytes);
    read_audio_with::<T, _>(cursor)
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

    let bytes = T::to_bytes_slice(samples);

    let wav_type = WavType::from(TypeId::of::<T>());

    crate::write_with(writer, &bytes, wav_type, sr, channels)
}

/// Writes audio samples from an ndarray to a WAV file.
///
/// # Arguments
/// - `path`: File path to write to.
/// - `samples`: 2D ndarray with shape (channels, samples).
/// - `sr`: Sample rate (Hz).
/// - `channels`: Optional number of channels (inferred from array shape if None).
pub fn write_ndarray<P: AsRef<Path>, T: AudioSample>(
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
    let (num_channels, num_samples) = samples.dim();
    let channels = channels.unwrap_or(num_channels as u16);

    // Convert to interleaved format
    let mut interleaved = Vec::with_capacity(num_channels * num_samples);
    for sample_idx in 0..num_samples {
        for channel_idx in 0..num_channels {
            interleaved.push(samples[(channel_idx, sample_idx)]);
        }
    }

    let boxed_samples = interleaved.into_boxed_slice();
    crate::write(path, &boxed_samples, sr, Some(channels))
}

/// Writes AudioSamples to a writer in WAV format.
///
/// This function provides a convenient way to write AudioSamples to any writer
/// (stdout, pipe, network, etc.) instead of just files.
///
/// # Type Parameters
/// - `T`: Sample type (i16, I24, i32, f32, f64).
/// - `W`: Writer type implementing `Write`.
///
/// # Arguments
/// - `writer`: Destination implementing `Write`.
/// - `audio`: AudioSamples instance to write.
///
/// # Example
/// ```rust,no_run
/// use wavers::{read_audio, write_audio_with};
/// use std::io;
///
/// fn main() -> wavers::WaversResult<()> {
///     let audio = read_audio::<f32>("input.wav")?;
///     
///     // Write to stdout
///     let stdout = io::stdout();
///     write_audio_with(stdout.lock(), &audio)?;
///     Ok(())
/// }
/// ```
pub fn write_audio_with<T: AudioSample, W: Write>(
    writer: W,
    audio: &AudioSamples<T>,
) -> WaversResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let bytes = if audio.is_mono() {
        let mono_data = audio
            .as_mono()
            .ok_or(WaversErrorKind::Internal(crate::error::InternalError::Bug(
                "Failed to extract mono data from AudioSamples".to_string(),
            )))?;
        T::to_bytes_slice(&mono_data.as_slice().unwrap())
    } else {
        let multi_data = audio
            .as_multi_channel()
            .ok_or(WaversErrorKind::Internal(crate::error::InternalError::Bug(
                "Failed to extract multi-channel data from AudioSamples".to_string(),
            )))?;
        
        // Convert to interleaved format for WAV
        let (num_channels, num_samples) = multi_data.dim();
        let mut interleaved = Vec::with_capacity(num_channels * num_samples);
        for sample_idx in 0..num_samples {
            for channel_idx in 0..num_channels {
                interleaved.push(multi_data[(channel_idx, sample_idx)]);
            }
        }
        T::to_bytes_slice(&interleaved)
    };

    let wav_type = WavType::from(std::any::TypeId::of::<T>());
    let channels = audio.channels() as u16;
    let sample_rate = audio.sample_rate();

    crate::write_with(writer, &bytes, wav_type, sample_rate, Some(channels))
}

/// Writes AudioSamples to a file in WAV format.
///
/// This is the recommended way to write audio data as it preserves all metadata
/// and provides type-safe sample format conversion.
///
/// # Type Parameters
/// - `T`: Sample type (i16, I24, i32, f32, f64).
/// - `P`: Path type implementing `AsRef<Path>`.
///
/// # Arguments
/// - `path`: File path to write to.
/// - `audio`: AudioSamples instance to write.
///
/// # Example
/// ```rust,no_run
/// use wavers::{read_audio, write_audio};
///
/// fn main() -> wavers::WaversResult<()> {
///     let audio = read_audio::<f32>("input.wav")?;
///     
///     // Write to file with automatic format conversion
///     write_audio("output.wav", &audio)?;
///     Ok(())
/// }
/// ```
pub fn write_audio<T: AudioSample, P: AsRef<Path>>(
    path: P,
    audio: &AudioSamples<T>,
) -> WaversResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    write_audio_with(writer, audio)
}

/// Writes samples to a writer in WAV format from a boxed slice.
///
/// # Type Parameters
/// - `T`: Sample type (i16, I24, i32, f32, f64).
/// - `W`: Writer type implementing `Write`.
///
/// # Arguments
/// - `writer`: Destination implementing `Write`.
/// - `samples`: Boxed slice of input samples.
/// - `sr`: Sample rate (Hz).
/// - `channels`: Optional number of channels.
///
/// # Example
/// ```rust,no_run
/// use wavers::write_samples_with;
/// use std::io;
///
/// fn main() -> wavers::WaversResult<()> {
///     let samples: Box<[f32]> = vec![0.0, 0.5, -0.5, 0.0].into_boxed_slice();
///     
///     // Write to stdout
///     let stdout = io::stdout();
///     write_samples_with(stdout.lock(), &samples, 44100, Some(1))?;
///     Ok(())
/// }
/// ```
pub fn write_samples_with<T: AudioSample, W: Write>(
    writer: W,
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
    let bytes = T::to_bytes_slice(samples);
    let wav_type = WavType::from(std::any::TypeId::of::<T>());
    crate::write_with(writer, &bytes, wav_type, sr, channels)
}

/// Writes samples to a writer in WAV format from an ndarray.
///
/// # Type Parameters
/// - `T`: Sample type (i16, I24, i32, f32, f64).
/// - `W`: Writer type implementing `Write`.
///
/// # Arguments
/// - `writer`: Destination implementing `Write`.
/// - `samples`: 2D ndarray with shape (channels, samples).
/// - `sr`: Sample rate (Hz).
/// - `channels`: Optional number of channels (inferred from array shape if None).
///
/// # Example
/// ```rust,no_run
/// use wavers::write_ndarray_with;
/// use ndarray::array;
/// use std::io;
///
/// fn main() -> wavers::WaversResult<()> {
///     let samples = array![[0.0f32, 0.5], [-0.5, 0.0]]; // 2 channels, 2 samples each
///     
///     // Write to stdout
///     let stdout = io::stdout();
///     write_ndarray_with(stdout.lock(), &samples, 44100, None)?;
///     Ok(())
/// }
/// ```
pub fn write_ndarray_with<T: AudioSample, W: Write>(
    writer: W,
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
    let (num_channels, num_samples) = samples.dim();
    let channels = channels.unwrap_or(num_channels as u16);

    // Convert to interleaved format
    let mut interleaved = Vec::with_capacity(num_channels * num_samples);
    for sample_idx in 0..num_samples {
        for channel_idx in 0..num_channels {
            interleaved.push(samples[(channel_idx, sample_idx)]);
        }
    }

    let boxed_samples = interleaved.into_boxed_slice();
    write_samples_with(writer, &boxed_samples, sr, Some(channels))
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

/// Extracts metadata from INFO LIST chunks in a WAV file.
///
/// # Arguments
/// - `path`: Path to the WAV file.
///
/// # Returns
/// An optional [WavMetadata] struct containing metadata from INFO LIST chunks.
/// Returns None if no INFO chunks are present or if they contain no data.
///
/// # Example
/// ```no_run
/// use wavers::metadata;
///
/// fn main() -> wavers::WaversResult<()> {
///     if let Some(metadata) = metadata("audio.wav")? {
///         println!("Artist: {:?}", metadata.artist);
///         println!("Title: {:?}", metadata.title);
///         println!("Copyright: {:?}", metadata.copyright);
///     } else {
///         println!("No metadata found");
///     }
///     Ok(())
/// }
/// ```
pub fn metadata<P: AsRef<Path>>(path: P) -> WaversResult<Option<WavMetadata>> {
    let file = std::fs::File::open(path)?;
    let buf_reader = BufReader::new(file);
    let wav = Wav::from_reader(buf_reader)?;
    wav.metadata()
}

#[cfg(test)]
pub(crate) mod wav_api_tests {
    use super::*;
    use audio_samples::AudioStatistics;
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

    #[test]
    fn test_read_audio_with() {
        // Test the new read_audio_with function
        let f = File::open(ONE_CHANNEL_WAV_I16).expect("open file");
        let reader = BufReader::new(f);
        let audio = read_audio_with::<f32, _>(reader).expect("read_audio_with failed");

        // Verify metadata
        assert_eq!(audio.sample_rate(), 16000, "sample rate");
        assert_eq!(audio.channels(), 1, "channels");
        assert!(audio.is_mono(), "should be mono");

        // Verify sample count matches expected
        let expected = expected_i16();
        assert_eq!(audio.samples_per_channel(), expected.len(), "sample count");

        // Test that we can access audio processing methods
        let rms = audio.rms();
        let peak = audio.peak();
        assert!(rms > 0.0, "RMS should be positive");
        assert!(peak > 0.0, "Peak should be positive");
    }

    #[test]
    fn test_read_audio() {
        // Test the new read_audio function
        let audio = read_audio::<i16, _>(ONE_CHANNEL_WAV_I16).expect("read_audio failed");

        // Verify metadata matches what we expect
        assert_eq!(audio.sample_rate(), 16000, "sample rate");
        assert_eq!(audio.channels(), 1, "channels");
        assert_eq!(audio.duration_seconds(), 10.0, "duration");

        // Verify sample data accuracy by comparing with expected data
        let expected = expected_i16();
        assert_eq!(audio.samples_per_channel(), expected.len(), "sample count");

        if let Some(mono_array) = audio.as_mono() {
            for (i, (&actual, &expected)) in mono_array.iter().zip(expected.iter()).enumerate() {
                assert_eq!(actual, expected, "sample mismatch at index {}", i);
            }
        } else {
            panic!("Expected mono audio");
        }
    }

    #[test]
    fn test_read_audio_statistics() {
        // Test that AudioSamples provides statistical methods
        let audio = read_audio::<f32, _>(ONE_CHANNEL_WAV_I16).expect("read_audio failed");

        // Test basic statistics
        let rms = audio.rms();
        let peak = audio.peak();
        let min = audio.min_native();
        let max = audio.max_native();

        assert!(rms > 0.0 && rms <= 1.0, "RMS should be reasonable: {}", rms);
        assert!(
            peak > 0.0 && peak <= 1.0,
            "Peak should be reasonable: {}",
            peak
        );
        assert!(min >= -1.0, "Min should be >= -1.0: {}", min);
        assert!(max <= 1.0, "Max should be <= 1.0: {}", max);

        // Test zero crossings
        let crossings = audio.zero_crossings();
        assert!(
            crossings > 0,
            "Should have some zero crossings: {}",
            crossings
        );

        let zcr = audio.zero_crossing_rate();
        assert!(zcr > 0.0, "Zero crossing rate should be positive: {}", zcr);
    }

    #[test]
    fn test_wav_read_audio_method() {
        // Test the new Wav::read_audio method
        let f = File::open(ONE_CHANNEL_WAV_I16).expect("open file");
        let mut wav = crate::wav::Wav::from_reader(BufReader::new(f)).expect("from_reader");

        let audio = wav.read_audio::<f32>().expect("read_audio failed");

        // Verify it works the same as the free function
        assert_eq!(audio.sample_rate(), 16000);
        assert_eq!(audio.channels(), 1);
        assert!(audio.is_mono());

        // Verify we can call it multiple times (caching should work)
        let audio2 = wav
            .read_audio::<f32>()
            .expect("read_audio failed second time");
        assert_eq!(audio.sample_rate(), audio2.sample_rate());
        assert_eq!(audio.total_samples(), audio2.total_samples());
    }

    #[test]
    fn test_read_audio_different_types() {
        // Test that we can read as different sample types
        let audio_i16 = read_audio::<i16, _>(ONE_CHANNEL_WAV_I16).expect("read as i16");
        let audio_i32 = read_audio::<i32, _>(ONE_CHANNEL_WAV_I16).expect("read as i32");
        let audio_f32 = read_audio::<f32, _>(ONE_CHANNEL_WAV_I16).expect("read as f32");
        let audio_f64 = read_audio::<f64, _>(ONE_CHANNEL_WAV_I16).expect("read as f64");

        // All should have same metadata
        assert_eq!(audio_i16.sample_rate(), audio_i32.sample_rate());
        assert_eq!(audio_i16.sample_rate(), audio_f32.sample_rate());
        assert_eq!(audio_i16.sample_rate(), audio_f64.sample_rate());

        assert_eq!(audio_i16.channels(), audio_i32.channels());
        assert_eq!(audio_i16.channels(), audio_f32.channels());
        assert_eq!(audio_i16.channels(), audio_f64.channels());

        assert_eq!(
            audio_i16.samples_per_channel(),
            audio_i32.samples_per_channel()
        );
        assert_eq!(
            audio_i16.samples_per_channel(),
            audio_f32.samples_per_channel()
        );
        assert_eq!(
            audio_i16.samples_per_channel(),
            audio_f64.samples_per_channel()
        );
    }
}
