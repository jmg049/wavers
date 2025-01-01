# WaveRs

A fast, ergonomic WAV file manipulation library for Rust

[![Crates.io][crate-img]][crate] [![Documentation][docs-img]][docs]

[crate]: https://crates.io/crates/wavers
[crate-img]: https://img.shields.io/crates/v/wavers.svg
[docs]: https://docs.rs/wavers
[docs-img]: https://docs.rs/wavers/badge.svg

## Overview

WaveRs is a high-performance library for reading, writing, and manipulating WAV audio files in Rust. It provides an ergonomic API focused on safety and performance, while supporting advanced WAV format features that many other libraries miss.

### Key Features

- **High Performance**: Optimized for speed with zero-copy operations where possible
- **Type Safety**: Leverages Rust's type system to prevent common audio manipulation errors
- **Format Support**: Handles standard PCM, IEEE float, and Extensible WAV formats
- **Flexible Conversions**: Seamless conversion between common audio sample types (i16, i24, i32, f32, f64)
- **Memory Efficient**: Streaming support for processing large files without loading them entirely into memory
- **Modern Ergonomics**: Iterator-based APIs for frame and channel-wise processing
- **Optional Features**: Integration with ndarray, SIMD optimizations, and more

### Why WaveRs?

While other WAV libraries exist for Rust, WaveRs differentiates itself through:

1. **Extensive Format Support**: Handles many WAV variants including PCM, IEEE float, and the Extensible format 
2. **Modern API Design**: Ergonomic Rust APIs that prevent common errors
3. **Performance Focus**: Optimized for both throughput and memory efficiency
4. **Open Development**: Community contributions welcome, with periodic maintenance and updates
5. **Rich Feature Set**: Support for advanced use cases like streaming and audio resampling

## Quick Start

Add WaveRs to your project:

```bash
cargo add wavers
```

### Basic Usage

Reading a WAV file:

```rust
use wavers::{Wav, read};

// Load and read audio data
let (samples, sample_rate) = read::<f32, _>("audio.wav")?;

// Or for more control:
let mut wav = Wav::<f32>::from_path("audio.wav")?;
println!("Channels: {}", wav.n_channels());
println!("Sample Rate: {}", wav.sample_rate());
println!("Duration: {}s", wav.duration());

let samples = wav.read()?;
```

Writing a WAV file:

```rust
use wavers::write;

// Generate a 440Hz sine wave
let sample_rate = 44100;
let duration = 1.0; // seconds
let mut samples: Vec<f32> = (0..sample_rate)
    .map(|x| {
        let t = x as f32 / sample_rate as f32;
        (t * 440.0 * 2.0 * std::f32::consts::PI).sin()
    })
    .collect();

// Write as mono 32-bit float WAV
write("sine.wav", &samples, sample_rate as i32, 1)?;
```

### Advanced Features

Frame-wise processing (process all channels at each time point):

```rust
let mut wav = Wav::<f32>::from_path("stereo.wav")?;

for frame in wav.frames() {
    // Each frame contains one sample per channel
    let left = frame[0];
    let right = frame[1];
    // Process stereo frame...
}
```

Channel-wise processing (process each channel independently):

```rust
for channel in wav.channels() {
    // Process entire channel as a continuous signal
    // Useful for per-channel operations like filtering
}
```

Block-wise processing with overlap (useful for spectral analysis, windowing):

```rust
let block_size = 1024;
let overlap = 512;

for block in wav.blocks(block_size, overlap) {
    // Process each overlapping block
    // block.len() == block_size * n_channels
    // (except possibly the final block)
}
```

Sample type conversion:

```rust
// Read as i16 and convert to f32
let wav = Wav::<i16>::from_path("audio.wav")?;
let samples = wav.read()?.convert::<f32>();
```

## Anatomy of a WAV File

WAV files use the Resource Interchange File Format (RIFF) structure, organizing data into chunks. Each chunk has an identifier, size field, and data. Here's a detailed look at the core chunks:

### RIFF Chunk
The outer container of the WAV file:

```
Offset  Size    Description
0       4       "RIFF" identifier
4       4       File size (minus 8 bytes for this header)
8       4       "WAVE" format identifier
12      *       Sub-chunks
```

### Format (fmt) Chunk
Describes the audio format:

```
Offset  Size    Description
0       4       "fmt " identifier
4       4       Chunk size (16, 18, or 40 bytes)
8       2       Format code (1=PCM, 3=IEEE float, 0xFFFE=Extensible)
10      2       Number of channels
12      4       Sample rate (Hz)
16      4       Byte rate (bytes per second)
20      2       Block align (bytes per frame)
22      2       Bits per sample
[24     2       Extension size (if present)]
[26     *       Format-specific extension]
```

For Extensible format (chunk size = 40):
```
24      2       Extension size (22)
26      2       Valid bits per sample
28      4       Channel mask
32      16      SubFormat GUID
```

### Data Chunk
Contains the actual audio samples:

```
Offset  Size    Description
0       4       "data" identifier
4       4       Chunk size (bytes of audio data)
8       *       Audio data
```

### Additional Chunks

Beyond the core RIFF, fmt, and data chunks, WAV files can contain several other chunk types. WaveRs provides varying levels of support for these additional chunks:

#### FACT Chunk
The FACT chunk is required for compressed formats and optional for PCM. It provides additional format-dependent information:

```
Offset  Size    Description
0       4       "fact" identifier
4       4       Chunk size (usually 4)
8       4       Number of samples per channel
```

This chunk is particularly important for compressed formats as it stores the actual number of samples, which might differ from what you'd calculate from the data chunk size.

#### LIST-INFO Chunk
The LIST chunk can contain various types of metadata, with INFO being the most common type. While WaveRs has initial support for LIST chunks, this feature is still under development:

```
Offset  Size    Description
0       4       "LIST" identifier
4       4       Chunk size
8       4       List type ("INFO" for metadata)
12      *       Metadata entries
```

Each INFO metadata entry follows this format:
```
0       4       Identifier (e.g., "INAM"=title, "IART"=artist)
4       4       Entry size
8       *       UTF-8 text
```

Common INFO identifiers include:
- INAM: Title
- IART: Artist
- ICMT: Comments
- ICRD: Creation date
- ISFT: Software

Note that while WaveRs can detect and parse these chunks, full read/write support for LIST chunk manipulation is still in development.

## Optional Features

WaveRs provides several optional features that can be enabled using cargo features:

### NDArray Integration
```rust
use wavers::{AsNdarray, IntoNdarray, Wav};
use ndarray::Array2;

// Enable with: cargo add wavers --features ndarray

let wav = Wav::<f32>::from_path("audio.wav")?;
let (audio_matrix, sample_rate): (Array2<f32>, i32) = wav.into_ndarray()?;

// audio_matrix shape: (samples, channels)
// Useful for DSP operations that benefit from matrix operations
```

### Resampling Support (Still WIP)
```rust
use wavers::{resample, Wav};

// Enable with: cargo add wavers --features resampling

let mut wav = Wav::<f32>::from_path("input.wav")?;
let target_sample_rate = 44100;
let resampled = resample(&mut wav, target_sample_rate)?;
```

### SIMD Optimizations
The `simd` feature enables SIMD acceleration for the resampling functions. Note that while other operations may benefit from compiler auto-vectorization, explicit SIMD optimizations are currently only implemented for resampling operations.

### Additional Features
- **colored**: Adds colored output for debug and display implementations
- **logging**: Enables detailed logging via the log crate

Enable features like this:
```bash
cargo add wavers --features "ndarray resampling simd"
```

## Contributing

Contributions to WaveRs are welcome! While development occurs periodically rather than continuously, I review pull requests when available and appreciate community involvement. Feel free to:

- Report bugs or suggest features through issues
- Submit pull requests for bug fixes or enhancements
- Improve documentation or examples
- Share your experiences using the library

The codebase aims to maintain high standards for performance and correctness, so please include tests with your contributions when relevant.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Benchmarks (Update Required)
The benchmarks below were recorded using ``Criterion`` and each benchmark was run on a small dataset of wav files taken from the GenSpeech data set. The durations vary between approx. 7s and 15s and each file is encoded as PCM_16. The results below are the time taken to load all the wav files in the data set. So the time per file is the total time divided by the number of files in the data set. The data set contains 10 files. There are some suspected anomalies in the benchmarks which warrant further investigation. The benchmarks were run on a desktop PC with the following (relevant) specs: 

- CPU: 13th Gen Intel® Core™ i7-13700KF
- RAM: 32Gb DDR4
- Storage: 1Tb SSD


### Hound vs Wavers - native i16
| benchmark                    | name       | min_time   | mean_time   | max_time   |
|:-----------------------------|:-----------|:-----------|:------------|:-----------|
| Hound vs Wavers - native i16 | Hound Read i16   | 7.4417 ms  | 7.4441 ms   | 7.4466 ms  |
| Hound vs Wavers - native i16 | Wavers Read i16   | 122.42 µs  | 122.56 µs   | 122.72 µs  |
| Hound vs Wavers - native i16 | Hound Write i16  | 2.1900 ms  | 2.2506 ms   | 2.3201ms   |
| Hound vs Wavers - native i16 | Wavers Write i16 | 5.9484 ms  | 6.2091 ms   | 6.5018 ms  |

### Reading
| benchmark   | name                       | min_time   | mean_time   | max_time   |
|:------------|:---------------------------|:-----------|:------------|:-----------|
| Reading     | Native i16 - Read          | 121.28 µs  | 121.36 µs   | 121.44 µs  |
| Reading     | Native i16 - Read Wav File | 121.56 µs  | 121.79 µs   | 122.08 µs  |
| Reading     | Native i16 As f32          | 287.63 µs  | 287.78 µs   | 287.97 µs  |


### Writing
| benchmark   | name                      | min_time   | mean_time   | max_time   |
|:------------|:--------------------------|:-----------|:------------|:-----------|
| Writing     | Slice - Native i16        | 5.9484 ms  | 6.2091 ms   | 6.5018 ms  |
| Writing     | Slice - Native i16 As f32 | 30.271 ms  | 33.773 ms   | 37.509 ms  |
| Writing     | Write native f32          | 11.286 ms  | 11.948 ms   | 12.648 ms  |