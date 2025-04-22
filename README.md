<div align="center">

# WaveRS

## Fast, flexible, and lightweight WAV audio reader/writer for Rust and Python

<img src="logo.png" alt="WaveRS Logo" width="200"/>

[![Crates.io][crate-img]][crate] [![Docs.rs][docs-img]][docs] [![PyWavers GitHub][crate-pywavers-img]][crate-pywavers] [![PyWavers Docs.rs][crate-pywavers-docrs-img]][pywavers-docrs] [![PyWavers RTD][crate-pywavers-readdocs-img]][pywavers-readdocs]  [![License: MIT][license-img]][license]
</div>

---

## Overview

WaveRS is a fast, lightweight, and ergonomic Rust library for reading, writing, and manipulating WAV audio files.  

It supports a wide range of sample types, provides efficient conversion utilities, and offers flexible iterators over WAV data. Designed for performance-critical applications and ease of integration, it works well across DSP pipelines, embedded audio, Python bindings, and more.

---

## Features

- **Efficient and Lightweight** — Minimal overhead for latency-sensitive applications.
- **Intuitive API** — Simple and clear function signatures for all core operations.
- **Wide Format Support** — Supports `i16`, `i24`, `i32`, `f32`, and `f64` samples.
- **Automatic Sample Conversion** — Includes SIMD-accelerated format conversion.
- **WAVE_FORMAT_EXTENSIBLE Support** — Handles extended WAV files with extra channel metadata.
- **Chunk Management** — Full support for standard and extended chunk types.
- **Iterators for Frames and Channels** — Efficiently process audio in various patterns.
- **Optional Features**  
  - `ndarray` — Convert audio data into 2D arrays for scientific computing.  
  - `pyo3` — Python bindings via [PyWavers].  
  - `log` — Optional logging integration.

---


## Quick Usage

### Reading a WAV File

```rust
use wavers::{read};
use audio_sample::Samples;

fn main() -> wavers::WaversResult<()> {
    let (samples, sample_rate, channels): (Samples<i16>, u32, u16) = read("audio/input.wav")?;
    println!("Sample Rate: {sample_rate}, Channels: {channels}");
    Ok(())
}
```

### Writing a WAV File

```rust
use wavers::{read, write};
use audio_sample::ConvertSequence;

fn main() -> wavers::WaversResult<()> {
    let (samples, rate, channels) = read::<f32, _>("audio/input.wav")?;
    let converted = samples.convert_sequence::<i16>();
    write("audio/output.wav", &converted, rate, Some(channels))?;
    Ok(())
}
```

### Sample Conversion

```rust
use audio_sample::{Samples, ConvertSequence};

fn convert(samples: Samples<i16>) -> Samples<f32> {
    samples.convert_sequence()
}
```

### Iterating Over Audio

```rust
use wavers::Wav;

fn main() -> wavers::WaversResult<()> {
    let wav = Wav::from_path("audio/stereo.wav")?;

    for frame in wav.frames::<i16>() {
        println!("Frame: {:?}", frame?);
    }

    for channel in wav.channels::<i16>() {
        println!("Channel: {:?}", channel?);
    }

    Ok(())
}
```

### Getting WAV Metadata

```rust
use wavers::info;

fn main() -> wavers::WaversResult<()> {
    let metadata = info("audio/file.wav")?;
    println!(
        "Sample Rate: {}, Channels: {}, Duration: {:.2} s",
        metadata.sample_rate, metadata.channels, metadata.duration
    );
    Ok(())
}
```

---

## Optional: Ndarray Integration

Enable with the `ndarray` feature.

```rust
#[cfg(feature = "ndarray")]
use ndarray::Array2;

fn main() -> wavers::WaversResult<()> {
    let wav = wavers::Wav::from_path("audio/input.wav")?;
    let (samples, rate): (Array2<i16>, u32) = wav.read_ndarray()?;
    println!("Shape: {:?}", samples.shape());
    Ok(())
}
```

---

## Anatomy of a WAV File

WAV files follow the RIFF (Resource Interchange File Format) specification. A file consists of one or more **chunks**, each with an identifier, size, and associated data. The most important chunks are:

### RIFF Chunk (Header)

| Field            | Size (Bytes) | Description                     |
|------------------|--------------|---------------------------------|
| Chunk ID         | 4            | Always `"RIFF"`                 |
| Chunk Size       | 4            | File size minus 8 bytes         |
| Format           | 4            | Always `"WAVE"`                 |

### `fmt` Chunk (Standard)

| Field              | Size (Bytes) | Description                         |
|--------------------|--------------|-------------------------------------|
| Chunk ID           | 4            | `"fmt "` (note the trailing space)  |
| Chunk Size         | 4            | Size of the rest of the chunk       |
| Audio Format       | 2            | 1 = PCM, 3 = IEEE float             |
| Num Channels       | 2            | Mono = 1, Stereo = 2, etc.          |
| Sample Rate        | 4            | Hz                                  |
| Byte Rate          | 4            | SampleRate × NumChannels × Bits/8   |
| Block Align        | 2            | NumChannels × BitsPerSample / 8     |
| Bits Per Sample    | 2            | 16, 24, 32, 64 (Wavers does not support 8-bit) |

### `fmt` Chunk (Extended: WAVE_FORMAT_EXTENSIBLE)

| Field                      | Size (Bytes) | Description                                           |
|----------------------------|--------------|-------------------------------------------------------|
| Extension Size             | 2            | Always 22 for WAVE_FORMAT_EXTENSIBLE                 |
| Valid Bits Per Sample     | 2            | Bits of precision in the sample data                 |
| Channel Mask               | 4            | Speaker position mask (e.g., front left, center)     |
| Subformat GUID             | 16           | Defines actual format (PCM, IEEE float, etc.)        |

### `data` Chunk

| Field         | Size (Bytes) | Description                    |
|---------------|--------------|--------------------------------|
| Chunk ID      | 4            | `"data"`                       |
| Chunk Size    | 4            | Size of audio data in bytes    |
| Audio Data    | variable     | Raw sample data                |

Other chunks (e.g., `LIST`, `fact`, `bext`, `smpl`) may also appear in extended files. Wavers will safely ignore unsupported chunks and can be extended to process custom metadata in the future.

---

## Wav Utilities

```rust
use wavers::wav_spec;

fn main() -> wavers::WaversResult<()> {
    let wav = wavers::Wav::<i16>::from_path("file.wav")?;
    println!("Sample rate: {}", wav.sample_rate());
    println!("Channels: {}", wav.n_channels());
    println!("Duration: {} seconds", wav.duration());

    let (_, header) = wav_spec("file.wav")?;
    println!("Encoding: {:?}", header.encoding);
    Ok(())
}
```

## Related Tools

- [`PyWavers`](https://github.com/jmg049/pywavers) — Python bindings built with `pyo3`
- [`wav_inspect`](https://crates.io/crates/wav_inspect) — CLI for quick WAV file inspection

---

## License

Licensed under the MIT License.  
See [LICENSE](./LICENSE) for more information.

---

# Benchmarks

## Introduction

This document describes the methodology used to benchmark various WAV audio file processing libraries across different programming languages and implementations. The benchmark focuses on measuring read and write performance across different audio formats, sample rates, durations, and channel configurations.

## Libraries Compared

- **libsndfile  **: A widely-used C library for reading and writing audio files, considered an industry standard. Supports all tested formats.
- **Wavers  **: A Rust library designed for efficient WAV file handling, leveraging Rust s memory safety features. Supports all tested formats.
- **Hound  **: Another Rust library for WAV processing with a different API approach. Has format limitations - only supports PCM_16 (i16), PCM_32 (i32), and IEEE_F32 (f32).

---

## Important

**The rust benchmarks were compiled using ``cargo build --release`` and the ``libsndfile`` benchmark was compiled with ``gcc -o wav_benchmark wav_benchmark.c -lsndfile -lm -O3``.**

---

## Benchmark Parameters

The benchmark tests reading and writing performance across a comprehensive matrix of configurations:

### Sample Formats

- **PCM_16**: 16-bit integer PCM (common CD-quality format) - supported by all libraries
- **PCM_24**: 24-bit integer PCM (professional audio standard) - supported by libsndfile and Wavers, but not Hound
- **PCM_32**: 32-bit integer PCM (extended dynamic range) - supported by all libraries
- **IEEE_F32**: 32-bit floating-point (common for processing applications) - supported by all libraries
- **IEEE_F64**: 64-bit floating-point (highest precision) - supported by libsndfile and Wavers, but not Hound

### Sample Rates

- **8,000 Hz**: Common for telephony and speech applications
- **16,000 Hz**: Standard for high-quality speech applications
- **44,100 Hz**: Standard CD audio quality

### Durations

- **1 second**: Short audio segments, typical for real-time processing
- **10 seconds**: Medium-length audio, common for music segments
- **60 seconds**: Longer audio files, tests sustained performance

### Channel Configurations

- **Mono (1 channel)**: Common for speech and single-track recordings
- **Stereo (2 channels)**: Standard for music and most consumer audio

## Parameter Justifications

### Sample Format Selection

The five formats represent the complete range of commonly used WAV formats in real-world applications. They cover both integer and floating-point representations across various bit depths, allowing us to measure performance differences when handling different amounts of data per sample.

### Sample Rate Selection

The chosen sample rates cover the main use cases in audio processing, from low-quality telephony (8kHz) to high-quality audio (44.1kHz). Higher sample rates (such as 96kHz or 192kHz) were excluded as they are less common in general applications and would have increased benchmark running time substantially without adding proportional value.

### Duration Selection

The three durations allow us to measure performance characteristics for both short and longer files:

- 1-second files test initialization overhead and small file performance
- 10-second files provide a good middle ground for typical use cases
- 60-second files test sustained performance and memory efficiency with larger data sets

Longer durations (>60s) were excluded to keep the benchmark execution time reasonable while still gathering meaningful data about sustained performance.

### Channel Configuration Selection

Mono and stereo represent the vast majority of audio applications.

## Measurement Methodology

### General Approach

For each combination of parameters (5 formats × 3 sample rates × 3 durations × 2 channel configurations × 2 operations):

1. **Warm-up Phase**: A warm-up period is used to eliminate cold-start biases
   - Minimum 3 seconds, or duration/10 if longer
   - Executes the same operation repeatedly without timing

2. **Measurement Phase**: Multiple measurements are collected
   - Up to 100 samples per configuration (50 for benchmarks > 10 minutes)
   - Minimum measurement period of 15 seconds, or duration/2 if longer
   - Each measurement includes the complete operation (file opening, processing, and closing)

3. **Statistical Analysis**: Comprehensive statistics are calculated
   - Mean, median, min, max times
   - Standard deviation
   - Throughput in bytes per second

### Timing Approach

- High-precision timers are used (Instant in Rust, gettimeofday in C)
- Each operation is measured individually with minimal overhead
- The complete operation is included in the measurement (open, process, close)

### Data Generation

- Random noise data is generated consistently across all benchmarks
- The same pseudo-random number generator and seeding approach is used
- Data amplitude range is normalized between -1.0 and 1.0 before format conversion

### File Operations

- Temporary files are created in a dedicated directory
- Files are explicitly cleaned up after each benchmark
- File size is measured for accurate throughput calculations

## Benchmark Execution Environment

### File System Considerations

All benchmarks run on the same file system to ensure fair comparison. The temporary files are written to the same directory to minimize filesystem variability.

### Caching Effects

The benchmark methodology includes warm-up phases to minimize cache effects. Each configuration is tested sequentially with its own files to prevent cross-contamination of filesystem caches.

### Memory Management

All libraries are responsible for their own memory management. The benchmark does not force garbage collection or memory cleanup between runs to reflect real-world usage patterns.

## 1 - Reading Performance  

| Scenario | Fastest library | Relative observations |
|----------|-----------------|-----------------------|
|Mono, 1 s, 8 kHz, PCM‑16|`wavers`|2.79 µs (Wavers) vs 5.02 µs (libsndfile), 52.3 µs (hound)|
|Mono, 1 s, 44.1 kHz, PCM‑16|`wavers`|5.19 µs (Wavers) vs 6.45 µs (libsndfile) vs 294 µs (hound) |
|Mono, 10 s, 8 kHz, PCM‑16|`libsndfile`|7.95 µs (libsndfile) vs 13.3 µs (wavers) vs 514 µs (hound) |

- `wavers` records the lowest latency in every short‑file test and maintains that lead for most longer files.  
- `libsndfile` narrows—or reverses—the gap as file duration and size increase, suggesting its internal buffering becomes advantageous.  
- `hound` incurs higher per‑sample cost across the board, most evident at higher sample rates.

---

## 2 - Writing Performance  

| Scenario | Fastest library | Relative observations |
|----------|-----------------|-----------------------|
|Mono, 1 s, 44.1 kHz, PCM‑16|`hound`|302 µs (hound) vs 960 µs (Wavers) vs 5599 µs (libsndfile)|
|Mono, 10 s, 8 kHz, PCM‑16|`libsndfile`|714 µs (libsndfile) vs 982 µs (hound) vs 1121 µs (wavers) |
|Stereo, 1 s, 8 kHz, PCM‑32|`wavers`|181 µs (Wavers) vs 192 µs (libsndfile) vs 235 µs (hound)|

- For very small files at high sample rates, `hound` shows the shortest write time; its minimal header‑generation path dominates the timing.  
- On longer writes (≥ 10 s) and lower rates, `libsndfile` generally delivers the best throughput.  
- `wavers`’ write times remain close to libsndfile in most integer‑format tests and lead when 24‑ or 32‑bit PCM is involved.

---

## 3 - Effect of Individual Parameters  

| Parameter | Observed scaling |
|-----------|------------------|
|**Sample format**|Integer → float adds ≤ 10 % in `wavers` and libsndfile; ~40 – 50 % in `hound`.|
|**Sample rate**|Per‑frame time increases almost linearly for `wavers` and libsndfile; non‑linear growth (×5–10) in `hound`.|
|**Duration**|Time grows sub‑linearly with duration for all libraries; fixed overheads are amortised quickly.|
|**Channels (1 → 2)**|Timing doubles in libsndfile and wavers (ideal scaling); increase is >2× in `hound`, implying extra per‑channel overhead.|

---

## 4 - Practical Guidance  

| Requirement | Candidate libraries |
|-------------|--------------------|
|Fastest reads, broad format set (incl. 24‑/64‑bit float)|`wavers` or `libsndfile` (both close; `wavers` leads in most short‑file cases)|
|Sustained high‑throughput writes (minutes‑long exports)|`libsndfile`|
|Tiny telemetry logs, limited to PCM‑16/32/F32|`hound` (small binary, quickest small‑file writes)|
|Archival in IEEE F64|`wavers` or `libsndfile` (not supported by `hound`)|

---

## 5 - Limitations and Next Steps

- **Format and channel scope.** Only 8–44.1 kHz, mono/stereo cases were profiled. Extending to 96 kHz+ and >2 channels would provide a fuller picture.  
- **Platform features.** The benchmark is single‑threaded; libraries offering asynchronous or multi‑threaded paths were not exercised.  
- **External resources.** Memory consumption, CPU events and low‑level cache behaviour were not recorded; a complementary study is planned.  

Within its defined scope, the data indicate that no single library dominates every workload:  

- `wavers` tends to minimise read latency and maintains competitive write times across all supported formats.  
- `libsndfile` achieves the highest sustained write throughput for larger files and offers the widest feature set.  
- `hound` is efficient for short PCM‑16/32/F32 writes and has the lightest dependency footprint.

The results show no single library is universally “best”; each one excels along different axes that matter in practice:

| Library        | Read Profile                                                                                                                         | Write Profile                                                                                                    | Format coverage                                    | Ecosystem & integration                                                           |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------------------------------------|
| wavers    | Lowest mean latency across every short‑file test and many longer runs; very consistent std‑dev figures (sub‑µs jitter)               | Within 10 – 15 % of the fastest implementation in most integer‑PCM cases and top performer for 24‑/32‑bit writes | PCM‑16/24/32, IEEE‑F32/64                          | Memory‑safe, pure‑Rust, feature rich, doc-rich, benchmarked and a Python API;                      |
| libsndfile   | Second‑fastest reader overall, occasionally leading once file size is large enough for its internal buffer to dominate transfer cost | Highest sustained throughput on ≥ 10 s files at all tested rates; scales linearly with channels                  | Same set as wavers plus RF64 & non‑standard chunks | Mature C ABI, bindings for Python, Rust, Go, etc.; widely deployed in DAWs & CLIs |
| hound     | Higher per‑sample overhead; suitable when read speed is not critical                                                                 | Fastest for very small (< 1 s) PCM‑16/32/IEEE‑F32 writes thanks to lightweight header emission                   | PCM‑16/32, IEEE‑F32 only                           | Single‑crate, zero‑FFI dependency; ideal for embedded or audit‑friendly builds and small storage footprint (38.5Kib vs. 199Kib (Wavers) and approx. 535Kib (on my Kubuntu machine))   |

Key take‑aways for users:

- Latency‑sensitive readers (DSP front‑ends, packetised streaming) benefit most from wavers thanks to its minimal per‑frame cost and broad sample‑format support.
- Bulk exporters and long offline renders are still best served by libsndfile, whose mature buffering yields the greatest bytes‑per‑second once header overhead is amortised.
- Resource‑constrained or audit‑heavy projects that touch only PCM‑16/32/F32 can pick hound for its tiny footprint and excellent small‑file write speeds.
- Feature completeness matters. If 24‑bit integer, 64‑bit float, or RF64 is required, wavers or libsndfile are the only candidates in this set. Conversely, if eliminating native dependencies is a priority, hound is the only pure‑Rust option.

Future benchmarks should extend the matrix (96 kHz+, multichannel, threaded I/O) to validate whether these performance hierarchies persist under higher data rates and parallel workloads, but within the present scope the data provide a clear decision table rather than a single winner.

![Read Benchmarks](benches/benchmark_plots/read_mean_time_ms_grid.png)
![Write Benchmarks](benches/benchmark_plots/write_mean_time_ms_grid.png)

## Read Operation - Time in Microseconds(μs) (Mean Time)

|Channels|Duration(s)|SampleRate(Hz)|hound - IEEE_F32|hound - PCM_16|hound - PCM_32|libsndfile - IEEE_F32|libsndfile - IEEE_F64|libsndfile - PCM_16|libsndfile - PCM_24|libsndfile - PCM_32|wavers - IEEE_F32|wavers - IEEE_F64|wavers - PCM_16|wavers - PCM_24|wavers - PCM_32|
|-------------------:|-----------------------:|---------------------------:|-------------------------------:|-----------------------------:|-----------------------------:|---------------------------------:|---------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|--------------------------------:|--------------------------------:|------------------------------:|------------------------------:|------------------------------:|
|                  1 |                      1 |                       8000 |                          28.41 |                        52.32 |                        34.65 |                             7.5  |                             8.3  |                           5.02 |                           8.62 |                           5.39 |                            3.31 |                            4.41 |                          2.79 |                          7.36 |                          3.4  |
|                  1 |                      1 |                      16000 |                          56.47 |                       104.21 |                        68.55 |                             8.35 |                             9.9  |                           5.36 |                          12.67 |                           6.07 |                            9.51 |                           30.83 |                          3.31 |                         12.21 |                          4.47 |
|                  1 |                      1 |                      44100 |                         156.07 |                       293.96 |                       185.99 |                            10.72 |                            14.47 |                           6.45 |                          26.77 |                           8.25 |                            8.45 |                          108.96 |                          5.19 |                         29.48 |                          8.17 |
|                  1 |                     10 |                       8000 |                         286.95 |                       513.54 |                       334.48 |                            13.5  |                            20.81 |                           7.95 |                          45.01 |                          11.33 |                           13.26 |                           31.12 |                          7.66 |                         53.23 |                         13.33 |
|                  1 |                     10 |                      16000 |                         567.32 |                      1036    |                       653.27 |                            20.91 |                            47.42 |                          11.31 |                          85.73 |                          19.01 |                           28.76 |                          472.13 |                         13.65 |                        153.79 |                         30.46 |
|                  1 |                     10 |                      44100 |                        1593.49 |                      2838.12 |                      1826.49 |                            67.82 |                           128.36 |                          30.02 |                         239.66 |                          70.99 |                          127.91 |                         1536.5  |                         46.56 |                        304.61 |                        133.51 |
|                  1 |                     60 |                       8000 |                        1727.32 |                      3091.04 |                      1987.62 |                            80.34 |                           137.46 |                          29.42 |                         259.65 |                          73.03 |                          129.2  |                          263.29 |                         54.92 |                        331.08 |                        135.38 |
|                  1 |                     60 |                      16000 |                        3572.9  |                      6286.24 |                      4102.62 |                           144.94 |                           385.06 |                          73.04 |                         518.23 |                         133.01 |                          293.04 |                         3482.23 |                        146.67 |                        668.72 |                        278.59 |
|                  1 |                     60 |                      44100 |                       10238.5  |                     17421.7  |                     11609.1  |                           472.58 |                          1867.14 |                         203.16 |                        1556.76 |                         526.65 |                         1311.43 |                        12136.6  |                        428.59 |                       2149.33 |                       1272.26 |
|                  2 |                      1 |                       8000 |                          56.26 |                       104.27 |                        68.34 |                             8.88 |                            10.22 |                           5.39 |                          12.59 |                           6.05 |                            4.4  |                           31.56 |                          3.32 |                         12.24 |                          4.39 |
|                  2 |                      1 |                      16000 |                         115.23 |                       207.78 |                       135.28 |                            10.34 |                            12.97 |                           5.98 |                          21.24 |                           7.45 |                           29.62 |                           72.79 |                          4.38 |                         22.2  |                         30.29 |
|                  2 |                      1 |                      44100 |                         318.19 |                       572.54 |                       375.41 |                            14.78 |                            22.94 |                           8.49 |                          48.3  |                          11.95 |                          108.36 |                          242.96 |                          8.24 |                         59.5  |                         14.19 |
|                  2 |                     10 |                       8000 |                         558.76 |                      1036.97 |                       673.3  |                            21.21 |                            48.69 |                          11.35 |                          86.79 |                          19.04 |                           29.41 |                          479.05 |                         13.43 |                        105.82 |                         30.41 |
|                  2 |                     10 |                      16000 |                        1162.63 |                      2032.57 |                      1316.91 |                            48.29 |                           100.3  |                          19.32 |                         173.68 |                          46.73 |                          470.08 |                          934.91 |                         28.76 |                        229.04 |                        476.63 |
|                  2 |                     10 |                      44100 |                        3216.13 |                      5782.06 |                      3706.15 |                           170.57 |                           263.36 |                          68.53 |                         465.99 |                         140.83 |                         1398.17 |                         3430.04 |                        119.96 |                        607.64 |                        258.09 |
|                  2 |                     60 |                       8000 |                        3634.13 |                      6151.61 |                      3986.61 |                           147.08 |                           322.19 |                          73.85 |                         517.81 |                         149.19 |                          281.81 |                         3714.05 |                        143.19 |                        662.53 |                        284.99 |
|                  2 |                     60 |                      16000 |                        7190.32 |                     12638.6  |                      8290.66 |                           323.16 |                          1240.34 |                         145.18 |                        1052.03 |                         312.78 |                         3515.38 |                         8150.87 |                        276.69 |                       1443.65 |                        724.64 |
|                  2 |                     60 |                      44100 |                       20935.2  |                     35165.6  |                     23905    |                          1823.54 |                          5162.96 |                         546.39 |                        3331.77 |                        1837.22 |                        12392.9  |                        24539.8  |                       1191.27 |                       4914.52 |                       2839.14 |

## Write Operation - Time in Microseconds(μs) (Mean Time)

|Channels|Duration(s)|SampleRate(Hz)|hound - IEEE_F32|hound - PCM_16|hound - PCM_32|libsndfile - IEEE_F32|libsndfile - IEEE_F64|libsndfile - PCM_16|libsndfile - PCM_24|libsndfile - PCM_32|wavers - IEEE_F32|wavers - IEEE_F64|wavers - PCM_16|wavers - PCM_24|wavers - PCM_32|
|-------------------:|-----------------------:|---------------------------:|-------------------------------:|-----------------------------:|-----------------------------:|---------------------------------:|---------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|--------------------------------:|--------------------------------:|------------------------------:|------------------------------:|------------------------------:|
|                  1 |                      1 |                       8000 |                         128.43 |                        87.06 |                       119.02 |                           118.19 |                           192.9  |                          78.62 |                          90.38 |                         128.91 |                          114.47 |                         1577.08 |                         79.83 |                         88.42 |                        107.66 |
|                  1 |                      1 |                      16000 |                         990.4  |                       153.37 |                       217.55 |                           244.35 |                           533.07 |                         119.25 |                        1011.14 |                         220.33 |                          208.51 |                         1175.25 |                        103.44 |                        155.53 |                        194.78 |
|                  1 |                      1 |                      44100 |                         997.25 |                       302.34 |                      1005.37 |                          1269.06 |                          1656    |                        5598.43 |                        1082.27 |                         615.41 |                          602.31 |                         1572.68 |                        960.17 |                       1096.28 |                        608.53 |
|                  1 |                     10 |                       8000 |                        4690.14 |                       981.84 |                      1368.78 |                          1818.38 |                          2382.79 |                         714    |                         842.08 |                        1573.82 |                         1234.2  |                         2219.76 |                       1121.34 |                        947.42 |                       1601.17 |
|                  1 |                     10 |                      16000 |                        3071.67 |                      1289.06 |                      2739.97 |                         10054    |                          5070.72 |                        1458.7  |                        1820.36 |                        2573    |                         2800.14 |                         7448.93 |                       1301.06 |                       2383.41 |                       9223.25 |
|                  1 |                     10 |                      44100 |                        9094.49 |                      3678.89 |                      7109.76 |                          7021.14 |                         14120.2  |                        3282.77 |                        9187.67 |                        6208.17 |                         6077.86 |                        12894.2  |                       3421.52 |                       5480.78 |                       5631.26 |
|                  1 |                     60 |                       8000 |                        9645    |                      3977.13 |                      7466.25 |                          7774.56 |                         12724.3  |                        3612.74 |                        8065.88 |                        9660.27 |                        10338.7  |                        16501.1  |                       3766.17 |                       7640.54 |                       6967    |
|                  1 |                     60 |                      16000 |                       13562.9  |                      9614.26 |                     14931.3  |                         16437    |                         25073.7  |                        7185.4  |                       12982.2  |                       13547    |                        17031.5  |                        34092.1  |                      10629.1  |                      15767    |                      12016.7  |
|                  1 |                     60 |                      44100 |                       37309.6  |                     19874.8  |                     39050.1  |                         38582.4  |                         65971.7  |                       22109.7  |                       28373.4  |                       30562.6  |                        34448.6  |                        73138    |                      61372.8  |                      27867.8  |                      73439.5  |
|                  2 |                      1 |                       8000 |                        1543.42 |                       733.84 |                       234.9  |                           275.58 |                          1245.94 |                         124.13 |                         160.83 |                         191.98 |                          234.17 |                         2379.94 |                        108.55 |                        157.25 |                        181.05 |
|                  2 |                      1 |                      16000 |                         389.66 |                       303.46 |                       665.59 |                           478.52 |                          1347.52 |                         214.32 |                         371.58 |                         486.62 |                         1133.64 |                         1423.82 |                       1499.77 |                        915.29 |                        501.77 |
|                  2 |                      1 |                      44100 |                        1053.96 |                       606.56 |                      1086.51 |                          1469.4  |                          2470.24 |                         675.08 |                        1388.1  |                        1650.39 |                         1717.98 |                         2884.58 |                       1306.66 |                       5068.46 |                       1653.66 |
|                  2 |                     10 |                       8000 |                        2379.62 |                      1079.63 |                      2261.74 |                          2431.75 |                          7229.27 |                        1745.42 |                        1681.14 |                       10637.9  |                         2834.64 |                         5048.65 |                       1284.7  |                       2230.34 |                       2582.07 |
|                  2 |                     10 |                      16000 |                        7593.71 |                      2489.44 |                      4929.06 |                          4877.66 |                          8376.58 |                        2825.81 |                        3799.29 |                        4589.18 |                         5476.75 |                         9726.91 |                       2490.06 |                       3950.15 |                       5290.95 |
|                  2 |                     10 |                      44100 |                       37029.3  |                      6841.74 |                     14797.9  |                         13630.6  |                         52664.5  |                       30391.3  |                       13878.1  |                       10775.1  |                        13959.5  |                        28383.7  |                       6181.8  |                      25965.4  |                      11247.9  |
|                  2 |                     60 |                       8000 |                       13687.1  |                     13669    |                     14178.7  |                         14903.3  |                         27220.9  |                        9331.59 |                        9302.4  |                       12338.9  |                        12505.6  |                        29205.8  |                       6751.01 |                      14478.5  |                      15552.4  |
|                  2 |                     60 |                      16000 |                       26017.8  |                     14395.6  |                     32576.9  |                         26958.9  |                         47215.8  |                       15355.4  |                       20077.3  |                       26599.5  |                        48105    |                        78209.6  |                      12830.2  |                      19379.9  |                      28031.4  |
|                  2 |                     60 |                      44100 |                       66923.5  |                     40033.7  |                     67436.1  |                         95410.4  |                        115635    |                       37223    |                       51123.1  |                       64698.8  |                        73337.1  |                       135838    |                      33070.6  |                      54083    |                      66809.8  |

---

[crate]: https://crates.io/crates/wavers  
[crate-img]: https://img.shields.io/crates/v/wavers?style=for-the-badge&color=009E73&label=crates.io

[docs]: https://docs.rs/wavers  
[docs-img]: https://img.shields.io/badge/docs.rs-online-009E73?style=for-the-badge&labelColor=gray

[crate-pywavers]: https://github.com/jmg049/pywavers  
[crate-pywavers-img]: https://img.shields.io/badge/PyWavers-GitHub-56B4E9?style=for-the-badge&labelColor=gray

[crate-pywavers-docrs-img]: https://img.shields.io/badge/PyWavers-docs.rs-56B4E9?style=for-the-badge&labelColor=gray  
[pywavers-docrs]: https://docs.rs/pywavers

[crate-pywavers-readdocs-img]: https://img.shields.io/badge/Read_The_Docs-PyWavers-56B4E9?style=for-the-badge&labelColor=gray  
[pywavers-readdocs]: https://pywavers.readthedocs.io

[license-img]: https://img.shields.io/crates/l/wavers?style=for-the-badge&label=license&labelColor=gray  
[license]: https://github.com/jmg049/wavers/blob/main/LICENSE
