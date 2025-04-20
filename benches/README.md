# WAV Audio Library Benchmark Methodology

## Introduction

This document describes the methodology used to benchmark various WAV audio file processing libraries across different programming languages and implementations. The benchmark focuses on measuring read and write performance across different audio formats, sample rates, durations, and channel configurations.

## Libraries Compared

- **libsndfile (C)**: A widely-used C library for reading and writing audio files, considered an industry standard. Supports all tested formats.
- **Wavers (Rust)**: A Rust library designed for efficient WAV file handling, leveraging Rust's memory safety features. Supports all tested formats.
- **Hound (Rust)**: Another Rust library for WAV processing with a different API approach. Has format limitations - only supports PCM_16 (i16), PCM_32 (i32), and IEEE_F32 (f32).

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

Mono and stereo represent the vast majority of audio applications. More channels (5.1, 7.1, etc.) are less common outside of specialized applications and would significantly increase the benchmark running time without providing broadly applicable insights.

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

## Output and Analysis

### JSON Output Format

Results are stored in a standardized JSON format with consistent fields:

- `operation`: read or write
- `format`: sample format (PCM_16, PCM_24, etc.)
- `duration_sec`: duration in seconds
- `sample_rate`: sample rate in Hz
- `channels`: number of channels
- `mean_time`: average time in seconds
- `min_time`: minimum time in seconds
- `max_time`: maximum time in seconds
- `median_time`: median time in seconds
- `std_dev`: standard deviation in seconds
- `throughput_bytes_per_sec`: throughput in bytes per second
- `file_size_bytes`: size of the processed file in bytes
- `sample_count`: number of measurements taken



## Interpretation Guidelines

When interpreting the benchmark results, consider:

1. **Use Case Alignment**: Focus on the configurations that match your intended use case.

2. **Performance vs. Feature Tradeoffs**: The fastest library may not have all the features you need.

3. **Throughput vs. Latency**: For real-time applications, consistent performance (low standard deviation) may be more important than raw speed.

4. **Memory Usage**: While not directly measured, different formats have different memory implications.

5. **Error Handling**: The benchmark focuses on the happy path; error handling characteristics are not tested.

6. **Format Support Limitations**: Not all libraries support all formats. Specifically:
   - Hound does not support PCM_24 (24-bit audio) or IEEE_F64 (64-bit float)
   - When comparing libraries, consider only the formats supported by all libraries in the comparison
   - For format-specific needs, eliminate libraries that don't support your required format

## Conclusion

This benchmark methodology provides a fair, comprehensive comparison of WAV audio library performance across a range of real-world configurations. The approach balances thoroughness with practical constraints to deliver meaningful performance insights for different use cases.

By testing across multiple dimensions (format, sample rate, duration, channels) and operations (read, write), the benchmark results enable informed decisions about which library