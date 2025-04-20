# Hound Benchmark

This is a standalone benchmark for the Hound crate to measure read and write performance for WAV files.

## Features

- Benchmarks reading and writing for 5 sample formats (PCM_16, PCM_32, IEEE_F32, IEEE_F64)
- Tests with various durations (1, 10, and 60 seconds)
- Tests with different sample rates (8000, 16000, 44100 Hz)
- Tests with mono and stereo configurations
- Produces JSON output compatible with the analysis script

## Running the Benchmark

1. Make sure you have Rust installed (https://www.rust-lang.org/tools/install)
2. Clone this repository
3. Run the benchmark:

```bash
cargo run --release
```

The benchmark will:

1. Create a temporary directory `./tmp_bench_resources`
2. Run all benchmarks, showing progress
3. Output results to `hound_benchmark_results.json`
4. Clean up the temporary directory

## Benchmark Process

For each combination of format, duration, sample rate, and channels, the benchmark:

1. Performs a warm-up phase (minimum 3 seconds)
2. Takes multiple timing measurements (up to 100 samples)
3. Calculates statistics (mean, median, min, max, standard deviation)
4. Measures throughput in bytes per second

