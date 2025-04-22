# Compilation Instructions for WAV Benchmark

To compile the WAV benchmark program, you'll need to have the following dependencies installed:

1. GCC or clang compiler
2. libsndfile development package
3. math library

## Installing Dependencies

### On Debian/Ubuntu

```bash
sudo apt-get update
sudo apt-get install build-essential libsndfile1-dev
```

### On macOS (using Homebrew)

```bash
brew install libsndfile
```

### On Fedora/RHEL

```bash
sudo dnf install gcc libsndfile-devel
```

## Compilation Command

Save the two source code files into a single file named `wav_benchmark.c`, then compile with:

```bash
gcc -o wav_benchmark wav_benchmark.c -lsndfile -lm -O3
```

## Running the Benchmark

Execute the compiled program:

```bash
./wav_benchmark
```

The benchmark will:

1. Create a temporary directory `./tmp_bench_resources`
2. Run all combinations of:
   - 5 audio formats (PCM_16, PCM_24, PCM_32, IEEE_F32, IEEE_F64)
   - 3 durations (1, 10, 60 seconds)
   - 3 sample rates (8000, 16000, 44100 Hz)
   - 2 channel configurations (mono, stereo)
3. Output results to `libsndfile_benchmark_results.json`
4. Clean up the temporary directory automatically

## Using the Results

The JSON output file can be processed with Python for analysis and comparison with the Rust benchmark results. The file structure is designed to be easily parsed and analyzed.
