use audio_samples::{AudioSample, ConvertTo, ConvertCollection, Samples, I24, AudioSamples};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use std::time::{Duration, Instant};
use rand::Rng;
use serde_json::{json, Value};
use wavers::Wav;

// Define the same benchmark parameters as in the C benchmark
const DURATIONS: [Duration; 3] = [
    Duration::from_secs(1),
    Duration::from_secs(10),
    Duration::from_secs(60),
];
const SAMPLE_RATES: [usize; 3] = [8000, 16000, 44100];
const N_CHANNELS: [usize; 2] = [1, 2];

// Statistics structure similar to the C benchmark
struct BenchmarkStats {
    mean: f64,
    min: f64,
    max: f64,
    standard_deviation: f64,
    median: f64,
    throughput: f64,
    file_size: u64,
    sample_count: usize,
}

// Generate random noise samples
fn noise(duration_sec: Duration, sample_rate: usize, n_channels: usize) -> Box<[f32]> {
    let n_samples = duration_sec.as_secs_f32() * sample_rate as f32 * n_channels as f32;
    let n_samples = n_samples.ceil() as usize;

    let mut rng = rand::rng();
    let data = (0..n_samples)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect::<Vec<f32>>();
    data.into_boxed_slice()
}

// Calculate benchmark statistics
fn calculate_stats(measurements: &[f64], file_size: u64) -> BenchmarkStats {
    if measurements.is_empty() {
        return BenchmarkStats {
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            standard_deviation: 0.0,
            median: 0.0,
            throughput: 0.0,
            file_size,
            sample_count: 0,
        };
    }

    // Sort measurements for median calculation
    let mut sorted_measurements = measurements.to_vec();
    sorted_measurements.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate mean
    let sum: f64 = measurements.iter().sum();
    let mean = sum / measurements.len() as f64;

    // Min and max
    let min = sorted_measurements[0];
    let max = sorted_measurements[sorted_measurements.len() - 1];

    // Calculate standard deviation
    let variance: f64 = measurements
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / measurements.len() as f64;
    let std_dev = variance.sqrt();

    // Calculate median
    let median = if measurements.len() % 2 == 0 {
        (sorted_measurements[measurements.len() / 2 - 1] + sorted_measurements[measurements.len() / 2]) / 2.0
    } else {
        sorted_measurements[measurements.len() / 2]
    };

    // Calculate throughput (bytes per second)
    let throughput = file_size as f64 / mean;

    BenchmarkStats {
        mean,
        min,
        max,
        standard_deviation: std_dev,
        median,
        throughput,
        file_size,
        sample_count: measurements.len(),
    }
}

// Get the format name as a string
fn get_format_name<T:'static>() -> &'static str {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>() {
        "PCM_16"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<I24>() {
        "PCM_24"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        "PCM_32"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        "IEEE_F32"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        "IEEE_F64"
    } else {
        "UNKNOWN"
    }
}

// Calculate warm-up and measurement times based on duration
fn get_benchmark_times(duration_sec: u64) -> (u64, u64, usize) {
    let warm_up_time = std::cmp::max(3, duration_sec / 10);
    let measurement_time = std::cmp::max(15, duration_sec / 2);
    let sample_size = if duration_sec > 600 { 50 } else { 100 };
    (warm_up_time, measurement_time, sample_size)
}

// Benchmark writing operation
fn benchmark_write<T: AudioSample + 'static>(
    duration: Duration,
    sample_rate: usize,
    n_channels: usize,
) -> BenchmarkStats 

    where i16: ConvertTo<T>,
          I24: ConvertTo<T>,
          i32: ConvertTo<T>,
          f32: ConvertTo<T>,
          f64: ConvertTo<T>,
          T: AudioSample
{
    let format_name = get_format_name::<T>();
    let tmp_filename = format!(
        "./tmp_bench_resources/Wavers_writing_{}_{}s_{}Hz_{}ch.wav",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels
    );

    // Generate samples
    let samples: Box<[T]> = noise(duration, sample_rate, n_channels).convert();

    // Calculate warm-up and measurement times
    let (warm_up_time, measurement_time, sample_size) = get_benchmark_times(duration.as_secs());

    println!(
        "Writing {} - {}s - {}Hz - {}ch (Warm-up: {}s, Measurement: {}s, Samples: {})...",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels,
        warm_up_time,
        measurement_time,
        sample_size
    );

    // Perform warm-up
    let warm_up_start = Instant::now();
    while warm_up_start.elapsed().as_secs() < warm_up_time {
        wavers::write(
            &tmp_filename,
            &samples,
            sample_rate as u32,
            Some(n_channels as u16),
        )
        .unwrap();
    }

    // Perform measurement
    let mut measurements = Vec::with_capacity(sample_size);
    let measurement_start = Instant::now();

    while measurements.len() < sample_size && measurement_start.elapsed().as_secs() < measurement_time {
        let start_sample = Instant::now();
        wavers::write(
            &tmp_filename,
            &samples,
            sample_rate as u32,
            Some(n_channels as u16),
        )
        .unwrap();
        let elapsed = start_sample.elapsed().as_secs_f64();
        measurements.push(elapsed);
    }

    // Get file size
    let file_size = fs::metadata(&tmp_filename).unwrap().len();

    // Calculate statistics
    calculate_stats(&measurements, file_size)
}

// Benchmark reading operation using AudioSamples
fn benchmark_read_audio_samples<T: AudioSample + 'static>(
    duration: Duration,
    sample_rate: usize,
    n_channels: usize,
) -> BenchmarkStats
    where i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>
{
    let format_name = get_format_name::<T>();
    let tmp_filename = format!(
        "./tmp_bench_resources/Wavers_reading_{}_{}s_{}Hz_{}ch.wav",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels
    );

    // Generate and write a file for reading benchmark
    let samples: Box<[T]> = noise(duration, sample_rate, n_channels).convert();
    wavers::write(
        &tmp_filename,
        &samples,
        sample_rate as u32,
        Some(n_channels as u16),
    )
    .unwrap();

    // Calculate warm-up and measurement times
    let (warm_up_time, measurement_time, sample_size) = get_benchmark_times(duration.as_secs());

    println!(
        "Reading {} - {}s - {}Hz - {}ch (Warm-up: {}s, Measurement: {}s, Samples: {})...",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels,
        warm_up_time,
        measurement_time,
        sample_size
    );

    // Perform warm-up
    let warm_up_start = Instant::now();
    while warm_up_start.elapsed().as_secs() < warm_up_time {
        let _: AudioSamples<T> = wavers::read_audio(&tmp_filename).unwrap();
    }

    // Perform measurement
    let mut measurements = Vec::with_capacity(sample_size);
    let measurement_start = Instant::now();

    while measurements.len() < sample_size && measurement_start.elapsed().as_secs() < measurement_time {
        let start_sample = Instant::now();
        let _: AudioSamples<T> = wavers::read_audio(&tmp_filename).unwrap();
        let elapsed = start_sample.elapsed().as_secs_f64();
        measurements.push(elapsed);
    }

    // Get file size
    let file_size = fs::metadata(&tmp_filename).unwrap().len();

    // Calculate statistics
    calculate_stats(&measurements, file_size)
}

// Benchmark Wav::to_audio_samples operation
fn benchmark_wav_to_audio_samples<T: AudioSample + 'static>(
    duration: Duration,
    sample_rate: usize,
    n_channels: usize,
) -> BenchmarkStats
    where i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>
{
    let format_name = get_format_name::<T>();
    let tmp_filename = format!(
        "./tmp_bench_resources/Wavers_wav_to_audio_samples_{}_{}s_{}Hz_{}ch.wav",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels
    );

    // Generate and write a file for the benchmark
    let samples: Box<[T]> = noise(duration, sample_rate, n_channels).convert();
    wavers::write(
        &tmp_filename,
        &samples,
        sample_rate as u32,
        Some(n_channels as u16),
    )
    .unwrap();

    // Calculate warm-up and measurement times
    let (warm_up_time, measurement_time, sample_size) = get_benchmark_times(duration.as_secs());

    println!(
        "Wav::to_audio_samples {} - {}s - {}Hz - {}ch (Warm-up: {}s, Measurement: {}s, Samples: {})...",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels,
        warm_up_time,
        measurement_time,
        sample_size
    );

    // Perform warm-up
    let warm_up_start = Instant::now();
    while warm_up_start.elapsed().as_secs() < warm_up_time {
        let mut wav: Wav<BufReader<File>> = Wav::<BufReader<File>>::from_path(&tmp_filename).unwrap();
        let _: AudioSamples<T> = wav.to_audio_samples().unwrap();
    }

    // Perform measurement
    let mut measurements = Vec::with_capacity(sample_size);
    let measurement_start = Instant::now();

    while measurements.len() < sample_size && measurement_start.elapsed().as_secs() < measurement_time {
        let start_sample = Instant::now();
        let mut wav: Wav<BufReader<File>> = Wav::<BufReader<File>>::from_path(&tmp_filename).unwrap();
        let _: AudioSamples<T> = wav.to_audio_samples().unwrap();
        let elapsed = start_sample.elapsed().as_secs_f64();
        measurements.push(elapsed);
    }

    // Get file size
    let file_size = fs::metadata(&tmp_filename).unwrap().len();

    // Calculate statistics
    calculate_stats(&measurements, file_size)
}

// Convert benchmark stats to JSON
fn stats_to_json(
    write_stats: &BenchmarkStats,
    read_stats: &BenchmarkStats,
    wav_to_audio_samples_stats: &BenchmarkStats,
    format_name: &str,
    duration_sec: u64,
    sample_rate: usize,
    n_channels: usize,
) -> Vec<Value> {
    vec![
        json!({
            "operation": "write",
            "format": format_name,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "channels": n_channels,
            "mean_time": write_stats.mean,
            "min_time": write_stats.min,
            "max_time": write_stats.max,
            "median_time": write_stats.median,
            "std_dev": write_stats.standard_deviation,
            "throughput_bytes_per_sec": write_stats.throughput,
            "file_size_bytes": write_stats.file_size,
            "sample_count": write_stats.sample_count
        }),
        json!({
            "operation": "read",
            "format": format_name,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "channels": n_channels,
            "mean_time": read_stats.mean,
            "min_time": read_stats.min,
            "max_time": read_stats.max,
            "median_time": read_stats.median,
            "std_dev": read_stats.standard_deviation,
            "throughput_bytes_per_sec": read_stats.throughput,
            "file_size_bytes": read_stats.file_size,
            "sample_count": read_stats.sample_count
        }),
        json!({
            "operation": "wav_to_audio_samples",
            "format": format_name,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "channels": n_channels,
            "mean_time": wav_to_audio_samples_stats.mean,
            "min_time": wav_to_audio_samples_stats.min,
            "max_time": wav_to_audio_samples_stats.max,
            "median_time": wav_to_audio_samples_stats.median,
            "std_dev": wav_to_audio_samples_stats.standard_deviation,
            "throughput_bytes_per_sec": wav_to_audio_samples_stats.throughput,
            "file_size_bytes": wav_to_audio_samples_stats.file_size,
            "sample_count": wav_to_audio_samples_stats.sample_count
        }),
    ]
}

fn main() {
    // Create temp directory
    if !Path::new("./tmp_bench_resources").exists() {
        fs::create_dir_all("./tmp_bench_resources").unwrap();
    }

    // Prepare JSON output
    let mut results = Vec::new();

    // Track progress
    let total_benchmarks = 5 * DURATIONS.len() * SAMPLE_RATES.len() * N_CHANNELS.len() * 3; // *3 for read, write, and wav_to_audio_samples
    let mut completed_benchmarks = 0;

    // Run benchmarks for each format, duration, sample rate, and channel count
    for duration in &DURATIONS {
        for &sample_rate in &SAMPLE_RATES {
            for &n_channels in &N_CHANNELS {
                // PCM_16 (i16)
                {
                    println!(
                        "\n[{}/{}] Benchmarking PCM_16 - {}s - {}Hz - {}ch",
                        completed_benchmarks + 1,
                        total_benchmarks,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let write_stats = benchmark_write::<i16>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let read_stats = benchmark_read_audio_samples::<i16>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let wav_to_audio_samples_stats = benchmark_wav_to_audio_samples::<i16>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    let json_stats = stats_to_json(
                        &write_stats,
                        &read_stats,
                        &wav_to_audio_samples_stats,
                        "PCM_16",
                        duration.as_secs(),
                        sample_rate,
                        n_channels,
                    );
                    results.extend(json_stats);
                }

                // PCM_24 (I24)
                {
                    println!(
                        "\n[{}/{}] Benchmarking PCM_24 - {}s - {}Hz - {}ch",
                        completed_benchmarks + 1,
                        total_benchmarks,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let write_stats = benchmark_write::<I24>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let read_stats = benchmark_read_audio_samples::<I24>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let wav_to_audio_samples_stats = benchmark_wav_to_audio_samples::<I24>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    let json_stats = stats_to_json(
                        &write_stats,
                        &read_stats,
                        &wav_to_audio_samples_stats,
                        "PCM_24",
                        duration.as_secs(),
                        sample_rate,
                        n_channels,
                    );
                    results.extend(json_stats);
                }

                // PCM_32 (i32)
                {
                    println!(
                        "\n[{}/{}] Benchmarking PCM_32 - {}s - {}Hz - {}ch",
                        completed_benchmarks + 1,
                        total_benchmarks,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let write_stats = benchmark_write::<i32>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let read_stats = benchmark_read_audio_samples::<i32>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let wav_to_audio_samples_stats = benchmark_wav_to_audio_samples::<i32>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    let json_stats = stats_to_json(
                        &write_stats,
                        &read_stats,
                        &wav_to_audio_samples_stats,
                        "PCM_32",
                        duration.as_secs(),
                        sample_rate,
                        n_channels,
                    );
                    results.extend(json_stats);
                }

                // IEEE_F32 (f32)
                {
                    println!(
                        "\n[{}/{}] Benchmarking IEEE_F32 - {}s - {}Hz - {}ch",
                        completed_benchmarks + 1,
                        total_benchmarks,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let write_stats = benchmark_write::<f32>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let read_stats = benchmark_read_audio_samples::<f32>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let wav_to_audio_samples_stats = benchmark_wav_to_audio_samples::<f32>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    let json_stats = stats_to_json(
                        &write_stats,
                        &read_stats,
                        &wav_to_audio_samples_stats,
                        "IEEE_F32",
                        duration.as_secs(),
                        sample_rate,
                        n_channels,
                    );
                    results.extend(json_stats);
                }

                // IEEE_F64 (f64)
                {
                    println!(
                        "\n[{}/{}] Benchmarking IEEE_F64 - {}s - {}Hz - {}ch",
                        completed_benchmarks + 1,
                        total_benchmarks,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let write_stats = benchmark_write::<f64>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let read_stats = benchmark_read_audio_samples::<f64>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let wav_to_audio_samples_stats = benchmark_wav_to_audio_samples::<f64>(*duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    let json_stats = stats_to_json(
                        &write_stats,
                        &read_stats,
                        &wav_to_audio_samples_stats,
                        "IEEE_F64",
                        duration.as_secs(),
                        sample_rate,
                        n_channels,
                    );
                    results.extend(json_stats);
                }

                // Print progress
                println!("Progress: {:.1}%", (completed_benchmarks as f32 / total_benchmarks as f32) * 100.0);
            }
        }
    }

    // Write results to JSON file
    let json_string = serde_json::to_string_pretty(&results).unwrap();
    fs::write("wavers_benchmark_results.json", json_string).unwrap();

    // Clean up temp directory
    fs::remove_dir_all("./tmp_bench_resources").unwrap();

    println!("\nBenchmark complete. Results saved to wavers_benchmark_results.json");
}