use std::path::Path;
use std::time::{Duration, Instant};
use rand::Rng;
use serde_json::{json, Value};
use std::fs;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

// Define the same benchmark parameters as in the previous benchmarks
const DURATIONS: [Duration; 3] = [
    Duration::from_secs(1),
    Duration::from_secs(10),
    Duration::from_secs(60),
];
const SAMPLE_RATES: [usize; 3] = [8000, 16000, 44100];
const N_CHANNELS: [usize; 2] = [1, 2];

// Sample format enum
#[derive(Clone, Copy)]
enum SampleType {
    PCM16,
    PCM32,
    Float32,
}

impl SampleType {
    fn name(&self) -> &'static str {
        match self {
            SampleType::PCM16 => "PCM_16",
            SampleType::PCM32 => "PCM_32",
            SampleType::Float32 => "IEEE_F32",
        }
    }

    fn bits(&self) -> u16 {
        match self {
            SampleType::PCM16 => 16,
            SampleType::PCM32 => 32,
            SampleType::Float32 => 32,
        }
    }

    fn is_float(&self) -> bool {
        match self {
            SampleType::Float32  => true,
            _ => false,
        }
    }

    fn sample_format(&self) -> SampleFormat {
        match self {
            SampleType::PCM16 | SampleType::PCM32 => SampleFormat::Int,
            SampleType::Float32 => SampleFormat::Float,
        }
    }
}

// Statistics structure
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

// Calculate warm-up and measurement times based on duration
fn get_benchmark_times(duration_sec: u64) -> (u64, u64, usize) {
    let warm_up_time = std::cmp::max(3, duration_sec / 10);
    let measurement_time = std::cmp::max(15, duration_sec / 2);
    let sample_size = if duration_sec > 600 { 50 } else { 100 };
    (warm_up_time, measurement_time, sample_size)
}

// Generate a test WAV file for benchmarking
fn generate_test_wav(
    filename: &str,
    sample_type: SampleType,
    duration_sec: Duration,
    sample_rate: usize,
    n_channels: usize,
) -> WavSpec {
    let spec = WavSpec {
        channels: n_channels as u16,
        sample_rate: sample_rate as u32,
        bits_per_sample: sample_type.bits(),
        sample_format: sample_type.sample_format(),
    };

    let num_samples = duration_sec.as_secs() as usize * sample_rate * n_channels;
    let mut rng = rand::rng();

    let mut writer = WavWriter::create(filename, spec).unwrap();

    match sample_type {
        SampleType::PCM16 => {
            for _ in 0..num_samples {
                let sample = (rng.random_range(-1.0..1.0) * 32767.0) as i16;
                writer.write_sample(sample).unwrap();
            }
        },
        SampleType::PCM32 => {
            for _ in 0..num_samples {
                let sample = (rng.random_range(-1.0..1.0) * 2147483647.0) as i32;
                writer.write_sample(sample).unwrap();
            }
        },
        SampleType::Float32 => {
            for _ in 0..num_samples {
                let sample = rng.random_range(-1.0..1.0) as f32;
                writer.write_sample(sample).unwrap();
            }
        },

    }

    writer.finalize().unwrap();
    spec
}

// Benchmark writing operation
fn benchmark_write(
    sample_type: SampleType,
    duration: Duration,
    sample_rate: usize,
    n_channels: usize,
) -> BenchmarkStats {
    let format_name = sample_type.name();
    let input_filename = format!(
        "./tmp_bench_resources/Hound_input_{}_{}s_{}Hz_{}ch.wav",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels
    );
    let output_filename = format!(
        "./tmp_bench_resources/Hound_writing_{}_{}s_{}Hz_{}ch.wav",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels
    );

    // Generate test WAV file
    let spec = generate_test_wav(&input_filename, sample_type, duration, sample_rate, n_channels);

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

    // Read input file to get samples for writing benchmark
    let mut reader = WavReader::open(&input_filename).unwrap();
    let all_samples = match sample_type {
        SampleType::PCM16 => {
            let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
            Box::new(samples) as Box<dyn std::any::Any>
        },
        SampleType::PCM32 => {
            let samples: Vec<i32> = reader.samples().map(|s| s.unwrap()).collect();
            Box::new(samples) as Box<dyn std::any::Any>
        },
        SampleType::Float32 => {
            let samples: Vec<f32> = reader.samples().map(|s| s.unwrap()).collect();
            Box::new(samples) as Box<dyn std::any::Any>
        },

    };

    // Perform warm-up
    let warm_up_start = Instant::now();
    while warm_up_start.elapsed().as_secs() < warm_up_time {
        match sample_type {
            SampleType::PCM16 => {
                let samples = all_samples.downcast_ref::<Vec<i16>>().unwrap();
                let mut writer = WavWriter::create(&output_filename, spec).unwrap();
                for &sample in samples {
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
            },
            SampleType::PCM32 => {
                let samples = all_samples.downcast_ref::<Vec<i32>>().unwrap();
                let mut writer = WavWriter::create(&output_filename, spec).unwrap();
                for &sample in samples {
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
            },
            SampleType::Float32 => {
                let samples = all_samples.downcast_ref::<Vec<f32>>().unwrap();
                let mut writer = WavWriter::create(&output_filename, spec).unwrap();
                for &sample in samples {
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
            },

        }
    }

    // Perform measurement
    let mut measurements = Vec::with_capacity(sample_size);
    let measurement_start = Instant::now();

    while measurements.len() < sample_size && measurement_start.elapsed().as_secs() < measurement_time {
        let start_sample = Instant::now();
        match sample_type {
            SampleType::PCM16 => {
                let samples = all_samples.downcast_ref::<Vec<i16>>().unwrap();
                let mut writer = WavWriter::create(&output_filename, spec).unwrap();
                for &sample in samples {
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
            },
            SampleType::PCM32 => {
                let samples = all_samples.downcast_ref::<Vec<i32>>().unwrap();
                let mut writer = WavWriter::create(&output_filename, spec).unwrap();
                for &sample in samples {
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
            },
            SampleType::Float32 => {
                let samples = all_samples.downcast_ref::<Vec<f32>>().unwrap();
                let mut writer = WavWriter::create(&output_filename, spec).unwrap();
                for &sample in samples {
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
            },

        }
        let elapsed = start_sample.elapsed().as_secs_f64();
        measurements.push(elapsed);
    }

    // Get file size
    let file_size = fs::metadata(&output_filename).unwrap().len();

    // Calculate statistics
    let stats = calculate_stats(&measurements, file_size);

    // Clean up
    if Path::new(&input_filename).exists() {
        let _ = fs::remove_file(&input_filename);
    }
    if Path::new(&output_filename).exists() {
        let _ = fs::remove_file(&output_filename);
    }

    stats
}

// Benchmark reading operation
fn benchmark_read(
    sample_type: SampleType,
    duration: Duration,
    sample_rate: usize,
    n_channels: usize,
) -> BenchmarkStats {
    let format_name = sample_type.name();
    let tmp_filename = format!(
        "./tmp_bench_resources/Hound_reading_{}_{}s_{}Hz_{}ch.wav",
        format_name,
        duration.as_secs(),
        sample_rate,
        n_channels
    );

    // Generate test WAV file
    generate_test_wav(&tmp_filename, sample_type, duration, sample_rate, n_channels);

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
        match sample_type {
            SampleType::PCM16 => {
                let reader = WavReader::open(&tmp_filename).unwrap();
                let _: Vec<i16> = reader.into_samples().map(|s| s.unwrap()).collect();
            },
            SampleType::PCM32 => {
                let reader = WavReader::open(&tmp_filename).unwrap();
                let _: Vec<i32> = reader.into_samples().map(|s| s.unwrap()).collect();
            },
            SampleType::Float32 => {
                let reader = WavReader::open(&tmp_filename).unwrap();
                let _: Vec<f32> = reader.into_samples().map(|s| s.unwrap()).collect();
            },

        }
    }

    // Perform measurement
    let mut measurements = Vec::with_capacity(sample_size);
    let measurement_start = Instant::now();

    while measurements.len() < sample_size && measurement_start.elapsed().as_secs() < measurement_time {
        let start_sample = Instant::now();
        match sample_type {
            SampleType::PCM16 => {
                let reader = WavReader::open(&tmp_filename).unwrap();
                let _: Vec<i16> = reader.into_samples().map(|s| s.unwrap()).collect();
            },
            SampleType::PCM32 => {
                let reader = WavReader::open(&tmp_filename).unwrap();
                let _: Vec<i32> = reader.into_samples().map(|s| s.unwrap()).collect();
            },
            SampleType::Float32 => {
                let reader = WavReader::open(&tmp_filename).unwrap();
                let _: Vec<f32> = reader.into_samples().map(|s| s.unwrap()).collect();
            },

        }
        let elapsed = start_sample.elapsed().as_secs_f64();
        measurements.push(elapsed);
    }

    // Get file size
    let file_size = fs::metadata(&tmp_filename).unwrap().len();

    // Calculate statistics
    let stats = calculate_stats(&measurements, file_size);

    // Clean up
    if Path::new(&tmp_filename).exists() {
        let _ = fs::remove_file(&tmp_filename);
    }

    stats
}

// Convert benchmark stats to JSON
fn stats_to_json(
    write_stats: &BenchmarkStats,
    read_stats: &BenchmarkStats,
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
    ]
}

fn main() {
    // Create temp directory
    if !Path::new("./tmp_bench_resources").exists() {
        fs::create_dir_all("./tmp_bench_resources").unwrap();
    }

    // Prepare JSON output
    let mut results = Vec::new();

    // Define sample types to test
    let sample_types = [
        SampleType::PCM16,
        SampleType::PCM32,
        SampleType::Float32,
    ];

    // Track progress
    let total_benchmarks = sample_types.len() * DURATIONS.len() * SAMPLE_RATES.len() * N_CHANNELS.len() * 2; // *2 for read and write
    let mut completed_benchmarks = 0;

    // Run benchmarks for each format, duration, sample rate, and channel count
    for duration in &DURATIONS {
        for &sample_rate in &SAMPLE_RATES {
            for &n_channels in &N_CHANNELS {
                for &sample_type in &sample_types {
                    println!(
                        "\n[{}/{}] Benchmarking {} - {}s - {}Hz - {}ch",
                        completed_benchmarks + 1,
                        total_benchmarks,
                        sample_type.name(),
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let write_stats = benchmark_write(sample_type, *duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    println!("[{}/{}] ", completed_benchmarks + 1, total_benchmarks);

                    let read_stats = benchmark_read(sample_type, *duration, sample_rate, n_channels);
                    completed_benchmarks += 1;

                    let json_stats = stats_to_json(
                        &write_stats,
                        &read_stats,
                        sample_type.name(),
                        duration.as_secs(),
                        sample_rate,
                        n_channels,
                    );
                    results.extend(json_stats);

                    // Print progress
                    println!("Progress: {:.1}%", (completed_benchmarks as f32 / total_benchmarks as f32) * 100.0);
                }
            }
        }
    }

    // Write results to JSON file
    let json_string = serde_json::to_string_pretty(&results).unwrap();
    fs::write("hound_benchmark_results.json", json_string).unwrap();

    // Clean up temp directory
    if Path::new("./tmp_bench_resources").exists() {
        let _ = fs::remove_dir_all("./tmp_bench_resources");
    }

    println!("\nBenchmark complete. Results saved to hound_benchmark_results.json");
}