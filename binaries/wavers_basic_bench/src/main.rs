use audio_samples::AudioSamples;
use audio_samples::{AudioSample, ConvertTo};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::collections::HashMap;
use termimad::{MadSkin, print_text};
use wavers::{read, read_audio, write, write_audio};
use rayon::prelude::*;
use indicatif::ParallelProgressIterator;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let skin = MadSkin::default();
    let preamble = include_str!("../preamble.md");
    print_text(preamble);

    // Configuration
    let config = WaversBenchConfig {
        max_runtime: None,
        n_iter: Some(10_000),
        n_warmup_iter: Some(10),
    };

    // Generate all benchmark conditions
    let conditions = generate_all_conditions();

    println!("Generated {} total benchmark conditions", conditions.len());

    // Setup benchmark files
    let temp_dir = PathBuf::from("temp_benchmark_files");
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");
    
    let file_map = setup_benchmark_files(&conditions, &temp_dir)?;
    println!("Pre-generated {} benchmark files", file_map.len());

    // Run all benchmarks
    let progress_bar = ProgressBar::new(conditions.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .expect("Failed to create progress bar template"),
    );

    std::fs::create_dir_all("benchmarks").expect("Failed to create benchmarks directory");

    let json_file = File::create("benchmarks/benchmark_results.txt").expect("Failed to create JSON file for results");
    let buf_writer = std::io::BufWriter::new(json_file);

    let buf_writer = Arc::new(Mutex::new(buf_writer));


    conditions.par_iter().progress_count(conditions.len() as u64).for_each(|condition| {
        let buf_writer = Arc::clone(&buf_writer);
        progress_bar.set_message(format!("Running: {}", condition));

        let test_file = file_map.get(condition);
        let result = run_benchmark_condition(&condition, &config, test_file).expect("Failed to run benchmark condition");
        
        let mut result_for_disk = result.clone();


        // write each result to JSON file as they are generated
        result_for_disk.insert(result_for_disk.len(), '\n');

        {
            let mut buf_writer = buf_writer.lock().expect("Failed to lock buffer writer");

            buf_writer
                .write_all(result_for_disk.as_bytes())
                .expect("Failed to write result to JSON file");

            buf_writer.flush().expect("Failed to flush buffer to JSON file");
        }


    });

    progress_bar.finish_with_message("All benchmarks completed");

    // Cleanup benchmark files
    cleanup_benchmark_files(&temp_dir)?;
    println!("Cleaned up temporary benchmark files");

    Ok(())
}

// Generate synthetic audio data
fn generate_synthetic_audio<T: AudioSample>(
    sample_rate: i32,
    duration: Duration,
    channels: usize,
) -> Vec<T>
where
    T: AudioSample,
    f32: ConvertTo<T>,
{
    let mut rng = rand::rng();
    let total_samples = (sample_rate as f64 * duration.as_secs_f64()) as usize * channels;

    (0..total_samples)
        .map(|_| {
            let sample_f32: f32 = rng.random_range(-1.0..1.0);
            sample_f32.convert_to().expect("Failed to convert sample")
        })
        .collect()
}

// Generate all benchmark conditions
fn generate_all_conditions() -> Vec<String> {
    let mut conditions = Vec::new();

    let sample_rates = [8000, 16000, 44100];
    let durations = [1, 5, 10, 30, 60]; // seconds
    let channels = [1, 2];
    let sample_types = ["i16", "i32", "f32", "f64"];
    let functions = ["read_audio", "write_audio", "read", "write"];

    for &sr in &sample_rates {
        for &duration in &durations {
            for &ch in &channels {
                for &sample_type in &sample_types {
                    for &func in &functions {
                        conditions.push(format!(
                            "{}_{}_{}Hz_{}s_{}ch",
                            func, sample_type, sr, duration, ch
                        ));
                    }
                }
            }
        }
    }

    conditions
}

// Setup benchmark files - pre-generate all test files to avoid race conditions
fn setup_benchmark_files(conditions: &[String], temp_dir: &Path) -> Result<HashMap<String, PathBuf>, Box<dyn std::error::Error>> {
    let mut file_map = HashMap::new();
    
    for condition in conditions {
        // Parse condition to get parameters
        let mut parts: Vec<&str> = condition.rsplitn(5, '_').collect::<Vec<&str>>();
        parts.reverse();
        
        if parts.len() != 5 {
            continue;
        }
        
        let function = parts[0];
        let sample_type = parts[1];
        let sample_rate = parts[2].trim_end_matches("Hz").parse::<i32>()?;
        let duration_secs = parts[3].trim_end_matches("s").parse::<u64>()?;
        let channels = parts[4].trim_end_matches("ch").parse::<usize>()?;
        
        let duration = Duration::from_secs(duration_secs);
        
        // Only generate files for read operations (write operations create their own files)
        if function == "read_audio" || function == "read" {
            let filename = format!("{}_{}.wav", condition, std::process::id());
            let file_path = temp_dir.join(&filename);
            
            // Generate test file based on sample type
            match sample_type {
                "i16" => {
                    let samples = generate_synthetic_audio::<i16>(sample_rate, duration, channels);
                    write(&file_path, &samples.into_boxed_slice(), sample_rate as u32, Some(channels as u16))?;
                }
                "i32" => {
                    let samples = generate_synthetic_audio::<i32>(sample_rate, duration, channels);
                    write(&file_path, &samples.into_boxed_slice(), sample_rate as u32, Some(channels as u16))?;
                }
                "f32" => {
                    let samples = generate_synthetic_audio::<f32>(sample_rate, duration, channels);
                    write(&file_path, &samples.into_boxed_slice(), sample_rate as u32, Some(channels as u16))?;
                }
                "f64" => {
                    let samples = generate_synthetic_audio::<f64>(sample_rate, duration, channels);
                    write(&file_path, &samples.into_boxed_slice(), sample_rate as u32, Some(channels as u16))?;
                }
                _ => continue,
            }
            
            file_map.insert(condition.clone(), file_path);
        }
    }
    
    Ok(file_map)
}

// Cleanup benchmark files
fn cleanup_benchmark_files(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if temp_dir.exists() {
        std::fs::remove_dir_all(temp_dir)?;
    }
    Ok(())
}

// Run benchmark for a specific condition
fn run_benchmark_condition(
    condition: &str,
    config: &WaversBenchConfig,
    test_file: Option<&PathBuf>,
) -> Result<String, Box<dyn std::error::Error>> {
    // Parse condition string
    let mut parts: Vec<&str> = condition.rsplitn(5, '_').collect::<Vec<&str>>();
    parts.reverse();
    // print_text(&format!("Running benchmark for condition: {:?}", parts));
    if parts.len() != 5 {
        return Err("Invalid condition format".into());
    }

    let function = parts[0];
    let sample_type = parts[1];
    let sample_rate = parts[2].trim_end_matches("Hz").parse::<i32>()?;
    let duration_secs = parts[3].trim_end_matches("s").parse::<u64>()?;
    let channels = parts[4].trim_end_matches("ch").parse::<usize>()?;

    let duration = Duration::from_secs(duration_secs);
    let n_warmup = config.n_warmup_iter.unwrap_or(10);
    let n_iter = config.n_iter.unwrap_or(1000);

    // Run the actual benchmark based on sample type and function
    match (sample_type, function) {
        ("i16", "read_audio") => {
            let file_path = test_file.ok_or("Pre-generated file not found for read operation")?;
            run_read_audio_benchmark::<i16>(sample_rate, duration, channels, n_warmup, n_iter, file_path)
        }
        ("i16", "write_audio") => {
            run_write_audio_benchmark::<i16>(sample_rate, duration, channels, n_warmup, n_iter)
        }
        ("i16", "read") => {
            let file_path = test_file.ok_or("Pre-generated file not found for read operation")?;
            run_read_benchmark::<i16>(sample_rate, duration, channels, n_warmup, n_iter, file_path)
        }
        ("i16", "write") => {
            run_write_benchmark::<i16>(sample_rate, duration, channels, n_warmup, n_iter)
        }
        ("i32", "read_audio") => {
            let file_path = test_file.ok_or("Pre-generated file not found for read operation")?;
            run_read_audio_benchmark::<i32>(sample_rate, duration, channels, n_warmup, n_iter, file_path)
        }
        ("i32", "write_audio") => {
            run_write_audio_benchmark::<i32>(sample_rate, duration, channels, n_warmup, n_iter)
        }
        ("i32", "read") => {
            let file_path = test_file.ok_or("Pre-generated file not found for read operation")?;
            run_read_benchmark::<i32>(sample_rate, duration, channels, n_warmup, n_iter, file_path)
        }
        ("i32", "write") => {
            run_write_benchmark::<i32>(sample_rate, duration, channels, n_warmup, n_iter)
        }
        ("f32", "read_audio") => {
            let file_path = test_file.ok_or("Pre-generated file not found for read operation")?;
            run_read_audio_benchmark::<f32>(sample_rate, duration, channels, n_warmup, n_iter, file_path)
        }
        ("f32", "write_audio") => {
            run_write_audio_benchmark::<f32>(sample_rate, duration, channels, n_warmup, n_iter)
        }
        ("f32", "read") => {
            let file_path = test_file.ok_or("Pre-generated file not found for read operation")?;
            run_read_benchmark::<f32>(sample_rate, duration, channels, n_warmup, n_iter, file_path)
        }
        ("f32", "write") => {
            run_write_benchmark::<f32>(sample_rate, duration, channels, n_warmup, n_iter)
        }
        ("f64", "read_audio") => {
            let file_path = test_file.ok_or("Pre-generated file not found for read operation")?;
            run_read_audio_benchmark::<f64>(sample_rate, duration, channels, n_warmup, n_iter, file_path)
        }
        ("f64", "write_audio") => {
            run_write_audio_benchmark::<f64>(sample_rate, duration, channels, n_warmup, n_iter)
        }
        ("f64", "read") => {
            let file_path = test_file.ok_or("Pre-generated file not found for read operation")?;
            run_read_benchmark::<f64>(sample_rate, duration, channels, n_warmup, n_iter, file_path)
        }
        ("f64", "write") => {
            run_write_benchmark::<f64>(sample_rate, duration, channels, n_warmup, n_iter)
        }
        _ => Err(format!("Unsupported combination: {} {}", sample_type, function).into()),
    }
}

// Individual benchmark functions
fn run_read_audio_benchmark<T: AudioSample>(
    sample_rate: i32,
    duration: Duration,
    channels: usize,
    n_warmup: usize,
    n_iter: usize,
    test_file: &Path,
) -> Result<String, Box<dyn std::error::Error>>
where
    T: AudioSample,
    f32: ConvertTo<T>,
    i16: ConvertTo<T>,
    audio_samples::I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    // Use pre-generated test file
    // Warmup
    for _ in 0..n_warmup {
        let _ = read_audio::<T, _>(test_file)?;
    }

    // Benchmark
    let mut timings = Vec::new();
    for _ in 0..n_iter {
        let timing = time_fn(|| {
            let _ = read_audio::<T, _>(test_file).unwrap();
        });
        timings.push(timing);
    }

    let avg_time = timings.iter().sum::<Duration>() / timings.len() as u32;
    Ok(format!(
        "read_audio<{}>: sample_rate {} Hz, duration: {} seconds, channels: {}, avg {} µs",
        std::any::type_name::<T>(),
        sample_rate,
        duration.as_secs_f64(),
        channels,
        avg_time.as_micros(),
    ))
}

fn run_write_audio_benchmark<T: AudioSample>(
    sample_rate: i32,
    duration: Duration,
    channels: usize,
    n_warmup: usize,
    n_iter: usize,
) -> Result<String, Box<dyn std::error::Error>>
where
    T: AudioSample,
    f32: ConvertTo<T>,
    i16: ConvertTo<T>,
    audio_samples::I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    // Generate test data
    let samples = generate_synthetic_audio::<T>(sample_rate, duration, channels);
    let audio_samples = match channels {
        1 => AudioSamples::new_mono(Array1::from_vec(samples), sample_rate as u32),
        2 => AudioSamples::new_multi_channel(
            Array2::from_shape_vec(
                (2, samples.len() / 2),
                samples.into_iter().collect::<Vec<_>>(),
            )
            .unwrap(),
            sample_rate as u32,
        ),
        _ => return Err("Unsupported number of channels".into()),
    };

    // Warmup
    for i in 0..n_warmup {
        let warmup_file = format!("temp_warmup_{}_{}.wav", std::process::id(), i);
        write_audio(&warmup_file, &audio_samples)?;
        std::fs::remove_file(&warmup_file).ok();
    }

    // Benchmark
    let mut timings = Vec::new();
    for i in 0..n_iter {
        let test_file = format!("temp_test_{}_{}.wav", std::process::id(), i);
        let timing = time_fn(|| {
            write_audio(&test_file, &audio_samples).unwrap();
        });
        timings.push(timing);
        std::fs::remove_file(&test_file).ok();
    }

    let avg_time = timings.iter().sum::<Duration>() / timings.len() as u32;
    Ok(format!(
        "write_audio<{}>: sample_rate {} Hz, duration: {} seconds, channels: {}, avg {} µs",
        std::any::type_name::<T>(),
        sample_rate,
        duration.as_secs_f64(),
        channels,
        avg_time.as_micros(),
    ))
}

fn run_read_benchmark<T: AudioSample>(
    sample_rate: i32,
    duration: Duration,
    channels: usize,
    n_warmup: usize,
    n_iter: usize,
    test_file: &Path,
) -> Result<String, Box<dyn std::error::Error>>
where
    T: AudioSample,
    f32: ConvertTo<T>,
    i16: ConvertTo<T>,
    audio_samples::I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    // Use pre-generated test file
    // Warmup
    for _ in 0..n_warmup {
        let _: (Box<[T]>, u32, u16) = read(test_file)?;
    }

    // Benchmark
    let mut timings = Vec::new();
    for _ in 0..n_iter {
        let timing = time_fn(|| {
            let _: (Box<[T]>, u32, u16) = read(test_file).unwrap();
        });
        timings.push(timing);
    }

    let avg_time = timings.iter().sum::<Duration>() / timings.len() as u32;
    Ok(format!(
        "read<{}>: sample_rate {} Hz, duration: {} seconds, channels: {}, avg {} µs",
        std::any::type_name::<T>(),
        sample_rate,
        duration.as_secs_f64(),
        channels,
        avg_time.as_micros(),
    ))
}

fn run_write_benchmark<T: AudioSample>(
    sample_rate: i32,
    duration: Duration,
    channels: usize,
    n_warmup: usize,
    n_iter: usize,
) -> Result<String, Box<dyn std::error::Error>>
where
    T: AudioSample,
    f32: ConvertTo<T>,
    i16: ConvertTo<T>,
    audio_samples::I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    // Generate test data
    let samples = generate_synthetic_audio::<T>(sample_rate, duration, channels);
    let boxed_samples = samples.into_boxed_slice();

    
    // Warmup
    for i in 0..n_warmup {
        let warmup_file = format!("temp_warmup_{}_{}.wav", std::process::id(), i);
        write(
            &warmup_file,
            &boxed_samples,
            sample_rate as u32,
            Some(channels as u16),
        )?;
        std::fs::remove_file(&warmup_file).ok();
    }

    // Benchmark
    let mut timings = Vec::new();
    for i in 0..n_iter {
        let test_file = format!("temp_test_{}_{}.wav", std::process::id(), i);
        let timing = time_fn(|| {
            write(
                &test_file,
                &boxed_samples,
                sample_rate as u32,
                Some(channels as u16),
            )
            .unwrap();
        });
        timings.push(timing);
        std::fs::remove_file(&test_file).ok();
    }

    let avg_time = timings.iter().sum::<Duration>() / timings.len() as u32;
    Ok(format!(
        "write<{}>: sample_rate {} Hz, duration: {} seconds, channels: {}, avg {} µs",
        std::any::type_name::<T>(),
        sample_rate,
        duration.as_secs_f64(),
        channels,
        avg_time.as_micros(),
    ))
}

// Generate markdown report
fn generate_markdown_report(results: &[String]) {
    println!("\n## Benchmark Results\n");
    println!("Total benchmark conditions executed: {}", results.len());
    println!("\n### Results Summary\n");

    for result in results {
        println!("- {}", result);
    }

    println!("\n*Detailed statistical analysis would be implemented here*\n");
}

// Save all results to disk
fn save_all_results_to_disk(results: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    let filename = format!("benchmark_results_{}.json", timestamp);
    let json_results = serde_json::to_string_pretty(results)?;

    std::fs::write(&filename, json_results)?;
    println!("Results saved to: {}", filename);

    Ok(())
}

pub fn time_fn<F: FnOnce() -> T, T>(f: F) -> Duration {
    // Measures the time taken to execute a function `f`.
    let start = std::time::Instant::now();
    let _result = f();
    start.elapsed()
}

pub fn save_bench_results_to_disk<T: AudioSample>(
    results: WaversBenchResult<T>,
    fp: &Path,
) -> std::io::Result<()> {
    // Saves the results of a benchmark to a file in JSON format.
    let file = File::create(fp)?;
    let mut writer = std::io::BufWriter::new(file);
    let serialized = serde_json::to_string(&results)?;
    writer.write_all(serialized.as_bytes())?;
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WaversBenchConfig {
    pub max_runtime: Option<Duration>,
    pub n_iter: Option<usize>,
    pub n_warmup_iter: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WaversBenchResult<T: AudioSample>
where
    T: 'static + Deserialize<'static>,
{
    pub condition: WaversBenchCondition<T>,
    pub timings: Vec<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct WaversBenchCondition<T: AudioSample> {
    pub sr: i32,
    pub duration: Duration,
    pub sample_type: PhantomData<T>,
    pub channels: usize,
}

impl<T: AudioSample> WaversBenchCondition<T> {
    pub fn new(sr: i32, duration: Duration, channels: usize) -> Self {
        Self {
            sr,
            duration,
            sample_type: PhantomData,
            channels,
        }
    }

    pub fn sample_rate(&self) -> i32 {
        self.sr
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl<T: AudioSample> Display for WaversBenchCondition<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let md_str = format!(
            "``Sample Rate``: {} Hz\n``Duration``: {:.2} seconds\n``Channels``: {}\n``Sample Type``: {}",
            self.sr,
            self.duration.as_secs_f64(),
            self.channels,
            std::any::type_name::<T>()
        );
        write!(f, "{}", md_str)
    }
}
