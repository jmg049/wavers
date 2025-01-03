use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use hound;
use std::time::Duration;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use wavers::{Samples, Wav};

const CONFIGS: &[TestConfig] = &[
    TestConfig {
        duration: Duration::from_secs(1),
        sample_rate: 8000,
        channels: 1,
    },
    TestConfig {
        duration: Duration::from_secs(1),
        sample_rate: 44100,
        channels: 2,
    },
    TestConfig {
        duration: Duration::from_secs(10),
        sample_rate: 16000,
        channels: 1,
    },
    TestConfig {
        duration: Duration::from_secs(10),
        sample_rate: 48000,
        channels: 2,
    },
];

const SIGNALS: &[TestSignal] = &[
    TestSignal::Sine(440.0),
    TestSignal::WhiteNoise,
    TestSignal::Silence,
];

use std::fmt;

#[derive(Copy, Clone, Debug)]
pub enum TestSignal {
    Sine(f32), // Frequency in Hz
    WhiteNoise,
    Silence,
}

#[derive(Copy, Clone)]
pub struct TestConfig {
    pub duration: Duration,
    pub sample_rate: usize,
    pub channels: usize,
}

impl fmt::Display for TestConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}s_{}Hz_{}ch",
            self.duration.as_secs(),
            self.sample_rate,
            self.channels
        )
    }
}

pub fn generate_test_signal(signal: TestSignal, config: &TestConfig) -> Samples<f32> {
    let samples_per_channel = config.duration.as_secs_f32() * config.sample_rate as f32;
    let total_samples = (samples_per_channel * config.channels as f32).ceil() as usize;

    let data: Vec<f32> = match signal {
        TestSignal::Sine(freq) => (0..total_samples)
            .map(|i| {
                let t = (i / config.channels) as f32 / config.sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect(),
        TestSignal::WhiteNoise => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..total_samples)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        }
        TestSignal::Silence => vec![0.0; total_samples],
    };

    Samples::from(data.into_boxed_slice())
}

#[derive(Debug, Default)]
pub struct MemoryMetrics {
    pub peak: usize,
    pub allocated: usize,
    pub deallocated: usize,
}

pub fn track_memory<F, R>(f: F) -> (R, MemoryMetrics)
where
    F: FnOnce() -> R,
{
    
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
    static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);

    let start = MemoryMetrics {
        allocated: ALLOCATED.load(Ordering::SeqCst),
        deallocated: DEALLOCATED.load(Ordering::SeqCst),
        peak: 0,
    };

    let result = f();

    let end = MemoryMetrics {
        allocated: ALLOCATED.load(Ordering::SeqCst),
        deallocated: DEALLOCATED.load(Ordering::SeqCst),
        peak: 0, // Would need more sophisticated tracking for peak
    };

    (
        result,
        MemoryMetrics {
            allocated: end.allocated - start.allocated,
            deallocated: end.deallocated - start.deallocated,
            peak: end.allocated.max(start.allocated),
        },
    )
}

pub fn compare_read_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Read Performance Comparison");

    for config in CONFIGS {
        let name = config.to_string();
        let signal = TestSignal::Sine(440.0);
        let samples = generate_test_signal(signal, config);
        let test_file = format!("./bench_tmp/{}.wav", name);

        // Write test file
        wavers::write(
            &test_file,
            &samples,
            config.sample_rate as i32,
            config.channels as u16,
        )
        .unwrap();

        let file_size = std::fs::metadata(&test_file).unwrap().len();
        group.throughput(Throughput::Bytes(file_size));

        // Wavers benchmark
        group.bench_function(BenchmarkId::new("wavers", &name), |b| {
            b.iter(|| {
                let mut wav = Wav::<f32>::from_path(black_box(&test_file)).unwrap();
                black_box(wav.read().unwrap());
            })
        });

        // Hound benchmark
        group.bench_function(BenchmarkId::new("hound", &name), |b| {
            b.iter(|| {
                let mut reader = hound::WavReader::open(black_box(&test_file)).unwrap();
                black_box(
                    reader
                        .samples::<f32>()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                );
            })
        });

        // Symphonia benchmark
        group.bench_function(BenchmarkId::new("symphonia", &name), |b| {
            b.iter(|| {
                let file = std::fs::File::open(&test_file).unwrap();
                let mss = MediaSourceStream::new(Box::new(file), Default::default());

                let fmt_opts = FormatOptions::default();
                let mut hint = Hint::new();
                hint.with_extension("wav");

                let probed = symphonia::default::get_probe()
                    .format(&hint, mss, &fmt_opts, &Default::default())
                    .unwrap();

                let mut format = probed.format;
                let track = format.default_track().unwrap();
                let mut decoder = symphonia::default::get_codecs()
                    .make(&track.codec_params, &DecoderOptions::default())
                    .unwrap();

                while let Ok(packet) = format.next_packet() {
                    let decoded = decoder.decode(&packet).unwrap();
                    black_box(&decoded);
                }
            })
        });

        std::fs::remove_file(&test_file).unwrap();
    }
    group.finish();
}

pub fn compare_write_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Write Performance Comparison");

    for config in CONFIGS {
        let name = config.to_string();
        let signal = TestSignal::Sine(440.0);
        let samples = generate_test_signal(signal, config);

        let wavers_file = format!("./bench_tmp/wavers_{}.wav", name);
        let hound_file = format!("./bench_tmp/hound_{}.wav", name);

        // Wavers benchmark
        group.bench_function(BenchmarkId::new("wavers", &name), |b| {
            b.iter(|| {
                wavers::write(
                    black_box(&wavers_file),
                    black_box(&samples),
                    black_box(config.sample_rate as i32),
                    black_box(config.channels as u16),
                )
                .unwrap();
            })
        });

        // Hound benchmark
        group.bench_function(BenchmarkId::new("hound", &name), |b| {
            b.iter(|| {
                let spec = hound::WavSpec {
                    channels: config.channels as u16,
                    sample_rate: config.sample_rate as u32,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                };
                let mut writer = hound::WavWriter::create(black_box(&hound_file), spec).unwrap();
                for sample in samples.iter() {
                    writer.write_sample(*sample).unwrap();
                }
                black_box(writer.finalize().unwrap());
            })
        });

        // Cleanup
        if std::path::Path::new(&wavers_file).exists() {
            std::fs::remove_file(&wavers_file).unwrap();
        }
        if std::path::Path::new(&hound_file).exists() {
            std::fs::remove_file(&hound_file).unwrap();
        }
    }
    group.finish();
}

pub fn compare_seeking_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Seek Performance Comparison");

    for config in CONFIGS {
        let name = config.to_string();
        let signal = TestSignal::Sine(440.0);
        let samples = generate_test_signal(signal, config);
        let test_file = format!("./bench_tmp/{}.wav", name);

        // Write test file
        wavers::write(
            &test_file,
            &samples,
            config.sample_rate as i32,
            config.channels as u16,
        )
        .unwrap();

        // Generate seek positions
        let total_samples = samples.len();
        let seek_positions: Vec<u64> = (0..100)
            .map(|_| rand::random::<u64>() % total_samples as u64)
            .collect();

        // Wavers random seek benchmark
        group.bench_function(BenchmarkId::new("wavers_random_seek", &name), |b| {
            let mut wav = Wav::<f32>::from_path(&test_file).unwrap();
            b.iter(|| {
                for &pos in &seek_positions {
                    wav.seek_by_samples(pos).unwrap();
                    black_box(wav.read_sample().unwrap());
                }
            })
        });

        // Hound random seek benchmark
        group.bench_function(BenchmarkId::new("hound_random_seek", &name), |b| {
            let mut reader = hound::WavReader::open(&test_file).unwrap();
            b.iter(|| {
                for &pos in &seek_positions {
                    reader.seek(pos as u32).unwrap();
                    black_box(reader.samples::<f32>().next().unwrap().unwrap());
                }
            })
        });

        // Sequential seek benchmarks
        let step = total_samples / 100;

        group.bench_function(BenchmarkId::new("wavers_sequential_seek", &name), |b| {
            let mut wav = Wav::<f32>::from_path(&test_file).unwrap();
            b.iter(|| {
                for i in (0..total_samples).step_by(step) {
                    wav.seek_by_samples(i as u64).unwrap();
                    black_box(wav.read_sample().unwrap());
                }
            })
        });

        group.bench_function(BenchmarkId::new("hound_sequential_seek", &name), |b| {
            let mut reader = hound::WavReader::open(&test_file).unwrap();
            b.iter(|| {
                for i in (0..total_samples).step_by(step) {
                    reader.seek(i as u32).unwrap();
                    black_box(reader.samples::<f32>().next().unwrap().unwrap());
                }
            })
        });

        std::fs::remove_file(&test_file).unwrap();
    }
    group.finish();
}

criterion_group!(
    name = comparison_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(30))
        .sample_size(50)
        .noise_threshold(0.05);
    targets = compare_read_performance, compare_write_performance, compare_seeking_performance
);
