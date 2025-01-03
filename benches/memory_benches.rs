use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use std::time::Duration;
use wavers::{Samples, Wav};

use std::fmt;

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

pub fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Usage");

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

        // Memory usage during read operations
        group.bench_function(BenchmarkId::new("read_memory", &name), |b| {
            b.iter(|| {
                let (result, metrics) = track_memory(|| {
                    let mut wav = Wav::<f32>::from_path(&test_file).unwrap();
                    black_box(wav.read().unwrap())
                });
                black_box(result);
                black_box(metrics);
            })
        });

        // Memory usage during write operations
        group.bench_function(BenchmarkId::new("write_memory", &name), |b| {
            b.iter(|| {
                let output_file = format!("./bench_tmp/output_{}.wav", name);
                let (result, metrics) = track_memory(|| {
                    wavers::write(
                        &output_file,
                        &samples,
                        config.sample_rate as i32,
                        config.channels as u16,
                    )
                });
                black_box(result.unwrap());
                black_box(metrics);
                std::fs::remove_file(&output_file).unwrap();
            })
        });

        // Memory usage during seeking operations
        group.bench_function(BenchmarkId::new("seek_memory", &name), |b| {
            b.iter(|| {
                let mut wav = Wav::<f32>::from_path(&test_file).unwrap();
                let total_samples = wav.n_samples();
                let (result, metrics) = track_memory(|| {
                    for i in (0..total_samples).step_by(total_samples / 100) {
                        wav.seek_by_samples(i as u64).unwrap();
                        black_box(wav.read_sample().unwrap());
                    }
                });
                black_box(result);
                black_box(metrics);
            })
        });

        // Memory usage during format conversion
        group.bench_function(BenchmarkId::new("conversion_memory", &name), |b| {
            b.iter(|| {
                let (result, metrics) = track_memory(|| {
                    let i16_samples: Samples<i16> = samples.clone().convert();
                    black_box(i16_samples.convert::<f32>())
                });
                black_box(result);
                black_box(metrics);
            })
        });

        std::fs::remove_file(&test_file).unwrap();
    }
    group.finish();
}

pub fn bench_peak_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("Peak Memory Usage");

    // Test with increasingly large files
    let large_configs = vec![
        TestConfig {
            duration: Duration::from_secs(60),
            sample_rate: 44100,
            channels: 2,
        },
        TestConfig {
            duration: Duration::from_secs(300),
            sample_rate: 44100,
            channels: 2,
        },
    ];

    for config in large_configs {
        let name = config.to_string();
        let signal = TestSignal::Sine(440.0);
        let samples = generate_test_signal(signal, &config);
        let test_file = format!("./bench_tmp/large_{}.wav", name);

        wavers::write(
            &test_file,
            &samples,
            config.sample_rate as i32,
            config.channels as u16,
        )
        .unwrap();

        group.bench_function(BenchmarkId::new("large_file_memory", &name), |b| {
            b.iter(|| {
                let (result, metrics) = track_memory(|| {
                    let mut wav = Wav::<f32>::from_path(&test_file).unwrap();
                    black_box(wav.read().unwrap())
                });
                black_box(result);
                black_box(metrics);
            })
        });

        std::fs::remove_file(&test_file).unwrap();
    }
    group.finish();
}

criterion_group!(
    name = memory_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(30))
        .sample_size(25) // Fewer samples for memory tests as they're more resource intensive
        .noise_threshold(0.05);
    targets = bench_memory_usage, bench_peak_memory
);
