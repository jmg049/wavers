use i24::i24;
use std::{collections::HashMap, fmt::Display, path::PathBuf, time::Duration};
use wavers::{AudioSample, ConvertTo, Samples};
pub const BENCH_RESOURCES: &'static str = "./bench_resources";
// Define all supported sample types
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SampleType {
    I16,
    I24,
    I32,
    F32,
    F64,
}

impl Display for SampleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleType::I16 => write!(f, "i16"),
            SampleType::I24 => write!(f, "i24"),
            SampleType::I32 => write!(f, "i32"),
            SampleType::F32 => write!(f, "f32"),
            SampleType::F64 => write!(f, "f64"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BenchType {
    Read,
    Write,
    Conversion((SampleType, SampleType)),
    Undefined,
}

impl Display for BenchType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchType::Read => write!(f, "read"),
            BenchType::Write => write!(f, "write"),
            BenchType::Conversion((from, to)) => write!(f, "conversion_{}_to_{}", from, to),
            BenchType::Undefined => write!(f, "undefined"),
        }
    }
}

pub fn generate_test_signal<T: AudioSample>(config: &WaversBenchConfig) -> Samples<T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let samples_per_channel = config.duration.as_secs_f32() * config.sample_rate as f32;
    let total_samples = (samples_per_channel * config.num_channels as f32).ceil() as usize;

    let freq = 440.0; // A4

    let data: Vec<f32> = (0..total_samples)
        .map(|i| {
            let t = (i / config.num_channels) as f32 / config.sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    // Convert f32 samples to target type
    let f32_samples = Samples::from(data.into_boxed_slice());
    f32_samples.convert::<T>()
}

#[derive(Debug, Clone)]
pub struct WaversBenchConfig {
    pub core_bench_config: BenchConfig,
    pub duration: Duration,
    pub sample_rate: usize,
    pub num_channels: usize,
    pub sample_type: SampleType,
    pub fp: Option<PathBuf>,
}

impl WaversBenchConfig {
    pub fn new(
        core_bench_config: BenchConfig,
        duration: Duration,
        sample_rate: usize,
        num_channels: usize,
        sample_type: SampleType,
    ) -> Self {
        let mut cfg = WaversBenchConfig {
            core_bench_config,
            duration,
            sample_rate,
            num_channels,
            sample_type,
            fp: None,
        };

        let fp = format!("{}/{}.wav", BENCH_RESOURCES, cfg);
        let fp = PathBuf::from(fp);
        cfg.fp = Some(fp.clone());
        // Create the directory if it doesn't exist
        match std::fs::create_dir_all(fp.parent().unwrap()) {
            Ok(_) => (),
            Err(e) => panic!("Failed to create directory: {}", e),
        }

        // if the file doesn't exist and we are reading, generate the file
        if core_bench_config.bench_type == BenchType::Read {
            match std::fs::metadata(&fp) {
                Ok(_) => (), // File exists
                Err(_) => match sample_type {
                    SampleType::I16 => {
                        let samples = generate_test_signal::<i16>(&cfg);
                        wavers::write(fp, &samples, sample_rate as i32, num_channels as u16)
                            .unwrap();
                    }
                    SampleType::I24 => {
                        let samples = generate_test_signal::<i24>(&cfg);
                        wavers::write(fp, &samples, sample_rate as i32, num_channels as u16)
                            .unwrap();
                    }
                    SampleType::I32 => {
                        let samples = generate_test_signal::<i32>(&cfg);
                        wavers::write(fp, &samples, sample_rate as i32, num_channels as u16)
                            .unwrap();
                    }
                    SampleType::F32 => {
                        let samples = generate_test_signal::<f32>(&cfg);
                        wavers::write(fp, &samples, sample_rate as i32, num_channels as u16)
                            .unwrap();
                    }
                    SampleType::F64 => {
                        let samples = generate_test_signal::<f64>(&cfg);
                        wavers::write(fp, &samples, sample_rate as i32, num_channels as u16)
                            .unwrap();
                    }
                },
            }
        }

        cfg
    }
}

impl Display for WaversBenchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}_{}_{}s_{}Hz_{}ch",
            self.core_bench_config.bench_type,
            self.sample_type,
            self.duration.as_secs(),
            self.sample_rate,
            self.num_channels
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BenchConfig {
    pub bench_type: BenchType,
    pub measure_throughput: bool,
    pub sample_size: usize,
    pub noise_threshold: f64,
    pub measurement_time: Duration,
    pub warmup_time: Duration,
}

impl BenchConfig {
    pub fn new(
        bench_type: BenchType,
        measure_throughput: bool,
        sample_size: usize,
        noise_threshold: f64,
        measurement_time: Duration,
        warmup_time: Duration,
    ) -> Self {
        BenchConfig {
            bench_type,
            measure_throughput,
            sample_size,
            noise_threshold,
            measurement_time,
            warmup_time,
        }
    }
}

impl Display for BenchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BenchConfig {{ fp: {:?}, measure_throughput: {}, sample_size: {}, noise_threshold: {}, measurement_time: {:?}, warmup_time: {:?}, bench_type: {:?}}}", self.fp, self.measure_throughput, self.sample_size, self.noise_threshold, self.measurement_time, self.warmup_time, self.bench_type)
    }
}

pub fn generate_bench_configs() -> HashMap<BenchType, Vec<WaversBenchConfig>> {
    let mut configs = HashMap::new();

    let sample_rates = [8000, 16000, 44100];
    let channels = [1, 2];
    let durations = [
        Duration::from_secs(1),
        Duration::from_secs(10),
        Duration::from_secs(60),
    ];

    let mut read_configs = Vec::new();
    let mut write_configs = Vec::new();
    let mut conversion_configs = Vec::new();
    for sample_rate in sample_rates {
        for n_channels in channels {
            for durations in durations {
                for sample_type in [
                    SampleType::I16,
                    SampleType::I24,
                    SampleType::I32,
                    SampleType::F32,
                    SampleType::F64,
                ] {
                    let read_config = WaversBenchConfig::new(
                        BenchConfig::new(
                            BenchType::Read,
                            true,
                            100,
                            0.05,
                            Duration::from_secs(30),
                            Duration::from_secs(5),
                        ),
                        durations,
                        sample_rate,
                        n_channels,
                        sample_type,
                    );

                    read_configs.push(read_config);

                    let write_config = WaversBenchConfig::new(
                        BenchConfig::new(
                            BenchType::Write,
                            true,
                            100,
                            0.05,
                            Duration::from_secs(30),
                            Duration::from_secs(5),
                        ),
                        durations,
                        sample_rate,
                        n_channels,
                        sample_type,
                    );

                    write_configs.push(write_config);

                    for _sample_type in [
                        SampleType::I16,
                        SampleType::I24,
                        SampleType::I32,
                        SampleType::F32,
                        SampleType::F64,
                    ] {
                        if _sample_type == sample_type {
                            continue;
                        }
                        let conversion_config = WaversBenchConfig::new(
                            BenchConfig::new(
                                BenchType::Conversion((sample_type, _sample_type)),
                                true,
                                100,
                                0.05,
                                Duration::from_secs(30),
                                Duration::from_secs(5),
                            ),
                            durations,
                            sample_rate,
                            n_channels,
                            sample_type,
                        );

                        conversion_configs.push(conversion_config);
                    }
                }
            }
        }
    }

    configs.insert(BenchType::Read, read_configs);
    configs.insert(BenchType::Write, write_configs);
    configs.insert(
        BenchType::Conversion((SampleType::I16, SampleType::I24)),
        conversion_configs,
    );

    configs
}
