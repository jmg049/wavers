use criterion::{
    black_box, criterion_group, measurement::WallTime, AxisScale, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration, Throughput,
};
use i24::i24;
use std::{path::Path, time::Duration};
use wavers::{AudioSample, ConvertTo, Samples, Wav, DATA};

use crate::{
    common::{generate_test_signal, BenchType, SampleType, WaversBenchConfig},
    generate_bench_configs,
};

pub fn bench_read_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("Read/Write Operations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    group.finish();
}

fn benchmark_type<T: AudioSample>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    test_file: &str,
    samples: &Samples<T>,
    config: &TestConfig,
) where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    // Write benchmark
    group.bench_function(BenchmarkId::new("write", name), |b| {
        b.iter(|| {
            wavers::write(
                black_box(test_file),
                black_box(samples),
                black_box(config.sample_rate as i32),
                black_box(config.channels as u16),
            )
        })
    });

    // Write file for read test
    wavers::write(
        test_file,
        samples,
        config.sample_rate as i32,
        config.channels as u16,
    )
    .unwrap();

    let file_size = std::fs::metadata(test_file).unwrap().len();
    group.throughput(Throughput::Bytes(file_size));

    // Read benchmark
    group.bench_function(BenchmarkId::new("read", name), |b| {
        b.iter(|| {
            let mut wav = Wav::<T>::from_path(black_box(test_file)).unwrap();
            black_box(wav.read().unwrap());
        })
    });
}

pub fn bench_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sample Type Conversions");

    // Test all possible type conversions
    for config in CONFIGS {
        let name = format!("{}", config);

        // Generate test data as f32 first
        let f32_samples = generate_test_signal::<f32>(TestSignal::WhiteNoise, config);
        let i16_samples: Samples<i16> = f32_samples.clone().convert();
        let i24_samples: Samples<i24> = f32_samples.clone().convert();
        let i32_samples: Samples<i32> = f32_samples.clone().convert();
        let f64_samples: Samples<f64> = f32_samples.clone().convert();

        // Benchmark conversions between all types
        bench_conversion::<f32, i16>(&mut group, &name, &f32_samples);
        bench_conversion::<f32, i24>(&mut group, &name, &f32_samples);
        bench_conversion::<f32, i32>(&mut group, &name, &f32_samples);
        bench_conversion::<f32, f64>(&mut group, &name, &f32_samples);

        bench_conversion::<i16, i24>(&mut group, &name, &i16_samples);
        bench_conversion::<i16, i32>(&mut group, &name, &i16_samples);
        bench_conversion::<i16, f32>(&mut group, &name, &i16_samples);
        bench_conversion::<i16, f64>(&mut group, &name, &i16_samples);

        bench_conversion::<i24, i16>(&mut group, &name, &i24_samples);
        bench_conversion::<i24, i32>(&mut group, &name, &i24_samples);
        bench_conversion::<i24, f32>(&mut group, &name, &i24_samples);
        bench_conversion::<i24, f64>(&mut group, &name, &i24_samples);

        bench_conversion::<i32, i16>(&mut group, &name, &i32_samples);
        bench_conversion::<i32, i24>(&mut group, &name, &i32_samples);
        bench_conversion::<i32, f32>(&mut group, &name, &i32_samples);
        bench_conversion::<i32, f64>(&mut group, &name, &i32_samples);

        bench_conversion::<f32, i16>(&mut group, &name, &f32_samples);
        bench_conversion::<f32, i24>(&mut group, &name, &f32_samples);
        bench_conversion::<f32, i32>(&mut group, &name, &f32_samples);
        bench_conversion::<f32, f64>(&mut group, &name, &f32_samples);

        bench_conversion::<f64, i16>(&mut group, &name, &f64_samples);
        bench_conversion::<f64, i24>(&mut group, &name, &f64_samples);
        bench_conversion::<f64, i32>(&mut group, &name, &f64_samples);
        bench_conversion::<f64, f32>(&mut group, &name, &f64_samples);
    }

    group.finish();
}

fn bench_conversion<S: AudioSample, T: AudioSample>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    samples: &Samples<S>,
) where
    S: ConvertTo<T>,
{
    let conversion_name = format!(
        "{}_to_{}/{}",
        std::any::type_name::<S>(),
        std::any::type_name::<T>(),
        name
    );
    group.bench_function(BenchmarkId::new("convert", conversion_name), |b| {
        b.iter(|| black_box(samples.clone().convert::<T>()))
    });
}

// pub fn bench_seeking(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Seeking Operations");

//     for config in CONFIGS {
//         let name = format!("{}", config);

//         match config.sample_type {
//             SampleType::I16 => bench_seeking_type::<i16>(&mut group, &name, config),
//             SampleType::I24 => bench_seeking_type::<i24>(&mut group, &name, config),
//             SampleType::I32 => bench_seeking_type::<i32>(&mut group, &name, config),
//             SampleType::F32 => bench_seeking_type::<f32>(&mut group, &name, config),
//             SampleType::F64 => bench_seeking_type::<f64>(&mut group, &name, config),
//         }
//     }

//     group.finish();
// }

// fn bench_seeking_type<T: AudioSample>(
//     group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
//     name: &str,
//     config: &TestConfig,
// ) where
//     i16: ConvertTo<T>,
//     i24: ConvertTo<T>,
//     i32: ConvertTo<T>,
//     f32: ConvertTo<T>,
//     f64: ConvertTo<T>,
// {
//     let samples = generate_test_signal::<T>(TestSignal::Sine(440.0), config);
//     let test_file = format!("./bench_tmp/seeking_{}.wav", name);

//     // Write test file
//     wavers::write(
//         &test_file,
//         &samples,
//         config.sample_rate as i32,
//         config.channels as u16,
//     )
//     .unwrap();

//     // Random seeking
//     group.bench_function(BenchmarkId::new("random_seek", name), |b| {
//         let mut wav = Wav::<T>::from_path(&test_file).unwrap();
//         let data_chunk_offset = wav.header().get_chunk_info(DATA.into()).unwrap().offset;
//         let data_chunk_size = wav.header().get_chunk_info(DATA.into()).unwrap().size;

//         let seek_positions: Vec<u64> = (0..100)
//             .map(|_| {
//                 let pos = rand::random::<u64>() % data_chunk_size as u64;
//                 pos + data_chunk_offset as u64
//             })
//             .collect();

//         b.iter(|| {
//             for &pos in &seek_positions {
//                 wav.seek_by_samples(pos).unwrap();
//                 black_box(wav.read_sample().unwrap());
//             }
//         })
//     });

//     // Sequential seeking
//     group.bench_function(BenchmarkId::new("sequential_seek", name), |b| {
//         let mut wav = Wav::<T>::from_path(&test_file).unwrap();
//         let step = wav.n_samples() as u64 / 100;

//         b.iter(|| {
//             for i in (0..wav.n_samples() as u64).step_by(step as usize) {
//                 wav.seek_by_samples(i).unwrap();
//                 black_box(wav.read_sample().unwrap());
//             }
//         })
//     });

//     // Frame-by-frame seeking
//     group.bench_function(BenchmarkId::new("frame_seek", name), |b| {
//         let mut wav = Wav::<T>::from_path(&test_file).unwrap();
//         let n_channels = wav.n_channels() as u64;
//         let frame_count = wav.n_samples() as u64 / n_channels;
//         let step = frame_count / 100;

//         b.iter(|| {
//             for i in (0..frame_count).step_by(step as usize) {
//                 wav.seek_by_samples(i * n_channels).unwrap();
//                 // Read one full frame
//                 for _ in 0..n_channels {
//                     black_box(wav.read_sample().unwrap());
//                 }
//             }
//         })
//     });

//     // Clean up
//     std::fs::remove_file(&test_file).unwrap();
// }

fn bench_read_type<T: AudioSample>(group: &mut BenchmarkGroup<'_, WallTime>, fp: &Path, name: &str)
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    group.bench_function(BenchmarkId::new("read", name), |b| {
        b.iter(|| {
            let mut wav = Wav::<T>::from_path(black_box(fp)).unwrap();
            black_box(wav.read().unwrap());
        })
    });
}

fn bench_read(group: &mut BenchmarkGroup<'_, WallTime>, fp: &Path, config: &WaversBenchConfig) {
    let name = format!("{}", config);

    match config.sample_type {
        SampleType::I16 => bench_read_type::<i16>(group, fp, &name),
        SampleType::I24 => bench_read_type::<i24>(group, fp, &name),
        SampleType::I32 => bench_read_type::<i32>(group, fp, &name),
        SampleType::F32 => bench_read_type::<f32>(group, fp, &name),
        SampleType::F64 => bench_read_type::<f64>(group, fp, &name),
    }
}

pub fn benchmark_reading(configs: &[WaversBenchConfig]) {
    for config in configs {
        let test_file = config.fp.as_ref().unwrap();
        let warmup = config.core_bench_config.warmup_time;
        let measurement = config.core_bench_config.measurement_time;
        let sample_size = config.core_bench_config.sample_size;
        let noise_threshold = config.core_bench_config.noise_threshold;
        let measure_throughput = config.core_bench_config.measure_throughput;

        let mut C = Criterion::default()
            .warm_up_time(warmup)
            .measurement_time(measurement)
            .sample_size(sample_size)
            .noise_threshold(noise_threshold);

        let mut group = C.benchmark_group("Reading");

        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        if measure_throughput {
            let file_size = std::fs::metadata(test_file).unwrap().len();
            // File is generated by this point
            group.throughput(Throughput::Bytes(file_size));
        }

        bench_read(&mut group, test_file, config);

        group.finish();
    }
}

fn bench_write_type<T: AudioSample>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    fp: &Path,
    name: &str,
    samples: &Samples<T>,
    config: &WaversBenchConfig,
) where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    group.bench_function(BenchmarkId::new("write", name), |b| {
        b.iter(|| {
            wavers::write(
                black_box(fp),
                black_box(samples),
                black_box(config.sample_rate as i32),
                black_box(config.num_channels as u16),
            )
        })
    });
}

fn benchmark_writing(configs: &[WaversBenchConfig]) {
    for config in configs {
        let test_file = config.fp.as_ref().unwrap();
        let warmup = config.core_bench_config.warmup_time;
        let measurement = config.core_bench_config.measurement_time;
        let sample_size = config.core_bench_config.sample_size;
        let noise_threshold = config.core_bench_config.noise_threshold;

        let mut c = Criterion::default()
            .warm_up_time(warmup)
            .measurement_time(measurement)
            .sample_size(sample_size)
            .noise_threshold(noise_threshold);

        let mut group = c.benchmark_group("Writing");

        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        match config.sample_type {
            SampleType::I16 => {
                let samples = generate_test_signal::<i16>(config);
                bench_write_type::<i16>(
                    &mut group,
                    test_file,
                    &format!("{}", config),
                    &samples,
                    config,
                );
            }
            SampleType::I24 => {
                let samples = generate_test_signal::<i24>(config);
                bench_write_type::<i24>(
                    &mut group,
                    test_file,
                    &format!("{}", config),
                    &samples,
                    config,
                );
            }
            SampleType::I32 => {
                let samples = generate_test_signal::<i32>(config);
                bench_write_type::<i32>(
                    &mut group,
                    test_file,
                    &format!("{}", config),
                    &samples,
                    config,
                );
            }
            SampleType::F32 => {
                let samples = generate_test_signal::<f32>(config);
                bench_write_type::<f32>(
                    &mut group,
                    test_file,
                    &format!("{}", config),
                    &samples,
                    config,
                );
            }
            SampleType::F64 => {
                let samples = generate_test_signal::<f64>(config);
                bench_write_type::<f64>(
                    &mut group,
                    test_file,
                    &format!("{}", config),
                    &samples,
                    config,
                );
            }
        }

        group.finish();
    }
}

fn main() {
    let target_bench_type = BenchType::Read;
    let configs = generate_bench_configs();

    match target_bench_type {
        BenchType::Read => benchmark_reading(configs.get(&BenchType::Read).unwrap()),
        BenchType::Write => benchmark_writing(configs.get(&BenchType::Write).unwrap()),
        _ => (),
    }
}

// criterion_group!(
//     name = core_benches;
//     config = Criterion::default()
//         .warm_up_time(Duration::from_secs(5))
//         .measurement_time(Duration::from_secs(30))
//         .sample_size(50)
//         .noise_threshold(0.05);
//     targets =  bench_conversions
// );
