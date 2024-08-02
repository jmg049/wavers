use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, PlotConfiguration,
};

use rand::Rng;

use i24::i24;
use std::time::Duration;
use wavers::{Samples, Wav};

macro_rules! bench_sample_conversions {
    ($c:expr, $($from_type:ty => [$($to_type:ty),+]),+) => {
        const DURATIONS: [Duration; 3] = [
            Duration::from_secs(1),
            Duration::from_secs(10),
            Duration::from_secs(60),
        ];
        const SAMPLE_RATES: [usize; 3] = [8000, 16000, 44100];
        const N_CHANNELS: usize = 1;

        $(
            $(
                for duration in &DURATIONS {
                    for sample_rate in SAMPLE_RATES {
                        let group_name = format!(
                            "Samples conversion {} to {} - {}s - {}Hz - {}ch",
                            stringify!($from_type),
                            stringify!($to_type),
                            duration.as_secs(),
                            sample_rate,
                            N_CHANNELS
                        );

                        let warm_up_time = std::cmp::max(3, duration.as_secs() / 10);
                        let measurement_time = std::cmp::max(15, duration.as_secs() / 2);
                        let sample_size = if duration.as_secs() > 600 { 50 } else { 100 };

                        let mut group = $c.benchmark_group(&group_name);
                        group
                            .sample_size(sample_size)
                            .warm_up_time(std::time::Duration::from_secs(warm_up_time))
                            .measurement_time(std::time::Duration::from_secs(measurement_time))
                            .noise_threshold(0.05)
                            .significance_level(0.1)
                            .confidence_level(0.95);

                        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

                        let samples: Samples<$from_type> = noise(*duration, sample_rate, N_CHANNELS).convert();

                        group.bench_function(&group_name, |b| {
                            b.iter(|| {
                                let _samples = samples.clone().convert::<$to_type>();
                            });
                        });
                    }
                }
            )+
        )+
    };
}

macro_rules! bench_wavers_writing_for_type {
    ($c:expr, $sample_type:ty, $format_name:expr) => {
        for duration in &DURATIONS {
            for sample_rate in SAMPLE_RATES {
                for n_channels in N_CHANNELS {
                    let group_name = format!(
                        "Wavers writing {} - {}s - {}Hz - {}ch",
                        $format_name,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let warm_up_time = std::cmp::max(3, duration.as_secs() / 10);
                    let measurement_time = std::cmp::max(15, duration.as_secs() / 2);
                    let sample_size = if duration.as_secs() > 600 { 50 } else { 100 };

                    let mut group = $c.benchmark_group(&group_name);
                    group
                        .sample_size(sample_size)
                        .warm_up_time(std::time::Duration::from_secs(warm_up_time))
                        .measurement_time(std::time::Duration::from_secs(measurement_time))
                        .noise_threshold(0.05)
                        .significance_level(0.1)
                        .confidence_level(0.95);

                    group.plot_config(
                        PlotConfiguration::default().summary_scale(AxisScale::Logarithmic),
                    );

                    let tmp_file_name = format!("./tmp_bench_resources/{}.wav", group_name);
                    group.bench_function(&group_name, |b| {
                        // setup code
                        let samples: Samples<$sample_type> =
                            noise(*duration, sample_rate, n_channels).convert();

                        b.iter(|| {
                            wavers::write(
                                black_box(&tmp_file_name),
                                black_box(&samples),
                                black_box(sample_rate as i32),
                                black_box(n_channels as u16),
                            )
                            .unwrap();
                        });
                    });

                    // Calculate throughput after writing
                    let file_size = std::fs::metadata(&tmp_file_name).unwrap().len();
                    group.throughput(criterion::Throughput::Bytes(file_size));
                }
            }
        }
    };
}

macro_rules! bench_wavers_reading_for_type {
    ($c:expr, $sample_type:ty, $format_name:expr) => {
        for duration in &DURATIONS {
            for sample_rate in SAMPLE_RATES {
                for n_channels in N_CHANNELS {
                    let group_name = format!(
                        "Wavers reading {} - {}s - {}Hz - {}ch",
                        $format_name,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let warm_up_time = std::cmp::max(3, duration.as_secs() / 10);
                    let measurement_time = std::cmp::max(15, duration.as_secs() / 2);
                    let sample_size = if duration.as_secs() > 600 { 50 } else { 100 };

                    let mut group = $c.benchmark_group(&group_name);
                    group
                        .sample_size(sample_size)
                        .warm_up_time(std::time::Duration::from_secs(warm_up_time))
                        .measurement_time(std::time::Duration::from_secs(measurement_time))
                        .noise_threshold(0.05)
                        .significance_level(0.1)
                        .confidence_level(0.95);

                    group.plot_config(
                        PlotConfiguration::default().summary_scale(AxisScale::Logarithmic),
                    );

                    let tmp_file_name = format!("./tmp_bench_resources/{}.wav", group_name);
                    let file_size = std::fs::metadata(&tmp_file_name).unwrap().len();
                    group.throughput(criterion::Throughput::Bytes(file_size));
                    group.bench_function(&group_name, |b| {
                        // setup code
                        let samples: Samples<$sample_type> =
                            noise(*duration, sample_rate, n_channels).convert();

                        wavers::write(
                            &tmp_file_name,
                            &samples,
                            sample_rate as i32,
                            n_channels as u16,
                        )
                        .unwrap();

                        b.iter(|| {
                            let mut wav =
                                Wav::<$sample_type>::from_path(black_box(&tmp_file_name)).unwrap();
                            black_box(wav.read().unwrap())
                        });
                    });
                }
            }
        }
    };
}

macro_rules! bench_hound_reading_for_type {
    ($c:expr, $sample_type:ty, $format_name:expr) => {
        for duration in &DURATIONS {
            for sample_rate in SAMPLE_RATES {
                for n_channels in N_CHANNELS {
                    let group_name = format!(
                        "Hound reading {} - {}s - {}Hz - {}ch",
                        $format_name,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let warm_up_time = std::cmp::max(3, duration.as_secs() / 10);
                    let measurement_time = std::cmp::max(15, duration.as_secs() / 2);
                    let sample_size = if duration.as_secs() > 600 { 50 } else { 100 };

                    let mut group = $c.benchmark_group(&group_name);
                    group
                        .sample_size(sample_size)
                        .warm_up_time(std::time::Duration::from_secs(warm_up_time))
                        .measurement_time(std::time::Duration::from_secs(measurement_time))
                        .noise_threshold(0.05)
                        .significance_level(0.1)
                        .confidence_level(0.95);

                    group.plot_config(
                        PlotConfiguration::default().summary_scale(AxisScale::Logarithmic),
                    );

                    let tmp_file_name = format!("./tmp_bench_resources/{}.wav", group_name);
                    let file_size = std::fs::metadata(&tmp_file_name).unwrap().len();
                    group.throughput(criterion::Throughput::Bytes(file_size));
                    group.bench_function(&group_name, |b| {
                        // setup code
                        let samples: Samples<$sample_type> =
                            noise(*duration, sample_rate, n_channels).convert();

                        wavers::write(
                            &tmp_file_name,
                            &samples,
                            sample_rate as i32,
                            n_channels as u16,
                        )
                        .unwrap();

                        b.iter(|| {
                            let mut reader =
                                hound::WavReader::open(black_box(&tmp_file_name)).unwrap();
                            let _: &[$sample_type] = black_box(
                                &reader
                                    .samples::<$sample_type>()
                                    .filter_map(Result::ok)
                                    .collect::<Vec<_>>(),
                            );
                        });
                    });
                }
            }
        }
    };
}

macro_rules! bench_hound_writing_for_type {
    ($c:expr, $sample_type:ty, $format_name:expr) => {
        for duration in &DURATIONS {
            for sample_rate in SAMPLE_RATES {
                for n_channels in N_CHANNELS {
                    let group_name = format!(
                        "Hound writing {} - {}s - {}Hz - {}ch",
                        $format_name,
                        duration.as_secs(),
                        sample_rate,
                        n_channels
                    );

                    let warm_up_time = std::cmp::max(3, duration.as_secs() / 10);
                    let measurement_time = std::cmp::max(15, duration.as_secs() / 2);
                    let sample_size = if duration.as_secs() > 600 { 50 } else { 100 };

                    let mut group = $c.benchmark_group(&group_name);
                    group
                        .sample_size(sample_size)
                        .warm_up_time(std::time::Duration::from_secs(warm_up_time))
                        .measurement_time(std::time::Duration::from_secs(measurement_time))
                        .noise_threshold(0.05)
                        .significance_level(0.1)
                        .confidence_level(0.95);

                    group.plot_config(
                        PlotConfiguration::default().summary_scale(AxisScale::Logarithmic),
                    );

                    let input_file_name = format!("./tmp_bench_resources/input_{}.wav", group_name);
                    let output_file_name =
                        format!("./tmp_bench_resources/output_{}.wav", group_name);

                    // Create input file
                    let samples: Samples<$sample_type> =
                        noise(*duration, sample_rate, n_channels).convert();
                    wavers::write(
                        &input_file_name,
                        &samples,
                        sample_rate as i32,
                        n_channels as u16,
                    )
                    .unwrap();

                    // Read input file
                    let mut reader = hound::WavReader::open(black_box(&input_file_name)).unwrap();
                    let spec = reader.spec();
                    let samples: &[$sample_type] = &reader
                        .samples::<$sample_type>()
                        .filter_map(Result::ok)
                        .collect::<Vec<_>>();

                    group.bench_function(&group_name, |b| {
                        b.iter(|| {
                            let mut writer =
                                hound::WavWriter::create(black_box(&output_file_name), spec)
                                    .unwrap();
                            for sample in samples {
                                writer.write_sample(*sample).unwrap();
                            }
                            black_box(writer.finalize().unwrap());
                        });
                    });

                    // Calculate throughput after writing
                    let file_size = std::fs::metadata(&output_file_name).unwrap().len();
                    group.throughput(criterion::Throughput::Bytes(file_size));

                    // Clean up temporary files
                    std::fs::remove_file(&input_file_name).unwrap();
                    std::fs::remove_file(&output_file_name).unwrap();
                }
            }
        }
    };
}

fn bench_wavers_reading(c: &mut Criterion) {
    // set up the benchmark files
    const DURATIONS: [Duration; 3] = [
        Duration::from_secs(1),
        Duration::from_secs(10),
        Duration::from_secs(60),
    ];
    const SAMPLE_RATES: [usize; 3] = [8000, 16000, 44100];
    const N_CHANNELS: [usize; 2] = [1, 2];

    std::fs::create_dir_all("./tmp_bench_resources").unwrap();

    bench_wavers_reading_for_type!(c, i16, "PCM_16");
    bench_wavers_reading_for_type!(c, i24, "PCM_24");
    bench_wavers_reading_for_type!(c, i32, "PCM_32");
    bench_wavers_reading_for_type!(c, f32, "IEEE_F32");
    bench_wavers_reading_for_type!(c, f64, "IEEE_F64");

    std::fs::remove_dir_all("./tmp_bench_resources").unwrap();
}

fn bench_hound_reading(c: &mut Criterion) {
    // set up the benchmark files
    const DURATIONS: [Duration; 3] = [
        Duration::from_secs(1),
        Duration::from_secs(10),
        Duration::from_secs(60),
    ];
    const SAMPLE_RATES: [usize; 3] = [8000, 16000, 44100];
    const N_CHANNELS: [usize; 2] = [1, 2];

    std::fs::create_dir_all("./tmp_bench_resources").unwrap();

    bench_hound_reading_for_type!(c, i16, "PCM_16");
    // bench_hound_reading_for_type!(c, i24, "PCM_24");
    bench_hound_reading_for_type!(c, i32, "PCM_32");
    bench_hound_reading_for_type!(c, f32, "IEEE_F32");
    // bench_hound_reading_for_type!(c, f64, "IEEE_F64");

    std::fs::remove_dir_all("./tmp_bench_resources").unwrap();
}

fn bench_wavers_writing(c: &mut Criterion) {
    // set up the benchmark files
    const DURATIONS: [Duration; 3] = [
        Duration::from_secs(1),
        Duration::from_secs(10),
        Duration::from_secs(60),
    ];
    const SAMPLE_RATES: [usize; 3] = [8000, 16000, 44100];
    const N_CHANNELS: [usize; 2] = [1, 2];

    std::fs::create_dir_all("./tmp_bench_resources").unwrap();

    bench_wavers_writing_for_type!(c, i16, "PCM_16");
    bench_wavers_writing_for_type!(c, i24, "PCM_24");
    bench_wavers_writing_for_type!(c, i32, "PCM_32");
    bench_wavers_writing_for_type!(c, f32, "IEEE_F32");
    bench_wavers_writing_for_type!(c, f64, "IEEE_F64");

    std::fs::remove_dir_all("./tmp_bench_resources").unwrap();
}

fn bench_hound_writing(c: &mut Criterion) {
    // set up the benchmark files
    const DURATIONS: [Duration; 3] = [
        Duration::from_secs(1),
        Duration::from_secs(10),
        Duration::from_secs(60),
    ];
    const SAMPLE_RATES: [usize; 3] = [8000, 16000, 44100];
    const N_CHANNELS: [usize; 2] = [1, 2];

    std::fs::create_dir_all("./tmp_bench_resources").unwrap();

    bench_hound_writing_for_type!(c, i16, "PCM_16");
    // bench_hound_writing_for_type!(c, i24, "PCM_24");
    bench_hound_writing_for_type!(c, i32, "PCM_32");
    bench_hound_writing_for_type!(c, f32, "IEEE_F32");
    // bench_hound_writing_for_type!(c, f64, "IEEE_F64");

    std::fs::remove_dir_all("./tmp_bench_resources").unwrap();
}

fn bench_samples_conversion(c: &mut Criterion) {
    bench_sample_conversions!(
        c,
        i16 => [i24, i32, f32, f64],
        i24 => [i16, i32, f32, f64],
        i32 => [i16, i24, f32, f64],
        f32 => [i16, i24, i32, f64],
        f64 => [i16, i24, i32, f32]
    );
}

fn noise(duration_sec: Duration, sample_rate: usize, n_channels: usize) -> Samples<f32> {
    let n_samples = duration_sec.as_secs_f32() * sample_rate as f32 * n_channels as f32;
    let n_samples = n_samples.ceil() as usize;

    let mut rng = rand::thread_rng();
    let data = (0..n_samples as usize)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect::<Vec<f32>>();

    Samples::from(data.into_boxed_slice())
}

criterion_group!(
    benches,
    bench_wavers_reading,
    bench_wavers_writing,
    bench_samples_conversion,
    bench_hound_reading,
    bench_hound_writing
);
criterion_main!(benches);
