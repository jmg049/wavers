use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hound::WavReader;
use ndarray::Array2;
use std::path::Path;
use std::time::Duration;

use wavers::{read, write_wav_as, SampleType, Sample};

const BENCHMARK_SIZE: usize = 20;
const FILES: [&'static str; BENCHMARK_SIZE] = [
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir1/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir1/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir1/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir1/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir2/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir2/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir2/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir2/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir3/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir3/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir3/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir3/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir4/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir4/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir4/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir4/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir5/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir5/lpcnq.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir5/ref.wav",
    "./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir5/lpcnq.wav",
];

fn bench_i16_read(c: &mut Criterion) {
    let mut i16_group = c.benchmark_group("i16_read");
    i16_group.significance_level(0.1).sample_size(500).measurement_time(Duration::from_secs(20));
    i16_group.bench_function("Wavers - Read i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _ = read(black_box(Path::new(file)), Some(SampleType::I16(0))).unwrap();
            }
        })
    });

    i16_group.bench_function("Hound - Read i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _ = Array2::from_shape_vec(
                    (reader.len() as usize, 1 as usize),
                    reader.samples::<i16>().collect::<Vec<_>>(),
                )
                .unwrap();
            }
        })
    });

    i16_group.bench_function("Wavers - Read i16 as Vec", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _ = read(black_box(Path::new(file)), None).unwrap().mapv(|sample| sample.convert_to(SampleType::I16(0))).into_raw_vec();
            }
        })
    });

    i16_group.bench_function("Hound - Read i16 as Vec", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _ = reader.samples::<i16>().filter_map(Result::ok).collect::<Vec<_>>();   
            }
        })
    });
    i16_group.finish();
}

fn bench_i16_write(c: & mut Criterion) {
    let mut i16_group = c.benchmark_group("i16_write");
    i16_group.significance_level(0.1).sample_size(500).measurement_time(Duration::from_secs(20));
    i16_group.bench_function("Wavers - Write i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let data = read(black_box(Path::new(file)), Some(SampleType::I16(0))).unwrap();
                write_wav_as(black_box(Path::new("test.wav")), &data, None, 16000).unwrap();
            }
        })
    });

    i16_group.bench_function("Hound - Write i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let wav_data = Array2::from_shape_vec(
                    (reader.len() as usize, 1 as usize),
                    reader.samples::<i16>().filter_map(Result::ok).collect::<Vec<_>>(),
                )
                .unwrap();

                let spec = hound::WavSpec {
                    channels: 1,
                    sample_rate: 16000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                };
                let mut writer = hound::WavWriter::create(black_box(Path::new("test.wav")), spec).unwrap();
                for sample in wav_data.iter() {
                    writer.write_sample(*sample).unwrap();
                }
            }
        })
    });
    i16_group.finish();

    // delete the test file
    std::fs::remove_file("test.wav").unwrap();
}

fn i16_to_f32(sample: i16) -> f32 {
    (sample as f32 / 32768.0).clamp(-1.0, 1.0)
}

fn bench_f32_read(c: &mut Criterion) {
    let mut f32_group = c.benchmark_group("f32_read");
    f32_group.significance_level(0.1).sample_size(500).measurement_time(Duration::from_secs(20));
    f32_group.bench_function("Wavers - Read f32", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _ = read(black_box(Path::new(file)), Some(SampleType::F32(0.0))).unwrap();
            }
        })
    });

    f32_group.bench_function("Hound - Read f32", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _ = Array2::from_shape_vec(
                    (reader.len() as usize, 1 as usize),
                    reader.samples::<i16>().filter_map(Result::ok).map(|int| i16_to_f32(int)).collect::<Vec<_>>(),
                )
                .unwrap();
            }
        })
    });

    f32_group.bench_function("Wavers - Read f32 as Vec", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let samples = read(black_box(Path::new(file)), None).unwrap().mapv(|sample| sample.convert_to_f32()).into_raw_vec();
            }
        })
    });

    f32_group.bench_function("Hound - Read f32 as Vec", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _ = reader.samples::<i16>().filter_map(Result::ok).map(|int| i16_to_f32(int)).collect::<Vec<_>>();
            }
        })
    });
    f32_group.finish();
}

fn bench_f32_write(c: & mut Criterion) {
    let mut f32_group = c.benchmark_group("f32_write");
    f32_group.significance_level(0.1).sample_size(500).measurement_time(Duration::from_secs(20));
    f32_group.bench_function("Wavers - Write f32", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let data = read(black_box(Path::new(file)), Some(SampleType::F32(0.0))).unwrap();
                write_wav_as(black_box(Path::new("test.wav")), &data, Some(SampleType::F32(0.0)), 16000).unwrap();
            }
        })
    });

    f32_group.bench_function("Hound - Write f32", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let wav_data = Array2::from_shape_vec(
                    (reader.len() as usize, 1 as usize),
                    reader.samples::<i16>().filter_map(Result::ok).map(|int| i16_to_f32(int)).collect::<Vec<_>>()
                )
                .unwrap();

                let spec = hound::WavSpec {
                    channels: 1,
                    sample_rate: 16000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                };
                let mut writer = hound::WavWriter::create(black_box(Path::new("test.wav")), spec).unwrap();
                for sample in wav_data.iter() {
                    writer.write_sample(*sample).unwrap();
                }
            }
        })
    });
    f32_group.finish();

    // delete the test file
    std::fs::remove_file("test.wav").unwrap();
}

criterion_group!(benches, bench_i16_read, bench_i16_write, bench_f32_read, bench_f32_write);
// criterion_group!(benches, bench_i16_read);
criterion_main!(benches);
