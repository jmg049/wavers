use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hound::WavReader;
use std::path::Path;
use std::time::Duration;

use wavers::{read, Sample, IterAudioConversion};

const BENCHMARK_SIZE: usize = 20;
const FILES: [&'static str; BENCHMARK_SIZE] = [
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir1/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir1/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir1/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir1/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir2/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir2/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir2/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir2/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir3/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir3/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir3/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir3/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir4/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir4/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir4/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir4/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir5/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir5/lpcnq.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir5/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir5/lpcnq.wav",
];

fn bench_i16_vec(c: & mut Criterion) {
    let mut group = c.benchmark_group("Vec - i16");
    group.significance_level(0.01).sample_size(400).measurement_time(Duration::from_secs(25));

    group.bench_function("Wavers - Vec - Sample(i16)", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _: Vec<Sample> = read(black_box(Path::new(file)), None).unwrap();
            }
        })
    });



    group.bench_function("Wavers - Vec - i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _: Vec<i16> = read(black_box(Path::new(file)), None).unwrap().as_i16();
            }
        })
    });

    
    group.bench_function("Hound - Vec - i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _: Vec<i16> = reader.samples::<i16>().filter_map(Result::ok).collect::<Vec<_>>();
            }
        })
    });
    group.finish();
    
}

fn bench_f32_vec(c: & mut Criterion) {
    let mut group = c.benchmark_group("Vec - f32");
    group.significance_level(0.01).sample_size(400).measurement_time(Duration::from_secs(25));

    group.bench_function("Wavers - Vec - Sample(f32)", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _: Vec<Sample> = read(black_box(Path::new(file)), None).unwrap().as_f32_samples();
            }
        })
    });
            

    group.bench_function("Wavers - Vec - f32", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _: Vec<f32> = read(black_box(Path::new(file)), None).unwrap().as_f32();
            }
        })
    });

    group.bench_function("Hound - Vec - f32", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _: Vec<f32> = reader.samples::<f32>().filter_map(Result::ok).collect::<Vec<_>>();
            }
        })
    });

    group.finish();
}

#[cfg(feature = "ndarray")]
fn bench_vec_to_ndarray(c: &mut Criterion) {
    use wavers::wave::IntoArray;

    use ndarray::Array2;
    let mut group = c.benchmark_group("Vec to Ndarray");
    let n_channels = 1;
    group.significance_level(0.01).sample_size(400).measurement_time(Duration::from_secs(25));

    group.bench_function("Wavers - Ndarray - Sample(i16)", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _: Array2<Sample> = read(black_box(Path::new(file)), None).unwrap().into_array(n_channels).unwrap();
            }
        })
    });

    group.bench_function("Wavers - Ndarray - i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let _: Array2<i16> = read(black_box(Path::new(file)), None).unwrap().as_i16().into_array(n_channels).unwrap();
            }
        })
    });

    group.bench_function("Hound - Ndarray - i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _ = reader.samples::<i16>().filter_map(Result::ok).collect::<Vec<_>>().into_array(n_channels).unwrap();            
            }
        })
    });
} 

#[cfg(feature = "ndarray")]
criterion_group!(benches, bench_vec_to_ndarray, bench_i16_vec, bench_f32_vec);

#[cfg(not(feature = "ndarray"))]
criterion_group!(benches, bench_i16_vec, bench_f32_vec);

criterion_main!(benches);