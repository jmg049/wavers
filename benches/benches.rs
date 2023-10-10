use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hound::WavReader;

use std::{fs::File, path::Path, time::Duration};
use wavers::{read, ReadSeek, Samples, Wav};

const BENCHMARK_SIZE: usize = 10;

/// All files below are natively encooded as PCM-16 and have varying durations between approx. 6s and 15s
const FILES: [&'static str; BENCHMARK_SIZE] = [
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir1/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir1/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir2/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir2/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir3/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir3/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir4/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir4/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir5/ref.wav",
    "./test_resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir5/ref.wav",
];

const ONE_CHANNEL_WAV_I16: &'static str = "./test_resources/one_channel_i16.wav";
const ONE_CHANNEL_WAV_F32: &'static str = "./test_resources/one_channel_f32.wav";

fn bench_read_one_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("Reading Single File");

    group.bench_function("Read I16 as f32", |b| {
        b.iter(|| {
            let (_, _): (Samples<f32>, i32) =
                read::<f32, _>(black_box(ONE_CHANNEL_WAV_I16)).unwrap();
        })
    });

    group.bench_function(" Slice - Read I16 ", |b| {
        b.iter(|| {
            let (_, _): (Samples<i16>, i32) =
                read::<i16, _>(black_box(ONE_CHANNEL_WAV_I16)).unwrap();
        })
    });
}

fn bench_reading(c: &mut Criterion) {
    let mut group = c.benchmark_group("Reading");

    group.bench_function("Native i16 - Read", |b| {
        b.iter(|| {
            for file in FILES {
                let (_, _): (Samples<i16>, i32) = read::<i16, &_>(black_box(file)).unwrap();
            }
        })
    });

    group.bench_function("Native i16 - Read Wav File", |b| {
        b.iter(|| {
            for file in FILES {
                let _: &[i16] = Wav::<i16>::from_path(black_box(file))
                    .unwrap()
                    .read()
                    .unwrap()
                    .as_ref();
            }
        })
    });

    group.bench_function("Native i16 As f32", |b| {
        b.iter(|| {
            for file in FILES {
                let (_, _) = read::<f32, _>(black_box(file)).unwrap();
            }
        })
    });

    group.finish();
}

fn bench_writing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Writing");
    let tmp_out = Path::new("./test_resources/tmp");

    if !tmp_out.exists() {
        std::fs::create_dir(tmp_out).unwrap();
    }

    group.sample_size(400).sample_size(400);

    group.bench_function("Slice - Native i16", |b| {
        b.iter(|| {
            for file in FILES {
                let mut wav: Wav<i16> = Wav::<i16>::from_path(black_box(file)).unwrap();
                let out = tmp_out.join(file);
                let out = out.file_name().unwrap();
                wav.write::<i16, _>(black_box(&out)).unwrap();
            }
        })
    });

    group.bench_function("Slice - Native i16 As f32", |b| {
        b.iter(|| {
            for file in FILES {
                let mut wav: Wav<f32> = Wav::<f32>::from_path(black_box(file)).unwrap();
                let out = tmp_out.join(file);
                let out = out.file_name().unwrap();
                wav.write::<f32, _>(black_box(out)).unwrap();
            }
        })
    });

    group.bench_function("Write native f32", |b| {
        b.iter(|| {
            let mut wav: Wav<f32> = Wav::from_path(black_box(ONE_CHANNEL_WAV_F32)).unwrap();
            wav.write::<f32, _>("./test_resources/tmp/one_channel_f32.wav")
                .unwrap();
        })
    });

    group.bench_function("Hound", |b| {
        b.iter(|| {
            let mut reader = WavReader::open(black_box(ONE_CHANNEL_WAV_I16)).unwrap();
            let spec = reader.spec();
            let mut writer =
                hound::WavWriter::create("./test_resources/tmp/one_channel_i16.wav", spec).unwrap();
            for sample in reader.samples::<i16>() {
                writer.write_sample(sample.unwrap()).unwrap();
            }
        })
    });
    group.finish();
}

fn bench_wavers_vs_hound_native_i16(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hound vs Wavers - native i16");
    group.sample_size(400).sample_size(400);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("Hound i16", |b| {
        b.iter(|| {
            for file in FILES {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _: &[i16] = &reader
                    .samples::<i16>()
                    .filter_map(Result::ok)
                    .collect::<Vec<_>>();
            }
        })
    });

    group.bench_function("Wavers i16", |b| {
        b.iter(|| {
            for file in FILES {
                let _: &[i16] = &Wav::from_path(black_box(file)).unwrap().read().unwrap();
            }
        })
    });

    group.finish();
}

#[cfg(feature = "ndarray")]
fn bench_as_ndarray(c: &mut Criterion) {
    use wavers::{AsNdarray, IntoNdarray};

    let mut group = c.benchmark_group("Ndarray");
    group.sample_size(400).sample_size(400);

    group.bench_function("i16 as ndarray", |b| {
        b.iter(|| {
            let _ = Wav::<i16>::from_path(black_box(ONE_CHANNEL_WAV_I16))
                .unwrap()
                .as_ndarray()
                .unwrap();
        })
    });

    group.bench_function("i16 into ndarray", |b| {
        b.iter(|| {
            let _ = Wav::<i16>::from_path(black_box(ONE_CHANNEL_WAV_I16))
                .unwrap()
                .into_ndarray()
                .unwrap();
        })
    });
}

#[cfg(feature = "ndarray")]
criterion_group!(benches, bench_as_ndarray);

#[cfg(not(feature = "ndarray"))]
criterion_group!(
    benches,
    bench_read_one_file,
    bench_reading,
    bench_writing,
    bench_wavers_vs_hound_native_i16
);
criterion_main!(benches);
