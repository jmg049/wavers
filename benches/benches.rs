use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hound::WavReader;
use std::path::Path;
use std::time::Duration;

use wavers::{read, write_wav_as, SampleType};

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
                let _: Vec<i16> = read(black_box(Path::new(file)), Some(SampleType::I16(0))).unwrap()
                .iter().map(|sample| sample.to_i16()).collect();
            }
        })
    });

    i16_group.bench_function("Hound - Read i16", |b| {
        b.iter(|| {
            for file in FILES.iter() {
                let mut reader = WavReader::open(black_box(file)).unwrap();
                let _: Vec<i16> = reader.samples::<i16>().filter_map(Result::ok).collect::<Vec<_>>();
            }
        })
    });
    i16_group.finish();
}


fn i16_to_f32(sample: i16) -> f32 {
    (sample as f32 / 32768.0).clamp(-1.0, 1.0)
}



// criterion_group!(benches, bench_i16_read, bench_i16_write, bench_f32_read, bench_f32_write);
criterion_group!(benches, bench_i16_read);
criterion_main!(benches);