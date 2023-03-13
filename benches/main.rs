
#![feature(iter_collect_into)]
mod wav;
use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let p = Path::new("./test.wav");
    c.bench_function("one minute 440hz sine wave", |b| b.iter(|| wav::read_wav_i16_as_f32(p).unwrap()));
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);