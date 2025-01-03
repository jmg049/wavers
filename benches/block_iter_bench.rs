use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::Path;
use wavers::iter::BlockIterator;
use wavers::{Wav, WaversResult};

fn create_test_wav(
    path: &Path,
    duration_secs: f32,
    sample_rate: i32,
    n_channels: u16,
) -> WaversResult<()> {
    let n_samples = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(n_samples * n_channels as usize);

    // Generate test signal - different frequencies for each channel
    for i in 0..n_samples {
        for ch in 0..n_channels {
            let t = i as f32 / sample_rate as f32;
            let freq = 440.0 * (ch + 1) as f32;
            let sample = (t * freq * 2.0 * std::f32::consts::PI).sin();
            samples.push(sample);
        }
    }

    wavers::write(path, &samples, sample_rate, n_channels)
}

fn benchmark_block_iterators(c: &mut Criterion) {
    let test_cases = vec![
        // (name, channels, duration, block_size, overlap)
        ("mono_small_blocks", 1, 5.0, 512, 128),
        ("stereo_small_blocks", 2, 5.0, 512, 128),
        ("mono_large_blocks", 1, 5.0, 4096, 1024),
        ("stereo_large_blocks", 2, 5.0, 4096, 1024),
        ("mono_huge_overlap", 1, 5.0, 8192, 7168),
        ("stereo_huge_overlap", 2, 5.0, 8192, 7168),
    ];

    let temp_dir = tempfile::tempdir().unwrap();
    let mut group = c.benchmark_group("Block Iterators");

    for (name, channels, duration, block_size, overlap) in test_cases {
        let wav_path = temp_dir.path().join(format!("test_{}.wav", name));
        create_test_wav(&wav_path, duration, 44100, channels).unwrap();

        group.bench_with_input(
            BenchmarkId::new("optimized", name),
            &(&wav_path, block_size, overlap),
            |b, (path, block_size, overlap)| {
                b.iter(|| {
                    let mut wav = Wav::<f32>::from_path(path).unwrap();
                    let iter = BlockIterator::new(&mut wav, *block_size, *overlap);
                    iter.count()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_block_iterators);
criterion_main!(benches);
