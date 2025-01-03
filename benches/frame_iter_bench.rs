use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use i24::i24;
use std::path::Path;
use wavers::{iter::FrameIterator, AudioSample, ConvertTo, Samples, Wav, WaversResult};

// Original FrameIterator implementation for comparison
pub struct OriginalFrameIterator<'a, T: 'a + AudioSample>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    max_pos: u64,
    wav: &'a mut Wav<T>,
}

impl<'a, T: 'a + AudioSample> Iterator for OriginalFrameIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Item = Samples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let current_pos = match self.wav.current_pos() {
            Ok(pos) => pos,
            Err(_) => {
                match self.wav.to_data() {
                    Ok(_) => (),
                    Err(e) => {
                        eprintln!("Error: {}", e);
                    }
                }
                return None;
            }
        };

        if current_pos >= self.max_pos {
            match self.wav.to_data() {
                Ok(_) => (),
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
            return None;
        }

        let n_samples = self.wav.n_channels() as usize;
        let frame = match self.wav.read_samples(n_samples) {
            Ok(frame) => frame,
            Err(_) => return None,
        };
        Some(frame)
    }
}

fn create_test_wav(
    path: &Path,
    duration_secs: f32,
    sample_rate: i32,
    n_channels: u16,
) -> WaversResult<()> {
    let n_samples = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(n_samples * n_channels as usize);

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

fn benchmark_frame_iterators(c: &mut Criterion) {
    let test_cases = vec![("mono", 1), ("stereo", 2), ("5.1", 6)];
    let temp_dir = tempfile::tempdir().unwrap();
    let mut group = c.benchmark_group("Frame Iterators");

    // Configure the benchmark group
    group
        .sample_size(200)
        .measurement_time(std::time::Duration::from_secs(60))
        .warm_up_time(std::time::Duration::from_secs(10));

    for (name, channels) in test_cases {
        let wav_path = temp_dir.path().join(format!("test_{}.wav", name));
        create_test_wav(&wav_path, 5.0, 44100, channels).unwrap();

        // Benchmark original implementation
        group.bench_with_input(BenchmarkId::new("original", name), &wav_path, |b, path| {
            b.iter(|| {
                let mut wav = Wav::<f32>::from_path(path).unwrap();
                let iter = OriginalFrameIterator {
                    max_pos: wav.max_data_pos(),
                    wav: &mut wav,
                };
                iter.count()
            })
        });

        // Benchmark optimized implementation
        group.bench_with_input(BenchmarkId::new("optimized", name), &wav_path, |b, path| {
            b.iter(|| {
                let mut wav = Wav::<f32>::from_path(path).unwrap();
                let iter = FrameIterator::new(wav.max_data_pos(), &mut wav);
                iter.count()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_frame_iterators);
criterion_main!(benches);
