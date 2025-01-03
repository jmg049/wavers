use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use i24::i24;
use std::path::Path;
use std::{alloc::Layout, fmt::Debug};
use wavers::iter::ChannelIterator;
use wavers::{AudioSample, ConvertTo, Samples, Wav, WaversResult};

fn alloc_sample_buffer<T>(len: usize) -> Box<[T]>
where
    T: AudioSample + Copy + Debug,
{
    if len == 0 {
        return <Box<[T]>>::default();
    }

    let layout = match Layout::array::<T>(len) {
        Ok(layout) => layout,
        Err(_) => panic!("Failed to allocate buffer of size {}", len),
    };

    let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
    let slice_ptr: *mut [T] = core::ptr::slice_from_raw_parts_mut(ptr, len);
    unsafe { Box::from_raw(slice_ptr) }
}
// Original implementation for comparison
pub struct OriginalChannelIterator<'a, T: 'a + AudioSample>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    wav: &'a mut Wav<T>,
    current_channel: usize,
    n_samples_per_channel: usize,
}

impl<'a, T: 'a + AudioSample> Iterator for OriginalChannelIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Item = Samples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let n_channels = self.wav.n_channels() as u64;
        let current_channel = self.current_channel as u64;

        if current_channel >= n_channels {
            match self.wav.to_data() {
                Ok(_) => (),
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
            return None;
        }

        match self.wav.seek_by_samples(current_channel) {
            Ok(_) => (),
            Err(_) => return None,
        }

        let mut samples: Box<[T]> = alloc_sample_buffer(self.n_samples_per_channel);

        for i in 0..self.n_samples_per_channel {
            let sample = match self.wav.read_sample() {
                Ok(frame) => frame,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    return None;
                }
            };
            samples[i] = sample;
            match self.wav.seek_by_samples(n_channels - 1) {
                Ok(_) => (),
                Err(_) => {
                    break;
                }
            }
        }

        self.current_channel += 1;
        match self.wav.to_data() {
            Ok(_) => (),
            Err(_) => return None,
        }

        Some(Samples::from(samples))
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

    // Generate different frequency sine waves for each channel
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

fn benchmark_channel_iterators(c: &mut Criterion) {
    let test_cases = vec![
        // (name, channels, duration in seconds)
        ("mono_short", 1, 1.0),
        ("stereo_short", 2, 1.0),
        ("5.1_short", 6, 1.0),
        ("mono_long", 1, 10.0),
        ("stereo_long", 2, 10.0),
        ("5.1_long", 6, 10.0),
    ];

    let temp_dir = tempfile::tempdir().unwrap();
    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("Channel Iterators");
    group
        .sample_size(200)
        .measurement_time(std::time::Duration::from_secs(60))
        .warm_up_time(std::time::Duration::from_secs(10));

    for (name, channels, duration) in test_cases {
        let wav_path = temp_dir.path().join(format!("test_{}.wav", name));
        create_test_wav(&wav_path, duration, 44100, channels).unwrap();

        // Benchmark original implementation
        group.bench_with_input(BenchmarkId::new("original", name), &wav_path, |b, path| {
            b.iter(|| {
                let mut wav = Wav::<f32>::from_path(path).unwrap();
                let n_samples_per_channel = wav.n_samples() / wav.n_channels() as usize;
                let iter = OriginalChannelIterator {
                    wav: &mut wav,
                    current_channel: 0,
                    n_samples_per_channel,
                };
                iter.count()
            })
        });

        // Benchmark optimized implementation
        group.bench_with_input(BenchmarkId::new("optimized", name), &wav_path, |b, path| {
            b.iter(|| {
                let mut wav = Wav::<f32>::from_path(path).unwrap();
                let iter = ChannelIterator::new(&mut wav);
                iter.count()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_channel_iterators);
criterion_main!(benches);
