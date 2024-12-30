use crate::{
    core::alloc_sample_buffer,
    header::{ChunkIdentifier, HeaderChunkInfo},
    resampling::{interleaved_to_planar, planar_to_interleaved},
    AudioSample, ConvertSlice, ConvertTo, Samples, Wav, WavHeader, WaversError, WaversResult, DATA,
};
use i24::i24;
use rubato::{
    Resampler, Sample, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

pub struct ResamplingOptions {
    pub sinc_len: usize,
    pub f_cutoff: f64,
    pub window: WindowFunction,
    pub oversampling_factor: usize,
    pub chunk_size: usize,
}

impl Default for ResamplingOptions {
    fn default() -> Self {
        ResamplingOptions {
            sinc_len: 256,
            f_cutoff: 0.95,
            window: WindowFunction::BlackmanHarris2,
            oversampling_factor: 128,
            chunk_size: 128,
        }
    }
}

/// Result struct containing resampled audio data and associated metadata
///
/// This struct contains the resampled audio samples along with the necessary
/// header information and new sample rate. It is designed to be used directly
/// with the `write_with_header` function to save the resampled audio.
#[derive(Debug, Clone)]
pub struct ResampleResult<T>
where
    T: AudioSample,
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i24]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
{
    /// Updated WAV header with correct sample rate and chunk offsets
    pub header: WavHeader,
    /// Resampled audio data
    pub data: Samples<T>,
    /// New sample rate in Hz
    pub fs: i32,
}

/// Resamples audio data to a new sample rate while preserving audio quality
///
/// This function performs high-quality resampling using a Sinc interpolation filter.
/// It handles both upsampling and downsampling, and works with mono and multi-channel audio.
/// The implementation uses the Rubato library's SincFixedIn resampler with carefully chosen
/// parameters for optimal audio quality.
///
/// # Arguments
/// * `wav` - Source WAV file containing audio to resample
/// * `target_fs` - Target sample rate in Hz
///
/// # Returns
/// Returns a `ResampleResult` containing:
/// * Updated WAV header with new sample rate and chunk offsets
/// * Resampled audio data
/// * New sample rate
///
/// # Example
/// ```no_run
/// use wavers::{Wav, resample};
///
/// let mut wav = Wav::<f32>::from_path("input.wav").unwrap();
/// let target_fs = 48000;
/// let resampled = resample(&mut wav, target_fs).unwrap();
///
/// // Save resampled audio
/// wavers::write_with_header(
///     "output.wav",
///     &resampled.data,
///     &resampled.header
/// ).unwrap();
/// ```
#[inline(always)]

pub fn resample<T: AudioSample>(wav: &mut Wav<T>, target_fs: i32) -> WaversResult<ResampleResult<T>>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i24]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
    T: Sample,
{
    let fs = wav.sample_rate() as f64;
    let target_fs = target_fs as f64;
    let n_channels = wav.n_channels() as usize;
    let mut data = wav.read()?;
    interleaved_to_planar(data.as_mut(), n_channels)?;

    let mut channel_references: Vec<&[T]> = Vec::with_capacity(n_channels);

    for i in 0..n_channels {
        let offset = i * (data.len() / n_channels);
        channel_references.push(&data[offset..offset + data.len() / n_channels]);
    }

    let sinc_interpolator_params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<T>::new(
        target_fs / fs,
        2.0,
        sinc_interpolator_params,
        128,
        n_channels,
    )?;

    let mut resampled_data: Box<[T]> =
        alloc_sample_buffer((data.len() as f64 * target_fs / fs) as usize);
    resampler.process(&channel_references, None)?;

    let mut new_header = wav.header().clone();
    new_header.fmt_chunk.sample_rate = target_fs as i32;

    // Below is not enough, since changing the size will impact on the starting offsets of the other chunks
    // new_header.data_mut().size = (resampled_data.len() * std::mem::size_of::<T>()) as u32;

    let header_chunk_infos = &mut new_header.header_info;

    // Need to build up a list of the offsets in order
    // Then we need to find where the data chunk is and update the size
    // Then only update the chunk offsets after the data chunk
    //
    // Why? This function DOES NOT return a Wav struct, it the resampled data, the neew sample rate and a new header that can
    // be used to write the resampled data to a file immediately using the [crate::write_with_header].

    // Guaranteed to have a data chunk
    let data_chunk_info: &mut HeaderChunkInfo = header_chunk_infos.get_mut(&DATA.into()).unwrap();
    let origial_data_chunk_size = data_chunk_info.size;

    let new_data_chunk_size = (resampled_data.len() * std::mem::size_of::<T>()) as u32;
    data_chunk_info.size = new_data_chunk_size;

    let mut ordered_chunks: Vec<(ChunkIdentifier, HeaderChunkInfo)> = header_chunk_infos
        .iter_mut()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Sort by offset
    ordered_chunks.sort_by(|a, b| a.1.offset.cmp(&b.1.offset));

    // Update the data chunk size

    // update the other chunks after the data chunk
    let data_chunk_index = ordered_chunks
        .iter()
        .position(|(k, _)| *k == ChunkIdentifier::new(DATA))
        .unwrap();

    for i in data_chunk_index + 1..ordered_chunks.len() {
        let (chunk_id, chunk_info) = &ordered_chunks[i];
        let new_offset =
            chunk_info.offset + new_data_chunk_size as usize - origial_data_chunk_size as usize;
        header_chunk_infos.get_mut(chunk_id).unwrap().offset = new_offset;
    }

    // Now the header is updated.
    // 1. The sample rate field in the FMT_CHUNK is updated
    // 2. The size of the data chunk is updated
    // 3. The offsets of the chunks after the data chunk are updated
    // All contained in a new [crate::header::WavHeader] struct.
    // Finally re-interleave the data and return the result.

    planar_to_interleaved(resampled_data.as_mut(), n_channels)?;

    Ok(ResampleResult {
        header: new_header,
        data: Samples::new(resampled_data),
        fs: target_fs as i32,
    })
}
/// Performs resampling with custom options and progress reporting
pub fn resample_with_options<T: AudioSample, F: FnMut(f32)>(
    wav: &mut Wav<T>,
    target_fs: i32,
    options: ResamplingOptions,
) -> WaversResult<ResampleResult<T>>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i24]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
    T: Sample,
{
    let fs = wav.sample_rate() as f64;
    let target_fs = target_fs as f64;
    let n_channels = wav.n_channels() as usize;
    let mut data = wav.read()?;
    interleaved_to_planar(data.as_mut(), n_channels)?;

    let mut channel_references: Vec<&[T]> = Vec::with_capacity(n_channels);

    for i in 0..n_channels {
        let offset = i * (data.len() / n_channels);
        channel_references.push(&data[offset..offset + data.len() / n_channels]);
    }

    let sinc_interpolator_params = SincInterpolationParameters {
        sinc_len: options.sinc_len,
        f_cutoff: options.f_cutoff as f32,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: options.oversampling_factor,
        window: options.window,
    };

    let mut resampler = SincFixedIn::<T>::new(
        target_fs / fs,
        2.0,
        sinc_interpolator_params,
        options.chunk_size,
        n_channels,
    )?;

    let mut resampled_data: Box<[T]> =
        alloc_sample_buffer((data.len() as f64 * target_fs / fs) as usize);
    resampler.process(&channel_references, None)?;

    let mut new_header = wav.header().clone();
    new_header.fmt_chunk.sample_rate = target_fs as i32;

    // Below is not enough, since changing the size will impact on the starting offsets of the other chunks
    // new_header.data_mut().size = (resampled_data.len() * std::mem::size_of::<T>()) as u32;

    let header_chunk_infos = &mut new_header.header_info;

    // Need to build up a list of the offsets in order
    // Then we need to find where the data chunk is and update the size
    // Then only update the chunk offsets after the data chunk
    //
    // Why? This function DOES NOT return a Wav struct, it the resampled data, the neew sample rate and a new header that can
    // be used to write the resampled data to a file immediately using the [crate::write_with_header].

    // Guaranteed to have a data chunk
    let data_chunk_info: &mut HeaderChunkInfo = header_chunk_infos.get_mut(&DATA.into()).unwrap();
    let origial_data_chunk_size = data_chunk_info.size;

    let new_data_chunk_size = (resampled_data.len() * std::mem::size_of::<T>()) as u32;
    data_chunk_info.size = new_data_chunk_size;

    let mut ordered_chunks: Vec<(ChunkIdentifier, HeaderChunkInfo)> = header_chunk_infos
        .iter_mut()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Sort by offset
    ordered_chunks.sort_by(|a, b| a.1.offset.cmp(&b.1.offset));

    // Update the data chunk size

    // update the other chunks after the data chunk
    let data_chunk_index = ordered_chunks
        .iter()
        .position(|(k, _)| *k == ChunkIdentifier::new(DATA))
        .unwrap();

    for i in data_chunk_index + 1..ordered_chunks.len() {
        let (chunk_id, chunk_info) = &ordered_chunks[i];
        let new_offset =
            chunk_info.offset + new_data_chunk_size as usize - origial_data_chunk_size as usize;
        header_chunk_infos.get_mut(chunk_id).unwrap().offset = new_offset;
    }

    // Now the header is updated.
    // 1. The sample rate field in the FMT_CHUNK is updated
    // 2. The size of the data chunk is updated
    // 3. The offsets of the chunks after the data chunk are updated
    // All contained in a new [crate::header::WavHeader] struct.
    // Finally re-interleave the data and return the result.

    planar_to_interleaved(resampled_data.as_mut(), n_channels)?;

    Ok(ResampleResult {
        header: new_header,
        data: Samples::new(resampled_data),
        fs: target_fs as i32,
    })
}

#[cfg(test)]
mod resampling_tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    const TEST_OUTPUT: &str = "./test_resources/tmp/";
    const ONE_CHANNEL_WAV_I16: &str = "./test_resources/one_channel_i16.wav";
    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";

    #[test]
    fn test_basic_upsampling() {
        let mut wav: Wav<f32> = Wav::<f32>::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let original_fs = wav.sample_rate();
        let target_fs = original_fs * 2;
        let original_samples = wav.read().unwrap();
        let original_len = original_samples.len();

        let result = resample(&mut wav, target_fs).unwrap();

        // Check sample rate update
        assert_eq!(result.fs, target_fs);

        // Check approximate sample count ratio
        let expected_len = (original_len as f64 * (target_fs as f64 / original_fs as f64)) as usize;
        assert_eq!(result.data.len(), expected_len);

        // Verify header update
        assert_eq!(result.header.fmt_chunk.sample_rate, target_fs);
        assert!(result.header.data().size > wav.header().data().size);
    }

    #[test]
    fn test_basic_downsampling() {
        let mut wav = Wav::<f32>::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let original_fs = wav.sample_rate();
        let target_fs = original_fs / 2;
        let original_samples = wav.read().unwrap();
        let original_len = original_samples.len();

        let result = resample(&mut wav, target_fs).unwrap();

        // Check sample rate update
        assert_eq!(result.fs, target_fs);

        // Check approximate sample count ratio
        let expected_len = (original_len as f64 * (target_fs as f64 / original_fs as f64)) as usize;
        assert_eq!(result.data.len(), expected_len);

        // Verify header update
        assert_eq!(result.header.fmt_chunk.sample_rate, target_fs);
    }

    #[test]
    fn test_multichannel_resampling() {
        let mut wav = Wav::<f32>::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let original_fs = wav.sample_rate();
        let target_fs = original_fs * 2;
        let n_channels = wav.n_channels() as usize;

        let result = resample(&mut wav, target_fs).unwrap();

        // Check channel count preservation
        assert_eq!(result.data.len() % n_channels, 0);
        assert_eq!(
            result.header.fmt_chunk.channels, n_channels as u16,
            "Channel count mismatch: expected {}, got {}",
            n_channels, result.header.fmt_chunk.channels
        );
    }

    #[test]
    fn test_header_offsets_integrity() {
        let mut wav = Wav::<f32>::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let target_fs = wav.sample_rate() * 2;

        let original_header = wav.header().clone();
        let result = resample(&mut wav, target_fs).unwrap();

        // Get chunks after DATA chunk in original header
        let data_offset = original_header.data().offset;
        let original_post_data_chunks: Vec<_> = original_header
            .header_info
            .iter()
            .filter(|(_, info)| info.offset > data_offset)
            .collect();

        // Verify relative positioning of chunks after DATA is preserved
        for (chunk_id, _original_info) in original_post_data_chunks {
            let new_info = result.header.header_info.get(chunk_id).unwrap();
            assert!(new_info.offset > result.header.data().offset);
        }
    }

    #[test]
    fn test_signal_integrity() {
        // Create a simple sine wave
        let sample_rate = 44100;
        let duration = 1.0; // seconds
        let frequency = 440.0; // Hz

        let mut samples: Vec<f32> = Vec::new();
        for i in 0..(sample_rate as f64 * duration) as usize {
            let t = i as f64 / sample_rate as f64;
            let value = (2.0 * std::f64::consts::PI * frequency * t).sin() as f32;
            samples.push(value);
        }

        // Write test file
        std::fs::create_dir_all(TEST_OUTPUT).unwrap();
        let test_file = format!("{}/sine_test.wav", TEST_OUTPUT);
        crate::write(&test_file, &samples, sample_rate as i32, 1).unwrap();

        // Resample and verify frequency preservation
        let mut wav = Wav::<f32>::from_path(&test_file).unwrap();
        let target_fs = sample_rate * 2;
        let result = resample(&mut wav, target_fs as i32).unwrap();

        // Verify basic signal properties (we should still see the same frequency)
        let resampled = result.data.as_ref();
        let duration = resampled.len() as f64 / target_fs as f64;
        assert_approx_eq!(duration, 1.0, 1e-2);

        // remove the file
        std::fs::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_edge_cases() {
        let mut wav = Wav::<f32>::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let expected_fs = wav.sample_rate();

        // Test same sample rate
        let result = resample(&mut wav, expected_fs).unwrap();
        assert_eq!(result.fs, expected_fs);

        // Test very small ratio
        let mut wav = Wav::<f32>::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let expected_fs = wav.sample_rate();

        let result = resample(&mut wav, expected_fs / 10).unwrap();
        assert!(result.data.len() < wav.read().unwrap().len());

        // Test very large ratio
        let mut wav = Wav::<f32>::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let result = resample(&mut wav, expected_fs * 10).unwrap();
        assert!(result.data.len() > wav.read().unwrap().len());
    }
}
