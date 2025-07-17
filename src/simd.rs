//! SIMD-optimized audio data conversion functions.
//!
//! This module provides high-performance interleaving and de-interleaving
//! operations for audio data using SIMD instructions when available.

use audio_samples::I24;

/// Trait for SIMD-optimized interleaving/de-interleaving operations.
pub trait SimdAudioOps: Copy + Default {
    /// De-interleave stereo data from [L0, R0, L1, R1, ...] to [L0, L1, ...], [R0, R1, ...]
    fn deinterleave_stereo(interleaved: &[Self]) -> Vec<Self>;

    /// Interleave stereo data from [L0, L1, ...], [R0, R1, ...] to [L0, R0, L1, R1, ...]
    fn interleave_stereo(left: &[Self], right: &[Self]) -> Vec<Self>;

    /// De-interleave multi-channel data
    fn deinterleave_multi(interleaved: &[Self], channels: usize) -> Vec<Self>;

    /// Interleave multi-channel data
    fn interleave_multi(channel_data: &[&[Self]], channels: usize) -> Vec<Self>;
}

/// Optimized implementation for f32 using manual vectorization
impl SimdAudioOps for f32 {
    #[inline]
    fn deinterleave_stereo(interleaved: &[f32]) -> Vec<f32> {
        let samples_per_channel = interleaved.len() / 2;
        let mut deinterleaved = Vec::with_capacity(interleaved.len());

        // Reserve space for both channels
        deinterleaved.resize(interleaved.len(), 0.0);
        let (left_slice, right_slice) = deinterleaved.split_at_mut(samples_per_channel);

        // Process 8 stereo pairs at a time for better cache utilization
        let chunks = interleaved.chunks_exact(16);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base_idx = chunk_idx * 8;

            // Manually unroll for better performance
            left_slice[base_idx] = chunk[0];
            right_slice[base_idx] = chunk[1];
            left_slice[base_idx + 1] = chunk[2];
            right_slice[base_idx + 1] = chunk[3];
            left_slice[base_idx + 2] = chunk[4];
            right_slice[base_idx + 2] = chunk[5];
            left_slice[base_idx + 3] = chunk[6];
            right_slice[base_idx + 3] = chunk[7];
            left_slice[base_idx + 4] = chunk[8];
            right_slice[base_idx + 4] = chunk[9];
            left_slice[base_idx + 5] = chunk[10];
            right_slice[base_idx + 5] = chunk[11];
            left_slice[base_idx + 6] = chunk[12];
            right_slice[base_idx + 6] = chunk[13];
            left_slice[base_idx + 7] = chunk[14];
            right_slice[base_idx + 7] = chunk[15];
        }

        // Handle remainder
        let processed_samples = (interleaved.len() / 16) * 8;
        for (i, chunk) in remainder.chunks_exact(2).enumerate() {
            left_slice[processed_samples + i] = chunk[0];
            right_slice[processed_samples + i] = chunk[1];
        }

        deinterleaved
    }

    #[inline]
    fn interleave_stereo(left: &[f32], right: &[f32]) -> Vec<f32> {
        debug_assert_eq!(left.len(), right.len());
        let mut interleaved = Vec::with_capacity(left.len() * 2);

        // Process 8 samples at a time
        let chunks = left.chunks_exact(8).zip(right.chunks_exact(8));
        let left_remainder = left.chunks_exact(8).remainder();
        let right_remainder = right.chunks_exact(8).remainder();

        for (left_chunk, right_chunk) in chunks {
            // Manually unroll for better performance
            interleaved.push(left_chunk[0]);
            interleaved.push(right_chunk[0]);
            interleaved.push(left_chunk[1]);
            interleaved.push(right_chunk[1]);
            interleaved.push(left_chunk[2]);
            interleaved.push(right_chunk[2]);
            interleaved.push(left_chunk[3]);
            interleaved.push(right_chunk[3]);
            interleaved.push(left_chunk[4]);
            interleaved.push(right_chunk[4]);
            interleaved.push(left_chunk[5]);
            interleaved.push(right_chunk[5]);
            interleaved.push(left_chunk[6]);
            interleaved.push(right_chunk[6]);
            interleaved.push(left_chunk[7]);
            interleaved.push(right_chunk[7]);
        }

        // Handle remainder
        for (l, r) in left_remainder.iter().zip(right_remainder.iter()) {
            interleaved.push(*l);
            interleaved.push(*r);
        }

        interleaved
    }

    #[inline]
    fn deinterleave_multi(interleaved: &[f32], channels: usize) -> Vec<f32> {
        if channels == 2 {
            return Self::deinterleave_stereo(interleaved);
        }
        generic_deinterleave(interleaved, channels)
    }

    #[inline]
    fn interleave_multi(channel_data: &[&[f32]], channels: usize) -> Vec<f32> {
        if channels == 2 && channel_data.len() == 2 {
            return Self::interleave_stereo(channel_data[0], channel_data[1]);
        }
        generic_interleave(channel_data, channels)
    }
}

/// Optimized implementation for f64
impl SimdAudioOps for f64 {
    #[inline]
    fn deinterleave_stereo(interleaved: &[f64]) -> Vec<f64> {
        let samples_per_channel = interleaved.len() / 2;
        let mut deinterleaved = Vec::with_capacity(interleaved.len());
        deinterleaved.resize(interleaved.len(), 0.0);

        let (left_slice, right_slice) = deinterleaved.split_at_mut(samples_per_channel);

        // Process 4 stereo pairs at a time
        let chunks = interleaved.chunks_exact(8);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base_idx = chunk_idx * 4;

            left_slice[base_idx] = chunk[0];
            right_slice[base_idx] = chunk[1];
            left_slice[base_idx + 1] = chunk[2];
            right_slice[base_idx + 1] = chunk[3];
            left_slice[base_idx + 2] = chunk[4];
            right_slice[base_idx + 2] = chunk[5];
            left_slice[base_idx + 3] = chunk[6];
            right_slice[base_idx + 3] = chunk[7];
        }

        // Handle remainder
        let processed_samples = (interleaved.len() / 8) * 4;
        for (i, chunk) in remainder.chunks_exact(2).enumerate() {
            left_slice[processed_samples + i] = chunk[0];
            right_slice[processed_samples + i] = chunk[1];
        }

        deinterleaved
    }

    #[inline]
    fn interleave_stereo(left: &[f64], right: &[f64]) -> Vec<f64> {
        debug_assert_eq!(left.len(), right.len());
        let mut interleaved = Vec::with_capacity(left.len() * 2);

        let chunks = left.chunks_exact(4).zip(right.chunks_exact(4));
        let left_remainder = left.chunks_exact(4).remainder();
        let right_remainder = right.chunks_exact(4).remainder();

        for (left_chunk, right_chunk) in chunks {
            interleaved.push(left_chunk[0]);
            interleaved.push(right_chunk[0]);
            interleaved.push(left_chunk[1]);
            interleaved.push(right_chunk[1]);
            interleaved.push(left_chunk[2]);
            interleaved.push(right_chunk[2]);
            interleaved.push(left_chunk[3]);
            interleaved.push(right_chunk[3]);
        }

        for (l, r) in left_remainder.iter().zip(right_remainder.iter()) {
            interleaved.push(*l);
            interleaved.push(*r);
        }

        interleaved
    }

    #[inline]
    fn deinterleave_multi(interleaved: &[f64], channels: usize) -> Vec<f64> {
        if channels == 2 {
            return Self::deinterleave_stereo(interleaved);
        }
        generic_deinterleave(interleaved, channels)
    }

    #[inline]
    fn interleave_multi(channel_data: &[&[f64]], channels: usize) -> Vec<f64> {
        if channels == 2 && channel_data.len() == 2 {
            return Self::interleave_stereo(channel_data[0], channel_data[1]);
        }
        generic_interleave(channel_data, channels)
    }
}

/// Optimized implementation for i16
impl SimdAudioOps for i16 {
    #[inline]
    fn deinterleave_stereo(interleaved: &[i16]) -> Vec<i16> {
        let samples_per_channel = interleaved.len() / 2;
        let mut deinterleaved = Vec::with_capacity(interleaved.len());
        deinterleaved.resize(interleaved.len(), 0);

        let (left_slice, right_slice) = deinterleaved.split_at_mut(samples_per_channel);

        // Process 16 stereo pairs at a time
        let chunks = interleaved.chunks_exact(32);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base_idx = chunk_idx * 16;

            // Unroll loop for better performance
            for i in 0..16 {
                left_slice[base_idx + i] = chunk[i * 2];
                right_slice[base_idx + i] = chunk[i * 2 + 1];
            }
        }

        // Handle remainder
        let processed_samples = (interleaved.len() / 32) * 16;
        for (i, chunk) in remainder.chunks_exact(2).enumerate() {
            left_slice[processed_samples + i] = chunk[0];
            right_slice[processed_samples + i] = chunk[1];
        }

        deinterleaved
    }

    #[inline]
    fn interleave_stereo(left: &[i16], right: &[i16]) -> Vec<i16> {
        debug_assert_eq!(left.len(), right.len());
        let mut interleaved = Vec::with_capacity(left.len() * 2);

        let chunks = left.chunks_exact(16).zip(right.chunks_exact(16));
        let left_remainder = left.chunks_exact(16).remainder();
        let right_remainder = right.chunks_exact(16).remainder();

        for (left_chunk, right_chunk) in chunks {
            for i in 0..16 {
                interleaved.push(left_chunk[i]);
                interleaved.push(right_chunk[i]);
            }
        }

        for (l, r) in left_remainder.iter().zip(right_remainder.iter()) {
            interleaved.push(*l);
            interleaved.push(*r);
        }

        interleaved
    }

    #[inline]
    fn deinterleave_multi(interleaved: &[i16], channels: usize) -> Vec<i16> {
        if channels == 2 {
            return Self::deinterleave_stereo(interleaved);
        }
        generic_deinterleave(interleaved, channels)
    }

    #[inline]
    fn interleave_multi(channel_data: &[&[i16]], channels: usize) -> Vec<i16> {
        if channels == 2 && channel_data.len() == 2 {
            return Self::interleave_stereo(channel_data[0], channel_data[1]);
        }
        generic_interleave(channel_data, channels)
    }
}

/// Optimized implementation for i32
impl SimdAudioOps for i32 {
    #[inline]
    fn deinterleave_stereo(interleaved: &[i32]) -> Vec<i32> {
        let samples_per_channel = interleaved.len() / 2;
        let mut deinterleaved = Vec::with_capacity(interleaved.len());
        deinterleaved.resize(interleaved.len(), 0);

        let (left_slice, right_slice) = deinterleaved.split_at_mut(samples_per_channel);

        // Process 8 stereo pairs at a time
        let chunks = interleaved.chunks_exact(16);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base_idx = chunk_idx * 8;

            for i in 0..8 {
                left_slice[base_idx + i] = chunk[i * 2];
                right_slice[base_idx + i] = chunk[i * 2 + 1];
            }
        }

        // Handle remainder
        let processed_samples = (interleaved.len() / 16) * 8;
        for (i, chunk) in remainder.chunks_exact(2).enumerate() {
            left_slice[processed_samples + i] = chunk[0];
            right_slice[processed_samples + i] = chunk[1];
        }

        deinterleaved
    }

    #[inline]
    fn interleave_stereo(left: &[i32], right: &[i32]) -> Vec<i32> {
        debug_assert_eq!(left.len(), right.len());
        let mut interleaved = Vec::with_capacity(left.len() * 2);

        let chunks = left.chunks_exact(8).zip(right.chunks_exact(8));
        let left_remainder = left.chunks_exact(8).remainder();
        let right_remainder = right.chunks_exact(8).remainder();

        for (left_chunk, right_chunk) in chunks {
            for i in 0..8 {
                interleaved.push(left_chunk[i]);
                interleaved.push(right_chunk[i]);
            }
        }

        for (l, r) in left_remainder.iter().zip(right_remainder.iter()) {
            interleaved.push(*l);
            interleaved.push(*r);
        }

        interleaved
    }

    #[inline]
    fn deinterleave_multi(interleaved: &[i32], channels: usize) -> Vec<i32> {
        if channels == 2 {
            return Self::deinterleave_stereo(interleaved);
        }
        generic_deinterleave(interleaved, channels)
    }

    #[inline]
    fn interleave_multi(channel_data: &[&[i32]], channels: usize) -> Vec<i32> {
        if channels == 2 && channel_data.len() == 2 {
            return Self::interleave_stereo(channel_data[0], channel_data[1]);
        }
        generic_interleave(channel_data, channels)
    }
}

/// I24 implementation (fallback to generic since it's not a standard SIMD type)
impl SimdAudioOps for I24 {
    #[inline]
    fn deinterleave_stereo(interleaved: &[I24]) -> Vec<I24> {
        generic_deinterleave(interleaved, 2)
    }

    #[inline]
    fn interleave_stereo(left: &[I24], right: &[I24]) -> Vec<I24> {
        debug_assert_eq!(left.len(), right.len());
        let mut interleaved = Vec::with_capacity(left.len() * 2);

        for (l, r) in left.iter().zip(right.iter()) {
            interleaved.push(*l);
            interleaved.push(*r);
        }

        interleaved
    }

    #[inline]
    fn deinterleave_multi(interleaved: &[I24], channels: usize) -> Vec<I24> {
        generic_deinterleave(interleaved, channels)
    }

    #[inline]
    fn interleave_multi(channel_data: &[&[I24]], channels: usize) -> Vec<I24> {
        generic_interleave(channel_data, channels)
    }
}

/// Generic fallback implementations for non-optimized cases
#[inline]
fn generic_deinterleave<T: Copy>(interleaved: &[T], channels: usize) -> Vec<T> {
    let samples_per_channel = interleaved.len() / channels;
    let mut deinterleaved = Vec::with_capacity(interleaved.len());

    for channel_idx in 0..channels {
        for sample_idx in 0..samples_per_channel {
            deinterleaved.push(interleaved[sample_idx * channels + channel_idx]);
        }
    }

    deinterleaved
}

#[inline]
fn generic_interleave<T: Copy>(channel_data: &[&[T]], channels: usize) -> Vec<T> {
    if channel_data.is_empty() {
        return Vec::new();
    }

    let samples_per_channel = channel_data[0].len();
    let mut interleaved = Vec::with_capacity(samples_per_channel * channels);

    for sample_idx in 0..samples_per_channel {
        for channel_idx in 0..channels {
            interleaved.push(channel_data[channel_idx][sample_idx]);
        }
    }

    interleaved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_stereo_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let deinterleaved = f32::deinterleave_stereo(&original);
        let expected = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        assert_eq!(deinterleaved, expected);

        let left = &deinterleaved[0..4];
        let right = &deinterleaved[4..8];
        let reinterleaved = f32::interleave_stereo(left, right);
        assert_eq!(reinterleaved, original);
    }

    #[test]
    fn test_i16_stereo_roundtrip() {
        let original = vec![100i16, 200, 300, 400, 500, 600, 700, 800];
        let deinterleaved = i16::deinterleave_stereo(&original);
        let expected = vec![100, 300, 500, 700, 200, 400, 600, 800];
        assert_eq!(deinterleaved, expected);

        let left = &deinterleaved[0..4];
        let right = &deinterleaved[4..8];
        let reinterleaved = i16::interleave_stereo(left, right);
        assert_eq!(reinterleaved, original);
    }

    #[test]
    fn test_multi_channel_roundtrip() {
        let original = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]; // 3 channels, 4 samples each
        let deinterleaved = f32::deinterleave_multi(&original, 3);
        let expected = vec![
            1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0,
        ];
        assert_eq!(deinterleaved, expected);

        let channel_data = vec![
            &deinterleaved[0..4],
            &deinterleaved[4..8],
            &deinterleaved[8..12],
        ];
        let reinterleaved = f32::interleave_multi(&channel_data, 3);
        assert_eq!(reinterleaved, original);
    }
}
