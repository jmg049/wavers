use crate::{AudioSample, WaversError, WaversResult};

/// Converts audio data from planar format (samples grouped by channel) to interleaved format.
///
/// In planar format, samples are grouped by channel:
/// [ch1_s1, ch1_s2, ch1_s3, ch2_s1, ch2_s2, ch2_s3]
///
/// After conversion, samples are interleaved:
/// [ch1_s1, ch2_s1, ch1_s2, ch2_s2, ch1_s3, ch2_s3]
///
/// # Arguments
/// * `planar_data` - Audio data in planar format (mutable slice)
/// * `channels` - Number of channels in the audio data
///
/// # Panics
/// Will not panic, but returns early if data length is not divisible by channel count
///
/// # Example
/// ```
/// let mut data = vec![1, 2, 3, 4, 5, 6]; // Two channels: [1,2,3] and [4,5,6]
/// planar_to_interleaved(&mut data, 2);
/// assert_eq!(data, vec![1, 4, 2, 5, 3, 6]);
/// ```
pub fn planar_to_interleaved<T: AudioSample + Copy, A: AsMut<[T]>>(
    mut planar_data: A,
    channels: usize,
) -> WaversResult<()> {
    let data = planar_data.as_mut();
    let total_samples = data.len();
    if total_samples % channels != 0 {
        return Err(WaversError::PlanarToInterleavedError(
            format!("Data length is not divisible by channel count: Total Samples = {}, number of channels = {}", total_samples, channels),
        )); // Invalid input
    }
    let samples_per_channel = total_samples / channels;

    let temp = data.to_vec();

    for (i, chunk) in temp.chunks(samples_per_channel).enumerate() {
        for (j, &sample) in chunk.iter().enumerate() {
            data[j * channels + i] = sample;
        }
    }
    Ok(())
}

/// Converts audio data from interleaved format to planar format (samples grouped by channel).
///
/// In interleaved format, samples alternate between channels:
/// [ch1_s1, ch2_s1, ch1_s2, ch2_s2, ch1_s3, ch2_s3]
///
/// After conversion, samples are grouped by channel:
/// [ch1_s1, ch1_s2, ch1_s3, ch2_s1, ch2_s2, ch2_s3]
///
/// # Arguments
/// * `interleaved_data` - Audio data in interleaved format (mutable slice)
/// * `channels` - Number of channels in the audio data
///
/// # Panics
/// Will not panic, but returns early if data length is not divisible by channel count
///
/// # Example
/// ```
/// let mut data = vec![1, 4, 2, 5, 3, 6]; // Interleaved 2 channels
/// interleaved_to_planar(&mut data, 2);
/// assert_eq!(data, vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn interleaved_to_planar<T: AudioSample + Copy, A: AsMut<[T]>>(
    mut interleaved_data: A,
    channels: usize,
) -> WaversResult<()> {
    let data = interleaved_data.as_mut();
    let total_samples = data.len();
    if total_samples % channels != 0 {
        return Err(WaversError::InterleavedToPlanarError(
            format!("Data length is not divisible by channel count: Total Samples = {}, number of channels = {}", total_samples, channels), 
        )); // Invalid input
    }
    let samples_per_channel = total_samples / channels;

    let temp = data.to_vec();

    for i in 0..samples_per_channel {
        for c in 0..channels {
            data[c * samples_per_channel + i] = temp[i * channels + c];
        }
    }

    Ok(())
}
#[cfg(feature = "simd")]
use std::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

#[cfg(feature = "simd")]
/// Not guaranteed to work! Use with caution.
pub fn deinterleave_simd<T: SimdElement + Default + AudioSample, const LANES: usize>(
    data: &mut [T],
    channels: usize,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    let total_samples = data.len();
    if total_samples % channels != 0 {
        return; // Invalid input
    }
    let samples_per_channel = total_samples / channels;
    let full_chunks = samples_per_channel / LANES;
    let remainder = samples_per_channel % LANES;

    let temp = data.to_vec();

    for c in 0..channels {
        for chunk in 0..full_chunks {
            let mut simd_chunk = Simd::<T, LANES>::default();
            for i in 0..LANES {
                simd_chunk[i] = temp[chunk * LANES * channels + i * channels + c];
            }
            simd_chunk.copy_to_slice(&mut data[c * samples_per_channel + chunk * LANES..]);
        }
    }

    // Handle remaining elements
    if remainder > 0 {
        let start = full_chunks * LANES * channels;
        for c in 0..channels {
            for i in 0..remainder {
                data[c * samples_per_channel + full_chunks * LANES + i] =
                    temp[start + i * channels + c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConvertTo;
    use approx_eq::assert_approx_eq;

    // Helper function to create test data
    fn create_test_data<T: AudioSample>(len: usize) -> Vec<T>
    where
        i32: ConvertTo<T>,
    {
        let mut out_vec: Vec<T> = Vec::with_capacity(len);
        for i in 0..len {
            out_vec.push((i as i32).convert_to());
        }
        out_vec
    }

    fn approx_eq(x: f32, y: f32) -> bool {
        (x - y).abs() < f32::EPSILON
    }

    fn is_interleaved<T: AudioSample + PartialEq + std::fmt::Debug>(
        data: &[T],
        channels: usize,
    ) -> bool {
        if data.len() % channels != 0 {
            return false;
        }
        let samples_per_channel = data.len() / channels;
        let result = (0..samples_per_channel).all(|i| {
            (0..channels).all(|c| {
                let condition = approx_eq(
                    data[i * channels + c].convert_to(),
                    data[c * samples_per_channel + i].convert_to(),
                );
                if !condition {
                    println!(
                        "Interleave mismatch at i={}, c={}: {:?} != {:?}",
                        i,
                        c,
                        data[i * channels + c],
                        data[c * samples_per_channel + i]
                    );
                }
                condition
            })
        });
        println!("Is interleaved: {}", result);
        result
    }
    fn is_deinterleaved<T: AudioSample + PartialEq + std::fmt::Debug>(
        data: &[T],
        channels: usize,
    ) -> bool {
        if data.len() % channels != 0 {
            return false;
        }
        let samples_per_channel = data.len() / channels;
        let result = (0..channels).all(|c| {
            (0..samples_per_channel).all(|i| {
                let condition = approx_eq(
                    data[c * samples_per_channel + i].convert_to(),
                    data[i * channels + c].convert_to(),
                );
                if !condition {
                    println!(
                        "Deinterleave mismatch at c={}, i={}: {:?} != {:?}",
                        c,
                        i,
                        data[c * samples_per_channel + i],
                        data[i * channels + c]
                    );
                }
                condition
            })
        });
        println!("Is deinterleaved: {}", result);
        result
    }
    macro_rules! test_interleave_deinterleave {
        ($name:ident, $type:ty) => {
            mod $name {
                use super::*;

                #[test]
                fn test_interleave_in_place() {
                    let mut data: Vec<$type> = create_test_data(16);
                    let original = data.clone();
                    println!("Original data: {:?}", original);
                    planar_to_interleaved(&mut data, 2).unwrap();
                    println!("Interleaved data: {:?}", data);
                    assert!(is_interleaved(&data, 2), "Data is not properly interleaved");
                    assert_approx_eq!(data[0].convert_to(), original[0].convert_to());
                    assert_approx_eq!(data[1].convert_to(), original[8].convert_to());
                }

                #[test]
                fn test_deinterleave_in_place() {
                    let mut data: Vec<$type> = create_test_data(16);
                    planar_to_interleaved(&mut data, 2).unwrap();
                    let interleaved = data.clone();
                    println!("Interleaved data: {:?}", interleaved);
                    interleaved_to_planar(&mut data, 2).unwrap();
                    println!("Deinterleaved data: {:?}", data);
                    assert!(
                        is_deinterleaved(&data, 2),
                        "Data is not properly deinterleaved"
                    );
                    assert_approx_eq!(data[0].convert_to(), interleaved[0].convert_to());
                    assert_approx_eq!(data[8].convert_to(), interleaved[1].convert_to());
                }

                #[test]
                fn test_interleave_deinterleave_three_channels() {
                    let original: Vec<$type> = create_test_data(15);
                    let mut data = original.clone();
                    println!("Original data: {:?}", original);
                    planar_to_interleaved(&mut data, 3).unwrap();
                    println!("Interleaved data: {:?}", data);
                    assert!(
                        is_interleaved(&data, 3),
                        "Data is not properly interleaved for 3 channels"
                    );
                    interleaved_to_planar(&mut data, 3).unwrap();
                    println!("Deinterleaved data: {:?}", data);
                    assert_eq!(
                        data, original,
                        "Roundtrip with 3 channels did not result in original data"
                    );
                }
            }
        };
    }
    test_interleave_deinterleave!(i16_tests, i16);
    test_interleave_deinterleave!(i32_tests, i32);
    test_interleave_deinterleave!(f32_tests, f32);
    test_interleave_deinterleave!(f64_tests, f64);

    #[cfg(feature = "simd")]
    mod simd_tests {
        use super::*;

        macro_rules! test_simd_deinterleave {
            ($name:ident, $type:ty, $lanes:expr) => {
                mod $name {
                    use super::*;
                    #[test]
                    fn test_deinterleave_in_place_simd() {
                        let mut data: Vec<$type> = create_test_data(16);
                        planar_to_interleaved(&mut data, 2);
                        let interleaved = data.clone();
                        println!("Interleaved data: {:?}", interleaved);
                        deinterleave_simd::<$type, $lanes>(&mut data, 2);
                        println!("Deinterleaved data: {:?}", data);
                        assert!(
                            is_deinterleaved(&data, 2),
                            "Data is not properly deinterleaved"
                        );
                        assert_approx_eq!(data[0].convert_to(), interleaved[0].convert_to());
                        assert_approx_eq!(data[8].convert_to(), interleaved[1].convert_to());
                    }

                    #[test]
                    fn test_interleave_deinterleave_three_channels() {
                        let original: Vec<$type> = create_test_data(15);
                        let mut data = original.clone();
                        println!("Original data: {:?}", original);
                        planar_to_interleaved(&mut data, 3);
                        println!("Interleaved data: {:?}", data);
                        assert!(
                            is_interleaved(&data, 3),
                            "Data is not properly interleaved for 3 channels"
                        );
                        deinterleave_simd::<$type, $lanes>(&mut data, 3);
                        println!("Deinterleaved data: {:?}", data);
                        assert_eq!(
                            data, original,
                            "Roundtrip with 3 channels did not result in original data"
                        );
                    }

                    #[test]
                    fn test_simd_deinterleave_with_remainder() {
                        let mut data: Vec<$type> = create_test_data(18); // Not divisible by 4 * 2
                        planar_to_interleaved(&mut data, 2);
                        deinterleave_simd::<$type, $lanes>(&mut data, 2);
                        assert!(is_deinterleaved(&data, 2));
                    }
                }
            };
        }

        test_simd_deinterleave!(test_simd_deinterleave_i16, i16, 8);
        test_simd_deinterleave!(test_simd_deinterleave_i32, i32, 4);
        test_simd_deinterleave!(test_simd_deinterleave_f32, f32, 4);
        test_simd_deinterleave!(test_simd_deinterleave_f64, f64, 2);
    }
}
