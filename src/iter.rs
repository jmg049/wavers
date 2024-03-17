//! Module containing the two different type of iterators for the Wav struct.
//! The FrameIterator iterates over the frames of the Wav file, while the ChannelIterator iterates over the channels of the Wav file.
//! Both iterators are implemented for the Wav struct and require AudioSample trait.
use crate::{core::alloc_sample_buffer, i24, AudioSample, ConvertSlice, ConvertTo, Samples, Wav};

/// A frame iterator for the Wav struct.
/// WaveRs defines a frame as a collection of samples, where each sample is a single value from a single channel.
/// So one frame contains n_channel samples and there are n_samples frames in the Wav file.
/// The FrameIterator takes a max_pos value which is used to limit where in the file the iterator should stop.
/// This should only be used via the ``frames`` function on the Wav struct.
///
/// Note: This iterator *should* reset the Wav struct to the beginning of the data chunk when it is done iterating.
pub struct FrameIterator<'a, T: 'a + AudioSample>
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
{
    max_pos: u64,
    wav: &'a mut Wav<T>,
}

impl<'a, T: 'a + AudioSample> FrameIterator<'a, T>
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
{
    pub fn new(max_pos: u64, wav: &'a mut Wav<T>) -> FrameIterator<'a, T> {
        FrameIterator { max_pos, wav }
    }
}

impl<'a, T: 'a + AudioSample> Iterator for FrameIterator<'a, T>
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

/// A channel iterator for the Wav struct.
/// The ChannelItertor returns an iterator over the channels of the Wav file.
/// The ChannelIterator takes a max_pos value which is used to limit where in the file the iterator should stop.
/// This should only be used via the ``channels`` function on the Wav struct.
///
/// Note: This iterator *should* reset the Wav struct to the beginning of the data chunk when it is done iterating.
pub struct ChannelIterator<'a, T: 'a + AudioSample>
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
{
    max_pos: u64,
    wav: &'a mut Wav<T>,
    current_channel: usize,
}

impl<'a, T: 'a + AudioSample> ChannelIterator<'a, T>
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
{
    pub fn new(max_pos: u64, wav: &'a mut Wav<T>) -> ChannelIterator<'a, T> {
        ChannelIterator {
            max_pos,
            wav,
            current_channel: 0,
        }
    }
}

impl<'a, T: 'a + AudioSample> Iterator for ChannelIterator<'a, T>
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

        let n_channels = self.wav.n_channels() as usize;

        if self.current_channel >= n_channels {
            return None;
        }

        let total_n_samples = self.wav.n_samples() as usize;
        let samples_per_channel = total_n_samples / n_channels;

        let mut samples: Box<[T]> = alloc_sample_buffer(samples_per_channel);

        match self.wav.advance_pos(self.current_channel as u64) {
            Ok(_) => (),
            Err(_) => return None,
        }

        for i in 0..samples_per_channel {
            let channel_samples: T = match self.wav.read_sample() {
                Ok(frame) => frame,
                Err(_) => return None,
            };
            match self.wav.advance_pos(n_channels as u64 - 1) {
                Ok(_) => (),
                Err(_) => return None,
            }
            samples[i] = channel_samples;
        }

        self.current_channel += 1;

        match self.wav.to_data() {
            Ok(_) => (),
            Err(_) => return None,
        }

        Some(Samples::from(samples))
    }
}

/// A window iterator for a wav file.
/// Iterates over the wav file in either overlapping or non-overlapping windows.
/// This should be accessed by either the ``Wav::windows`` function or ``Wav::windows_overlapping`` functon.
pub struct WindowIterator<'a, T: 'a + AudioSample>
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
{
    max_pos: u64,
    wav: &'a mut Wav<T>,
    window_size: u64,
    overlap: u64,
    offsets: Vec<u64>,
    current_offset: usize,
    remaining_window_size: u64,
}

impl<'a, T: 'a + T> WindowIterator<'a, T> {
    pub fn new(
        max_pos: u64,
        wav: &'a mut Wav<T>,
        window_size: u64,
        overlap: u64,
    ) -> WaversResult<Self> {
        match window_size > wav.n_samples() {
            false => (),
            true => return Err(WaversError::InvalidWindowSize(wav.n_samples, window_size)),
        }

        match overlap > window_size {
            false => (),
            true => return Err(WaversError::InvalidWindowOverlap(window_size, overlap)),
        }

        let remaining_window_size = 0;

        Ok(Self {
            max_pos,
            wav,
            window_size,
            overlap,
            offsets: calc_offsets(wav.n_samples(), window_size, overlap),
            current_offset: 0,
            remaining_window_size,
        })
    }

    pub(crate) fn new_no_overlap(
        max_pos: u64,
        wav: &'a mut Wav<T>,
        window_size: u64,
    ) -> WaversResult<Self> {
        Self::new(max_pos, wav, window_size, 0)
    }

    // n = (k - r) * m + r
    // where    n = number of samples
    //          k = window length
    //          m = number of windows
    //          r = size of overlap
    //
    // Rearranging
    // m = (n - r) / (k - r)
    #[inline(always)]
    fn calc_n_overlapping_windows(n: u64, k: u64, r: u64) -> u64 {
        (n - r) / (k - r)
    }

    fn calc_offsets(n: u64, k: u64, r: u64) -> Vec<u64> {
        let total_offsets = WindowIterator::calc_n_overlapping_windows(n, k, r);
        let mut offsets = Vec::with_capacity(total_offsets);
        for i in 0..total_offsets {
            offsets.push(i * self.window_size - self.overlap);
        }
        offsets
    }
}

impl<'a, T: 'a + AudioSample> Iterator for WindowIterator<'a, T> {
    type Item = Samples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let current_pos: u64 = match self.wav.current_pos() {
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

        if current_pos + (window_size * std::mem::size_of::<T>()) > self.max_pos {
            match self.wav.to_data() {
                Ok(_) => (),
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
            return None;
        }

        let current_offset = self.current_offset;
        let offset_pos = self.offsets[current_offset];
        self.wav.to_pos_within_data(offset_pos)?;
        let samples: Samples<T> = match wav.read_samples(self.window_size) {
            Ok(s) => s,
            Err(e) => return None,
        };

        Some(samples)
    }
}

#[cfg(test)]
mod iter_tests {
    use crate::DATA;

    use super::*;

    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";

    #[test]
    fn test_calc_n_overlapping_windows() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let n_windows = WindowIterator::calc_n_overlapping_windows(data.len() as u64, 2, 0);
        assert_eq!(
            n_windows,
            4,
            "Full data of length 8 should have 4 non-overlapping window of length 2. Expected {} got {}", 4, n_windows
        );

        let n_windows = WindowIterator::calc_n_overlapping_windows(data.len() as u64, 2, 1);
        assert_eq!(
            n_windows, 7,
            "There should be 4 non-overlappig windows + 3 overlapping windows, expected {} got {}",
            7, n_windows
        );
    }

    #[test]
    fn test_frame_iterator() {
        let mut wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let max_pos = wav.max_data_pos();
        let frame_iter = FrameIterator::new(max_pos, &mut wav);

        for frame in frame_iter {
            assert_eq!(frame.len(), 2, "Frame length should be 2");
        }

        let mut wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let max_pos = wav.max_data_pos();
        let frame_iter = FrameIterator::new(max_pos, &mut wav);

        let mut shadow_wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let data: Samples<i16> = shadow_wav.read().unwrap();
        let mut idx = 0;
        for frame in frame_iter {
            assert_eq!(
                frame[0], data[idx],
                "Frame[0] should be equal to data at index {}",
                idx
            );
            assert_eq!(
                frame[1],
                data[idx + 1],
                "Frame[1] should be equal to data at index {}",
                idx + 1
            );
            idx += 2;
        }
    }

    #[test]
    fn test_frame_iterator_resets() {
        let mut wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();

        for _ in wav.frames() {}

        let current_pos = wav.current_pos().unwrap();
        let expected_pos = wav.header().get_chunk_info(DATA.into()).unwrap().offset + 8;

        assert_eq!(
            current_pos, expected_pos as u64,
            "Current position {} should be {}",
            current_pos, expected_pos
        );
    }

    #[test]
    fn test_channel_iterator() {
        let mut wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let max_pos = wav.max_data_pos();
        let channel_iter = ChannelIterator::new(max_pos, &mut wav);

        let mut shadow_wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let data: Samples<i16> = shadow_wav.read().unwrap();

        let mut curr_channel = 0;
        for channel in channel_iter {
            let mut idx = 0;
            for sample in channel.iter() {
                assert_eq!(
                    *sample,
                    data[curr_channel + idx],
                    "Sample should be equal to data at index {}",
                    curr_channel + idx
                );
                idx += 2;
            }
            curr_channel += 1;
        }
    }

    #[test]
    fn test_channel_iterator_resets() {
        let mut wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();

        for _ in wav.channels() {}

        let current_pos = wav.current_pos().unwrap();
        let expected_pos = wav.header().get_chunk_info(DATA.into()).unwrap().offset + 8;

        assert_eq!(
            current_pos, expected_pos as u64,
            "Current position {} should be {}",
            current_pos, expected_pos
        );
    }
}
