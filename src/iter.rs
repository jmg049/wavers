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

#[cfg(test)]
mod iter_tests {
    use crate::DATA;

    use super::*;

    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";

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
