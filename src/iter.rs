//! Module containing the different type of iterators for the Wav struct.
//! - The FrameIterator iterates over the frames of the Wav file
//! - The ChannelIterator iterates over the channels of the Wav file.
//! - The BlockIterator iterates over blocks of the Wav file with an optional overlap.

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
    wav: &'a mut Wav<T>,
    current_channel: usize,
    n_samples_per_channel: usize,
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
    pub fn new(wav: &'a mut Wav<T>) -> ChannelIterator<'a, T> {
        let n_samples_per_channel = wav.n_samples() as usize / wav.n_channels() as usize;
        ChannelIterator {
            wav,
            current_channel: 0,
            n_samples_per_channel: n_samples_per_channel,
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

        // Seek to the start of the channel.
        // Since we just seeked to the start, we can seek by the current channel (samples) to get to the next channel.
        match self.wav.seek_by_samples(current_channel) {
            Ok(_) => (),
            Err(_) => return None,
        }
        let mut samples: Box<[T]> = alloc_sample_buffer(self.n_samples_per_channel);

        for i in 0..self.n_samples_per_channel {
            // read a sample
            let sample: T = match self.wav.read_sample() {
                Ok(frame) => frame,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    return None;
                }
            };
            samples[i] = sample;
            // seek to the next sample in the channel, if we go beyond the data chunk, break. This is indicated by the error.
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

/// **NOTE** This iterator is experimental and may not work as expected. I would prefer to have more tests and for it to be tested in the wild with some feedback.
/// A block iterator for the Wav struct.
/// Really just a frame iterator with a block size and overlap.
#[derive(Debug)]
pub struct BlockIterator<'a, T: 'a + AudioSample>
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
    wav: &'a mut Wav<T>,
    block_size: usize,
    overlap: usize,
    current_block: usize,
    total_blocks: usize,
    final_block_size: usize,
}
impl<'a, T: 'a + AudioSample> BlockIterator<'a, T>
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
    pub fn new(wav: &'a mut Wav<T>, block_size: usize, overlap: usize) -> BlockIterator<'a, T> {
        assert!(block_size > 0, "Block size must be positive");
        assert!(overlap < block_size, "Overlap must be less than block size");
        let n_channels = wav.n_channels() as usize;
        let signal_length = wav.n_samples() as usize;
        let (total_blocks, remainder_size) =
            BlockIterator::n_blocks(signal_length, block_size, overlap, n_channels);

        BlockIterator {
            wav,
            block_size,
            overlap,
            current_block: 0,
            total_blocks,
            final_block_size: remainder_size,
        }
    }

    /// Returns the number of blocks that will be returned by the iterator and the size of the final block since it may be smaller than the block size.
    pub fn n_blocks(
        signal_length: usize,
        block_size: usize,
        overlap: usize,
        n_channels: usize,
    ) -> (usize, usize) {
        let block_size = block_size * n_channels;
        let overlap = overlap * n_channels;

        // Calculate the number of complete blocks
        let complete_blocks =
            ((signal_length - block_size) as f64 / (block_size - overlap) as f64).floor() as usize;

        // Calculate the total number of blocks
        let total_blocks = complete_blocks + 1;

        // Calculate the size of the final block
        let size_of_final_block =
            signal_length - (complete_blocks * (block_size - overlap) + overlap);
        match size_of_final_block {
            0 => (total_blocks, size_of_final_block),
            _ => (total_blocks + 1, size_of_final_block),
        }
    }
}

impl<'a, T: 'a + AudioSample> Iterator for BlockIterator<'a, T>
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
        let n_channels = self.wav.n_channels() as usize;
        let n_samples_to_read = if self.current_block != self.total_blocks - 1 {
            n_channels * self.block_size
        } else {
            self.final_block_size
        };

        // seek to the relevant position
        if self.current_block == 0 {
            match self.wav.to_data() {
                Ok(_) => (),
                Err(e) => {
                    eprintln!("Seek To Data Error: {}", e);
                    return None;
                }
            }
        } else {
            let seek_pos = (self.block_size - self.overlap) * self.wav.n_channels() as usize;
            self.wav.seek_relative(-(seek_pos as i64)).unwrap();
        }

        if self.current_block >= self.total_blocks {
            match self.wav.to_data() {
                Ok(_) => (),
                Err(e) => {
                    eprintln!("Seek To Data Error: {}", e);
                    return None;
                }
            }
            return None;
        }

        let block = match self.wav.read_samples(n_samples_to_read) {
            Ok(block) => block,
            Err(e) => {
                eprintln!("Read Error: {}", e);
                return None;
            }
        };

        // Seek back by the overlap amount
        let seek_back = (self.overlap * n_channels) as i64;
        if seek_back > 0 && self.current_block < self.total_blocks - 1 {
            match self.wav.seek_relative(-seek_back) {
                Ok(_) => (),
                Err(e) => {
                    eprintln!("Seek Error: {}", e);
                    return None;
                }
            }
        }

        self.current_block += 1;
        Some(block)
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
        let channel_iter = ChannelIterator::new(&mut wav);

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

    #[test]
    fn test_block_iterator() {
        // the test file is sampled at 16Khz and has 2 channels. It is 10 seconds long.
        let mut wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let n_channels = wav.n_channels();
        let n_samples = wav.n_samples();
        let block_iter = BlockIterator::new(&mut wav, 1024, 512);

        let (full_blocks, remainder_size) =
            BlockIterator::<i16>::n_blocks(n_samples, 1024, 512, n_channels as usize);
        let block_size = block_iter.block_size * n_channels as usize;

        let mut blocks = vec![];

        for block in block_iter {
            if block.len() == block_size {
                assert!(true)
            } else if block.len() == remainder_size {
                println!("remainder size: {}", block.len());

                assert!(true)
            } else {
                assert!(
                    false,
                    "Block size should be {} or {}, not {}",
                    block_size,
                    remainder_size,
                    block.len()
                )
            }
            blocks.push(block);
        }

        assert_eq!(
            blocks.len(),
            full_blocks,
            "Block count should be {}",
            full_blocks
        );
    }

    #[test]
    fn test_block_iterator_resets() {
        let mut wav = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();

        for _ in wav.blocks(1024, 512) {}

        let current_pos = wav.current_pos().unwrap();
        let expected_pos = wav.header().get_chunk_info(DATA.into()).unwrap().offset + 8;

        assert_eq!(
            current_pos, expected_pos as u64,
            "Current position {} should be {}",
            current_pos, expected_pos
        );
    }
}
