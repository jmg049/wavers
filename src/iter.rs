//! Module containing the different type of iterators for the Wav struct.
//! - The FrameIterator iterates over the frames of the Wav file
//! - The ChannelIterator iterates over the channels of the Wav file.
//! - The BlockIterator iterates over blocks of the Wav file with an optional overlap.

use crate::{
    core::alloc_sample_buffer, i24, AudioSample, ConvertTo, Samples, Wav,
};

/// An iterator that yields frames of audio data, where each frame contains
/// one sample from each channel at a given time point.
pub struct FrameIterator<'a, T: 'a + AudioSample>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    wav: &'a mut Wav<T>,
    max_pos: u64,
    current_pos: u64,
    buffer: Box<[T]>,     // Reusable buffer for reading chunks
    buffer_size: usize,   // Size of the buffer in frames
    buffer_offset: usize, // Current position in buffer
    buffer_filled: usize, // Number of valid frames in buffer
    n_channels: usize,    // Cache channel count to avoid repeated lookups
}

impl<'a, T: 'a + AudioSample> FrameIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    pub fn new(max_pos: u64, wav: &'a mut Wav<T>) -> FrameIterator<'a, T> {
        let n_channels = wav.n_channels() as usize;
        // Choose a reasonable buffer size (e.g., enough for ~100ms at 44.1kHz)
        let frames_per_buffer = 4410;
        let buffer_size = frames_per_buffer * n_channels;

        // Get initial position
        let current_pos = wav.current_pos().unwrap_or_else(|_| {
            wav.to_data().expect("Failed to seek to data chunk start");
            wav.current_pos().expect("Failed to get current position")
        });

        FrameIterator {
            wav,
            max_pos,
            current_pos,
            buffer: alloc_sample_buffer(buffer_size),
            buffer_size: frames_per_buffer,
            buffer_offset: 0,
            buffer_filled: 0,
            n_channels,
        }
    }

    /// Refills the internal buffer with new audio data
    fn refill_buffer(&mut self) -> bool {
        // Calculate how many frames we can read
        let remaining_bytes = self.max_pos - self.current_pos;
        let remaining_frames =
            (remaining_bytes as usize) / (std::mem::size_of::<T>() * self.n_channels);
        let frames_to_read = std::cmp::min(self.buffer_size, remaining_frames);

        if frames_to_read == 0 {
            return false;
        }

        let samples_to_read = frames_to_read * self.n_channels;

        match self.wav.read_samples(samples_to_read) {
            Ok(samples) => {
                // Copy new samples into our buffer
                self.buffer[..samples_to_read].copy_from_slice(&samples);
                self.buffer_filled = frames_to_read;
                self.buffer_offset = 0;
                self.current_pos += (samples_to_read * std::mem::size_of::<T>()) as u64;
                true
            }
            Err(e) => {
                eprintln!("Error reading samples: {}", e);
                false
            }
        }
    }

    /// Creates a frame from the current buffer position
    fn current_frame(&self) -> Option<Samples<T>> {
        if self.buffer_offset >= self.buffer_filled {
            return None;
        }

        let start = self.buffer_offset * self.n_channels;
        let end = start + self.n_channels;
        let frame = &self.buffer[start..end];

        Some(Samples::from(frame))
    }
}

impl<'a, T: 'a + AudioSample> Iterator for FrameIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Item = Samples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // If we've consumed all buffered frames, try to refill
        if self.buffer_offset >= self.buffer_filled {
            if !self.refill_buffer() {
                // Reset the reader position before returning None
                let _ = self.wav.to_data();
                return None;
            }
        }

        // Get current frame and advance buffer offset
        let frame = self.current_frame();
        self.buffer_offset += 1;
        frame
    }
}

/// An iterator that yields each channel of audio data independently.
pub struct ChannelIterator<'a, T: 'a + AudioSample>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    wav: &'a mut Wav<T>,
    current_channel: usize,
    n_channels: usize,
    n_samples_per_channel: usize,
    buffer: Box<[T]>,        // Buffer for reading interleaved data
    buffer_size: usize,      // Size in frames (not samples)
    output_buffer: Box<[T]>, // Pre-allocated buffer for deinterleaved channel data
}

impl<'a, T: 'a + AudioSample> ChannelIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    pub fn new(wav: &'a mut Wav<T>) -> Self {
        let n_channels = wav.n_channels() as usize;
        let n_samples_per_channel = wav.n_samples() / n_channels;

        // Buffer size tuned for common use cases (~100ms at 44.1kHz)
        let buffer_size = std::cmp::min(4410, n_samples_per_channel);
        let buffer = alloc_sample_buffer(buffer_size * n_channels);
        let output_buffer = alloc_sample_buffer(n_samples_per_channel);

        // Ensure we're at the start of the data chunk
        let _ = wav.to_data();

        Self {
            wav,
            current_channel: 0,
            n_channels,
            n_samples_per_channel,
            buffer,
            buffer_size,
            output_buffer,
        }
    }

    /// Read and deinterleave a chunk of audio data for the current channel
    fn process_current_channel(&mut self) -> Option<Samples<T>> {
        if self.current_channel >= self.n_channels {
            return None;
        }

        // Seek to start of data for current channel
        if let Err(_) = self.wav.to_data() {
            return None;
        }

        let mut samples_processed = 0;
        let channel_idx = self.current_channel;

        while samples_processed < self.n_samples_per_channel {
            let frames_remaining = self.n_samples_per_channel - samples_processed;
            let frames_to_read = std::cmp::min(self.buffer_size, frames_remaining);
            let samples_to_read = frames_to_read * self.n_channels;

            // Read a chunk of interleaved data
            match self
                .wav
                .read_samples_into(&mut self.buffer[..samples_to_read])
            {
                Ok(_) => {
                    // Deinterleave from buffer
                    for frame in 0..frames_to_read {
                        let src_idx = frame * self.n_channels + channel_idx;
                        self.output_buffer[samples_processed + frame] = self.buffer[src_idx];
                    }
                    samples_processed += frames_to_read;
                }
                Err(_) => break,
            }
        }

        self.current_channel += 1;
        let _ = self.wav.to_data(); // Reset position for next channel

        // Return the deinterleaved channel data
        Some(Samples::new(Box::from(
            &self.output_buffer[..self.n_samples_per_channel],
        )))
    }
}

impl<'a, T: 'a + AudioSample> Iterator for ChannelIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Item = Samples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.process_current_channel()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_channels - self.current_channel;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a + AudioSample> ExactSizeIterator for ChannelIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
}

/// (EXPERIMENTAL! - Use with caution) Iterator that yields blocks of audio data with configurable overlap.
/// Useful for spectral processing and windowing operations.
pub struct BlockIterator<'a, T: 'a + AudioSample>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    wav: &'a mut Wav<T>,
    block_size: usize,
    overlap: usize,
    current_block: usize,
    total_blocks: usize,
    n_channels: usize,
    final_block_size: usize,
}

impl<'a, T: 'a + AudioSample> BlockIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    pub fn new(wav: &'a mut Wav<T>, block_size: usize, overlap: usize) -> Self {
        assert!(block_size > 0, "Block size must be positive");
        assert!(overlap < block_size, "Overlap must be less than block size");

        let n_channels = wav.n_channels() as usize;
        let signal_length = wav.n_samples();
        let (total_blocks, final_block_size) =
            Self::calculate_blocks(signal_length, block_size, overlap, n_channels);

        BlockIterator {
            wav,
            block_size,
            overlap,
            current_block: 0,
            total_blocks,
            n_channels,
            final_block_size,
        }
    }

    fn calculate_blocks(
        signal_length: usize,
        block_size: usize,
        overlap: usize,
        n_channels: usize,
    ) -> (usize, usize) {
        let samples_per_channel = signal_length / n_channels;
        let hop_size = block_size - overlap;

        if samples_per_channel <= block_size {
            return (1, signal_length);
        }

        let n_blocks = ((samples_per_channel - overlap) as f64 / hop_size as f64).ceil() as usize;
        let last_block_start = (n_blocks - 1) * hop_size;
        let samples_in_last_block = samples_per_channel - last_block_start;

        (n_blocks, samples_in_last_block * n_channels)
    }
}

impl<'a, T: 'a + AudioSample> Iterator for BlockIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Item = Samples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_block >= self.total_blocks {
            let _ = self.wav.to_data();
            return None;
        }

        let n_samples_to_read = if self.current_block == self.total_blocks - 1 {
            self.final_block_size
        } else {
            self.n_channels * self.block_size
        };

        // Position for next read
        if self.current_block == 0 {
            if let Err(_) = self.wav.to_data() {
                return None;
            }
        } else {
            let seek_back = ((self.block_size - self.overlap) * self.n_channels) as i64;
            unsafe {
                if let Err(_) = self.wav.seek_relative(-seek_back) {
                    return None;
                }
            }
        }

        match self.wav.read_samples(n_samples_to_read) {
            Ok(block) => {
                if self.current_block < self.total_blocks - 1 {
                    let overlap_seek = (self.overlap * self.n_channels) as i64;
                    unsafe {
                        let _ = self.wav.seek_relative(-overlap_seek);
                    }
                }
                self.current_block += 1;
                Some(block)
            }
            Err(_) => {
                let _ = self.wav.to_data();
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_blocks - self.current_block;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a + AudioSample> ExactSizeIterator for BlockIterator<'a, T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
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
            BlockIterator::<i16>::calculate_blocks(n_samples, 1024, 512, n_channels as usize);
        let block_size = block_iter.block_size * n_channels as usize;

        let mut blocks = vec![];

        for block in block_iter {
            println!("block size: {}", block.len());
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
        println!("blocks: {:?}", blocks.len());
        println!("full_blocks: {:?}", full_blocks);
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
