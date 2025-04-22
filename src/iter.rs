//! Iterators over WAV data: frames, channels, and blocks.
//!
//! This module defines iterator types for structured access to WAV file content,
//! such as frame-wise, channel-wise, or overlapping block access.
//!
//! Each iterator is built on top of the `Wav` struct and provides a convenient,
//! allocation-safe way to walk through audio samples. These iterators also account
//! for interleaved data layout and allow consistent zero-padding or tail-discarding
//! strategies when data is shorter than requested block sizes.
//!
//! All iterators require the target sample type `T` to implement [`AudioSample`],
//! and conversions are handled internally via the `audio_sample` crate.
//!
//! # Provided iterators
//!
//! - [`FrameIterator`] – yields one multi-channel frame at a time
//! - [`ChannelIterator`] – yields one channel's samples in sequence
//! - [`BlockIterator`] – yields blocks (with optional overlap and zero-padding)
//!
//! Use the [`WavIterators`] extension trait for easy access to all iterator types.


use crate::{
    chunks::DATA, error::{OperationError, ParseError}, wav::{alloc_sample_buffer, Wav}, ReadSeek, WaversErrorKind, WaversResult
};
use audio_sample::{AudioSample, ConvertTo, Samples, I24};
use std::io::SeekFrom;

/// A frame iterator for the Wav struct.
///
/// In WAV files, a frame consists of one sample from each channel.
/// This iterator yields each frame (all samples across channels at a given time point).
///
/// # Type Parameters
///
/// * `R` - The reader type that implements ReadSeek
/// * `T` - The audio sample type that the frames are converted to
pub struct FrameIterator<'a, R: ReadSeek, T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// The position to stop reading at
    max_pos: u64,
    /// Number of channels in the WAV file
    n_channels: usize,
    /// Reference to the Wav object being iterated
    wav: &'a mut Wav<R>,
    /// Phantom data for the audio sample type
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, R: ReadSeek, T: AudioSample> FrameIterator<'a, R, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Creates a new FrameIterator.
    ///
    /// # Arguments
    ///
    /// * `max_pos` - The maximum position to read to
    /// * `wav` - The Wav object to iterate over
    ///
    /// # Returns
    ///
    /// A new FrameIterator instance
    pub fn new(wav: &'a mut Wav<R>) -> Self {
        let fmt_chunk = wav.fmt_chunk();
        let n_channels = fmt_chunk.channels as usize;
        let max_pos = wav.max_data_pos();
        Self {
            max_pos,
            n_channels,
            wav,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, R: ReadSeek, T: AudioSample> Iterator for FrameIterator<'a, R, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Item = WaversResult<Samples<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Get current position
        let current_pos = self.wav.current_pos(); // guaranteed to be at least 0. Infallible (in theory)

        // Check if we've reached the maximum position and if so return None
        if current_pos >= self.max_pos {
            return None;
        }

        // Return the next frame
        Some(self.wav.read_samples(self.n_channels))
    }
}

/// A channel iterator for the Wav struct.
///
/// This iterator yields all samples from each channel separately.
/// For a stereo file, it would first yield all samples from the left channel,
/// then all samples from the right channel.
///
/// # Type Parameters
///
/// * `R` - The reader type that implements ReadSeek
/// * `T` - The audio sample type that the channels are converted to
pub struct ChannelIterator<'a, R: ReadSeek, T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Reference to the Wav object being iterated
    wav: &'a mut Wav<R>,
    /// The current channel being processed
    current_channel: usize,
    /// Number of samples per channel
    n_samples_per_channel: usize,
    /// Number of channels
    n_channels: usize,
    /// Phantom data for the audio sample type
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, R: ReadSeek, T: AudioSample> ChannelIterator<'a, R, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Creates a new ChannelIterator.
    ///
    /// # Arguments
    ///
    /// * `wav` - The Wav object to iterate over
    ///
    /// # Returns
    ///
    /// A new ChannelIterator instance
    pub fn new(wav: &'a mut Wav<R>) -> Self {
        let data_chunk_info = *wav.chunk_info.get_chunk_info(&DATA).unwrap(); // SAFETY: DATA chunk is guaranteed to exist by this point

        let fmt_chunk = wav.fmt_chunk(); // SAFETY: fmt_chunk is guaranteed to exist by this point
        let n_channels = fmt_chunk.channels as usize;

        let bytes_per_sample = fmt_chunk.bytes_per_sample() as usize;
        let total_samples = data_chunk_info.size as usize / bytes_per_sample;
        let n_samples_per_channel = total_samples / n_channels;

        Self {
            wav,
            current_channel: 0,
            n_samples_per_channel,
            n_channels,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, R: ReadSeek, T: AudioSample> Iterator for ChannelIterator<'a, R, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Item = WaversResult<Samples<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've processed all channels
        if self.current_channel >= self.n_channels {
            return None;
        }

        // Seek to the start of the data chunk
        self.wav.to_data(); // SAFETY: DATA chunk is guaranteed to exist by this point

        // Seek to the start of the current channel
        match self.wav.seek_by_samples(self.current_channel as u64) {
            Ok(_) => (),
            Err(e) => return Some(Err(e)),
        }
        // Allocate buffer for channel samples
        let mut channel_samples = alloc_sample_buffer(self.n_samples_per_channel);

        // Read samples for this channel
        for ch in 0..self.n_samples_per_channel {
            // Read one sample from the current channel
            match self.wav.read_sample::<T>() {
                Ok(sample) => channel_samples[ch] = sample,
                Err(e) => return Some(Err(e)),
            }

            // Skip other channels to get to the next sample for this channel
            if self.n_channels > 1 {
                match self.wav.seek_by_samples((self.n_channels - 1) as u64) {
                    Ok(_) => (),
                    Err(_) => {
                        // If we hit an error seeking, we've likely reached the end of the file
                        break;
                    }
                }
            }
        }

        // Move to the next channel for the next call
        self.current_channel += 1;
        Some(Ok(Samples::from(channel_samples)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailHandling {
    /// Discard any incomplete hop that does not fill a full frame.
    Discard,
    /// Zero‑pad the final (or single) frame to full length.
    Pad,
}

impl Default for TailHandling {
    fn default() -> Self {
        TailHandling::Pad
    }
}

pub struct BlockIterator<'a, R: ReadSeek, T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    wav: &'a mut Wav<R>,
    block_size: usize, // in frames
    _overlap: usize,    // in frames
    hop_len: usize,    // block_size - overlap  (in frames)
    n_channels: usize,
    total_blocks: usize,
    current_block: usize,
    tail: TailHandling,
    center: bool,
    pad: usize,           // frame_len/2 when center=true, otherwise 0
    samples_total: usize, // total samples in file (all channels)
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, R: ReadSeek, T: AudioSample> BlockIterator<'a, R, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    pub fn new(
        wav: &'a mut Wav<R>,
        block_size: usize,
        overlap: usize,
        tail: Option<TailHandling>,
        center: Option<bool>,
    ) -> WaversResult<Self> {
        if block_size == 0 {
            return Err(WaversErrorKind::Operation(OperationError::InvalidOp {
                reason: "Block size must be strictly positive".into()}
            ));
        }
        if overlap >= block_size {
            return Err(WaversErrorKind::Operation(OperationError::InvalidOp {
                reason: "Overlap must be smaller than block size".into()}
            ));
        }

        let fmt = wav.fmt_chunk();
        let n_channels = fmt.channels as usize;
        let bytes_ps = fmt.bytes_per_sample() as usize;

        let data_chunk = wav
            .chunk_info
            .get_chunk_info(&DATA)
            .ok_or(WaversErrorKind::Parse(ParseError::ChunkNotFound{id: DATA}))?;
        let samples_total = data_chunk.size as usize / bytes_ps;

        let tail = tail.unwrap_or_default();
        let center = center.unwrap_or(false);
        let total_blocks = Self::calculate_blocks(samples_total, block_size, overlap, tail, center);

        Ok(Self {
            wav,
            block_size,
            _overlap: overlap,
            hop_len: block_size - overlap,
            n_channels,
            total_blocks,
            current_block: 0,
            tail,
            center,
            pad: if center { block_size / 2 } else { 0 },
            samples_total,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn calculate_blocks(
        signal_len: usize,
        frame_len: usize,
        overlap: usize,
        _tail: TailHandling,
        center: bool,
    ) -> usize {
        assert!(frame_len > 0 && overlap < frame_len);

        let hop_len = frame_len - overlap;
        let pad = if center { frame_len / 2 } else { 0 };

        // When the signal is shorter than one frame (after optional padding)
        // still return exactly one frame.
        if signal_len + 2 * pad <= frame_len {
            return 1;
        }

        1 + (signal_len + 2 * pad - frame_len) / hop_len
    }

    // Number of *samples* (across all channels) that constitute one full frame.
    #[inline(always)]
    fn samples_per_frame(&self) -> usize {
        self.block_size * self.n_channels
    }
}

impl<'a, R: ReadSeek, T: AudioSample> Iterator for BlockIterator<'a, R, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Item = WaversResult<Samples<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_block >= self.total_blocks {
            // rewind to the start of DATA for potential re‑use
            let _ = self.wav.to_data();
            return None;
        }

        // Where does this frame start (in frames, *before* channel expansion)?
        // librosa’s `center=True` shifts every frame by -pad half‑frames.
        let pos_frames = match self.center {
            false => self.current_block * self.hop_len,
            true => self.current_block * self.hop_len.saturating_sub(self.pad), // may under‑flow on the first frames
        };

        let start_sample = pos_frames * self.n_channels;
        let frame_samples = self.samples_per_frame();

        // Seek to the beginning of the frame *if any part lies inside the file*.
        let leading_zeros = if start_sample >= self.samples_total {
            // Entire frame is beyond EOF — happens only with center=True + padding
            frame_samples
        } else {
            // There will be at least some real samples.
            self.wav.to_data();
            match self.wav.seek_by_samples(start_sample as u64) {
                Ok(_) => (),
                Err(e) => {
                    // If we hit an error seeking, we've likely reached the end of the file
                    self.current_block = self.total_blocks; // abort future calls
                    return Some(Err(e));
                }
            }
            0
        };

        // Read as many real samples as we can get for this frame.
        let remaining_in_file = self.samples_total.saturating_sub(start_sample);
        let max_real = frame_samples
            .saturating_sub(leading_zeros)
            .min(remaining_in_file);

        let mut frame = match self.wav.read_samples(max_real) {
            Ok(samples) => samples,
            Err(e) => {
                self.current_block = self.total_blocks;
                return Some(Err(e));
            }
        };

        let real_len = leading_zeros + frame.len();
        if real_len < frame_samples {
            match self.tail {
                TailHandling::Pad => {
                    // prepend zeros (for negative offset) or append zeros (for tail)
                    // so that len == frame_samples
                    let mut padded = Vec::<T>::with_capacity(frame_samples);
                    padded.extend(std::iter::repeat(T::zero()).take(leading_zeros));
                    padded.append(&mut frame.to_vec());
                    padded.extend(std::iter::repeat(T::zero()).take(frame_samples - padded.len()));
                    frame = Samples::from(padded.into_boxed_slice());
                }
                TailHandling::Discard => {
                    // If we’re discarding tails, return nothing when there is
                    // no real data at all, or return the shorter chunk.
                    if real_len == 0 {
                        self.current_block = self.total_blocks;
                        return None;
                    }
                    // For partial frames we simply leave them as‑is (same
                    // behaviour as `frame(y, pad_mode=None)`).
                    // No extra work required.
                }
            }
        } else if leading_zeros > 0 {
            // center=True, frame overlaps the virtual left‑side padding.
            let mut prefixed = Vec::<T>::with_capacity(frame_samples);
            prefixed.extend(std::iter::repeat(T::zero()).take(leading_zeros));
            prefixed.append(&mut frame.to_vec());
            frame = Samples::from(prefixed.into_boxed_slice());
        }

        self.current_block += 1;
        Some(Ok(frame))
    }
}

/// Extension trait for the Wav struct to provide iterator methods
pub trait WavIterators<R: ReadSeek> {
    /// Gets an iterator over frames in the Wav file
    ///
    /// A frame consists of one sample from each channel.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The audio sample type to convert to
    ///
    /// # Returns
    ///
    /// A FrameIterator instance
    fn frames<T: AudioSample>(&mut self) -> FrameIterator<R, T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>;

    /// Gets an iterator over channels in the Wav file
    ///
    /// This yields all samples from each channel separately.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The audio sample type to convert to
    ///
    /// # Returns
    ///
    /// A ChannelIterator instance
    fn channels<T: AudioSample>(&mut self) -> ChannelIterator<R, T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>;

    /// Gets an iterator over blocks in the Wav file
    ///
    /// This yields blocks of samples with an optional overlap.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The audio sample type to convert to
    ///
    /// # Arguments
    ///
    /// * `block_size` - The size of each block in frames
    /// * `overlap` - The number of overlapping frames between blocks
    ///
    /// # Returns
    ///
    /// A BlockIterator instance
    fn blocks<T: AudioSample>(
        &mut self,
        block_size: usize,
        overlap: usize,
    ) -> WaversResult<BlockIterator<R, T>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>;

    /// Gets the current position in the reader
    ///
    /// # Returns
    ///
    /// The current position as a u64
    fn current_pos(&mut self) -> u64;

    /// Gets the maximum position in the data chunk
    ///
    /// # Returns
    ///
    /// The maximum position as a u64
    fn max_data_pos(&self) -> u64;

    /// Seeks to the start of the data chunk
    fn to_data(&mut self);

    /// Seeks forward by the specified number of samples
    ///
    /// # Arguments
    ///
    /// * `samples` - The number of samples to seek forward
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or an error
    fn seek_by_samples(&mut self, samples: u64) -> WaversResult<()>;

    /// Seeks relative to the current position
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset to seek by
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or an error
    fn seek_relative(&mut self, offset: i64) -> WaversResult<u64>;
}

impl<R: ReadSeek> WavIterators<R> for Wav<R> {
    fn frames<T: AudioSample>(&mut self) -> FrameIterator<R, T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        FrameIterator::new(self)
    }

    fn channels<T: AudioSample>(&mut self) -> ChannelIterator<R, T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        ChannelIterator::new(self)
    }

    fn blocks<T: AudioSample>(
        &mut self,
        block_size: usize,
        overlap: usize,
    ) -> WaversResult<BlockIterator<R, T>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        BlockIterator::new(self, block_size, overlap, None, None)
    }

    /// Returns the current position in the reader.
    ///
    /// SAFETY: The  underlying reader.stream_position(), which is really reader.seek(SeekFrom::Curnent(0)) call can only fail if:
    ///
    /// From the std::io::Seek trait documentation:
    /// "Seeking can fail, for example because it might involve flushing a buffer." - Wavers does not flush the buffer, so this is not a concern.
    /// "Seeking to a negative offset is considered an error." - Wavers does not allow negative offsets, so this is not a concern.
    /// This is enforced early by the type system. Probably not compilable if this is not the case.
    fn current_pos(&mut self) -> u64 {
        self.reader.stream_position().unwrap_or(0)
    }

    fn max_data_pos(&self) -> u64 {
        if let Some(data_info) = self.chunk_info.get_chunk_info(&DATA) {
            (data_info.offset + 8 + data_info.size as usize) as u64
        } else {
            0
        }
    }

    fn to_data(&mut self) {
        let data_info = self.chunk_info.get_chunk_info(&DATA).unwrap(); // SAFETY: DATA chunk is guaranteed to exist by this point
        let data_start = (data_info.offset + 8) as u64; // +8 to skip the chunk header
        self.reader.seek(SeekFrom::Start(data_start)).unwrap();
    }

    fn seek_by_samples(&mut self, samples: u64) -> WaversResult<()> {
        // Get bytes per sample from the WAV type
        let bytes_per_sample = self.wav_type.n_bytes() as u64;
        let offset = samples * bytes_per_sample;
        
        let current_pos = self.current_pos();

        
        if current_pos + offset > self.max_data_pos() {
            return Err(WaversErrorKind::Operation(OperationError::InvalidOp {
                reason: "Offset exceeds maximum data position".into(),
            }));
        }

        match self.reader.seek(std::io::SeekFrom::Current(offset as i64)) {
            Ok(_) => Ok(()),
            Err(e) => Err(WaversErrorKind::Io(e)),
        }
    }

    fn seek_relative(&mut self, offset: i64) -> WaversResult<u64> {
        match self.reader.seek(SeekFrom::Current(offset)) {
            Ok(pos) => Ok(pos),
            Err(e) => Err(WaversErrorKind::Io(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{chunks::DATA, wav::Wav};
    use audio_sample::AudioSample;
    use crate::iter::WavIterators;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    // Constants for test files
    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";

    // Helper function to create a wav for testing
    fn create_test_wav<T: AudioSample>(path: &str) -> Wav<BufReader<File>>
    where
        i16: audio_sample::ConvertTo<T>,
        audio_sample::I24: audio_sample::ConvertTo<T>,
        i32: audio_sample::ConvertTo<T>,
        f32: audio_sample::ConvertTo<T>,
        f64: audio_sample::ConvertTo<T>,
    {
        let file = File::open(path).expect("Failed to open test file");
        let buf_reader = BufReader::new(file);
        Wav::from_reader(buf_reader).expect("Failed to create Wav from reader")
    }

    #[test]
    fn test_frame_iterator() {
        // This test is conditionally compiled if the test resources are available
        if Path::new(TWO_CHANNEL_WAV_I16).exists() {
            let mut wav = create_test_wav::<i16>(TWO_CHANNEL_WAV_I16);
            let fmt_chunk = wav.fmt_chunk();
            let expected_channels = fmt_chunk.channels as usize;

            // Test that frames have the correct number of samples (channels)
            let mut frame_iterator = wav.frames::<i16>();
            let frame = frame_iterator.next().unwrap();
            assert!(frame.is_ok(), "Failed to get frame");
            let frame: Samples<i16> = frame.unwrap();
            assert_eq!(
                frame.len(),
                expected_channels,
                "Frame should have {} channels",
                expected_channels
            );

            // Test that position is reset after iteration
            let current_pos =
                wav.current_pos() + ((expected_channels * std::mem::size_of::<i16>()) as u64); // should be data chunk start + 4 since we read two 2 byte samples
            let expected_pos = wav.chunk_info.get_chunk_info(&DATA).unwrap().offset + 8;
            assert_eq!(
                current_pos, expected_pos as u64,
                "Position should reset to start of data chunk"
            );
        }
    }

    #[test]
    fn test_channel_iterator() {
        if Path::new(TWO_CHANNEL_WAV_I16).exists() {
            let mut wav = create_test_wav::<i16>(TWO_CHANNEL_WAV_I16);
            let fmt_chunk = wav.fmt_chunk();
            let expected_channels = fmt_chunk.channels as usize;

            // Test that we get the right number of channels
            let mut channel_count = 0;
            for channel_result in wav.channels::<i16>() {
                let _channel = channel_result.expect("Failed to get channel");
                channel_count += 1;
            }
            assert_eq!(
                channel_count, expected_channels,
                "Should iterate over {} channels",
                expected_channels
            );
        }
    }

    #[test]
    fn test_block_iterator() {
        if Path::new(TWO_CHANNEL_WAV_I16).exists() {
            let mut wav = create_test_wav::<i16>(TWO_CHANNEL_WAV_I16);
            let fmt_chunk = wav.fmt_chunk();
            let n_channels = fmt_chunk.channels as usize;

            // Test block iterator with no overlap
            let block_size = 1024;
            let overlap = 0;

            // Count the blocks
            let mut block_count = 1; // Guarantee at least one block
            for block_result in wav.blocks::<i16>(block_size, overlap).unwrap() {
                let block = block_result.expect("Failed to get block");
                // For most blocks, the size should be block_size * n_channels
                // The last block might be smaller
                assert!(
                    block.len() <= block_size * n_channels,
                    "Block size should not exceed blockSize * channels"
                );
                block_count += 1;
            }

            // Calculate expected blocks
            let data_chunk_info = wav.chunk_info.get_chunk_info(&DATA).unwrap();
            let bytes_per_sample = wav.wav_type.n_bytes();
            let total_samples = data_chunk_info.size as usize / bytes_per_sample;
            let expected_blocks =
                BlockIterator::<std::io::BufReader<std::fs::File>, i16>::calculate_blocks(
                    total_samples,
                    block_size,
                    overlap,
                    TailHandling::Pad,
                    true,
                );

            assert_eq!(
                block_count, expected_blocks,
                "Should iterate over {} blocks",
                expected_blocks
            );
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{BlockIterator, TailHandling};
        use std::fs::File;
        use std::io::BufReader;

        #[test]
        fn test_calculate_blocks() {
            let blocks = BlockIterator::<BufReader<File>, i16>::calculate_blocks(
                1000, // signal length (samples)
                100,  // frame / block size
                0,    // overlap
                TailHandling::Pad,
                /*center=*/ false,
            );
            assert_eq!(blocks, 10, "Should yield 10 blocks with no overlap");

            let blocks = BlockIterator::<BufReader<File>, i16>::calculate_blocks(
                1000, // signal length
                100,
                50, // 50 % overlap
                TailHandling::Pad,
                false,
            );
            assert_eq!(blocks, 19, "Should yield 19 blocks with 50 % overlap");

            let samples_per_channel = 1000; // 2000 interleaved samples total
            let blocks = BlockIterator::<BufReader<File>, i16>::calculate_blocks(
                samples_per_channel,
                100,
                0,
                TailHandling::Pad,
                false,
            );
            assert_eq!(blocks, 10, "Should yield 10 blocks for 2‑ch file");
            let blocks = BlockIterator::<BufReader<File>, i16>::calculate_blocks(
                1050,
                100,
                0,
                TailHandling::Pad,
                false,
            );
            assert_eq!(
                blocks, 10,
                "1050‑sample signal should yield 10 full‑length blocks"
            );
            let blocks = BlockIterator::<BufReader<File>, i16>::calculate_blocks(
                50, // shorter than frame_len
                100,
                0,
                TailHandling::Pad,
                false,
            );
            assert_eq!(
                blocks, 1,
                "Signals shorter than one frame should still return exactly one block"
            );
        }
    }
}
