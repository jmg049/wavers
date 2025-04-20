use crate::chunks::{fmt::FmtChunk, ChunkID, ChunkInfo, ChunkManager, RIFFChunk, DATA, FMT, RIFF};
use crate::iter::{BlockIterator, ChannelIterator, FrameIterator, TailHandling, WavIterators};
use crate::{mmap_reader::MMapReader, wav_type::WavType, WaversErrorKind, WaversResult};
use crate::{FormatCode, ReadSeek};

#[cfg(feature = "colored_output")]
use colored::Colorize;

use audio_sample::I24;
use audio_sample::{AudioSample, ConvertTo, Samples};
use std::any::{type_name, TypeId};
use std::fs::File;
use std::io::BufReader;
use std::{alloc::Layout, path::Path};

pub struct Wav<R: ReadSeek> {
    pub reader: R,
    pub chunk_info: ChunkManager, // chunks are just offsets to the data in the reader
    pub wav_type: WavType,
    pub bytes_read: usize,
    pub bytes_remaining: usize,
    pub fmt_chunk: FmtChunk,
}

impl<R: ReadSeek> Wav<R> {
    pub fn new(mut reader: R) -> WaversResult<Self> {
        let chunks = find_all_chunks(&mut reader)?;

        let data_info = chunks.get_chunk_info(&DATA).unwrap();
        let data_size = data_info.size;

        let fmt_info = chunks.get_chunk_info(&FMT).unwrap();
        let fmt_offset = fmt_info.offset as u64;
        let fmt_size = fmt_info.size;

        let mut fmt_chunk_buf = alloc_box_buffer(fmt_size as usize);

        reader.seek(std::io::SeekFrom::Start(fmt_offset + 8))?; // plus 8 to go from the start of
                                                                // the chunk to the actual data of the chunk
        reader.read_exact(&mut fmt_chunk_buf)?;
        let bytes: &[u8] = fmt_chunk_buf.as_ref();
        let fmt_chunk: FmtChunk = FmtChunk::try_from(bytes)?;

        let wav_type = WavType::format_info_to_wav_type((
            fmt_chunk.format,
            fmt_chunk.effective_bits_per_sample()?,
            fmt_chunk.effective_format(),
        ));

        if wav_type == WavType::Unknown {
            return Err(WaversErrorKind::InvalidWavFile(
                format!("Unknown format: {}", fmt_chunk.format),
            ));
        }

        Ok(Self {
            reader,
            chunk_info: chunks,
            wav_type,
            bytes_read: 0,
            bytes_remaining: data_size as usize,
            fmt_chunk
        })
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> WaversResult<Wav<BufReader<File>>> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let wav = Wav::new(reader)?;
        Ok(wav)
    }

    /// Resets the reader to the start of the data chunk and reinitializes tracking counters.
    pub fn reset(&mut self) -> WaversResult<()> {
        let data_chunk_info = self.chunk_info.get_chunk_info(&DATA).unwrap();
        let data_offset = data_chunk_info.offset as u64;
        let data_size = data_chunk_info.size as usize;

        self.reader
            .seek(std::io::SeekFrom::Start(data_offset + 8))?;
        self.bytes_read = 0;
        self.bytes_remaining = data_size;

        Ok(())
    }

    /// Returns the current progress through the data chunk as a value from 0.0 to 1.0.
    pub fn progress(&self) -> f32 {
        if self.bytes_read + self.bytes_remaining == 0 {
            return 0.0;
        }

        self.bytes_read as f32 / (self.bytes_read + self.bytes_remaining) as f32
    }

    /// Returns the total number of bytes in the data chunk.
    pub fn total_data_bytes(&self) -> usize {
        self.bytes_read + self.bytes_remaining
    }

    /// Returns the number of bytes read so far from the data chunk.
    pub fn bytes_read(&self) -> usize {
        self.bytes_read
    }

    /// Returns the number of bytes remaining to be read from the data chunk.
    pub fn bytes_remaining(&self) -> usize {
        self.bytes_remaining
    }
    pub fn from_reader(reader: R) -> WaversResult<Self> {
        Self::new(reader)
    }

    pub fn fmt_chunk(&mut self) -> &FmtChunk {
        &self.fmt_chunk
    }

    /// Moves the position of the reader to the start of the data chunk and resets tracking counters.
    /// 
    /// SAFETY: The  underlying reader.stream_position(), which is really reader.seek(SeekFrom::Start(data_offset + 8)) call can only fail if:
    /// 
    /// From the std::io::Seek trait documentation:
    /// "Seeking can fail, for example because it might involve flushing a buffer." - Wavers does not flush the buffer, so this is not a concern.
    /// "Seeking to a negative offset is considered an error." - Wavers does not allow negative offsets, so this is not a concern. 
    /// This is enforced early by the type system. Probably not compilable if this is not the case.
    pub fn to_data(&mut self) -> WaversResult<()> {
        let data_chunk_info = self.chunk_info.get_chunk_info(&DATA).unwrap(); // SAFETY: The DATA chunk has to be present in the file, API will ensure this by early failure.
        let data_offset = data_chunk_info.offset as u64;
        let data_size = data_chunk_info.size as usize;

        self.reader
            .seek(std::io::SeekFrom::Start(data_offset + 8)).unwrap();
        self.bytes_read = 0;
        self.bytes_remaining = data_size;

        Ok(())
    }

    /// Seeks forward by the specified number of samples. Updates tracking counters.
    pub fn seek_by_samples(&mut self, n_samples: u64) -> WaversResult<()> {
        let bytes_per_sample = self.wav_type.n_bytes() as u64;
        let bytes_to_skip = n_samples * bytes_per_sample;

        // Ensure we don't seek past the end of the data chunk
        if bytes_to_skip as usize > self.bytes_remaining {
            return Err(WaversErrorKind::IOErrorKind(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Cannot seek beyond end of data: requested {} samples but only {} samples remain",
                    n_samples,
                    self.bytes_remaining / self.wav_type.n_bytes()
                ),
            )));
        }

        // Perform the seek
        self.reader
            .seek(std::io::SeekFrom::Current(bytes_to_skip as i64))?;

        // Update tracking counters
        self.bytes_read += bytes_to_skip as usize;
        self.bytes_remaining -= bytes_to_skip as usize;

        Ok(())
    }

    pub fn read<T: AudioSample>(&mut self) -> WaversResult<Samples<T>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let data_chunk_info = self.chunk_info.get_chunk_info(&DATA).unwrap();

        // Get the total size of the data chunk
        let data_size = data_chunk_info.size as usize;

        // Get current position in the file
        let current_pos = self.reader.stream_position()?;

        // Calculate offset from the start of data
        let data_start = (data_chunk_info.offset + 8) as u64; // +8 to skip chunk header

        // If current position is before data start, seek to start of data
        if current_pos < data_start {
            self.reader.seek(std::io::SeekFrom::Start(data_start))?;
            self.bytes_read = 0;
            self.bytes_remaining = data_size;
        } else if current_pos > data_start + data_size as u64 {
            // If we're past the end of data, reset to the start
            self.reader.seek(std::io::SeekFrom::Start(data_start))?;
            self.bytes_read = 0;
            self.bytes_remaining = data_size;
        } else {
            // We're somewhere in the data chunk
            self.bytes_read = (current_pos - data_start) as usize;
            self.bytes_remaining = data_size - self.bytes_read;
        }

        if self.bytes_remaining == 0 {
            // All data has been read, return empty samples
            let empty_samples: Box<[T]> = Box::new([]);
            return Ok(Samples::from(empty_samples));
        }

        // Calculate number of samples we can read based on remaining bytes
        let native_type = self.wav_type;
        let bytes_per_sample = native_type.n_bytes();
        let samples_remaining = self.bytes_remaining / bytes_per_sample;

        // Read all remaining samples
        let samples = self.read_samples(samples_remaining)?;

        // Position should now be at the end of data, reset to start
        self.reader.seek(std::io::SeekFrom::Start(data_start))?;
        self.bytes_read = 0;
        self.bytes_remaining = data_size;

        Ok(samples)
    }

    pub fn write<P: AsRef<Path>, T: AudioSample>(&mut self, path: P) -> WaversResult<()>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        // Read all samples as type T
        let samples: Samples<T> = self.read()?;

        // Get format information
        let fmt_chunk = self.fmt_chunk();
        let sample_rate = fmt_chunk.sample_rate as u32;
        let channels = fmt_chunk.channels;

        // Use the standalone write function to avoid duplicating code
        crate::write(path, &samples, sample_rate, Some(channels))
    }

    /// Reads a specific number of samples from the data chunk.
    /// Updates the bytes_read and bytes_remaining counters.
    /// If n_samples would read beyond the end of the data chunk, only the available samples are read.
    pub fn read_samples<T: AudioSample>(&mut self, n_samples: usize) -> WaversResult<Samples<T>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let native_type = self.wav_type;
        let target_type_id = TypeId::of::<T>();
        let target_type = WavType::from(target_type_id);

        if target_type == WavType::Unknown {
            return Err(WaversErrorKind::InvalidTargetType(
                native_type,
                type_name::<T>().to_string(),
            ));
        }

        // Calculate how many bytes we need to read
        let bytes_per_sample = native_type.n_bytes();
        let bytes_needed = bytes_per_sample * n_samples;

        // Ensure we don't read past the end of the data chunk
        let bytes_to_read = std::cmp::min(bytes_needed, self.bytes_remaining);
        let samples_to_read = bytes_to_read / bytes_per_sample;

        // If no samples to read, return empty samples
        if samples_to_read == 0 {
            let empty_samples: Box<[T]> = Box::new([]);
            return Ok(Samples::from(empty_samples));
        }

        // Allocate buffer and read
        let mut samples: Box<[u8]> = alloc_box_buffer(bytes_to_read);
        self.reader.read_exact(&mut samples)?;

        // Update tracking counters
        self.bytes_read += bytes_to_read;
        self.bytes_remaining -= bytes_to_read;

        // Force i24 tp be handled later
        if target_type == native_type && native_type != WavType::Pcm24 {
            let samples_t = cast_bytes::<T>(&samples)?;
            return Ok(Samples::from(samples_t));
        }

        match native_type {
            WavType::Pcm16 | WavType::EPcm16 => {
                let samples = cast_bytes::<i16>(&samples)?;
                Ok(Samples::from(samples).convert())
            }
            WavType::Pcm24 | WavType::EPcm24 => {
                let samples: Samples<I24> = read_i24(&samples);
                Ok(samples.convert())
            }
            WavType::Pcm32 | WavType::EPcm32 => {
                let samples = cast_bytes::<i32>(&samples)?;
                Ok(Samples::from(samples).convert())
            }
            WavType::Float32 | WavType::EFloat32 => {
                let samples = cast_bytes::<f32>(&samples)?;
                Ok(Samples::from(samples).convert())
            }
            WavType::Float64 | WavType::EFloat64 => {
                let samples = cast_bytes::<f64>(&samples)?;
                Ok(Samples::from(samples).convert())
            }
            _ => unreachable!("If unknown would have returned above."),
        }
    }
    /// Read a single sample from the DATA chunk
    pub fn read_sample<T: AudioSample>(&mut self) -> WaversResult<T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let samples = self.read_samples(1)?;
        if samples.is_empty() {
            return Err(WaversErrorKind::InvalidOp(
                "No samples available to read".to_string(),
            ));
        }
        Ok(samples[0])
    }

    /// Read a single frame from the DATA chunk
    pub fn read_frame<T: AudioSample>(&mut self) -> WaversResult<Samples<T>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let fmt_chunk = self.fmt_chunk();
        let channels = fmt_chunk.channels as usize;
        self.read_samples(channels)
    }

    /// Read some number of frames from the DATA chunk
    pub fn read_frames<T: AudioSample>(&mut self, n_frames: usize) -> WaversResult<Samples<T>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let fmt_chunk = self.fmt_chunk();
        let channels = fmt_chunk.channels as usize;
        self.read_samples(n_frames * channels)
    }

    /// Get information about the WAV file
    pub fn info(&mut self) -> WavInfo {
        let data_chunk_info = *self.chunk_info.get_chunk_info(&DATA).unwrap();

        let fmt_chunk = self.fmt_chunk();

        let sample_rate = fmt_chunk.sample_rate as u32;
        let channels = fmt_chunk.channels;
        let format = fmt_chunk.format;
        let bits_per_sample = fmt_chunk.bits_per_sample;
        let bytes_per_sample = bits_per_sample / 8;
        let n_samples = data_chunk_info.size / (bytes_per_sample as u32 * channels as u32);
        let duration = n_samples as f32 / sample_rate as f32;

        WavInfo {
            sample_rate,
            channels,
            format,
            bits_per_sample,
            bytes_per_sample,
            n_samples,
            duration,
        }
    }
}

pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: FormatCode,
    pub bits_per_sample: u16,
    pub bytes_per_sample: u16,
    pub n_samples: u32,
    pub duration: f32,
}

impl std::fmt::Display for WavInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(feature = "colored_output")]
        {
            // Define color theme with colored crate
            let title_style = "WAV Info".bright_blue().bold();
            let label_style = |s: &str| s.yellow();
            let value_style = |s: String| s.green();

            writeln!(f, "{}", title_style)?;
            writeln!(
                f,
                "  {}: {} Hz",
                label_style("Sample Rate"),
                value_style(self.sample_rate.to_string())
            )?;
            writeln!(
                f,
                "  {}: {}",
                label_style("Channels"),
                value_style(self.channels.to_string())
            )?;
            writeln!(
                f,
                "  {}: {} ({})",
                label_style("Format"),
                value_style(self.format.to_string()),
                value_style(format!("{}", self.format as u16))
            )?;
            writeln!(
                f,
                "  {}: {} bits",
                label_style("Bits Per Sample"),
                value_style(self.bits_per_sample.to_string())
            )?;
            writeln!(
                f,
                "  {}: {} bytes",
                label_style("Bytes Per Sample"),
                value_style(self.bytes_per_sample.to_string())
            )?;
            writeln!(
                f,
                "  {}: {}",
                label_style("Total Samples"),
                value_style(self.n_samples.to_string())
            )?;
            writeln!(
                f,
                "  {}: {:.2} seconds",
                label_style("Duration"),
                value_style(format!("{:.2}", self.duration))
            )?;
        }

        #[cfg(not(feature = "colored_output"))]
        {
            writeln!(f, "WAV Info")?;
            writeln!(f, "  Sample Rate: {} Hz", self.sample_rate)?;
            writeln!(f, "  Channels: {}", self.channels)?;
            writeln!(f, "  Format: {} ({})", self.format, self.format as u16)?;
            writeln!(f, "  Bits Per Sample: {} bits", self.bits_per_sample)?;
            writeln!(f, "  Bytes Per Sample: {} bytes", self.bytes_per_sample)?;
            writeln!(f, "  Total Samples: {}", self.n_samples)?;
            writeln!(f, "  Duration: {:.2} seconds", self.duration)?;
        }

        Ok(())
    }
}

fn read_i24(samples: &Box<[u8]>) -> Samples<I24> {
    let n_samples = samples.len() / 3;

    let mut i24_samples: Box<[I24]> = alloc_sample_buffer(n_samples);
    for i in 0..n_samples {
        let sample = &samples[i * 3..(i + 1) * 3];
        let i32_sample = i32::from_le_bytes([0, sample[0], sample[1], sample[2]]);
        i24_samples[i] = I24::saturating_from_i32(i32_sample);
    }
    i24_samples.into()
}

impl Wav<MMapReader> {
    pub fn from_mmap(mmap: memmap2::Mmap) -> WaversResult<Self> {
        let reader = MMapReader::new(mmap)?;
        Self::from_reader(reader)
    }


}

fn find_all_chunks<'c, R: ReadSeek>(reader: &mut R) -> WaversResult<ChunkManager> {
    // Have to find the RIFF chunk first
    let mut buf: [u8; 4] = [0; 4];

    reader.read_exact(&mut buf)?;

    while ChunkID::new(buf) != RIFF {
        // take 4 more bytes
        reader.read_exact(&mut buf)?;
    }

    let riff_chunk_offset = reader.stream_position()?;
    // found the RIFF chunk

    let mut riff_buf: [u8; 12] = [0; 12];
    // Copy over the RIFF chunk
    riff_buf[0..4].copy_from_slice(&buf);

    reader.read_exact(&mut riff_buf[4..12])?;

    let riff_chunk = RIFFChunk::parse(riff_buf)?; // Does the validation for us

    let mut chunk_manager = ChunkManager::new();

    chunk_manager.add_chunk_info(ChunkInfo::new(RIFF, 12, riff_chunk_offset as u32));

    // Now at position 12 - relative to start of the Wav file. We know the next four bytes will be
    // a chunk tag, followed by a fours bytes for a chunk size.
    // Need to keep track of bytes read so don't go beyond the wav file limits
    let max_position = riff_chunk_offset + riff_chunk.file_size as u64;
    let mut current_pos = reader.stream_position()?;

    while current_pos < max_position {
        let offset = current_pos;
        reader.read_exact(&mut buf)?;
        let tag = ChunkID::new(buf);

        match crate::chunks::validate_tag(tag) {
            Ok(()) => (),
            Err(e) => {
                eprintln!(
                    "{:?} is not a valid tag",
                    std::str::from_utf8(&tag.id).unwrap()
                );
                return Err(e);
            }
        }

        reader.read_exact(&mut buf)?;
        let chunk_size = u32::from_le_bytes(buf.try_into().map_err(|e| {
            WaversErrorKind::InvalidWavFile(format!(
                "Failed to create a valid u32 from buffer {:?} {:?}",
                buf, e
            ))
        })?);

        current_pos = reader.stream_position()? + chunk_size as u64;

        chunk_manager.add_chunk_info(ChunkInfo::new(tag, offset as usize, chunk_size));
        reader.seek(std::io::SeekFrom::Start(current_pos))?;
    }

    Ok(chunk_manager)
}

/// Helper function to allocate a fixed sized, heap allocated buffer of type T.
pub(crate) fn alloc_sample_buffer<T: AudioSample>(len: usize) -> Box<[T]> {
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

/// Helper function to allocate a fixed sized, heap allocated buffer of bytes.
pub(crate) fn alloc_box_buffer(len: usize) -> Box<[u8]> {
    if len == 0 {
        return <Box<[u8]>>::default();
    }
    let layout = match Layout::array::<u8>(len) {
        Ok(layout) => layout,
        Err(_) => panic!("Failed to allocate buffer of size {}", len),
    };

    let ptr = unsafe { std::alloc::alloc(layout) };
    let slice_ptr = core::ptr::slice_from_raw_parts_mut(ptr, len);
    unsafe { Box::from_raw(slice_ptr) }
}

/// Reinterpret a byte slice as a slice of type `T`.
/// Returns an error if the alignment is incorrect or the length isn't a multiple of T.
///
/// # Safety
/// T must be a plain-old-data (POD) type with no padding-sensitive layout or drop logic.
pub(crate) fn cast_bytes<T>(bytes: &[u8]) -> WaversResult<&[T]> {
    use std::mem::{align_of, size_of};

    if size_of::<T>() == 0 {
        return Err(WaversErrorKind::InvalidOp(
            "Cannot cast to zero-sized type".to_string(),
        ));
    }

    if bytes.as_ptr().align_offset(align_of::<T>()) != 0 {
        return Err(WaversErrorKind::InvalidOp(
            "Input bytes are not properly aligned for target type".to_string(),
        ));
    }

    if bytes.len() % size_of::<T>() != 0 {
        return Err(WaversErrorKind::InvalidOp(
            "Input length is not a multiple of target type size".to_string(),
        ));
    }

    let len = bytes.len() / size_of::<T>();
    let ptr = bytes.as_ptr() as *const T;

    unsafe { Ok(std::slice::from_raw_parts(ptr, len)) }
}

// in src/lib.rs (or wherever your Wav type lives)
#[cfg(test)]
mod wav_tests {
    use super::*;
    use crate::wav_api_tests::read_text_to_vec;
    use audio_sample::{Samples, I24};
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    // common constants
    const ONE_CHANNEL_WAV_I16: &str = "./test_resources/one_channel_i16.wav";
    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";
    const ONE_CHANNEL_WAV_I24: &str = "./test_resources/one_channel_i24.wav";
    const ONE_CHANNEL_WAV_I32: &str = "./test_resources/one_channel_i32.wav";
    const ONE_CHANNEL_WAV_F32: &str = "./test_resources/one_channel_f32.wav";
    const ONE_CHANNEL_WAV_F64: &str = "./test_resources/one_channel_f64.wav";    

    const ONE_CHANNEL_EXPECTED_I16: &str = "./test_resources/one_channel_i16.txt";
    const ONE_CHANNEL_EXPECTED_I24: &str = "./test_resources/one_channel_i24.txt";
    const ONE_CHANNEL_EXPECTED_I32: &str = "./test_resources/one_channel_i32.txt";

    const ONE_CHANNEL_EXPECTED_F32: &str = "./test_resources/one_channel_f32.txt";
    const ONE_CHANNEL_EXPECTED_F64: &str = "./test_resources/one_channel_f64.txt";

    const TEST_OUTPUT: &str = "./test_resources/tmp/";

    macro_rules! read_tests {
        (
            $(
                $name:ident: ($wav:path, $expected:path, $ty:ty)
            ),+ $(,)?
        ) => {
            $(
                #[test]
                fn $name() {
                    let f = File::open($wav).expect("open");
                    let mut wav = Wav::from_reader(BufReader::new(f)).expect("from_reader");
                    let samples: Samples<$ty> = wav.read().expect("read");
                    let expected: Vec<$ty> =
                        read_text_to_vec(Path::new($expected)).expect("read expected");
                    assert_eq!(samples.len(), expected.len(),
                        "len mismatch {} vs {}", samples.len(), expected.len());
                    for (i, (e, a)) in expected.iter().zip(samples.iter()).enumerate() {
                        // use approx_eq for f32/f64
                        if std::any::TypeId::of::<$ty>() == TypeId::of::<f32>() || std::any::TypeId::of::<$ty>() == TypeId::of::<f64>() {
                            let e: f64 = e.convert_to(); // should be f32 anyway but need to convince the compiler
                            let a: f64 = a.convert_to();

                            let diff: f64 = (e - a).abs();
                            assert!(diff < 1e-4, "at idx {} diff {}", i, diff);
                        } else {
                            assert_eq!(*e, *a, "idx {}", i);
                        }
                    }
                }
            )+
        };
    }

    macro_rules! write_tests {
        (
            $(
                $name:ident: ($ty:ty, $suffix:expr)
            ),+ $(,)?
        ) => {
            $(
                #[test]
                fn $name() {
                    // ── setup ────────────────────────────────────────────────────
                    let out_dir = Path::new(TEST_OUTPUT);
                    if !out_dir.exists() {
                        std::fs::create_dir_all(out_dir).unwrap();
                    }
                    let out = out_dir.join(format!("test_write_{}.wav", $suffix));
    
                    // read the original file as $ty
                    let f = File::open(ONE_CHANNEL_WAV_I16).expect("open");
                    let mut wav = Wav::from_reader(BufReader::new(f)).expect("from_reader");
                    let samples: Samples<$ty> = wav.read().expect("read");
                    let fmt = wav.fmt_chunk();
    
                    // ── write & read back ─────────────────────────────────────────
                    crate::write(&out, &samples, fmt.sample_rate as u32, Some(fmt.channels))
                        .expect("write");
                    let mut reread = Wav::from_reader(BufReader::new(File::open(&out).unwrap()))
                        .expect("open reread");
                    let new_samples: Samples<$ty> = reread.read().expect("read reread");
    
                    // ── compare sample data ───────────────────────────────────────
                    assert_eq!(samples.len(), new_samples.len(), "len {}", $suffix);
                    for (i, (orig, got)) in samples.iter().zip(new_samples.iter()).enumerate() {
                        assert_eq!(*orig, *got, "sample mismatch at idx {} ({})", i, $suffix);
                    }
    
                    // ── verify header definitely matches the sample type ───────────
                    let new_fmt = reread.fmt_chunk();
                    let expected_bits = (std::mem::size_of::<$ty>() * 8) as u16;
                    assert_eq!(
                        new_fmt.bits_per_sample,
                        expected_bits,
                        "fmt.bits_per_sample should be {} for {}",
                        expected_bits,
                        $suffix
                    );
    
                    // ── clean up ─────────────────────────────────────────────────
                    std::fs::remove_file(&out).unwrap();
                }
            )+
        };
    }


    // ——— opening tests —————————————————————————————————————
    mod wav_open_tests {
        use super::*;
        #[test]
        pub fn can_open_wav_file() {
            let f = File::open(TWO_CHANNEL_WAV_I16).unwrap();
            assert!(Wav::new(BufReader::new(f)).is_ok(),
                "Failed to open WAV file");
        }

        #[test]
        fn test_open_wav_file() {
            let f = File::open(TWO_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let fmt = wav.fmt_chunk();
            assert_eq!(fmt.channels, 2);
            assert_eq!(fmt.sample_rate, 16000);
            assert_eq!(fmt.bits_per_sample, 16);
        }
    }

    // ——— read tests ——————————————————————————————————————
    mod wav_read_tests {
        use super::*;

        #[test]
        fn test_read_single_channel() {
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let samples: Samples<i16> = wav.read().unwrap();
            let expected: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();
            assert_eq!(samples.len(), expected.len());
            for (i, (e, a)) in expected.iter().zip(samples.iter()).enumerate() {
                assert_eq!(*e, *a, "at idx {}", i);
            }
        }

        #[test]
        fn test_read_two_channel() {
            let f = File::open(TWO_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let samples: Samples<i16> = wav.read().unwrap();
            // stereo interleaved == two copies of mono
            let mono: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();
            let mut expected = Vec::with_capacity(mono.len() * 2);
            mono.iter().for_each(|&s| {
                expected.push(s);
                expected.push(s);
            });
            assert_eq!(samples.len(), expected.len());
            for (i, (e, a)) in expected.iter().zip(samples.iter()).enumerate() {
                assert_eq!(*e, *a, "at idx {}", i);
            }
        }

        #[test]
        fn test_read_i24() {
            if !Path::new(ONE_CHANNEL_WAV_I24).exists() {
                eprintln!("skip i24");
                return;
            }
            let f = File::open(ONE_CHANNEL_WAV_I24).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            assert_eq!(wav.fmt_chunk().bits_per_sample, 24);
            let samples: Samples<I24> = wav.read().unwrap();
            let expected_i32: Vec<i32> =
                read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I24)).unwrap();
            let expected: Vec<I24> = expected_i32
                .into_iter()
                .map(I24::saturating_from_i32)
                .collect();
            assert_eq!(samples.len(), expected.len());
            for (i, (e, a)) in expected.iter().zip(samples.iter()).enumerate() {
                assert_eq!(e.to_i32(), a.to_i32(), "at idx {}", i);
            }
        }

        #[test]
        fn test_convert_formats() {
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let samples: Samples<f32> = wav.read().unwrap();
            let expected: Vec<f32> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_F32)).unwrap();
            assert_eq!(samples.len(), expected.len());
            for (i, (e, a)) in expected.iter().zip(samples.iter()).enumerate() {
                let diff = (e - a).abs();
                assert!(diff < 1e-5, "at idx {} diff {}", i, diff);
            }
        }

        #[test]
        fn test_read_single_sample() {
            let f1 = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f1)).unwrap();
            let f2 = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut cmp = Wav::from_reader(BufReader::new(f2)).unwrap();
            let all: Samples<i16> = cmp.read().unwrap();
            let pos: usize = 1_000 ;
            wav.to_data().unwrap();
            wav.seek_by_samples(pos as u64).unwrap();
            let single = wav.read_sample::<i16>().unwrap();
            assert_eq!(single, all[pos]);
        }

        read_tests! {
            read_i16: (ONE_CHANNEL_WAV_I16, ONE_CHANNEL_EXPECTED_I16, i16),
            read_i16_i32: (ONE_CHANNEL_WAV_I16, ONE_CHANNEL_EXPECTED_I32, i32),
            read_i16_f32: (ONE_CHANNEL_WAV_I16, ONE_CHANNEL_EXPECTED_F32, f32),
            read_i16_f64: (ONE_CHANNEL_WAV_I16, ONE_CHANNEL_EXPECTED_F64, f64),
            read_i32: (ONE_CHANNEL_WAV_I32, ONE_CHANNEL_EXPECTED_I32, i32),
            read_i32_i16: (ONE_CHANNEL_WAV_I32, ONE_CHANNEL_EXPECTED_I16, i16),
            read_i32_f32: (ONE_CHANNEL_WAV_I32, ONE_CHANNEL_EXPECTED_F32, f32),
            read_i32_f64: (ONE_CHANNEL_WAV_I32, ONE_CHANNEL_EXPECTED_F64, f64),
            read_f32: (ONE_CHANNEL_WAV_F32, ONE_CHANNEL_EXPECTED_F32, f32),
            read_f32_i16: (ONE_CHANNEL_WAV_F32, ONE_CHANNEL_EXPECTED_I16, i16),
            read_f32_i32: (ONE_CHANNEL_WAV_F32, ONE_CHANNEL_EXPECTED_I32, i32),
            read_f32_f64: (ONE_CHANNEL_WAV_F32, ONE_CHANNEL_EXPECTED_F64, f64),
            read_f64: (ONE_CHANNEL_WAV_F64, ONE_CHANNEL_EXPECTED_F64, f64),
            read_f64_i16: (ONE_CHANNEL_WAV_F64, ONE_CHANNEL_EXPECTED_I16, i16),
            read_f64_i32: (ONE_CHANNEL_WAV_F64, ONE_CHANNEL_EXPECTED_I32, i32),
            read_f64_f32: (ONE_CHANNEL_WAV_F64, ONE_CHANNEL_EXPECTED_F32, f32),
            
        }
    }

    // ——— seek tests ——————————————————————————————————————
    mod wav_seek_tests {
        use super::*;

        #[test]
        fn test_seek_by_samples() {
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let sr = wav.fmt_chunk().sample_rate as u64;
            wav.to_data().unwrap();
            wav.seek_by_samples(sr).unwrap();
            let chunk: Samples<i16> = wav.read_samples(100).unwrap();
            let expected: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();
            let slice = &expected[sr as usize..sr as usize + 100];
            assert_eq!(chunk.as_ref(), slice);
        }

        #[test]
        fn test_seek_past_end() {
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let data_info = wav.chunk_info.get_chunk_info(&DATA).unwrap();
            let total = (data_info.size as usize) / wav.wav_type.n_bytes();
            wav.to_data().unwrap();
            let res = wav.seek_by_samples(total as u64 + 1);
            assert!(res.is_err());
        }
    }

    // ——— write tests —————————————————————————————————————
    mod wav_write_tests {
        use super::*;

        #[test]
        fn test_write_and_read() {
            let out_dir = Path::new(TEST_OUTPUT);
            if !out_dir.exists() {
                std::fs::create_dir_all(out_dir).unwrap();
            }
            let out = out_dir.join("test_write.wav");

            // read original
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let samples: Samples<i16> = wav.read().unwrap();
            let fmt = wav.fmt_chunk();

            // write
            crate::write(&out, &samples, fmt.sample_rate as u32, Some(fmt.channels)).unwrap();

            // read back
            let mut reread = Wav::from_reader(BufReader::new(File::open(&out).unwrap())).unwrap();
            let new: Samples<i16> = reread.read().unwrap();
            assert_eq!(samples.len(), new.len());
            for (i, (o, n)) in samples.iter().zip(new.iter()).enumerate() {
                assert_eq!(*o, *n, "at idx {}", i);
            }
            std::fs::remove_file(&out).unwrap();
        }

        #[test]
        fn test_write_different_format() {
            let out_dir = Path::new(TEST_OUTPUT);
            if !out_dir.exists() {
                std::fs::create_dir_all(out_dir).unwrap();
            }
            let out = out_dir.join("test_write_f32.wav");

            // read & convert
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let samples: Samples<f32> = wav.read().unwrap();
            let fmt = wav.fmt_chunk();

            // write float
            crate::write(&out, &samples, fmt.sample_rate as u32, Some(fmt.channels)).unwrap();

            // verify header
            let mut new_wav = Wav::from_reader(BufReader::new(File::open(&out).unwrap()))
                .unwrap();
            let new_fmt = new_wav.fmt_chunk();
            assert_eq!(new_fmt.format, FormatCode::WavFormatIeeeFloat);
            assert_eq!(new_fmt.bits_per_sample, 32);

            // read back & compare
            let mut r = Wav::from_reader(BufReader::new(File::open(&out).unwrap())).unwrap();
            let new: Samples<f32> = r.read().unwrap();
            assert_eq!(new.len(), samples.len());
            for (i, (a, b)) in samples.iter().zip(new.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "at idx {}", i);
            }
            std::fs::remove_file(&out).unwrap();
        }

        write_tests! {
            write_i16: (i16, "i16"),
            write_i24: (i32, "i32"),
            write_f32: (f32, "f32"),
            write_f64: (f64, "f64"),
        }
    }

    // ——— misc & error tests ———————————————————————————————
    mod wav_misc_tests {
        use super::*;

        #[test]
        fn test_wav_info() {
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let info = wav.info();
            assert_eq!(info.sample_rate, 16000);
            assert_eq!(info.channels, 1);
            assert_eq!(info.bits_per_sample, 16);
            assert_eq!(info.duration, 10.0);
            assert_eq!(info.n_samples, 160000);
        }

        #[test]
        fn test_invalid_wav_file() {
            let bad = std::io::Cursor::new(&b"not really a wav"[..]);
            assert!(Wav::new(bad).is_err());
        }

        #[test]
        fn test_empty_wav() {
            let empty = std::io::Cursor::new(&[] as &[u8]);
            assert!(Wav::new(empty).is_err());
        }
    }

}