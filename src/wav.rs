use crate::chunks::{ChunkID, ChunkInfo, ChunkManager, DATA, FMT, RIFF, RIFFChunk, fmt::FmtChunk};
use crate::error::InternalError;
use crate::{FormatCode, ReadSeek, WavIterators};
use crate::{WaversErrorKind, WaversResult, wav_type::WavType};

use audio_sample::{AudioSample, ConvertCollection, ConvertTo, I24, I24DiskMethods, Samples};

#[cfg(feature = "ndarray")]
use audio_sample::{ChannelLayout, ToNdarray};

use std::any::{TypeId, type_name};
use std::fmt::Display;
use std::fs::File;
use std::io::BufReader;
use std::{alloc::Layout, path::Path};

/// Represents a WAV file loaded from any readable and seekable source.
pub struct Wav<R: ReadSeek> {
    pub reader: R,
    pub chunk_info: ChunkManager,
    pub wav_type: WavType,
    pub bytes_read: usize,
    pub bytes_remaining: usize,
    pub fmt_chunk: FmtChunk,
}

impl<R: ReadSeek> Wav<R> {
    /// Constructs a WAV reader from an input stream.
    ///
    /// This will parse the chunk headers and validate structure. Returns an error if:
    /// - The format is unsupported.
    /// - The necessary chunks are missing.
    /// - The stream cannot be read or parsed.
    pub fn new(mut reader: R) -> WaversResult<Self> {
        let chunks = find_all_chunks(&mut reader)?;

        let data_info = chunks.get_chunk_info(&DATA).ok_or(WaversErrorKind::Parse(
            crate::error::ParseError::ChunkNotFound { id: DATA },
        ))?;
        let data_size = data_info.size;

        let fmt_info = chunks.get_chunk_info(&FMT).ok_or(WaversErrorKind::Parse(
            crate::error::ParseError::ChunkNotFound { id: FMT },
        ))?;
        let fmt_offset = fmt_info.offset as u64;
        let fmt_size = fmt_info.size;

        let mut fmt_chunk_buf = alloc_box_buffer(fmt_size as usize);
        reader.seek(std::io::SeekFrom::Start(fmt_offset + 8))?;
        reader.read_exact(&mut fmt_chunk_buf)?;
        let fmt_chunk: FmtChunk = FmtChunk::try_from(fmt_chunk_buf.as_ref())?;

        let wav_type = WavType::format_info_to_wav_type((
            fmt_chunk.format,
            fmt_chunk.effective_bits_per_sample()?,
            fmt_chunk.effective_format(),
        ));

        if wav_type == WavType::Unknown {
            return Err(WaversErrorKind::Validate(
                crate::error::ValidationError::UnsupportedFormat(fmt_chunk.format),
            ));
        }

        Ok(Self {
            reader,
            chunk_info: chunks,
            wav_type,
            bytes_read: 0,
            bytes_remaining: data_size as usize,
            fmt_chunk,
        })
    }

    /// Opens a WAV file from the given path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> WaversResult<Wav<BufReader<File>>> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        Wav::new(reader)
    }

    /// Resets the reading state to the beginning of the data chunk.
    pub fn reset(&mut self) -> WaversResult<()> {
        let data_chunk_info =
            self.chunk_info
                .get_chunk_info(&DATA)
                .ok_or(WaversErrorKind::Parse(
                    crate::error::ParseError::ChunkNotFound { id: DATA },
                ))?;

        self.reader
            .seek(std::io::SeekFrom::Start(data_chunk_info.offset as u64 + 8))?;
        self.bytes_read = 0;
        self.bytes_remaining = data_chunk_info.size as usize;

        Ok(())
    }

    /// Returns progress through the data chunk as a ratio from 0.0 to 1.0.
    pub fn progress(&self) -> f32 {
        let total = self.bytes_read + self.bytes_remaining;
        if total == 0 {
            0.0
        } else {
            self.bytes_read as f32 / total as f32
        }
    }

    pub fn total_data_bytes(&self) -> usize {
        self.bytes_read + self.bytes_remaining
    }

    pub fn bytes_read(&self) -> usize {
        self.bytes_read
    }

    pub fn bytes_remaining(&self) -> usize {
        self.bytes_remaining
    }

    pub fn from_reader(reader: R) -> WaversResult<Self> {
        Self::new(reader)
    }

    pub fn fmt_chunk(&mut self) -> &FmtChunk {
        &self.fmt_chunk
    }

    /// Reads all samples in the data chunk.
    pub fn read<T: AudioSample>(&mut self) -> WaversResult<Box<[T]>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let data_info = self
            .chunk_info
            .get_chunk_info(&DATA)
            .ok_or(WaversErrorKind::Parse(
                crate::error::ParseError::ChunkNotFound { id: DATA },
            ))?;
        let data_start = data_info.offset as u64 + 8;
        let data_size = data_info.size as usize;

        self.to_data();

        self.bytes_read = 0;
        self.bytes_remaining = data_size;

        let samples = self.read_samples::<T>(self.bytes_remaining / self.wav_type.n_bytes())?;

        self.reader.seek(std::io::SeekFrom::Start(data_start))?;
        self.bytes_read = 0;
        self.bytes_remaining = data_size;

        Ok(samples)
    }

    #[cfg(feature = "ndarray")]
    /// Reads all samples in the data chunk and converts them into an [ndarray::Array2] and the sample rate.
    /// Mirrors how a lot of Python libraries work. The ndarray is in the shape of (n_channels, n_samples).
    pub fn read_ndarray<T: AudioSample>(&mut self) -> WaversResult<(ndarray::Array2<T>, i32)>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let samples = self.read::<T>()?;
        let channels = self.fmt_chunk.channels as usize;
        let samples = ChannelSamples::new(samples, channels);

        Ok((
            samples.to_ndarray(channels, ChannelLayout::Interleaved)?,
            self.fmt_chunk.sample_rate,
        ))
    }

    /// Writes samples to a new WAV file using the provided type.
    pub fn write<P: AsRef<Path>, From: AudioSample, To: AudioSample>(
        &mut self,
        path: P,
    ) -> WaversResult<()>
    where
        To: ConvertTo<To>,
        i16: ConvertTo<To>,
        I24: ConvertTo<To>,
        i32: ConvertTo<To>,
        f32: ConvertTo<To>,
        f64: ConvertTo<To>,
    {
        let samples: Box<[To]> = self.read::<To>()?;
        let fmt_chunk = self.fmt_chunk();

        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        let bytes = samples.into_bytes();

        let wav_type = WavType::from(TypeId::of::<To>());

        crate::write_with(
            writer,
            &bytes,
            wav_type,
            fmt_chunk.sample_rate as u32,
            Some(fmt_chunk.channels),
        )
    }

    /// Reads a specific number of samples from the data chunk.
    pub fn read_samples<T: AudioSample>(&mut self, n_samples: usize) -> WaversResult<Box<[T]>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let native_type = self.wav_type;
        let target_type = WavType::from(TypeId::of::<T>());

        if target_type == WavType::Unknown {
            return Err(WaversErrorKind::Convert(
                crate::error::ConversionError::Unsupported {
                    src: native_type,
                    dst: type_name::<T>().to_string(),
                },
            ));
        }

        let bytes_per_sample = native_type.n_bytes();
        let bytes_needed = bytes_per_sample * n_samples;
        let bytes_to_read = bytes_needed.min(self.bytes_remaining);
        let samples_to_read = bytes_to_read / bytes_per_sample;

        if samples_to_read == 0 {
            return Ok(Box::from([]));
        }

        let mut raw = alloc_box_buffer(bytes_to_read);
        let raw_slice = &mut *raw;
        self.reader.read_exact(raw_slice)?;
        self.bytes_read += bytes_to_read;
        self.bytes_remaining -= bytes_to_read;

        if target_type == native_type && native_type != WavType::Pcm24 {
            // First verify that our data is compatible with the target type

            // Transmute the Box<[u8]> into Box<[T]>
            let boxed_t = unsafe {
                // Convert Box<[u8]> to raw pointer
                let raw_ptr = Box::into_raw(raw);

                // Calculate the correct length for T
                let len_in_t = bytes_to_read / std::mem::size_of::<T>();

                // Create a new Box<[T]> from the raw pointer
                Box::from_raw(std::slice::from_raw_parts_mut(raw_ptr as *mut T, len_in_t))
            };

            return Ok(boxed_t);
        }

        match native_type {
            WavType::Pcm16 | WavType::EPcm16 => {
                let boxed_data = transmute_box::<i16>(raw, bytes_to_read);
                Ok(boxed_data.convert_into())
            }
            WavType::Pcm24 | WavType::EPcm24 => {
                let boxed_data = match I24::read_i24s_le(&raw) {
                    Some(i24s) => i24s.into_boxed_slice(),
                    None => {
                        return Err(WaversErrorKind::Internal(InternalError::Bug(
                            "Failed to read I24 samples".to_string(),
                        )));
                    }
                };
                Ok(boxed_data.convert_into())
            }
            WavType::Pcm32 | WavType::EPcm32 => {
                let boxed_data = transmute_box::<i32>(raw, bytes_to_read);
                Ok(boxed_data.convert_into())
            }
            WavType::Float32 | WavType::EFloat32 => {
                let boxed_data = transmute_box::<f32>(raw, bytes_to_read);
                Ok(boxed_data.convert_into())
            }
            WavType::Float64 | WavType::EFloat64 => {
                let boxed_data = transmute_box::<f64>(raw, bytes_to_read);
                Ok(boxed_data.convert_into())
            }
            _ => {
                return Err(WaversErrorKind::Internal(InternalError::Bug(format!(
                    "Unsupported format {}",
                    native_type
                ))));
            }
        }
    }

    pub fn read_sample<T: AudioSample>(&mut self) -> WaversResult<T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let samples: Box<[T]> = self.read_samples(1)?;
        samples.first().copied().ok_or(WaversErrorKind::Operation(
            crate::error::OperationError::EmptyChunks { op: "read_sample" },
        ))
    }

    pub fn read_frame<T: AudioSample>(&mut self) -> WaversResult<Box<[T]>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let samples = self.read_samples(self.fmt_chunk.channels as usize)?;
        Ok(samples)
    }

    pub fn read_frames<T: AudioSample>(&mut self, n_frames: usize) -> WaversResult<Box<[T]>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        self.read_samples(n_frames * self.fmt_chunk.channels as usize)
    }

    pub fn info(&mut self) -> WavInfo {
        let data_chunk = *self.chunk_info.get_chunk_info(&DATA).unwrap();
        let fmt = self.fmt_chunk();

        let bytes_per_sample = fmt.bits_per_sample / 8;
        let total_samples = data_chunk.size / (bytes_per_sample as u32 * fmt.channels as u32);
        let duration = total_samples as f32 / fmt.sample_rate as f32;

        WavInfo {
            sample_rate: fmt.sample_rate as u32,
            channels: fmt.channels,
            format: fmt.format,
            bits_per_sample: fmt.bits_per_sample,
            bytes_per_sample,
            n_samples: total_samples,
            duration,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChannelSamples<T: AudioSample> {
    pub samples: Box<[T]>,
    pub channels: usize,
}

impl<T: AudioSample> ChannelSamples<T> {
    pub fn new(samples: Box<[T]>, channels: usize) -> Self {
        Self { samples, channels }
    }
}

impl<T: AudioSample> Default for ChannelSamples<T> {
    fn default() -> Self {
        Self {
            samples: Box::new([]),
            channels: 1,
        }
    }
}

impl<T: AudioSample> IntoIterator for ChannelSamples<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.samples.into_iter()
    }
}
impl<T: AudioSample> AsRef<[T]> for ChannelSamples<T> {
    fn as_ref(&self) -> &[T] {
        &self.samples
    }
}

impl<T: AudioSample> AsMut<[T]> for ChannelSamples<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.samples
    }
}

impl<T: AudioSample> Samples for ChannelSamples<T> {
    fn len(&self) -> usize {
        self.samples.len()
    }
}

#[cfg(feature = "ndarray")]
impl<T: AudioSample> ToNdarray<T> for ChannelSamples<T> {
    fn to_ndarray(
        self,
        channels: usize,
        layout: ChannelLayout,
    ) -> audio_sample::AudioSampleResult<ndarray::Array2<T>> {
        self.samples.to_ndarray(channels, layout)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: FormatCode,
    pub bits_per_sample: u16,
    pub bytes_per_sample: u16,
    pub n_samples: u32,
    pub duration: f32,
}

impl Display for WavInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WAV Info: {} channels, {} Hz, {} bits per sample, {} bytes per sample, {} samples, {:.2} seconds",
            self.channels,
            self.sample_rate,
            self.bits_per_sample,
            self.bytes_per_sample,
            self.n_samples,
            self.duration
        )
    }
}

fn transmute_box<U>(raw: Box<[u8]>, bytes_to_read: usize) -> Box<[U]> {
    let raw_ptr = Box::into_raw(raw);
    let len = bytes_to_read / std::mem::size_of::<U>();
    unsafe { Box::from_raw(std::slice::from_raw_parts_mut(raw_ptr as *mut U, len)) }
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
            let err_msg = format!("Failed to create a valid u32 from buffer {:?} {:?}", buf, e);

            WaversErrorKind::Validate(crate::error::ValidationError::InvalidWavFile {
                reason: err_msg,
            })
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

// in src/lib.rs (or wherever your Wav type lives)
#[cfg(test)]
mod wav_tests {
    use super::*;
    use crate::wav_api_tests::read_text_to_vec;
    use audio_sample::{I24, Samples};
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
                    let samples: Box<[$ty]> = wav.read().expect("read");
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
                    let samples: Box<[$ty]> = wav.read().expect("read");
                    let fmt = wav.fmt_chunk();

                    // ── write & read back ─────────────────────────────────────────
                    crate::write::<_, $ty, $ty>(&out, &samples, fmt.sample_rate as u32, Some(fmt.channels))
                        .expect("write");
                    let mut reread = Wav::from_reader(BufReader::new(File::open(&out).unwrap()))
                        .expect("open reread");
                    let new_samples: Box<[$ty]> = reread.read().expect("read reread");

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
            assert!(
                Wav::new(BufReader::new(f)).is_ok(),
                "Failed to open WAV file"
            );
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
    #[cfg(not(feature = "ndarray"))]
    mod wav_read_tests {
        use crate::iter::WavIterators;

        use super::*;

        #[test]
        fn test_read_single_channel() {
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let samples: Box<[i16]> = wav.read().unwrap();
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
            let samples: Box<[i16]> = wav.read().unwrap();
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
            let samples: Box<[I24]> = wav.read().unwrap();
            let expected_i32: Vec<i32> =
                read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I24)).unwrap();

            let expected: Vec<I24> = expected_i32
                .into_iter()
                .map(|x| x >> 8)
                .map(I24::wrapping_from_i32)
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
            let samples: Box<[i16]> = wav.read().unwrap();
            let expected: Vec<f32> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_F32)).unwrap();
            assert_eq!(samples.len(), expected.len());
            for (i, (e, a)) in expected.iter().zip(samples).enumerate() {
                let a: f32 = a.convert_to();
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
            let all: Box<[i16]> = cmp.read().unwrap();
            let pos: usize = 1_000;
            wav.to_data();
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
        use crate::iter::WavIterators;

        use super::*;

        #[test]
        fn test_seek_by_samples() {
            let f = File::open(ONE_CHANNEL_WAV_I16).unwrap();
            let mut wav = Wav::from_reader(BufReader::new(f)).unwrap();
            let sr = wav.fmt_chunk().sample_rate as u64;
            wav.to_data();
            wav.seek_by_samples(sr).unwrap();
            let chunk: Box<[i16]> = wav.read_samples(100).unwrap();
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
            wav.to_data();
            let res = wav.seek_by_samples(total as u64 + 1);
            assert!(res.is_err());
        }
    }

    // ——— write tests —————————————————————————————————————
    #[cfg(not(feature = "ndarray"))]
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
            let samples: Box<[i16]> = wav.read().unwrap();
            let fmt = wav.fmt_chunk();

            // write
            crate::write::<_, i16, i16>(&out, &samples, fmt.sample_rate as u32, Some(fmt.channels))
                .unwrap();

            // read back
            let mut reread = Wav::from_reader(BufReader::new(File::open(&out).unwrap())).unwrap();
            let new: Box<[i16]> = reread.read().unwrap();
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
            let samples: Box<[f32]> = wav.read().unwrap();
            let fmt = wav.fmt_chunk();

            // write float
            crate::write::<&Path, f32, f32>(
                &out,
                &samples,
                fmt.sample_rate as u32,
                Some(fmt.channels),
            )
            .unwrap();

            // verify header
            let mut new_wav = Wav::from_reader(BufReader::new(File::open(&out).unwrap())).unwrap();
            let new_fmt = new_wav.fmt_chunk();
            assert_eq!(new_fmt.format, FormatCode::WavFormatIeeeFloat);
            assert_eq!(new_fmt.bits_per_sample, 32);

            // read back & compare
            let mut r: Wav<BufReader<File>> =
                Wav::from_reader(BufReader::new(File::open(&out).unwrap())).unwrap();
            let new: Box<[f32]> = r.read().unwrap();
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
