/// Module contains the core structs, ``Wav`` and ``Samples`` for working working with wav files.
use std::alloc::Layout;
use std::any::TypeId;
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::{Deref, DerefMut};
use std::path::Path;

use bytemuck::cast_slice;

#[cfg(feature = "ndarray")]
use ndarray::{Array, Array2};

use crate::chunks::{read_chunk, Chunk, ListChunk};
use crate::chunks::{DATA, FACT, LIST};
use crate::conversion::ConvertSlice;

use crate::conversion::{AudioSample, ConvertTo};
use crate::error::{WaversError, WaversResult};
use crate::header::{read_header, ChunkIdentifier, HeaderChunkInfo, WavHeader};
use crate::wav_type::WavType;
use crate::{i24, FactChunk, FmtChunk, FormatCode};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Trait representing a type that can be used to read and seek.
pub trait ReadSeek: Read + Seek {}

impl<T: Read + Seek> ReadSeek for T {}

/// Struct representing a wav file.
/// The struct contains a boxed reader and the header information of the wav file.
///
/// The boxed reader can be any type that implements the ``ReadSeek`` trait and is used to read the audio samples from the wav file when desired.
/// The header information is available after instantiating the struct and can be used to inspect the wav file.
pub struct Wav<T: AudioSample>
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
    _phantom: std::marker::PhantomData<T>,
    reader: Box<dyn ReadSeek>,
    pub wav_info: WavInfo,
}

impl<T: AudioSample> Wav<T>
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
    /// Construct a new Wav struct from a boxed reader.
    pub fn new(mut reader: Box<dyn ReadSeek>) -> WaversResult<Self> {
        let wav_info = read_header(&mut reader)?;

        let (data_offset, _) = wav_info.wav_header.data().into();
        let data_offset = data_offset + 8;
        reader.seek(SeekFrom::Start(data_offset as u64))?;

        Ok(Self {
            _phantom: std::marker::PhantomData,
            reader,
            wav_info,
        })
    }

    /// Construct a new Wav struct from a path.
    /// Uses a BufReader to read the file.
    pub fn from_path<P: AsRef<Path>>(path: P) -> WaversResult<Self> {
        let f = std::fs::File::open(path)?;
        let buf_reader: Box<dyn ReadSeek> = Box::new(std::io::BufReader::new(f));
        Self::new(buf_reader)
    }

    /// Read the audio samples from the wav file.
    /// The function will read all the samples remaining. If data has already been read using read_samples, this function will only read the remaining samples.
    ///
    /// Resets the reader to the start of the data chunk after it has finished reading.
    ///
    /// Returns a ``WaversResult`` containing a ``Samples`` struct or an error.
    ///
    /// # Example
    /// ```no_run
    /// use wavers::Wav;
    ///
    /// fn main() {
    ///     let wav: Wav<i16> = Wav::from_path("path/to/wav.wav").unwrap();
    ///     let samples: &[i16] = &wav.read().unwrap();
    /// }
    ///
    #[inline(always)]
    pub fn read(&mut self) -> WaversResult<Samples<T>> {
        let (data_offset, data_size_bytes) = self.header().data().into();
        let data_offset = data_offset + 8;
        let native_type = self.wav_info.wav_type;
        let native_size_bytes: usize = native_type.n_bytes();
        let number_of_samples_already_read = (self.reader.seek(SeekFrom::Current(0))?
            - data_offset as u64)
            / native_size_bytes as u64;

        let n_samples = data_size_bytes as usize / native_size_bytes;
        let samples = self.read_samples(n_samples - number_of_samples_already_read as usize)?;
        self.reader.seek(SeekFrom::Start(data_offset as u64))?;

        Ok(samples)
    }

    /// Read n_samples from the wav file.
    /// The function will read n_samples from the current position in the file.
    ///
    /// Reading can later be resumed.
    ///
    /// Returns a ``WaversResult`` containing a ``Samples`` struct or an error.
    #[inline(always)]
    pub fn read_samples(&mut self, n_samples: usize) -> WaversResult<Samples<T>> {
        let native_type = self.wav_info.wav_type;

        let native_size_bytes: usize = native_type.n_bytes();
        let n_native_bytes: usize = n_samples * native_size_bytes;

        let mut samples = alloc_box_buffer(n_native_bytes);
        self.reader.read_exact(&mut samples)?;

        let wav_type_from_file = self.wav_info.wav_type;

        let desired_type = WavType::try_from(TypeId::of::<T>())?;

        if wav_type_from_file == desired_type {
            return Ok(Samples::from(cast_slice::<u8, T>(&samples)));
        }

        match wav_type_from_file {
            WavType::Pcm16 | WavType::EPcm16 => {
                let samples: &[i16] = cast_slice::<u8, i16>(&samples);
                Ok(Samples::from(samples).convert())
            }
            WavType::Pcm24 | WavType::EPcm24 => {
                // Read the samples as i32 first and then convert
                let samples: &[i24] = cast_slice::<u8, i24>(&samples);
                Ok(Samples::from(samples).convert())
            }
            WavType::Pcm32 | WavType::EPcm32 => {
                let samples: &[i32] = cast_slice::<u8, i32>(&samples);
                Ok(Samples::from(samples).convert())
            }
            WavType::Float32 | WavType::EFloat32 => {
                let samples: &[f32] = cast_slice::<u8, f32>(&samples);
                Ok(Samples::from(samples).convert())
            }
            WavType::Float64 | WavType::EFloat64 => {
                let samples: &[f64] = cast_slice::<u8, f64>(&samples);
                Ok(Samples::from(samples).convert())
            }
        }
    }

    #[inline(always)]
    pub fn write<F: AudioSample, P: AsRef<Path>>(&mut self, p: P) -> WaversResult<()>
    where
        T: ConvertTo<F>,
        Box<[T]>: ConvertSlice<F>,
    {
        let samples = self.read()?.convert::<F>();
        let sample_bytes = samples.as_bytes();
        let fmt_chunk = self.wav_info.wav_header.fmt_chunk;
        self.wav_info.wav_header =
            WavHeader::new_header::<F>(fmt_chunk.sample_rate, fmt_chunk.channels, samples.len())?;

        let f = std::fs::File::create(p)?;
        let mut buf_writer: BufWriter<File> = BufWriter::new(f);
        let data_size_bytes = sample_bytes.len() as u32; // write up to the data size

        match self.format() {
            (FormatCode::WAV_FORMAT_PCM, FormatCode::WAV_FORMAT_PCM) => {
                let header_bytes = self.wav_info.wav_header.as_base_bytes();
                buf_writer.write_all(&header_bytes)?;
            }
            (FormatCode::WAV_FORMAT_IEEE_FLOAT, _) => {
                let header_bytes = self.wav_info.wav_header.as_extended_bytes(); // need to check if the format should always be marked as extended
                buf_writer.write_all(&header_bytes)?;
            }
            (FormatCode::WAVE_FORMAT_EXTENSIBLE, FormatCode::WAV_FORMAT_PCM) => {
                let header_bytes = self.wav_info.wav_header.as_cb_bytes();
                buf_writer.write_all(&header_bytes)?;
            }
            (FormatCode::WAVE_FORMAT_EXTENSIBLE, _) => {
                let header_bytes = self.wav_info.wav_header.as_extended_bytes();
                buf_writer.write_all(&header_bytes)?;
            }
            _ => {
                return Err(WaversError::UnsupportedWriteFormat(
                    self.format().0,
                    self.format().1,
                ));
            }
        };

        buf_writer.write_all(&DATA)?;
        buf_writer.write_all(&data_size_bytes.to_ne_bytes())?; // write the data size
        buf_writer.write_all(&sample_bytes)?; // write the data
        Ok(())
    }

    /// Returns the Fact chunk of the wav file if it is present.
    /// This function will return an error if there is an issue loading the chunk.
    /// This function will return as Some if the chunk is present and None if it is not.
    pub fn get_fact_chunk(&mut self) -> WaversResult<Option<FactChunk>> {
        self.get_chunk(FACT.into())
    }

    /// Returns the List chunk of the wav file if it is present.
    /// This function will return an error if there is an issue loading the chunk.
    /// This function will return as Some if the chunk is present and None if it is not.
    pub fn get_list_chunk(&mut self) -> WaversResult<Option<ListChunk>> {
        self.get_chunk(LIST.into())
    }

    /// Returns a Result containing an optional so as to allow the possibility of a missing fact chunk, but also to allow for the possibility of an error in reading that chunk.
    pub fn get_chunk<C: Chunk>(&mut self, id: ChunkIdentifier) -> WaversResult<Option<C>> {
        let header: &WavHeader = self.header();
        let chunk_info: HeaderChunkInfo = match header.get_chunk_info(id) {
            Some(info) => info.clone(),
            None => return Ok(None),
        };
        let chunk = read_chunk::<C>(&mut self.reader, &chunk_info)?;
        Ok(Some(chunk))
    }

    /// Returns a reference to the fmt chunk of the wav file.
    pub fn get_fmt_chunk(&self) -> &FmtChunk {
        &self.wav_info.wav_header.fmt_chunk
    }

    /// Returns the format of the wav file.
    /// Includes the main format and then a sub format which is important for the extensible format.
    pub fn format(&self) -> (FormatCode, FormatCode) {
        let fmt_chunk = self.header().fmt_chunk;
        let format = fmt_chunk.format;
        let sub_format = fmt_chunk.format();
        (format, sub_format)
    }

    /// Returns a reference header of the wav file.
    pub fn header(&self) -> &WavHeader {
        &self.wav_info.wav_header
    }

    /// Returns a mutable reference to the header of the wav file.
    pub fn header_mut(&mut self) -> &mut WavHeader {
        &mut self.wav_info.wav_header
    }

    /// Returns the encoding of the wav file.
    pub fn encoding(&self) -> WavType {
        self.wav_info.wav_type
    }

    /// Returns the sample rate of the wav file.
    pub fn sample_rate(&self) -> i32 {
        self.header().fmt_chunk.sample_rate
    }

    /// Returns the number of channels of the wav file.
    pub fn n_channels(&self) -> u16 {
        self.header().fmt_chunk.channels
    }

    /// Returns the number of samples in the wav file.
    pub fn n_samples(&self) -> usize {
        let (_, native_data_size_bytes) = self.header().data().into();
        let size_of_native_bytes = self.header().fmt_chunk.bits_per_sample as usize / 8;
        native_data_size_bytes as usize / size_of_native_bytes
    }

    /// Returns the duration of the wav file in seconds.
    pub fn duration(&self) -> u32 {
        let data_size = self.header().data().size;

        let sample_rate = self.sample_rate() as u32;
        let n_channels = self.n_channels() as u32;
        let bytes_per_sample = (self.header().fmt_chunk.bits_per_sample / 8) as u32;

        data_size / (sample_rate * n_channels * bytes_per_sample)
    }

    /// Returns the sample rate, number of channels, duration and encoding of a wav file.
    pub fn wav_spec(&self) -> (u32, WavHeader) {
        let duration = self.duration();
        (duration, self.header().clone())
    }
}

/// Returns the sample rate, number of channels, duration and encoding of a wav file.
/// Convenmience function which opens the wav file and reads the header.
pub fn wav_spec<P: AsRef<Path>>(p: P) -> WaversResult<(u32, WavHeader)> {
    let wav = Wav::<i16>::from_path(p)?;
    Ok(wav.wav_spec())
}

/// Struct representing the information of a wav file.
#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavInfo {
    pub wav_type: WavType, // the type of the wav file
    pub wav_header: WavHeader,
}

/// Struct representing the information of a wav file.
/// This struct is used when the pyo3 feature is enabled.
#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavInfo {
    pub wav_type: WavType, // the type of the wav file
    pub wav_header: WavHeader,
}

#[cfg(feature = "ndarray")]
use crate::conversion::{AsNdarray, IntoNdarray};

#[cfg(feature = "ndarray")]
impl<T: AudioSample> IntoNdarray for Wav<T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Target = T;

    fn into_ndarray(mut self) -> WaversResult<(Array2<Self::Target>, i32)> {
        let n_channels = self.header().fmt_chunk.channels as usize;

        let copied_data: &[T] = &self.read()?.samples;
        let length = copied_data.len();
        let shape = (length / n_channels, n_channels);

        let arr: Array2<T> = Array::from_shape_vec(shape, copied_data.to_vec())?;
        Ok((arr, self.sample_rate()))
    }
}

#[cfg(feature = "ndarray")]
impl<T: AudioSample> AsNdarray for Wav<T>
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Target = T;

    fn as_ndarray(&mut self) -> WaversResult<(Array2<Self::Target>, i32)> {
        let n_channels = self.header().fmt_chunk.channels as usize;
        let copied_data: Box<[T]> = self.read()?.samples.to_owned();
        let copied_data: &[T] = &copied_data;
        let length = copied_data.len();

        let shape = (length / n_channels, n_channels);
        let arr: Array2<T> = Array::from_shape_vec(shape, copied_data.to_vec())?;
        Ok((arr, self.sample_rate()))
    }
}

/// Wrapper struct around a boxed slice of samples.
/// Wrapping allows the extension of the struct to include more functionality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Samples<T>
where
    T: AudioSample,
{
    pub(crate) samples: Box<[T]>,
}

impl<T> AsRef<[T]> for Samples<T>
where
    T: AudioSample,
{
    fn as_ref(&self) -> &[T] {
        &self.samples
    }
}

impl<T> AsMut<[T]> for Samples<T>
where
    T: AudioSample,
{
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.samples
    }
}

impl<T> Deref for Samples<T>
where
    T: AudioSample,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.samples
    }
}

impl<T> DerefMut for Samples<T>
where
    T: AudioSample,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.samples
    }
}

// From Vec
impl<T> From<Vec<T>> for Samples<T>
where
    T: AudioSample,
{
    fn from(samples: Vec<T>) -> Self {
        Samples {
            samples: samples.into_boxed_slice(),
        }
    }
}

// From Slice
impl<T> From<&[T]> for Samples<T>
where
    T: AudioSample,
{
    fn from(samples: &[T]) -> Self {
        Samples {
            samples: Box::from(samples),
        }
    }
}

// From boxed slice
impl<T> From<Box<[T]>> for Samples<T>
where
    T: AudioSample,
{
    fn from(samples: Box<[T]>) -> Self {
        Samples { samples }
    }
}

// From u8 Buffer
impl<T> From<&[u8]> for Samples<T>
where
    T: AudioSample,
{
    fn from(bytes: &[u8]) -> Self {
        let casted_samples: &[T] = cast_slice::<u8, T>(bytes);
        Samples {
            samples: Box::from(casted_samples),
        }
    }
}

impl<T> Display for Samples<T>
where
    T: AudioSample + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.samples)
    }
}

impl<T> Samples<T>
where
    T: AudioSample,
{
    /// Construct a new Samples struct from a boxed slice of audio samples.
    pub fn new(samples: Box<[T]>) -> Self {
        Self { samples }
    }

    /// Conversts the samples to the specified type ``F``. If the type is the same as the current type, the function will return self.
    /// The function will consume the current Samples struct and return a new Samples struct with the specified type.
    #[inline(always)]
    pub fn convert<F: AudioSample>(self) -> Samples<F>
    where
        T: ConvertTo<F>,
        Box<[T]>: ConvertSlice<F>,
    {
        // Quick check to see if we're converting to the same type, if so, just return self
        if TypeId::of::<T>() == TypeId::of::<F>() {
            let data: Box<[T]> = self.samples.clone();
            return Samples {
                samples: Box::from(cast_slice::<T, F>(&data)),
            };
        }
        let converted_samples = self.samples.convert_slice();
        Samples {
            samples: converted_samples,
        }
    }

    /// Converts the boxed slice of samples to the corresponding bytes.
    pub fn as_bytes(&self) -> &[u8] {
        cast_slice::<T, u8>(&self.samples)
    }
}

impl Samples<i16> {}
impl Samples<i24> {}
impl Samples<i32> {}
impl Samples<f32> {}
impl Samples<f64> {}

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

/// Helper function to allocate a fixed sized, heap allocated buffer of type T.
pub(crate) fn alloc_sample_buffer<T>(len: usize) -> Box<[T]>
where
    T: AudioSample + Copy + Debug,
{
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

#[cfg(test)]
mod core_tests {
    use super::*;

    #[cfg(feature = "ndarray")]
    use crate::IntoNdarray;

    use approx_eq::assert_approx_eq;
    use std::{io::BufRead, str::FromStr};

    const ONE_CHANNEL_WAV_I16: &str = "./test_resources/one_channel_i16.wav";
    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";

    const ONE_CHANNEL_WAV_I24: &str = "./test_resources/one_channel_i24.wav";
    const ONE_CHANNEL_EXPECTED_I24: &str = "./test_resources/one_channel_i24.txt";

    const ONE_CHANNEL_EXPECTED_I16: &str = "./test_resources/one_channel_i16.txt";
    const ONE_CHANNEL_EXPECTED_F32: &str = "./test_resources/one_channel_f32.txt";

    const TEST_OUTPUT: &str = "./test_resources/tmp/";

    #[test]
    pub fn duration_one_channel() {
        let wav: Wav<i16> = Wav::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let duration = wav.duration();
        assert_eq!(duration, 10, "Expected duration of 10 seconds");
    }

    #[test]
    pub fn duration_two_channel() {
        let wav: Wav<i16> = Wav::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let duration = wav.duration();
        assert_eq!(duration, 10, "Expected duration of 10 seconds");
    }

    /// Test that the number of samples is correct after converting.
    /// The wav file used is encoded as PCM_16 and has 2048 samples
    /// Comes from https://github.com/jmg049/wavers/issues/9
    #[test]
    pub fn n_samples_after_convert() {
        let wav: Wav<f32> = Wav::from_path("test_resources/n_samples_test.wav").unwrap();
        assert_eq!(wav.n_samples(), 2048, "Expected 2048 samples");
    }

    #[test]
    fn i16_i32_convert() {
        let mut wav: Wav<i32> = Wav::from_path(ONE_CHANNEL_WAV_I16).unwrap();

        let wav_i32: &[i32] = &wav.read().unwrap();

        let expected_i32_samples: &[i32] =
            &Wav::<i32>::from_path("test_resources/one_channel_i32.wav")
                .unwrap()
                .read()
                .unwrap();

        assert_eq!(
            expected_i32_samples.len(),
            wav.n_samples(),
            "Lengths not equal"
        );

        for (expected, actual) in expected_i32_samples.iter().zip(wav_i32.iter()) {
            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn wav_as_ndarray() {
        let wav: Wav<i16> =
            Wav::<i16>::from_path(ONE_CHANNEL_WAV_I16).expect("Failed to read file");

        let expected_wav: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();

        let (arr, sr) = wav.into_ndarray().unwrap();
        assert_eq!(arr.shape()[0], 1, "Expected 1 channels");
        for (expected, actual) in expected_wav.iter().zip(arr) {
            assert_eq!(*expected, actual, "{} != {}", expected, actual);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn two_channel_as_ndarray() {
        let wav: Wav<i16> =
            Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).expect("Failed to open file");
        let expected_wav: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();
        let mut new_expected = Vec::with_capacity(expected_wav.len() * 2);
        for sample in expected_wav {
            new_expected.push(sample);
            new_expected.push(sample);
        }

        let expected_wav = new_expected;

        let (two_channel_arr, sr): (Array2<i16>, i32) = wav.into_ndarray().unwrap();
        assert_eq!(two_channel_arr.shape()[0], 2, "Expected 2 channels");
        for (expected, actual) in expected_wav.iter().zip(two_channel_arr) {
            assert_eq!(*expected, actual, "{} != {}", expected, actual);
        }
    }

    #[test]
    fn primitive_to_u8_slice() {
        let mut test_data: Vec<i16> = Vec::with_capacity(8);
        for i in 0..8 {
            test_data.push(i);
        }
        let s_data: &[i16] = test_data.as_slice();
        let samples: Samples<i16> = Samples::from(s_data);
        let bytes = samples.as_bytes();

        let mut expected_bytes: Vec<u8> = Vec::with_capacity(16);
        for i in 0..8i16 {
            let b: [u8; 2] = i.to_ne_bytes();
            expected_bytes.extend_from_slice(&b);
        }

        for (expected, actual) in expected_bytes.iter().zip(bytes.iter()) {
            assert_eq!(
                *expected, *actual,
                "Expected: {}, Actual: {}",
                expected, actual
            );
        }
    }

    #[test]
    fn read_and_convert() {
        let expected_samples =
            read_text_to_vec::<f32>(Path::new(ONE_CHANNEL_EXPECTED_F32)).unwrap();

        let mut wav: Wav<f32> = Wav::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let samples: &[f32] = &wav.read().unwrap();

        assert_eq!(wav.n_samples(), expected_samples.len(), "Lengths not equal");
        for (expected, actual) in expected_samples.iter().zip(samples) {
            assert_approx_eq!(*expected as f64, *actual as f64, 1e-4);
        }
    }

    #[test]
    fn convert_write_read() {
        if !Path::new(TEST_OUTPUT).exists() {
            std::fs::create_dir(Path::new(TEST_OUTPUT)).unwrap();
        }

        let mut og_wav: Wav<f32> = Wav::<f32>::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let out_fp = format!("{}{}", TEST_OUTPUT, "convert_write_read.wav");
        og_wav.write::<f32, _>(Path::new(&out_fp)).unwrap();

        let mut wav: Wav<f32> = Wav::<f32>::from_path(&out_fp).unwrap();
        let actual_samples: &[f32] = &wav.read().unwrap();

        assert_ne!(
            wav.header().current_file_size,
            og_wav.header().current_file_size,
            "File sizes are equal and they shouldn't be"
        );

        let expected_samples =
            read_text_to_vec::<f32>(Path::new(ONE_CHANNEL_EXPECTED_F32)).unwrap();

        assert_eq!(wav.n_samples(), expected_samples.len(), "Lengths not equal");
        for (expected, actual) in expected_samples.iter().zip(actual_samples) {
            assert_approx_eq!(*expected as f64, *actual as f64, 1e-4);
        }
        std::fs::remove_file(Path::new(&out_fp)).unwrap();
    }

    #[test]
    fn can_read_two_channel() {
        let mut wav: Wav<i16> = Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).unwrap();
        let expected_samples =
            read_text_to_vec::<i16>(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();
        let mut new_expected = Vec::with_capacity(expected_samples.len() * 2);
        for sample in expected_samples {
            new_expected.push(sample);
            new_expected.push(sample);
        }

        let expected_samples = new_expected;
        assert_eq!(wav.n_samples(), expected_samples.len(), "Lengths not equal");
        for (expected, actual) in expected_samples.iter().zip(wav.read().unwrap().as_ref()) {
            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
        }
    }

    #[test]
    fn read_some_then_read_remainder() {
        let mut wav: Wav<i16> =
            Wav::from_path(ONE_CHANNEL_WAV_I16).expect("Failed to open wav file");
        let first_second_samples = wav.read_samples(wav.sample_rate() as usize).unwrap();
        assert_eq!(
            first_second_samples.len(),
            wav.sample_rate() as usize,
            "Lengths not equal"
        );

        let remaining_samples = wav.read().unwrap();
        assert_eq!(
            remaining_samples.len(),
            wav.n_samples() - wav.sample_rate() as usize
        );

        let all_samples = wav.read().unwrap();
        assert_eq!(all_samples.len(), wav.n_samples());
    }

    #[test]
    fn read_i24_correctly() {
        let mut wav: Wav<i24> = Wav::from_path(ONE_CHANNEL_WAV_I24).expect("Failed to open file");
        let samples: &[i24] = &wav.read().unwrap();

        // Expected values have been coming from SoundFile in Python as it is a tried and tested library
        // However, the values were written using Numpy and Numpy doesn't support i24 so it writes it as an i32
        // Hence the need to convert here. Relies on the assumption that the i24 conversion functions are correct.
        let expected_samples: Vec<i32> =
            read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I24)).unwrap();
        let expected_samples: Box<[i32]> = expected_samples.into_boxed_slice();
        let expected_samples: Box<[i24]> = expected_samples.convert_slice();

        assert_eq!(samples.len(), expected_samples.len(), "Lengths not equal");

        for (idx, (expected, actual)) in expected_samples.iter().zip(samples).enumerate() {
            assert_eq!(*expected, *actual, "{} != {} at {}", expected, actual, idx);
        }
    }

    #[test]
    fn write_i24_correctly() {
        let mut wav: Wav<i16> = Wav::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let out_fp = format!("{}{}", TEST_OUTPUT, "write_i24_correctly.wav");
        wav.write::<i24, _>(Path::new(&out_fp)).unwrap();

        let mut wav: Wav<i24> = Wav::from_path(&out_fp).unwrap();
        let samples: &[i24] = &wav.read().unwrap();

        let expected_samples: Vec<i32> =
            read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I24)).unwrap();
        let expected_samples: Box<[i32]> = expected_samples.into_boxed_slice();
        let expected_samples: Box<[i24]> = expected_samples.convert_slice();

        assert_eq!(samples.len(), expected_samples.len(), "Lengths not equal");

        for (idx, (expected, actual)) in expected_samples.iter().zip(samples).enumerate() {
            assert_eq!(*expected, *actual, "{} != {} at {}", expected, actual, idx);
        }

        std::fs::remove_file(Path::new(&out_fp)).unwrap();
    }

    fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(std::io::BufReader::new(file).lines())
    }

    fn read_text_to_vec<T: FromStr>(fp: &Path) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        <T as FromStr>::Err: std::error::Error + 'static,
    {
        let mut data = Vec::new();
        let lines = read_lines(fp)?;
        for line in lines {
            let line = line?;
            for sample in line.split(" ") {
                let parsed_sample: T = match sample.trim().parse::<T>() {
                    Ok(num) => num,
                    Err(err) => {
                        eprintln!("Failed to parse {}", sample);
                        panic!("{}", err)
                    }
                };
                data.push(parsed_sample);
            }
        }
        Ok(data)
    }
}
