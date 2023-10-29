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

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::conversion::ConvertSlice;

use crate::conversion::{AudioSample, ConvertTo};
use crate::error::{WaversError, WaversResult};
use crate::header::{read_header, WavHeader, DATA};

/// Trait used to indicate that the implementing type can be read and seeked through.
/// It is just a marker to allow for types that implement Read and Seek to be used.
///
/// For example: ``std::fs::File``, ``std::io::BufReader<T: ReadSeek>``
pub trait ReadSeek: Read + Seek {}

impl ReadSeek for std::fs::File {}
impl<T: ReadSeek> ReadSeek for std::io::BufReader<T> {}

/// Represents a wav file. It contains the header information and a reader (essentially a file pointer of some type).
/// Upon creation, does not read the entire file into memory, but only reads the header information.
/// The type T is the type that you want to read the samples as, NOT the type of the samples in the file.
///
/// Example
/// ```
/// use wavers::Wav;
///
/// fn main() {
///     // Creates a Wav struct which will read as samples as i16
///     let mut wav_i16: Wav<i16> = Wav::from_path("./test_resources/one_channel_i16.wav").unwrap();
///
///     // Creates a new Wav struct, using a file that we know is encoded as PCM_16, but which will read the samplea as 32-bit floats.
///     let mut wav_f32: Wav<f32> = Wav::from_path("./test_resources/one_channel_i16.wav").unwrap();
///
///     // We can then access the audio samples by calling read on the Wav struct. Needs to be mut as we update the reader position.
///     let samples_i16: &[i16] = &wav_i16.read().unwrap();
///
///     let samples_f32: &[f32] = &wav_f32.read().unwrap();
/// }
///
/// ```
///
pub struct Wav<T: AudioSample>
where
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
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
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
{
    /// Creates a new Wav struct from a reader that implements Read and Seek.
    ///
    /// Returns a result with the Wav struct if successful, or an error if not.
    ///
    /// # Example
    /// ```
    /// use wavers::Wav;
    /// use std::fs::File;
    ///
    /// fn main() {
    ///     let fp: File = File::open("./test_resources/one_channel_i16.wav").unwrap();
    ///     let wav = Wav::<i16>::new(Box::new(fp)).unwrap();
    /// }
    /// ```
    ///
    pub fn new(mut reader: Box<dyn ReadSeek>) -> WaversResult<Self> {
        let wav_info = read_header(&mut reader)?;

        let (data_offset, _) = wav_info.wav_header.data().into();

        reader.seek(SeekFrom::Start(data_offset as u64))?;
        let (_, _, _n_bytes) = wav_info.wav_type.into();

        Ok(Self {
            _phantom: std::marker::PhantomData,
            reader,
            wav_info,
        })
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> WaversResult<Self> {
        let f = std::fs::File::open(path)?;
        let buf_reader: Box<dyn ReadSeek> = Box::new(std::io::BufReader::new(f));
        Self::new(buf_reader)
    }

    /// Convenience function which creates a new Wav struct from anything that can be turned into a Path reference.
    /// This will open the file and create a BufReader from it.
    ///
    /// Returns a result with the Wav struct if successful, or an error if not.
    ///
    /// # Example
    /// ```no_run
    /// use wavers::Wav;
    ///
    /// fn main() {
    ///    let wav = Wav::<i16>::from_path("/path/to/wav.wav").unwrap();
    /// }
    /// ```
    pub fn from_path<P: AsRef<Path>>(path: P) -> WaversResult<Self> {
        let f = std::fs::File::open(path)?;
        let buf_reader: Box<dyn ReadSeek> = Box::new(std::io::BufReader::new(f));
        Self::new(buf_reader)
    }

    pub fn convert<F: AudioSample>(self) -> WaversResult<Wav<F>>
    where
        T: ConvertTo<F>,
        i16: ConvertTo<F>,
        i32: ConvertTo<F>,
        f32: ConvertTo<F>,
        f64: ConvertTo<F>,
    {
        let sr = self.sample_rate();
        let n_channels = self.n_channels();
        let n_samples = self.n_samples();
        let new_header = WavHeader::new_header::<F>(sr, n_channels, n_samples)?;
        println!("{:#?}", new_header);
        let new_type: WavType = TypeId::of::<F>().try_into()?;
        let new_info = WavInfo {
            wav_type: new_type,
            wav_header: new_header,
        };
        println!("samples = {}", n_samples);
        Ok(Wav::<F> {
            _phantom: std::marker::PhantomData,
            reader: self.reader,
            wav_info: new_info,
        })
    }

    /// Function used to read the audio samples contained with the audio file.
    /// Reads the entire file into memory and returns a ``Samples`` struct or an error.
    /// The ``Samples`` struct contains a boxed slice of the samples.
    #[inline(always)]
    pub fn read(&mut self) -> WaversResult<Samples<T>> {
        let (_, data_size_bytes) = self.header().data().into();
        let n_samples = data_size_bytes as usize / std::mem::size_of::<T>();
        self.read_samples(n_samples)
    }

    ///
    /// Function used to read a specified number of samples from the audio file.
    /// Reads the specified number of samples into memory and returns a ``Samples`` struct or an error.
    /// The file pointer will be updated to the position after the read samples. Therefore reading can be resumed from the current position.
    #[inline(always)]
    pub fn read_samples(&mut self, n_samples: usize) -> WaversResult<Samples<T>> {
        let n_bytes = n_samples * std::mem::size_of::<T>();
        println!("samples n_bytes: {}", n_bytes);
        println!("n_bytes = {}", n_bytes);
        println!("Size of T: {}", std::mem::size_of::<T>());
        let mut samples = alloc_box_buffer(n_bytes);
        self.reader.read_exact(&mut samples)?;

        let wav_type_from_file = self.wav_info.wav_type;

        let desired_type: WavType = TypeId::of::<T>().try_into()?;
        if wav_type_from_file == desired_type {
            return Ok(Samples::from(cast_slice::<u8, T>(&samples)));
        }

        match wav_type_from_file {
            WavType::Pcm16 => {
                let samples: &[i16] = cast_slice::<u8, i16>(&samples);
                Ok(Samples::from(samples).convert())
            }
            WavType::Pcm32 => {
                let samples: &[i32] = cast_slice::<u8, i32>(&samples);
                Ok(Samples::from(samples).convert())
            }
            WavType::Float32 => {
                let samples: &[f32] = cast_slice::<u8, f32>(&samples);
                Ok(Samples::from(samples).convert())
            }
            WavType::Float64 => {
                let samples: &[f64] = cast_slice::<u8, f64>(&samples);
                Ok(Samples::from(samples).convert())
            }
        }
    }

    ///
    /// Writes the current data of the Wav struct to a file.
    /// The type ``F`` is the type that the samples will be written as. This will be reflected in the encoding of the file in the Wav Header.
    ///
    /// Returns a result with nothing if successful, or an error if not.
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

        let header_bytes = self.header().as_bytes();

        let f = std::fs::File::create(p)?;
        let mut buf_writer: BufWriter<File> = BufWriter::new(f);
        let data_size_bytes = sample_bytes.len() as u32; // write up to the data size

        buf_writer.write_all(&header_bytes)?;
        buf_writer.write_all(&DATA)?;
        buf_writer.write_all(&data_size_bytes.to_ne_bytes())?; // write the data size
        buf_writer.write_all(&sample_bytes)?; // write the data
        Ok(())
    }

    /// Convenience function for accessing the header of the Wav file
    pub fn header(&self) -> &WavHeader {
        &self.wav_info.wav_header
    }

    /// Convenience function for accessing (and potentially mutating) the header of the Wav file
    pub fn header_mut(&mut self) -> &mut WavHeader {
        &mut self.wav_info.wav_header
    }

    /// Convenience function for accessing the encoding of the Wav file
    pub fn encoding(&self) -> WavType {
        self.wav_info.wav_type
    }

    /// Convenience function for accessing the sample rate of the Wav file
    pub fn sample_rate(&self) -> i32 {
        self.header().fmt_chunk.sample_rate
    }

    /// Convenience function for accessing the number of channels of the Wav file
    pub fn n_channels(&self) -> u16 {
        self.header().fmt_chunk.channels
    }

    /// Convenience function for accessing the number of samples of the Wav file
    pub fn n_samples(&self) -> usize {
        let (_, data_size_bytes) = self.header().data().into();
        data_size_bytes as usize / std::mem::size_of::<T>()
    }

    /// Convenience function for calculating the duration of the Wav file
    pub fn duration(&self) -> u32 {
        let data_size = self.header().data().size;

        let sample_rate = self.sample_rate() as u32;
        let n_channels = self.n_channels() as u32;
        let bytes_per_sample = (self.header().fmt_chunk.bits_per_sample / 8) as u32;

        data_size / (sample_rate * n_channels * bytes_per_sample)
    }

    /// Convenience function for accessing the sample rate, number of channels and duration of the Wav file
    pub fn wav_spec(&self) -> (i32, u16, u32) {
        let sample_rate = self.sample_rate();
        let n_channels = self.n_channels();
        let duration = self.duration();
        (sample_rate, n_channels, duration)
    }
}

/// Convenience function for accessing the sample rate, number of channels, duration and encoding of the Wav file without having to create one.
pub fn wav_spec<P: AsRef<Path>>(p: P) -> WaversResult<(i32, u16, u32, WavType)> {
    let wav = Wav::<i16>::from_path(p)?;
    let native_encoding = wav.wav_info.wav_type;
    let (sr, nc, d) = wav.wav_spec();
    Ok((sr, nc, d, native_encoding))
}

/// Struct containing metadata of a Wav file, i.e. it's encoding and its header
#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavInfo {
    pub wav_type: WavType, // the type of the wav file
    pub wav_header: WavHeader,
}

/// Struct containing metadata of a Wav file, i.e. it's encoding and its header
/// This struct is used when the pyo3 feature is enabled.
#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavInfo {
    pub wav_type: WavType, // the type of the wav file
    pub wav_header: WavHeader,
}

impl Into<(WavHeader, WavType)> for WavInfo {
    fn into(self) -> (WavHeader, WavType) {
        (self.wav_header, self.wav_type)
    }
}

/// Used to convert a TypeId into an encoding.
impl TryInto<WavType> for TypeId {
    type Error = WaversError;

    fn try_into(self) -> Result<WavType, Self::Error> {
        if self == TypeId::of::<i16>() {
            Ok(WavType::Pcm16)
        } else if self == TypeId::of::<i32>() {
            Ok(WavType::Pcm32)
        } else if self == TypeId::of::<f32>() {
            Ok(WavType::Float32)
        } else if self == TypeId::of::<f64>() {
            Ok(WavType::Float64)
        } else {
            Err(WaversError::InvalidType(format!("Invalid type {:?}", self)))
        }
    }
}

/// Enum which represents the encoding of a Wav file.
#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavType {
    Pcm16,
    Pcm32,
    Float32,
    Float64,
}

/// Enum which represents the encoding of a Wav file.
/// This enum is used when the pyo3 feature is enabled.
#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavType {
    Pcm16,
    Pcm32,
    Float32,
    Float64,
}

impl TryFrom<(u16, u16)> for WavType {
    type Error = WaversError;

    fn try_from(value: (u16, u16)) -> Result<Self, Self::Error> {
        Ok(match value {
            (1, 16) => WavType::Pcm16,
            (1, 32) => WavType::Pcm32,
            (3, 32) => WavType::Float32,
            (3, 64) => WavType::Float64,
            _ => return Err(WaversError::InvalidType("Invalid wav type".into())),
        })
    }
}

impl Into<TypeId> for WavType {
    fn into(self) -> TypeId {
        match self {
            WavType::Pcm16 => TypeId::of::<i16>(),
            WavType::Pcm32 => TypeId::of::<i32>(),
            WavType::Float32 => TypeId::of::<f32>(),
            WavType::Float64 => TypeId::of::<f64>(),
        }
    }
}

impl Into<(u16, u16, u16)> for WavType {
    fn into(self) -> (u16, u16, u16) {
        match self {
            WavType::Pcm16 => (1, 16, 2),
            WavType::Pcm32 => (1, 32, 4),
            WavType::Float32 => (3, 32, 4),
            WavType::Float64 => (3, 64, 8),
        }
    }
}

#[cfg(feature = "ndarray")]
use crate::conversion::{AsNdarray, IntoNdarray};

/// Implementation of the IntoNdarray trait for the Wav struct.
/// This function consumes the Wav struct and returns an ndarray and the sample rate.
///
/// Reads the Wav file and converts it to an ndarray.
/// The functionality is associated with the struct rather than
/// an individual function as the Wav struct contains channel
/// information which would otherwise have to be specified.
#[cfg(feature = "ndarray")]
impl<T: AudioSample> IntoNdarray for Wav<T>
impl<T: AudioSample> IntoNdarray for Wav<T>
where
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Target = T;

    fn into_ndarray(mut self) -> WaversResult<(Array2<Self::Target>, i32)> {
        let n_channels = self.n_channels() as usize;
        let sr = self.sample_rate();

        let copied_data: &[T] = &self.read()?.samples;
        let length = copied_data.len();
        let shape = (n_channels, length / n_channels);

        let arr: Array2<T> = Array::from_shape_vec(shape, copied_data.to_vec())?;
        Ok((arr, sr))
    }
}

/// Implementation of the AsNdarray trait for the Wav struct.
/// This function do not consume the Wav struct and returns an ndarray and the sample rate.
///
/// Reads the Wav file and converts it to an ndarray.
/// The functionality is associated with the struct rather than
/// an individual function as the Wav struct contains channel
/// information which would otherwise have to be specified.
#[cfg(feature = "ndarray")]
impl<T: AudioSample> AsNdarray for Wav<T>
impl<T: AudioSample> AsNdarray for Wav<T>
where
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Target = T;

    fn as_ndarray(&mut self) -> WaversResult<(Array2<Self::Target>, i32)> {
        let n_channels = self.n_channels() as usize;
        let sr = self.sample_rate();
        let copied_data: Box<[T]> = self.read()?.samples.to_owned();
        let copied_data: &[T] = &copied_data;
        let length = copied_data.len();

        let shape = (n_channels, length / n_channels);
        let arr: Array2<T> = Array::from_shape_vec(shape, copied_data.to_vec())?;
        Ok((arr, sr))
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
impl Samples<i32> {}
impl Samples<f32> {}
impl Samples<f64> {}

/// Function to create a fixed-size heap allocated block of bytes.
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

/// Function to create a fixed-size heap allocated block of samples (sizeof::<T> * len).
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

    use crate::read;

    #[cfg(feature = "ndarray")]
    use crate::IntoNdarray;

    use crate::read;

    use approx_eq::assert_approx_eq;
    use std::{fs::File, io::BufRead, path::Path, str::FromStr};

    const ONE_CHANNEL_WAV_I16: &str = "./test_resources/one_channel_i16.wav";
    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";

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

    #[test]
    fn i16_i32_convert() {
        let mut wav: Wav<i32> = Wav::from_path(ONE_CHANNEL_WAV_I16).unwrap();

        let wav_i32: &[i32] = &wav.read().unwrap();

        let expected_i32_samples: &[i32] =
            &Wav::<i32>::from_path("test_resources/one_channel_i32.wav")
                .unwrap()
                .read()
                .unwrap();

        for (expected, actual) in expected_i32_samples.iter().zip(wav_i32.iter()) {
            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn wav_as_ndarray() {
        let wav: Wav<i16> =
            Wav::<i16>::from_path(ONE_CHANNEL_WAV_I16).expect("Failed to read file");

        let wav: Wav<i16> =
            Wav::<i16>::from_path(ONE_CHANNEL_WAV_I16).expect("Failed to read file");

        let expected_wav: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();

        let (arr, sr) = wav.into_ndarray().unwrap();
        assert_eq!(sr, 16000, "sr != 16000");
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
        let wav: Wav<i16> =
            Wav::<i16>::from_path(TWO_CHANNEL_WAV_I16).expect("Failed to open file");
        let expected_wav: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();
        let mut new_expected = Vec::with_capacity(expected_wav.len() * 2);
        for sample in expected_wav {
            new_expected.push(sample);
            new_expected.push(sample);
        }

        let expected_wav = new_expected;

        let (two_channel_arr, sr) = wav.into_ndarray().unwrap();
        assert_eq!(sr, 16000, "sr != 16000");

        assert_eq!(two_channel_arr.shape()[0], 2, "Expected 2 channels");
        for (expected, actual) in std::iter::zip(expected_wav, two_channel_arr) {
            assert_eq!(expected, actual, "{} != {}", expected, actual);
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

    use std::stringify;
    macro_rules! read_tests {
        ($($T:ident), *) => {
            $(
                paste::item! {
                    #[test]
                    fn [<read_$T>]() {
                        let t_string: &str = stringify!($T);

                        let wav_str = format!("./test_resources/one_channel_{}.wav", t_string);
                        let expected_str = format!("./test_resources/one_channel_{}.txt", t_string);


                        let sample_data: &[$T] = &match read::<$T, _>(&wav_str) {
                            Ok(w) => w,
                            Err(e) => {eprintln!("{}\n{}", wav_str, e); panic!("Failed to read wav file")}
                        };

                        let expected_data: Vec<$T> = match read_text_to_vec(Path::new(&expected_str)) {
                            Ok(w) => w,
                            Err(e) => {eprintln!("{}\n{}", wav_str, e); panic!("Failed to read txt file")}
                        };

                        for (expected, actual) in expected_data.iter().zip(sample_data.iter()) {
                            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
                        }
                    }
                }
            )*
        }
    }
    read_tests!(i16, i32, f32, f64);

    macro_rules! write_tests {
        ($($T:ident), *) => {
            $(
                paste::item! {
                    #[test]
                    fn [<write_$T>]() {
                        if !Path::new(TEST_OUTPUT).exists() {
                            std::fs::create_dir(TEST_OUTPUT).unwrap();
                        }
                        let t_string: &str = stringify!($T);

                        let wav_str = format!("./test_resources/one_channel_{}.wav", t_string);
                        let expected_str = format!("./test_resources/one_channel_{}.txt", t_string);

                        let mut wav: Wav<$T> =
                            Wav::from_path(wav_str).expect("Failed to create wav file");
                        let expected_samples: Samples<$T> = Samples::from(
                            read_text_to_vec(&Path::new(&expected_str)).expect("Failed to read to vec"),
                        );


                        let out = format!("{}_one_channel_{}.wav", TEST_OUTPUT, t_string);
                        let out_path = Path::new(&out);

                        wav.write::<$T, _>(out_path)
                            .expect("Failed to write file");

                        let mut new_wav: Wav<$T> = Wav::<$T>::from_path(out_path).unwrap();

                        for (expected, actual) in expected_samples
                            .iter()
                            .zip(new_wav.read().unwrap().iter())
                        {
                            assert_eq!(expected, actual, "{} != {}", expected, actual);
                        }
                        std::fs::remove_file(Path::new(&out_path)).unwrap();
                    }
                }
            )*
        };
    }

    write_tests!(i16, i32, f32, f64);

    #[test]
    fn read_and_convert() {
        let expected_samples =
            read_text_to_vec::<f32>(Path::new(ONE_CHANNEL_EXPECTED_F32)).unwrap();

        let mut wav: Wav<f32> = Wav::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let samples: &[f32] = &wav.read().unwrap();
        for (expected, actual) in expected_samples.iter().zip(samples) {
            assert_approx_eq!(*expected as f64, *actual as f64, 1e-4);
        }
    }

    #[test]
    fn convert_write_read() {
        if !Path::new(TEST_OUTPUT).exists() {
            std::fs::create_dir(Path::new(TEST_OUTPUT)).unwrap();
        }

        let mut wav: Wav<f32> = Wav::<f32>::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        let out_fp = format!("{}{}", TEST_OUTPUT, "convert_write_read.wav");
        wav.write::<f32, _>(Path::new(&out_fp)).unwrap();

        let mut wav: Wav<f32> = Wav::<f32>::from_path(&out_fp).unwrap();
        let actual_samples: &[f32] = &wav.read().unwrap();

        let expected_samples =
            read_text_to_vec::<f32>(Path::new(ONE_CHANNEL_EXPECTED_F32)).unwrap();

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

        for (expected, actual) in expected_samples.iter().zip(wav.read().unwrap().as_ref()) {
            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
        }
    }

    #[test]
    fn can_convert_wav() {
        let p: &Path = Path::new(ONE_CHANNEL_EXPECTED_F32);
        let expected: Vec<f32> = read_text_to_vec(p).unwrap();

        let wav: Wav<i16> = Wav::from_path(ONE_CHANNEL_WAV_I16).unwrap();
        println!("Wav i16 samples: {}", wav.n_samples());

        let mut wav: Wav<f32> = wav.convert().unwrap();
        let actual: &[f32] = &wav.read().expect("Failed to read samples");
        println!("Read succes");
        for (exp, act) in expected.iter().zip(actual) {
            assert_approx_eq!(*exp as f64, *act as f64);
        }
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
