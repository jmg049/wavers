/// Module contains the core structs, ``Wav`` and ``Samples`` for working working with wav files.
use std::alloc::Layout;
use std::any::TypeId;
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::{Deref, DerefMut};
use std::path::Path;

use bytemuck::cast_slice;

use crate::conversion::ConvertSlice;

use crate::conversion::{AudioSample, ConvertTo};
use crate::error::{WaversError, WaversResult};
use crate::header::{read_header, WavHeader, DATA};
pub trait ReadSeek: Read + Seek {}

impl ReadSeek for std::fs::File {}
impl<T: ReadSeek> ReadSeek for std::io::BufReader<T> {}

pub struct Wav<T: AudioSample>
where
    Box<[i16]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,

    Box<[T]>: ConvertSlice<i16>,
    Box<[T]>: ConvertSlice<i32>,
    Box<[T]>: ConvertSlice<f32>,
    Box<[T]>: ConvertSlice<f64>,

    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    _phantom: std::marker::PhantomData<T>,
    reader: Box<dyn ReadSeek>,
    pub wav_info: WavInfo,
}

impl<T: AudioSample> Wav<T>
where
    Box<[i16]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
    Box<[T]>: ConvertSlice<i16>,
    Box<[T]>: ConvertSlice<i32>,
    Box<[T]>: ConvertSlice<f32>,
    Box<[T]>: ConvertSlice<f64>,

    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    pub fn header(&self) -> &WavHeader {
        &self.wav_info.wav_header
    }

    fn header_mut(&mut self) -> &mut WavHeader {
        &mut self.wav_info.wav_header
    }

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

    #[inline(always)]
    pub fn read(&mut self) -> WaversResult<Samples<T>> {
        let (_, data_size_bytes) = self.header().data().into();
        let n_samples = data_size_bytes as usize / std::mem::size_of::<T>();
        self.read_samples(n_samples)
    }

    #[inline(always)]
    pub fn read_samples(&mut self, n_samples: usize) -> WaversResult<Samples<T>> {
        let n_bytes = n_samples * std::mem::size_of::<T>();

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

    #[inline(always)]
    pub fn write<F: AudioSample, P: AsRef<Path>>(&mut self, p: P) -> WaversResult<()>
    where
        T: ConvertTo<F>,
        Box<[T]>: ConvertSlice<F>,
        Box<[i16]>: ConvertSlice<T>,
        Box<[i32]>: ConvertSlice<T>,
        Box<[f32]>: ConvertSlice<T>,
        Box<[f64]>: ConvertSlice<T>,
        Box<[T]>: ConvertSlice<i16>,
        Box<[T]>: ConvertSlice<i32>,
        Box<[T]>: ConvertSlice<f32>,
        Box<[T]>: ConvertSlice<f64>,
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

    pub fn from_path<P: AsRef<Path>>(path: P) -> WaversResult<Self>
    where
        Box<[i16]>: ConvertSlice<T>,
        Box<[i32]>: ConvertSlice<T>,
        Box<[f32]>: ConvertSlice<T>,
        Box<[f64]>: ConvertSlice<T>,
        Box<[T]>: ConvertSlice<i16>,
        Box<[T]>: ConvertSlice<i32>,
        Box<[T]>: ConvertSlice<f32>,
        Box<[T]>: ConvertSlice<f64>,
    {
        let f = std::fs::File::open(path)?;
        let buf_reader: Box<dyn ReadSeek> = Box::new(std::io::BufReader::new(f));
        Self::new(buf_reader)
    }
}

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

#[cfg(not(feature = "pyo3"))]
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
use crate::conversion::ndarray_conversion::{AsNdarray, IntoNdarray, IntoWav};
#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayBase, CowArray, ShapeError};

#[cfg(feature = "ndarray")]
impl<T> IntoNdarray for Wav<T>
where
    T: AudioSample,
{
    type Target = T;

    fn into_ndarray(self) -> Result<Array2<Self::Target>, ShapeError> {
        let n_channels = self.header.fmt_chunk.channels as usize;
        let shape = (self.len() / n_channels, n_channels);
        let copied_data = self.samples.as_ref();
        let arr = ArrayBase::from(copied_data.to_owned());
        arr.into_shape(shape)
    }
}

#[cfg(feature = "ndarray")]
impl<T> AsNdarray for Wav<T>
where
    T: AudioSample,
{
    type Target = T;

    fn as_ndarray(&self) -> Result<CowArray<Self::Target, ndarray::Ix2>, ShapeError> {
        let n_channels = self.header.fmt_chunk.channels as usize;
        let shape = (self.len() / n_channels, n_channels);
        let copied_data = &self.samples;
        let arr = CowArray::from(copied_data);
        arr.into_shape(shape)
    }
}

#[cfg(feature = "ndarray")]
impl<T> IntoWav for Array2<T>
where
    T: AudioSample + Debug + Copy + PartialEq,
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    type Target = T;
    fn into(self, sample_rate: i32) -> WaversResult<Wav<T>> {
        let n_channels = self.shape()[0];

        let samples: Samples<T> = Samples::from(self.into_raw_vec());

        let t_type = TypeId::of::<T>();
        let header = {
            if t_type == I16 {
                WavHeader::new_header::<i16>(sample_rate, n_channels as u16, samples.len())?
            } else if t_type == I32 {
                WavHeader::new_header::<i32>(sample_rate, n_channels as u16, samples.len())?
            } else if t_type == F32 {
                WavHeader::new_header::<f32>(sample_rate, n_channels as u16, samples.len())?
            } else if t_type == F64 {
                WavHeader::new_header::<f64>(sample_rate, n_channels as u16, samples.len())?
            } else {
                return Err(WaversError::InvalidType(format!("{:?}", t_type)));
            }
        };

        Ok(Wav { header, samples })
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

pub fn alloc_box_buffer(len: usize) -> Box<[u8]> {
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
    use crate::{read, write};

    use approx_eq::assert_approx_eq;
    use std::{fs::File, io::BufRead, path::Path, str::FromStr};

    const ONE_CHANNEL_WAV_I16: &str = "./test_resources/one_channel_i16.wav";
    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";

    const ONE_CHANNEL_EXPECTED_I16: &str = "./test_resources/one_channel_i16.txt";
    const ONE_CHANNEL_EXPECTED_F32: &str = "./test_resources/one_channel_f32.txt";

    const TEST_OUTPUT: &str = "./test_resources/tmp/";

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
        let wav = Wav::<i16>::read(ONE_CHANNEL_WAV_I16).unwrap();
        let expected_wav: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();

        let arr = wav.into_ndarray().unwrap();
        assert_eq!(arr.shape()[0], 1, "Expected 1 channels");
        for (expected, actual) in expected_wav.iter().zip(arr) {
            assert_eq!(*expected, actual, "{} != {}", expected, actual);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn two_channel_as_ndarray() {
        let wav: Wav<i16> = Wav::<i16>::read(Path::new(TWO_CHANNEL_WAV_I16)).unwrap();
        let expected_wav: Vec<i16> = read_text_to_vec(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();
        let mut new_expected = Vec::with_capacity(expected_wav.len() * 2);
        for sample in expected_wav {
            new_expected.push(sample);
            new_expected.push(sample);
        }

        let expected_wav = new_expected;

        let two_channel_arr = wav.into_ndarray().unwrap();
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
