/// Module contains the core structs, ``Wav`` and ``Samples`` for working working with wav files.
use std::alloc::Layout;
use std::any::TypeId;

use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::{Deref, DerefMut};
use std::path::Path;

use bytemuck::cast_slice;

use crate::conversion::{AudioSample, ConvertTo};
use crate::error::{WaversError, WaversResult};
use crate::header::{read_header, WavHeader, DATA};
use crate::FmtChunk;

pub const I16: TypeId = TypeId::of::<i16>();
pub const I32: TypeId = TypeId::of::<i32>();
pub const F32: TypeId = TypeId::of::<f32>();
pub const F64: TypeId = TypeId::of::<f64>();

/// Function to read a wav file from a given Path. The function will attempt to read the file and will convert the data to the specified type if necessary.
///
/// # Examples
/// ```no_run
/// use wavers::read;
/// use std::path::Path;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///    let fp: &Path = &Path::new("path/to/some/wav.wav");
///    let wav_i16 = read::<i16>(fp)?; // read a file as i16
///    let wav_f32 = read::<f32>(fp)?; // read a file as f32
///    Ok(())
/// }
/// ```
#[inline(always)]
pub fn read<T>(fp: &Path) -> WaversResult<Wav<T>>
where
    T: AudioSample + Debug + PartialEq,
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let mut file = File::open(fp)?;
    let (header, encoding) = read_header(&mut file)?;
    let file_encoding: TypeId = encoding.try_into()?;
    let target_encoding: TypeId = TypeId::of::<T>();

    let wav: Wav<T> = match file_encoding == target_encoding {
        true => Wav::<T>::from_file_and_header(&mut file, header)?,
        false => {
            if file_encoding == I16 {
                Wav::<i16>::from_file_and_header(&mut file, header)?.to::<T>()?
            } else if file_encoding == I32 {
                Wav::<i32>::from_file_and_header(&mut file, header)?.to::<T>()?
            } else if file_encoding == F32 {
                Wav::<f32>::from_file_and_header(&mut file, header)?.to::<T>()?
            } else if file_encoding == F64 {
                Wav::<f64>::from_file_and_header(&mut file, header)?.to::<T>()?
            } else {
                return Err(WaversError::InvalidType(format!("{:?}", file_encoding)));
            }
        }
    };
    Ok(wav)
}

/// Small struct that contains the necessary information to determine the exact encoding of a wav file.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WavEncoding {
    format: u16,
    bits_per_sample: u16,
}

impl TryInto<TypeId> for WavEncoding {
    type Error = WaversError;

    fn try_into(self) -> Result<TypeId, Self::Error> {
        match (self.format, self.bits_per_sample) {
            (1, 16) => Ok(I16),
            (1, 32) => Ok(I32),
            (3, 32) => Ok(F32),
            (3, 64) => Ok(F64),
            _ => Err(WaversError::InvalidType(
                format!(
                    "format: {}, bits_per_sample: {}",
                    self.format, self.bits_per_sample
                )
                .into(),
            )),
        }
    }
}

impl WavEncoding {
    /// Constructs a new WavEncoding struct
    pub fn new(format: u16, bits_per_sample: u16) -> Self {
        Self {
            format,
            bits_per_sample,
        }
    }
}

/// Struct containing information on the specification of the wav file
/// Avoids having to read the entire file to get the specification.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct WavSpec {
    pub fmt_data: FmtChunk,
    pub encoding: WavEncoding,
    pub duration: u32,
}

impl WavSpec {
    pub fn new(fmt_data: FmtChunk, duration: u32) -> Self {
        let encoding = WavEncoding::new(fmt_data.format, fmt_data.bits_per_sample);
        Self {
            fmt_data,
            encoding,
            duration,
        }
    }
}

impl TryFrom<&Path> for WavSpec {
    type Error = WaversError;

    fn try_from(value: &Path) -> Result<Self, Self::Error> {
        let mut file = File::open(value)?;
        let (header, encoding) = read_header(&mut file)?;
        let duration = header.get(DATA.into()).unwrap().size / header.fmt_chunk.block_align as u32;
        Ok(Self {
            fmt_data: header.fmt_chunk,
            encoding,
            duration,
        })
    }
}

/// Read the specification of a wav file. This includes the format information, duration and encoding.
pub fn read_spec(fp: &Path) -> WaversResult<WavSpec> {
    WavSpec::try_from(fp)
}

/// Struct representing an entire wav file. It contains the header information and the samples.
/// This is the priamry struct in the Wavers library.
///
/// Currently supports i16, i32, f32, and f64 samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wav<T>
where
    for<'a> T: 'a + AudioSample,
{
    pub header: WavHeader,
    pub samples: Samples<T>,
}

impl Wav<i16> {}
impl Wav<i32> {}
impl Wav<f32> {}
impl Wav<f64> {}

impl<T> Wav<T>
where
    for<'a> T: 'a + AudioSample + Debug + PartialEq + Copy,
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Constructs a new Wav struct from a WavHeader and Samples
    pub fn new(header: WavHeader, samples: Samples<T>) -> Self {
        Wav { header, samples }
    }

    /// Constructs a new Wav struct from a file and a header (also from the file).
    fn from_file_and_header(file: &mut File, header: WavHeader) -> WaversResult<Wav<T>> {
        let (data_offset, data_size_bytes) = match header.get(DATA.into()) {
            Some(e) => (e.offset, e.size),
            None => {
                return Err(WaversError::from(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "No data chunk found",
                )));
            }
        };

        let mut raw_samples: Box<[u8]> = alloc_box_buffer(data_size_bytes as usize);
        file.seek(SeekFrom::Start(data_offset as u64))?; // seek to data chunk
        file.read_exact(&mut raw_samples)?; // read data chunk into buffer

        let samples: Samples<T> = Samples::from(raw_samples.as_ref());
        Ok(Wav { header, samples })
    }

    /// Public function to read a wav file from a given Path. The function will attempt to read the file and will convert the data to the associated type ``T ``of the Wav struct.
    /// The header and data are read immediately.
    pub fn read(fp: &Path) -> WaversResult<Self> {
        let mut file = File::open(fp)?;
        let (header, _encoding) = read_header(&mut file)?;
        Self::from_file_and_header(&mut file, header)
    }

    /// Public function to write a wav file to the fiven file path.
    /// The function writes the data as is, if you want to write as different type call either to::\<type\> or as_::\<type\> before calling write.
    pub fn write<F: AsRef<Path>>(&self, fp: F) -> WaversResult<()> {
        let mut file = File::create(fp)?;
        let header_bytes = self.header.as_bytes();
        let total_header_size = header_bytes.len() + DATA.len() + 4; // RIFF to end of FMT chunk, then 4 bytes for "data", 4 bytes for the size of the data chunk

        let total_data_size = self.samples.len() * std::mem::size_of::<T>();

        let mut out_buf: Vec<u8> = Vec::with_capacity(total_header_size + total_data_size);
        unsafe { out_buf.set_len(total_header_size + total_data_size) }
        out_buf[0..header_bytes.len()].copy_from_slice(&header_bytes);

        // write data chunk
        out_buf[header_bytes.len()..header_bytes.len() + DATA.len()].copy_from_slice(DATA);
        // write data chunk size
        out_buf[header_bytes.len() + DATA.len()..header_bytes.len() + DATA.len() + 4]
            .copy_from_slice(&(total_data_size as u32).to_ne_bytes());

        // write data
        out_buf[header_bytes.len() + DATA.len() + 4..].copy_from_slice(self.samples.as_bytes());

        file.write_all(&out_buf)?;
        Ok(())
    }

    /// Used to convert between the different types of samples audio data.
    /// This function will consume the current Wav struct and return a new Wav struct with the specified type.
    pub fn to<F>(mut self) -> WaversResult<Wav<F>>
    where
        for<'a> T: 'a + ConvertTo<F> + Debug + ConvertTo<F>,
        for<'a> F: 'a + AudioSample + Debug + PartialEq + Copy + Sync + Send + ConvertTo<T>,
        i16: ConvertTo<F>,
        i32: ConvertTo<F>,
        f32: ConvertTo<F>,
        f64: ConvertTo<F>,
    {
        self.header.fmt_chunk.update_header(TypeId::of::<F>())?;
        let converted_samples = self.samples.convert::<F>();
        Ok(Wav {
            header: self.header,
            samples: converted_samples,
        })
    }

    /// Used to convert between the different types of samples audio data.
    /// This function will NOT consume the current Wav struct and return a new Wav struct with the specified type.
    pub fn as_<F>(&self) -> WaversResult<Wav<F>>
    where
        for<'a> T: 'a + ConvertTo<F> + Debug + ConvertTo<F>,
        for<'a> F: 'a + AudioSample + Debug + PartialEq + Copy + Sync + Send + ConvertTo<T>,
        i16: ConvertTo<F>,
        i32: ConvertTo<F>,
        f32: ConvertTo<F>,
        f64: ConvertTo<F>,
    {
        let sample_rate = self.header.fmt_chunk.sample_rate;
        let n_channels = self.header.fmt_chunk.channels;
        let sample_size = self.samples.len();
        let new_header = WavHeader::new_header::<F>(sample_rate, n_channels, sample_size)?;

        Ok(Wav {
            header: new_header,
            samples: self.samples.clone().convert::<F>(),
        })
    }
}

impl<T> Deref for Wav<T>
where
    for<'a> T: 'a + AudioSample,
{
    type Target = Samples<T>;

    fn deref(&self) -> &Self::Target {
        &self.samples
    }
}

impl<T> DerefMut for Wav<T>
where
    for<'a> T: 'a + AudioSample,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.samples
    }
}

impl<T> AsRef<Samples<T>> for Wav<T>
where
    for<'a> T: 'a + AudioSample + AsRef<Samples<T>>,
{
    fn as_ref(&self) -> &Samples<T> {
        &self.samples
    }
}

impl<T> AsMut<Samples<T>> for Wav<T>
where
    for<'a> T: 'a + AudioSample,
{
    fn as_mut(&mut self) -> &mut Samples<T> {
        &mut self.samples
    }
}

impl<T> Display for Wav<T>
where
    for<'a> T: 'a + AudioSample + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}\n{:?}", self.header, self.samples)
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
    for<'a> T: 'a + AudioSample,
{
    pub(crate) samples: Box<[T]>,
}

impl<T> AsRef<[T]> for Samples<T>
where
    for<'a> T: 'a + AudioSample,
{
    fn as_ref(&self) -> &[T] {
        &self.samples
    }
}

impl<T> AsMut<[T]> for Samples<T>
where
    for<'a> T: 'a + AudioSample,
{
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.samples
    }
}

impl<T> Deref for Samples<T>
where
    for<'a> T: 'a + AudioSample,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.samples
    }
}

impl<T> DerefMut for Samples<T>
where
    for<'a> T: 'a + AudioSample,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.samples
    }
}

// From Vec
impl<T> From<Vec<T>> for Samples<T>
where
    for<'a> T: 'a + AudioSample,
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
    for<'a> T: 'a + AudioSample,
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
    for<'a> T: 'a + AudioSample,
{
    fn from(samples: Box<[T]>) -> Self {
        Samples { samples }
    }
}

// From u8 Buffer
impl<T> From<&[u8]> for Samples<T>
where
    for<'a> T: 'a + AudioSample,
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
    for<'a> T: 'a + AudioSample + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.samples)
    }
}

impl<T> Samples<T>
where
    for<'a> T: 'a + AudioSample,
{
    /// Construct a new Samples struct from a boxed slice of audio samples.
    pub fn new(samples: Box<[T]>) -> Self {
        Self { samples }
    }

    /// Conversts the samples to the specified type ``F``. If the type is the same as the current type, the function will return self.
    /// The function will consume the current Samples struct and return a new Samples struct with the specified type.
    pub fn convert<F: AudioSample>(self) -> Samples<F>
    where
        for<'a> T: ConvertTo<F> + Debug + PartialEq + Copy + Sync,
        for<'a> F: 'a + AudioSample + Debug + Sync + Send,
        i16: ConvertTo<F>,
        i32: ConvertTo<F>,
        f32: ConvertTo<F>,
        f64: ConvertTo<F>,
    {
        // Quick check to see if we're converting to the same type, if so, just return self
        if TypeId::of::<T>() == TypeId::of::<F>() {
            let data: Box<[T]> = self.samples.clone();
            return Samples {
                samples: Box::from(cast_slice::<T, F>(&data)),
            };
        }

        let len = self.samples.len();

        let mut converted_samples: Box<[F]> = alloc_sample_buffer(len);

        for (i, sample) in self.samples.iter().enumerate() {
            converted_samples[i] = sample.convert_to();
        }
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

fn alloc_box_buffer(len: usize) -> Box<[u8]> {
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

fn alloc_sample_buffer<T>(len: usize) -> Box<[T]>
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

    use approx_eq::assert_approx_eq;
    use std::{fs::File, io::BufRead, path::Path, str::FromStr};

    const ONE_CHANNEL_WAV_I16: &str = "./test_resources/one_channel_i16.wav";
    const TWO_CHANNEL_WAV_I16: &str = "./test_resources/two_channel_i16.wav";

    const ONE_CHANNEL_EXPECTED_I16: &str = "./test_resources/one_channel_i16.txt";
    const ONE_CHANNEL_EXPECTED_F32: &str = "./test_resources/one_channel_f32.txt";

    const TEST_OUTPUT: &str = "./test_resources/tmp/";

    #[test]
    fn i16_i32_convert() {
        let wav_i16 = Wav::<i16>::read(Path::new(ONE_CHANNEL_WAV_I16))
            .unwrap()
            .samples;
        let wav_i32: Samples<i32> = wav_i16.convert();

        let expected_i32_samples =
            Wav::<i32>::read(Path::new("test_resources/one_channel_i32.wav"))
                .unwrap()
                .samples;

        for (expected, actual) in expected_i32_samples[0..10].iter().zip(wav_i32.iter()) {
            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn wav_as_ndarray() {
        let wav = Wav::<i16>::read(Path::new(ONE_CHANNEL_WAV_I16)).unwrap();
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


                        let wav: Wav<$T> = match Wav::<$T>::read(Path::new(&wav_str)) {
                            Ok(w) => w,
                            Err(e) => {eprintln!("{}\n{}", wav_str, e); panic!("Failed to read wav file")}
                        };
                        let expected_data: Vec<$T> = match read_text_to_vec(Path::new(&expected_str)) {
                            Ok(w) => w,
                            Err(e) => {eprintln!("{}\n{}", wav_str, e); panic!("Failed to read txt file")}
                        };

                        for (expected, actual) in expected_data.iter().zip(wav.samples.iter()) {
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
                            std::fs::create_dir(Path::new(TEST_OUTPUT)).unwrap();
                        }
                        let t_string: &str = stringify!($T);

                        let wav_str = format!("./test_resources/one_channel_{}.wav", t_string);
                        let expected_str = format!("./test_resources/one_channel_{}.txt", t_string);

                        let wav: Wav<$T> = match Wav::<$T>::read(Path::new(&wav_str)) {
                            Ok(w) => w,
                            Err(e) => {eprintln!("{}\n{}", wav_str, e); panic!("Failed to read wav file")}
                        };
                        let expected_data: Vec<$T> = match read_text_to_vec(Path::new(&expected_str)) {
                            Ok(w) => w,
                            Err(e) => {eprintln!("{}\n{}", wav_str, e); panic!("Failed to read txt file")}
                        };
                        for (expected, actual) in expected_data.iter().zip(wav.samples.iter()) {
                            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
                        }
                        let out_path = format!("{}{}", TEST_OUTPUT, "_one_channel_[<$T>].wav");
                        wav.write(Path::new(&out_path)).unwrap();

                        let wav: Wav<$T> = Wav::<$T>::read(Path::new(&out_path)).unwrap();

                        for (expected, actual) in expected_data.iter().zip(wav.samples.iter()) {
                            assert_eq!(*expected, *actual, "{} != {}", expected, actual);
                        }

                        std::fs::remove_file(Path::new(&out_path)).unwrap();
                    }
                }
            )*
        };
    }

    write_tests!(i32, f32, f64);

    #[test]
    fn read_and_convert() {
        let expected_samples =
            read_text_to_vec::<f32>(Path::new(ONE_CHANNEL_EXPECTED_F32)).unwrap();

        let wav: &[f32] = &read::<f32>(Path::new(ONE_CHANNEL_WAV_I16)).unwrap();

        for (expected, actual) in expected_samples.iter().zip(wav.iter()) {
            assert_approx_eq!(*expected as f64, *actual as f64, 1e-4);
        }
    }

    #[test]
    fn convert_write_read() {
        if !Path::new(TEST_OUTPUT).exists() {
            std::fs::create_dir(Path::new(TEST_OUTPUT)).unwrap();
        }
        let wav: Wav<i16> = Wav::<i16>::read(Path::new(ONE_CHANNEL_WAV_I16)).unwrap();
        let out_fp = format!("{}{}", TEST_OUTPUT, "convert_write_read.wav");
        wav.as_::<f32>().unwrap().write(&out_fp).unwrap();
        let wav: Wav<f32> = Wav::<f32>::read(Path::new(&out_fp)).unwrap();

        let expected_samples =
            read_text_to_vec::<f32>(Path::new(ONE_CHANNEL_EXPECTED_F32)).unwrap();

        for (expected, actual) in expected_samples.iter().zip(wav.as_ref()) {
            assert_approx_eq!(*expected as f64, *actual as f64, 1e-4);
        }
        std::fs::remove_file(Path::new(&out_fp)).unwrap();
    }

    #[test]
    fn can_read_two_channel() {
        let wav: Wav<i16> = Wav::<i16>::read(Path::new(TWO_CHANNEL_WAV_I16)).unwrap();
        let expected_samples =
            read_text_to_vec::<i16>(Path::new(ONE_CHANNEL_EXPECTED_I16)).unwrap();
        let mut new_expected = Vec::with_capacity(expected_samples.len() * 2);
        for sample in expected_samples {
            new_expected.push(sample);
            new_expected.push(sample);
        }

        let expected_samples = new_expected;

        for (expected, actual) in expected_samples.iter().zip(wav.samples.iter()) {
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
