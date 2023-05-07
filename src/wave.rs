use std::{
    alloc::Layout,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    u8,
};

use crate::sample::SampleType;

const RIFF: &[u8; 4] = b"RIFF";
const DATA: &[u8; 4] = b"data";
const WAVE: &[u8; 4] = b"WAVE";
const FMT: &[u8; 4] = b"fmt ";

#[allow(unused)] // Only the above chunks are guaranteed to be in a file/used
const LIST: &[u8; 4] = b"LIST";

///
/// A ``WavFile`` is a struct that contains the data and some metadata of a WAV file.
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavFile {
    pub fmt_chunk: FmtChunk,
    pub data: Box<[u8]>,
    pub seek_pos: u64,
}

impl WavFile {
    pub fn new(fmt_chunk: FmtChunk, data: Box<[u8]>, seek_pos: u64) -> WavFile {
        WavFile {
            fmt_chunk,
            data,
            seek_pos,
        }
    }

    ///
    /// Read a file from disk and return a ``WavFile``
    ///
    /// Returns a ``WavFile`` if the file is successfully read. Otherwise, returns an ``std::io::Error``.
    ///
    
    pub fn from_file(fp: &Path) -> Result<WavFile, std::io::Error> {
        let file = File::open(fp)?;
        let mut buf_reader = std::io::BufReader::new(file);
        let fmt_chunk = FmtChunk::from_buf_reader(&mut buf_reader)?;
        let (data_offset, data_len) = find_sub_chunk_id(&mut buf_reader, &b"data")?;
        let mut data = alloc_box_buffer(data_len);

        buf_reader.seek(SeekFrom::Start(data_offset as u64 + 4))?; // +4 to skip the length of the data chunk

        match buf_reader.read(&mut data) {
            Ok(_) => (),
            Err(err) => {
                eprintln!("Error reading data chunk: {}", err);
                return Err(err);
            }
        }
        Ok(WavFile::new(fmt_chunk, data, 0))
    }

    /// Read the underlying wave data into the desired sample type. If no sample type is given, the original sample type will be used.
    /// This funcions converts the underlying ``u8`` buffer to a concrete type ndarray.
    ///
    /// Returns a ndarray with the shape (n_samples, n_channels)
    ///
    /// Panics if the sample type is not supported.
    ///
    #[inline]
    pub fn read(&self, as_wav_type: Option<SampleType>) -> Vec<SampleType> {
        let bits_per_sample = self.bits_per_sample(); // This will dictate the original encoding. 16 for PCM, 32 for float or i32 ()
        let base_format = self.format();

        // Figure out the actual sample format. Neither the format nor the bits per sample are enough to determine the actual sample type.
        // For example, an IEEE Float, represented by the number 3, can be either 32-bit or 64-bit. A PCM sample can be either 16-bit, 24-bit or 32-bit.
        // Confusion can happen with a 32-bit float and a 32-bit int PCM.
        let sample_format = match base_format {
            1 => match bits_per_sample {
                16 => SampleType::I16(0),
                32 => SampleType::I32(0),
                _ => panic!("Unsupported bit depth for PCM: {}", bits_per_sample),
            },
            3 => match bits_per_sample {
                32 => SampleType::F32(0.0), 
                64 => SampleType::F64(0.0),
                _ => panic!("Unsupported bit depth for float: {}", bits_per_sample),
            },
            _ => panic!("Unsupported format: {}", base_format),
        };

        let data = match sample_format {
            SampleType::I16(_) => self.read_pcm_i16(),
            SampleType::I32(_) => self.read_pcm_i32(),
            SampleType::F32(_) => self.read_ieee_f32(),
            SampleType::F64(_) => self.read_ieee_f64(),
        };

        match as_wav_type {
            Some(wav_type) => match wav_type {
                SampleType::I16(_) => data.iter().map(|sample| sample.convert_to(wav_type)).collect(),
                SampleType::I32(_) => data.iter().map(|sample| sample.convert_to(wav_type)).collect(),
                SampleType::F32(_) => data.iter().map(|sample| sample.convert_to(wav_type)).collect(),
                SampleType::F64(_) => data.iter().map(|sample| sample.convert_to(wav_type)).collect(),
            },
            None => data,
        }
    }

    ///
    /// Will write the wave file data to the given path. If the file already exists, it will be overwritten.
    /// Writes the current state of the ``WavFile``, which may be different from the original file if it was read by ``wavers`` as only
    /// the ``FMT`` and ``Data`` chunks of the orignal wav file are presevered. The ``RIFF`` chunk is recreated and the length of the file is
    /// recalculated. Any modification of the wav data will be be written provided that the original format is preserved.
    ///
    /// To write a different type of sample, use the function write_wav_as and pass in the desired sample type.
    /// Returns an error if the file cannot be created or written to.
    ///
    ///
    
    pub fn write_wav(&self, fp: &Path) -> Result<(), std::io::Error> {
        let file = File::create(fp)?;
        let mut buf_writer = BufWriter::new(file);

        // Write header + Currently this is a lossy write with respect to the original file(if was originally read by wavers)
        buf_writer.write(RIFF)?;
        buf_writer.write(&(self.data.len() as u32 + 36).to_ne_bytes())?;
        buf_writer.write(WAVE)?;

        buf_writer.write(FMT)?;

        buf_writer.write_all(&self.fmt_chunk.as_bytes())?;

        buf_writer.write(DATA)?;
        buf_writer.write(&self.data.len().to_ne_bytes())?;

        // Write the underlying data
        buf_writer.write_all(self.data.as_ref())?;
        Ok(())
    }

    ///
    /// Reads the underlying wav data as 16-bit signed integer PCM samples.
    ///
    /// Panics if there is an error reading the data. Will likely change this to return a result in future.
    ///
    
    fn read_pcm_i16(&self) -> Vec<SampleType> {
        let n_channels = self.fmt_chunk.channels as usize;
        let mut channel_data: Vec<SampleType> =
            Vec::with_capacity((self.data.len() / 2) - self.seek_pos as usize);
        unsafe {
            channel_data.set_len((self.data.len() / 2) - self.seek_pos as usize);
        }

        let mut idx = 0;
        let iter_step = 2 * n_channels; // two bytes per sample per channel

        for samples in self.data.chunks(iter_step) {
                unsafe {
                    for channel_sample in samples.as_chunks_unchecked::<{std::mem::size_of::<i16>()}>() {
                        channel_data[idx] =
                            SampleType::I16(i16::from_ne_bytes(*channel_sample));
                        idx += 1;
                    }
                }
        }
        channel_data
    }

    ///
    /// Reads the underlying wav data as 32-bit signed integer PCM samples.
    ///
    /// Panics if there is an error reading the data. Will likely change this to return a result in future.
    ///
    
    fn read_pcm_i32(&self) -> Vec<SampleType> {
        let n_channels = self.fmt_chunk.channels as usize;
        let mut channel_data: Vec<SampleType> =
            Vec::with_capacity((self.data.len() / 4) - self.seek_pos as usize); // divide by 4 because 4 bytes per sample
        unsafe {
            channel_data.set_len((self.data.len() / 4) - self.seek_pos as usize);
        }

        let mut idx = 0;
        let iter_step = 4 * n_channels; // four bytes per sample per channel

        for samples in self.data.chunks(iter_step) {
            unsafe {
            for channel_sample in samples.as_chunks_unchecked::<{std::mem::size_of::<i32>()}>()  {
                channel_data[idx] =
                    SampleType::I32(i32::from_ne_bytes(*channel_sample));
                idx += 1;
            }
        }
        }
        channel_data
    }

    ///
    /// Reads the underlying wav data as 32-bit IEEE floating point PCM samples.
    ///
    /// Panics if there is an error reading the data. Will likely change this to return a result in future.
    ///
    
    fn read_ieee_f32(&self) -> Vec<SampleType> {
        let n_channels = self.fmt_chunk.channels as usize;
        let mut channel_data: Vec<SampleType> =
            Vec::with_capacity((self.data.len() / 4) - self.seek_pos as usize); // divide by 4 because 4 bytes per sample
        unsafe {
            channel_data.set_len((self.data.len() / 4) - self.seek_pos as usize);
        }

        let mut idx = 0;
        let iter_step = 4 * n_channels; // four bytes per sample per channel

        for samples in self.data.chunks(iter_step) {
            
            unsafe {
                for channel_sample in samples.as_chunks_unchecked::<{std::mem::size_of::<f32>()}>()  {
                channel_data[idx] =
                    SampleType::F32(f32::from_ne_bytes(*channel_sample));
                idx += 1;
            }
        }
        }
        channel_data
    }

    ///
    /// Reads the underlying wav data as 64-bit IEEE floating point PCM samples.
    ///
    /// Panics if there is an error reading the data. Will likely change this to return a result in future.
    ///
    
    fn read_ieee_f64(&self) -> Vec<SampleType> {
        let n_channels = self.fmt_chunk.channels as usize;
        let mut channel_data: Vec<SampleType> =
            Vec::with_capacity((self.data.len() / 8) - self.seek_pos as usize); // divide by 8 because 8 bytes per sample
        unsafe {
            channel_data.set_len((self.data.len() / 8) - self.seek_pos as usize);
        }

        let mut idx = 0;
        let iter_step = 8 * n_channels; // eight bytes per sample per channel

        for samples in self.data.chunks(iter_step) {
            unsafe {
                for channel_sample in samples.as_chunks_unchecked::<{std::mem::size_of::<f64>()}>()  {
                channel_data[idx] =
                    SampleType::F64(f64::from_ne_bytes(*channel_sample));
                idx += 1;
            }
        }
        }

        channel_data
    }

    ///
    /// Calculates the duration of the wav file in seconds using the ``WavFile`` data that is already loaded.
    ///
    
    pub fn duration(&self) -> u64 {
        self.data_size() as u64
            / (self.sample_rate() * self.channels() as i32 * (self.bits_per_sample() / 8) as i32)
                as u64
    }

    ///
    /// Returns the sample rate of the wav file.
    ///
    pub fn sample_rate(&self) -> i32 {
        self.fmt_chunk.sample_rate()
    }

    ///
    /// Returns the number of channels in the wav file.
    ///
    pub fn channels(&self) -> u16 {
        self.fmt_chunk.channels()
    }

    ///
    /// Returns the number of bits per sample in the wav file.
    ///
    fn bits_per_sample(&self) -> u16 {
        self.fmt_chunk.bits_per_sample()
    }

    fn format(&self) -> u16 {
        self.fmt_chunk.format()
    }

    ///
    /// Returns the number of bytes in the wav file data chunk less the size in bytes attributed to the offset.
    ///
    fn data_size(&self) -> usize {
        self.data.len() - self.seek_pos as usize
    }
}



///
/// Returns the duration of a wav file in seconds without reading the entire file, only the necessary header information.
///
/// Returns an error if the file is not a valid wav file or does not exist.
///  

pub fn signal_duration(signal_fp: &Path) -> Result<u64, std::io::Error> {
    let wav_file = File::open(signal_fp)?;
    let mut br = BufReader::new(wav_file);
    let fmt_chunk = FmtChunk::from_buf_reader(&mut br)?;

    let (data_offset, _) = find_sub_chunk_id(&mut br, &b"data")?;
    let mut data_size_buf: [u8; 4] = [0; 4];
    br.seek(SeekFrom::Start(data_offset as u64))?;
    br.read_exact(&mut data_size_buf)?;

    Ok(i32::from_ne_bytes(data_size_buf) as u64
        / (fmt_chunk.sample_rate()
            * fmt_chunk.channels() as i32
            * (fmt_chunk.bits_per_sample() / 8) as i32) as u64)
}

///
/// Returns the sample rate of a wav file without reading the entire file, only the necessary header information.
///
/// Returns an error if the file is not a valid wav file or does not exist.
///

pub fn signal_sample_rate(signal_fp: &Path) -> Result<i32, std::io::Error> {
    let wav_file = File::open(signal_fp)?;
    let mut br = BufReader::new(wav_file);
    let fmt_chunk = FmtChunk::from_buf_reader(&mut br)?;
    Ok(fmt_chunk.sample_rate())
}

///
/// Returns the number of channels of a wav file without reading the entire file, only the necessary header information.
///
/// Returns an error if the file is not a valid wav file or does not exist.
///

pub fn signal_channels(signal_fp: &Path) -> Result<u16, std::io::Error> {
    let wav_file = File::open(signal_fp)?;
    let mut br = BufReader::new(wav_file);
    let fmt_chunk = FmtChunk::from_buf_reader(&mut br)?;
    Ok(fmt_chunk.channels())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SignalInfo {
    pub sample_rate: i32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub duration: u64,
}

impl SignalInfo {
    pub fn new(sample_rate: i32, channels: u16, bits_per_sample: u16, duration: u64) -> Self {
        Self {
            sample_rate,
            channels,
            bits_per_sample,
            duration,
        }
    }
}

pub fn signal_info(signal_fp: &Path) -> Result<SignalInfo, std::io::Error> {
    let wav_file = File::open(signal_fp)?;
    let mut br = BufReader::new(wav_file);
    let fmt_chunk = FmtChunk::from_buf_reader(&mut br)?;

    let (data_offset, _) = find_sub_chunk_id(&mut br, &b"data")?;
    let mut data_size_buf: [u8; 4] = [0; 4];
    br.seek(SeekFrom::Start(data_offset as u64))?;
    br.read_exact(&mut data_size_buf)?;

    Ok(SignalInfo::new(
        fmt_chunk.sample_rate(),
        fmt_chunk.channels(),
        fmt_chunk.bits_per_sample(),
        i32::from_ne_bytes(data_size_buf) as u64
            / (fmt_chunk.sample_rate()
                * fmt_chunk.channels() as i32
                * (fmt_chunk.bits_per_sample() / 8) as i32) as u64,
    ))
}


pub fn read(fp: &Path, as_type: Option<SampleType>) -> Result<Vec<SampleType>, std::io::Error> {
    let wav_file = WavFile::from_file(fp)?;
    Ok(wav_file.read(as_type))
}

///
/// Write a wav array to a file with the given sample rate and as the given type if provided.
/// This function does not preserve the original file's header information. Only the RIFF, WAVE, FMT and DATA chunks are written.
///
/// Returns an error if the file cannot be created or if it cannot write to the file.
///

pub fn write_wav_as(
    fp: &Path,
    data: &Vec<SampleType>,
    as_type: Option<SampleType>,
    n_channels: u16,
    sample_rate: i32,
) -> Result<(), std::io::Error> {
    let file = File::create(fp)?;
    let mut buf_writer = BufWriter::new(file);

    // Write header + Currently this is a lossy write with respect to the original file(if was originally read by wavers)
    buf_writer.write(RIFF)?; // 4

    // Data len may be different if we change the type of the data
    let (data_len, block_align) = match as_type {
        Some(t) => match t {
            SampleType::I16(_) => (data.len() * 2, 2),
            SampleType::I32(_) => (data.len() * 4, 4),
            SampleType::F32(_) => (data.len() * 4, 4),
            SampleType::F64(_) => (data.len() * 8, 8),
        },
        None => (data.len() * 2, 2),
    };

    buf_writer.write(&((data_len as i32 + 36).to_ne_bytes()))?; // 4
    buf_writer.write(WAVE)?; // 4

    let byte_rate = sample_rate * n_channels as i32 * data[0].size_of_underlying() as i32; // sr * n_channels * sample resolution
    let block_align = block_align as u16; 

    let fmt_chunk = match as_type {
        // Match the type of the provided SampleType
        Some(t) => match t {
            SampleType::I16(_) => {
                FmtChunk::new(16, 1, n_channels, sample_rate, byte_rate, block_align, 16)
            }
            SampleType::I32(_) => {
                FmtChunk::new(16, 1, n_channels, sample_rate, byte_rate, block_align, 32)
            }
            SampleType::F32(_) => {
                FmtChunk::new(16, 3, n_channels, sample_rate, byte_rate, block_align, 32)
            }
            SampleType::F64(_) => {
                FmtChunk::new(16, 3, n_channels, sample_rate, byte_rate, block_align, 64)
            }
        },
        // Match the type of the first sample
        None => match data.first().expect("Empty array") {
            SampleType::I16(_) => {
                FmtChunk::new(16, 1, n_channels, sample_rate, byte_rate, block_align, 16)
            }
            SampleType::I32(_) => {
                FmtChunk::new(16, 1, n_channels, sample_rate, byte_rate, block_align, 32)
            }
            SampleType::F32(_) => {
                FmtChunk::new(16, 3, n_channels, sample_rate, byte_rate, block_align, 32)
            }
            SampleType::F64(_) => {
                FmtChunk::new(16, 3, n_channels, sample_rate, byte_rate, block_align, 64)
            }
        },
    };

    let fmt_bytes = fmt_chunk.as_bytes(); // 24
    buf_writer.write_all(&fmt_bytes)?;
    buf_writer.write(DATA)?; // 4

    buf_writer.write(&(data_len as u32).to_ne_bytes())?; // 4

    for sample in data.iter() {
        let sample_as_type = match as_type {
            Some(t) => sample.convert_to(t),
            None => *sample,
        };
        buf_writer.write_all(&sample_as_type.to_ne_bytes())?;
    }

    Ok(())
}


///
/// Create a boxed ``u8`` buffer of the given size. The buffer is zeroed. Used when reading and writing wav files.
///
/// Panics if the buffer cannot be allocated.
///

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

///
/// A struct for storing the necessary format information about a wav file.
///
/// In total the struct is 20 bytes. 4 bytes storing the size of the chunk
/// and 16 bytes for the format information.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FmtChunk {
    pub size: i32,
    pub format: u16,
    pub channels: u16,
    pub sample_rate: i32,
    pub byte_rate: i32,
    pub block_align: u16,
    pub bits_per_sample: u16,
}

impl FmtChunk {
    ///
    /// Create a new ``FmtChunk`` from the given parameters.
    ///
    pub fn new(
        size: i32,            // 4
        format: u16,          // 2
        channels: u16,        // 2
        sample_rate: i32,     // 4
        byte_rate: i32,       // 4
        block_align: u16,     // 2
        bits_per_sample: u16, //2
    ) -> FmtChunk {
        FmtChunk {
            size,
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        }
    }

    ///
    /// Create a new ``FmtChunk`` from the given file path.
    ///
    /// Returns an error if the file cannot be opened or the ``FmtChunk`` cannot be read.
    ///
    
    pub fn from_path(signal_fp: &Path) -> Result<FmtChunk, std::io::Error> {
        let wav_file = File::open(signal_fp)?;
        let mut br = BufReader::new(wav_file);
        FmtChunk::from_buf_reader(&mut br)
    }

    ///
    /// Create a new ``FmtChunk`` from the given ``BufReader``.
    ///
    /// Returns an error if the ``FmtChunk`` cannot be read.
    ///
    
    fn from_buf_reader(br: &mut BufReader<File>) -> Result<FmtChunk, std::io::Error> {
        let mut buf: [u8; 4] = [0; 4];
        let mut buf_two: [u8; 2] = [0; 2];
        let (offset, _) = find_sub_chunk_id(br, b"fmt ")?;
        br.seek(SeekFrom::Start(offset as u64))?;
        br.read_exact(&mut buf)?;
        let size = i32::from_ne_bytes(buf);

        br.read_exact(&mut buf_two)?;
        let format = u16::from_ne_bytes(buf_two);

        br.read_exact(&mut buf_two)?;
        let channels = u16::from_ne_bytes(buf_two);

        br.read_exact(&mut buf)?;
        let sample_rate = i32::from_ne_bytes(buf);

        br.read_exact(&mut buf)?;
        let byte_rate = i32::from_ne_bytes(buf);

        br.read_exact(&mut buf_two)?;
        let block_align = u16::from_ne_bytes(buf_two);

        br.read_exact(&mut buf_two)?;
        let bits_per_sample = u16::from_ne_bytes(buf_two);
        br.seek(SeekFrom::Start(0))?;
        Ok(FmtChunk::new(
            size,
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        ))
    }

    ///
    /// Convert the ``FmtChunk`` to a byte array.
    ///
    
    pub fn as_bytes(&self) -> [u8; 24] {
        let mut buf: [u8; 24] = [0; 24];
        buf[0..4].copy_from_slice(FMT);
        buf[4..8].copy_from_slice(&self.size.to_ne_bytes());
        buf[8..10].copy_from_slice(&self.format.to_ne_bytes());
        buf[10..12].copy_from_slice(&self.channels.to_ne_bytes());
        buf[12..16].copy_from_slice(&self.sample_rate.to_ne_bytes());
        buf[16..20].copy_from_slice(&self.byte_rate.to_ne_bytes());
        buf[20..22].copy_from_slice(&self.block_align.to_ne_bytes());
        buf[22..24].copy_from_slice(&self.bits_per_sample.to_ne_bytes());
        buf
    }

    // Helper functions to avoid having to use the fmt struct field directly

    ///
    /// Get the number of bytes per sample.
    ///
    
    pub fn get_sample_size(&self) -> usize {
        self.bits_per_sample as usize / 8
    }

    ///
    /// Get the wave file data format.
    ///
    
    pub fn format(&self) -> u16 {
        self.format
    }

    ///
    /// Get the number of channels.
    ///
    
    pub fn channels(&self) -> u16 {
        self.channels
    }

    ///
    /// Get the sample rate.
    ///
    
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    ///
    /// Get the byte rate.
    ///
    
    pub fn byte_rate(&self) -> i32 {
        self.byte_rate
    }

    ///
    /// Get the block align.
    ///
    
    pub fn block_align(&self) -> u16 {
        self.block_align
    }

    ///
    /// Get the number of bits per sample.
    ///
    
    pub fn bits_per_sample(&self) -> u16 {
        self.bits_per_sample
    }
}

///
/// A function which searchs for a sub-chunk id in a wave file currently stored in a ``BufReader``.
///
/// Returns the offset of the sub-chunk id and the size of the sub-chunk.
///
/// Returns an error if the sub-chunk id cannot be found.
///

pub fn find_sub_chunk_id(
    file: &mut BufReader<File>,
    chunk_id: &[u8; 4],
) -> Result<(usize, usize), std::io::Error> {
    let mut buf: [u8; 4] = [0; 4];
    // Find the RIFF Tag

    file.read_exact(&mut buf)?;
    if !buf_eq(&buf, RIFF) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to find RIFF tag in {:?}", file.get_ref()),
        ));
    }

    file.seek(SeekFrom::Current(8))?;
    let mut tag_offset: usize = 0;
    let mut bytes_traversed: usize = 12;
    loop {
        // First sub-chunk is guaranteed to begin at byte 12 so seek forward by 8.
        // No other chunk is at a guaranteed offset.
        let bytes_read = file.read(&mut buf)?;
        if bytes_read == 0 {
            break;
        }

        bytes_traversed += bytes_read;

        if buf_eq(&buf, chunk_id) {
            tag_offset = bytes_traversed;
        }

        let bytes_read = file.read(&mut buf)?;
        if bytes_read == 0 {
            break;
        }
        bytes_traversed += bytes_read;

        let chunk_len =
            buf[0] as u32 | (buf[1] as u32) << 8 | (buf[2] as u32) << 16 | (buf[3] as u32) << 24;
        if tag_offset > 0 {
            let chunk_size = chunk_len as usize;
            file.seek(SeekFrom::Start(0))?; // Reset the file offset to the beginning
            return Ok((tag_offset, chunk_size));
        }
        file.seek(SeekFrom::Current(chunk_len as i64))?;

        bytes_traversed += chunk_len as usize;
    }
    file.seek(SeekFrom::Start(0))?;

    Err(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!(
            "Failed to find {:?} tag in {:?}",
            std::str::from_utf8(chunk_id).unwrap(),
            file.get_ref()
        ),
    ))
}

///
/// Function to compare two 4-byte arrays for equality.
///

fn buf_eq(buf: &[u8; 4], chunk_id: &[u8; 4]) -> bool {
    buf[0] == chunk_id[0] && buf[1] == chunk_id[1] && buf[2] == chunk_id[2] && buf[3] == chunk_id[3]
}

