use std::{
    alloc::Layout,
    fs::File,
    io::{self, BufReader, Read, Seek},
    ops::Range,
    path::Path,
};

use itertools::Itertools;
use ndarray::Array1;

const RIFF: &[u8; 4] = b"RIFF";
const DATA: &[u8; 4] = b"data";
const WAVE: &[u8; 4] = b"WAVE";
const FMT: &[u8; 4] = b"fmt ";

#[derive(Debug, Clone)]
pub struct WavFile {
    header: WavHeader,
    data: Box<[u8]>,
    current_seek_pos: usize,
}

pub enum WavType {
    I16,
    I24,
    I32,
    F32,
    F64,
}

type i24 = i32;

impl WavFile {
    pub fn new(fp: &Path) -> Result<WavFile, std::io::Error> {
        let (header, data) = read_to_u8_buf(fp)?;

        Ok(WavFile {
            header,
            data,
            current_seek_pos: 0,
        })
    }

    pub fn new_with_pos(fp: &Path, initial_pos: usize) -> Result<WavFile, std::io::Error> {
        let (header, data) = read_section_to_u8_buf(fp, initial_pos as u64)?;
        Ok(WavFile {
            header,
            data,
            current_seek_pos: initial_pos,
        })
    }

    pub fn blocks(fp: &Path, block_size: usize, overlap: Option<usize>, dtype: Option<WavType>) {
        let _dtype = match dtype {
            Some(t) => t,
            None => WavType::I16,
        };

        let _overlap = match overlap {
            Some(o) => o,
            None => 0,
        };

        let (header, data) = read_to_u8_buf(fp).unwrap();
        let wav_file = WavFile {
            header,
            data,
            current_seek_pos: 0,
        };

        match _dtype {
            WavType::I16 => todo!(),
            WavType::I24 => todo!(),
            WavType::I32 => todo!(),
            WavType::F32 => todo!(),
            WavType::F64 => todo!(),
        }
    }

    #[inline(always)]
    pub fn read_pcm_i16(&self) -> Array1<i16> {
        let mut out_data: Vec<i16> =
            Vec::with_capacity(self.data.len() / 2 - self.current_seek_pos);
        unsafe {
            out_data.set_len(self.data.len() / 2 - self.current_seek_pos);
        }

        for i in 0..out_data.len() {
            out_data[i] = i16::from_le_bytes(
                self.data[i * 2..i * 2 + 2]
                    .try_into()
                    .expect("Failed to parse bytes as i16"),
            );
        }
        Array1::from(out_data)
    }

    pub fn read_pcm_i24(&self) -> Vec<i24> {
        let out_data: Vec<i32> = Vec::with_capacity(self.data.len() - self.current_seek_pos);

        out_data
    }

    pub fn read_pcm_i32(&self) -> Vec<i32> {
        let mut out_data: Vec<i32> = Vec::with_capacity(self.data.len() - self.current_seek_pos);
        unsafe { out_data.set_len(self.data.len() - self.current_seek_pos) };

        for i in 0..out_data.len() {
            out_data[i] = i32::from_le_bytes(
                self.data[i * 4..i * 4 + 4]
                    .try_into()
                    .expect("Failed to parse bytes as i32"),
            );
        }

        out_data
    }

    #[inline(always)]
    pub fn read_pcm_i16_as_f32(&self) -> Array1<f32> {
        Array1::from_vec(
            self.read_pcm_i16()
                .iter()
                .map(|sample| pcm_16_to_f32(*sample))
                .collect(),
        )
    }

    pub fn read_pcm_i24_as_f32(&self) -> Vec<i24> {
        let out_data: Vec<i32> = Vec::with_capacity(self.data.len() - self.current_seek_pos);

        out_data
    }

    pub fn read_pcm_i32_as_f32(&self) -> Vec<i32> {
        let out_data: Vec<i32> = Vec::with_capacity(self.data.len() - self.current_seek_pos);

        out_data
    }

    pub fn read_ieee_f32(&self) -> Vec<f32> {
        let out_data: Vec<f32> = Vec::with_capacity(self.data.len() - self.current_seek_pos);

        out_data
    }

    pub fn read_ieee_f64(&self) -> Vec<f64> {
        let out_data: Vec<f64> = Vec::with_capacity(self.data.len() - self.current_seek_pos);

        out_data
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct WavHeader {
    riff: [u8; 4],         // 4
    size: i32,             // 4
    format: [u8; 4],       // 4
    sub_chunk_id: [u8; 4], // 4
    sub_chunk_size: i32,   // 4
    audio_format: i16,     // 2
    n_channels: i16,       // 2
    sample_rate: i32,      // 4
    byte_rate: i32,        // 4
    block_align: i16,      // 2
    bits_per_sample: i16,  // 2
    data: [u8; 4],         // 4
    data_size: i32,        // 4
}

// What is this assumed to be safe? The format of a WAVE file specifies that the first 44 bytes is the header.
#[inline(always)]
unsafe fn read_wav_header(header_bytes: [u8; 44]) -> WavHeader {
    std::mem::transmute_copy::<[u8; 44], WavHeader>(&header_bytes)
}

// Reads the entire file into a byte buffer. The function returns a result containing the header and the data, or an IO error.
// The data is returned as a boxed slice, which is a heap allocated array.
#[inline(always)]
pub fn read_to_u8_buf(fp: &Path) -> Result<(WavHeader, Box<[u8]>), std::io::Error> {
    read_section_to_u8_buf(fp, 0)
}

// Reads only the bytes relating to the sample rate of a wav file. These bytes start at byte location 24.
// The function returns a result containing the sample rate, or an IO error.
#[inline(always)]
pub fn signal_sample_rate(signal_fp: &Path) -> Result<i32, io::Error> {
    let mut wav_file = File::open(signal_fp)?;
    wav_file.seek(io::SeekFrom::Start(24))?;
    let mut buf: [u8; 4] = [0; 4];
    wav_file.read_exact(&mut buf)?;
    let size: i32 = i32::from_le_bytes(buf);
    Ok(size)
}

#[inline(always)]
pub fn signal_duration(signal_fp: &Path) -> Result<u64, io::Error> {
    let mut wav_file = File::open(signal_fp)?;
    let mut buf: [u8; 44] = [0; 44];
    wav_file.read_exact(&mut buf)?;
    let wav_header = unsafe { read_wav_header(buf) };
    let rhs = (wav_header.sample_rate
        * wav_header.n_channels as i32
        * (wav_header.bits_per_sample / 8) as i32) as u64;
    let lhs = wav_header.data_size as u64;
    println!("lhs: {}, rhs: {}", lhs, rhs);
    let duration = lhs / rhs;
    Ok(duration)
}

#[inline(always)]
pub fn read_section_to_u8_buf(
    fp: &Path,
    initial_pos: u64,
) -> Result<(WavHeader, Box<[u8]>), std::io::Error> {
    let f_in: File = File::open(fp)?;
    let mut buffer = BufReader::new(&f_in);
    let mut header_bytes: [u8; 44] = [0; 44];
    buffer.read_exact(&mut header_bytes)?;

    let header = unsafe { read_wav_header(header_bytes) };
    buffer.seek(std::io::SeekFrom::Start(initial_pos + 44))?;

    let data_size = header.data_size - initial_pos as i32;
    let mut data_buf = alloc_box_buffer(data_size as usize);
    buffer.read_exact(&mut data_buf)?;
    Ok((header, data_buf))
}

#[inline(always)]
fn alloc_box_buffer(len: usize) -> Box<[u8]> {
    if len == 0 {
        return <Box<[u8]>>::default();
    }
    let layout = Layout::array::<u8>(len).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    let slice_ptr = core::ptr::slice_from_raw_parts_mut(ptr, len);
    unsafe { Box::from_raw(slice_ptr) }
}

#[inline(always)]
fn pcm_16_to_f32(sample: i16) -> f32 {
    (sample as f32 / 32768.0).clamp(-1.0, 1.0)
}

pub fn overlapping_chunks<T>(data: Vec<T>, chunk_size: usize, overlap_size: usize) -> Vec<Range<usize>>{
    assert!(
        chunk_size > overlap_size,
        "overlap_size must be less than chunk_size"
    );
    let n_windows = (data.len() - overlap_size) / (chunk_size - overlap_size);
    println!("n_windows: {}", n_windows);

    let mut ranges: Vec<Range<usize>> = Vec::new();

    let data_len = data.len();
    println!(
        "data_len: {} data_len % chunk_size: {}",
        data_len,
        data_len % chunk_size
    );
    if data_len % chunk_size == 0 {
        for i in (0..data_len).step_by(chunk_size) {
            let range = i..(i + chunk_size);
            ranges.push(range);
        }
    } else {
        for i in (0..data_len - chunk_size).step_by(chunk_size) {
            let range = i..(i + chunk_size);
            ranges.push(range);
        }
        ranges.push((data_len - chunk_size) + 1..data_len);
    }

    println!("{:?}", ranges);
    let mut overlapping_ranges: Vec<Range<usize>> = Vec::new();
    for i in (overlap_size..(data.len() - chunk_size)).step_by(chunk_size) {
        let range = i..(i + chunk_size);
        overlapping_ranges.push(range);
    }
    ranges.iter().interleave(overlapping_ranges)
}

