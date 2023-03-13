use std::{
    fs::File,
    path::Path,
};
use memmap2::Mmap;
use byteorder::{ByteOrder, LittleEndian};

#[derive(Debug)]
pub struct Wav<T> {
    header: WavHeader,
    pub data: Vec<T>,
}

impl Wav<i16> {
    pub fn new(wav_header: WavHeader, data: Vec<i16>) -> Wav<i16> {
        Wav {
            header: wav_header,
            data: data,
        }
    }
}

impl Wav<f32> {
    pub fn new(wav_header: WavHeader, data: Vec<f32>) -> Wav<f32> {
        Wav {
            header: wav_header,
            data: data,
        }
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

impl WavHeader {
    pub fn get_duration_float(&self) -> f32 {
        (self.bits_per_sample as f32 / (32 * self.sample_rate) as f32) * self.data_size as f32
    }

    pub fn get_required_samples(&self) -> usize {
        (self.sample_rate as f32 * self.get_duration_float()) as usize * self.n_channels as usize
    }
}

const PCM_DIV: f32 = 1.0 / 0x7FFF as f32;

#[inline(always)]
pub fn pcm_i16_to_f32(sample: i16) -> f32 {
    PCM_DIV * sample as f32
}

/// Expected wav structure HEADER (44 bytes) - DATA (bit depth * n_channels) per sample
pub fn read_wav_i16(file_path: &Path) -> Result<(Wav<i16>, i32), std::io::Error> {
    let  fp: File = File::open(file_path)?;
    let mut header_buf: [u8; 44] = [0; 44];

    let mmap = unsafe {Mmap::map(&fp)?};
    let header_bytes = &mmap[..44]; 
    header_buf.clone_from_slice(&header_bytes);
    let wav_header: WavHeader = unsafe { std::mem::transmute::<[u8; 44], WavHeader>(header_buf) };


    let data = &mmap[44..wav_header.data_size as usize + 44];

    let mut buf: Vec<u8> = Vec::with_capacity(wav_header.data_size as usize);
    let mut data_buffer: Vec<i16> = Vec::with_capacity(wav_header.get_required_samples());
    unsafe {
        buf.set_len(wav_header.data_size as usize);
        data_buffer.set_len(wav_header.get_required_samples());
    }


    LittleEndian::read_i16_into(data, &mut data_buffer);
    let res = Wav::<i16>::new(wav_header, data_buffer);
    Ok((res, wav_header.sample_rate))
}

pub fn read_wav_i16_as_f32(file_path: &Path) -> Result<(Wav<f32>, i32), std::io::Error> {
    read_wav_i16(file_path).map(|(wav, sample_rate)| {
        (
            Wav::<f32>::new(
                wav.header,
                wav.data.iter().cloned().map(pcm_i16_to_f32).collect(),
            ),
            sample_rate,
        )
    })
}
