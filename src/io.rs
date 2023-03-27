use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use memmap2::MmapOptions;
use std::io::Write;
use std::{fs::File, io::Read, path::Path};

use crate::{WavHeader, PCM_16};

pub fn read_wav_i16(file_path: &Path) -> Result<(WavHeader, Vec<i16>), std::io::Error> {
    let mut fp: File = File::open(file_path)?;

    let mut buf: Vec<u8> = vec![0; 44];
    fp.read_exact(&mut buf[..])?;
    let wav_header: WavHeader = unsafe {
        std::mem::transmute_copy::<[u8; 44], WavHeader>(
            buf[..].try_into().expect("Failed to read header!"),
        )
    };

    let mut data: Vec<i16> = Vec::with_capacity(wav_header.get_required_samples());
    unsafe {
        data.set_len(wav_header.get_required_samples());
    }

    // Very temporary if - need to determine bands where mmap becomes better for performance
    let mmap =
        unsafe { MmapOptions::new().map_copy_read_only(&fp) }.expect("Unable to create memmap");

    for (idx, sample) in mmap[44..wav_header.get_required_samples() + 44]
        .chunks_exact(2)
        .enumerate()
    {
        data[idx] = LittleEndian::read_i16(sample);
    }

    Ok((wav_header, data))
}

pub fn read_wav_f32(file_path: &Path) -> Result<(WavHeader, Vec<f32>), std::io::Error> {
    let (header, i16_data) = read_wav_i16(file_path)?;
    Ok((header, pcm_i16_to_ieee_f32(i16_data.as_slice())))
}

const PCM_DIV: f32 = 1.0 / 0x7FFF as f32;

fn pcm_i16_to_ieee_f32(i16_input: &[i16]) -> Vec<f32> {
    let mut dst: Vec<f32> = vec![0.0; i16_input.len()];
    for i in 0..i16_input.len() {
        dst[i] = PCM_DIV * i16_input[i] as f32
    }
    dst
}

fn ieee_f32_tp_pcm_16(data: &[f32]) -> Vec<i16> {
    let mut dst: Vec<i16> = vec![0; data.len()];
    for i in 0..data.len() {
        dst[i] = (data[i] * 0x7FFF as f32).clamp(-32767.0, 32767.0) as i16;
    }
    dst
}

pub fn write_pcm_i16(
    file_path: &Path,
    data: &[i16],
    sample_rate: i32,
    n_channels: i16,
) -> Result<(), std::io::Error> {
    println!("Data Size : {:?}", data.len());
    let mut f_out: File = File::create(file_path)?;
    let byte_rate = sample_rate * n_channels as i32 * (16 / 2) as i32;
    let header: WavHeader = WavHeader::new(
        36 + data.len() as i32,
        16,
        PCM_16,
        n_channels,
        sample_rate,
        byte_rate,
        n_channels * (16 / 2),
        PCM_16,
        data.len() as i32 * 16 as i32,
    );

    let header_bytes = header.as_bytes();
    File::write_all(&mut f_out, header_bytes)?;
    for sample in data.iter() {
        f_out.write_i16::<LittleEndian>(*sample)?;
    }
    Ok(())
}

// pub struct WavWriterOptionsBuilder<T> {

//     audio_format: i16,
//     n_channels: i16,
//     sample_rate: i32,
//     bits_per_sample: i16,
// }

// impl<T> WavWriterOptionsBuilder<T> {

//     pub fn new(data: Vec<i16>) -> WavWriterOptionsBuilder<T> {

//         WavWriterOptionsBuilder { audio_format: PCM_16, n_channels: (), sample_rate: (), bits_per_sample: () }
//     }

//     pub fn audio_format(&mut self, audio_format: i16) -> WavWriterOptions {
//         self.audio_format = audio_format;
//         self
//     }
//     pub fn n_channels(&mut self, n_channels: i16) -> WavWriterOptions {
//         self.n_channels = n_channels;
//         self
//     }
//     pub fn sample_rate(&mut self, sample_rate: i32) -> WavWriterOptions {
//         self.sample_rate = sample_rate;
//         self
//     }
//     pub fn bits_per_sample(&mut self, bits_per_sample: i16) -> WavWriterOptions {
//         self.bits_per_sample = bits_per_sample;
//         self
//     }

//     pub fn build(&self) -> WavWriterOptions {
//         WavWriterOptions::new(self.audio_format, self.n_channels, self.sample_rate, self.bits_per_sample)
//     }
// }

// pub struct WavWriterOptions {
//    audio_format: i16,
//     n_channels: i16,
//     sample_rate: i32,
//     bits_per_sample: i16,
// }

// impl WavWriterOptions {
//     pub fn new(audio_format: i16, n_channels: i16, sample_rate: i32, bits_per_sample: i16) -> WavWriterOptions {
//         WavWriterOptions { audio_format, n_channels, sample_rate, bits_per_sample }
//     }
// }
