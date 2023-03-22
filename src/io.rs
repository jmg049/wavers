use std::{fs::File, io::Read, path::Path};

use byteorder::{ByteOrder, LittleEndian};
use memmap2::MmapOptions;

use crate::WavHeader;

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
