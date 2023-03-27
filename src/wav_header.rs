use std::collections::HashSet;

use pyo3::prelude::pyclass;

pub const RIFF: [u8; 4] = ['R' as u8, 'I' as u8, 'F' as u8, 'F' as u8];
pub const DATA: [u8; 4] = ['D' as u8, 'A' as u8, 'T' as u8, 'A' as u8];
pub const FMT: [u8; 4] = ['F' as u8, 'M' as u8, 'T' as u8, ' ' as u8];
pub const WAVE: [u8; 4] = ['W' as u8, 'A' as u8, 'V' as u8, 'E' as u8];
pub const PCM_16: i16 = 0x01;



#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[pyclass]
pub struct WavHeader {
    riff: [u8; 4],         // 4
    size: i32,             // 4
    format: [u8; 4],       // 4
    sub_chunk_id: [u8; 4], // 4
    sub_chunk_size: i32,   // 4
    audio_format: i16,     // 2
    pub n_channels: i16,       // 2
    pub sample_rate: i32,      // 4
    byte_rate: i32,        // 4
    block_align: i16,      // 2
    bits_per_sample: i16,  // 2
    data: [u8; 4],         // 4
    data_size: i32,        // 4
}

impl WavHeader {
    pub fn new(
        size: i32,
        sub_chunk_size: i32,
        audio_format: i16,
        n_channels: i16,
        sample_rate: i32,
        byte_rate: i32,
        block_align: i16,
        bits_per_sample: i16,
        data_size: i32,
    ) -> WavHeader {
        let supported_formats: HashSet<i16> = HashSet::from([0x01, 0x02]);
        let supported_channels: HashSet<i16> = HashSet::from([0x01, 0x02]);
        assert!(
            supported_channels.contains(&n_channels),
            "{:?} channels are not currently supported",
            n_channels
        );
        assert!(
            supported_formats.contains(&audio_format),
            "{:?} is not a currently supported audio format",
            n_channels
        );

        WavHeader {
            riff: RIFF,
            size,
            format: FMT,
            sub_chunk_id: WAVE,
            sub_chunk_size,
            audio_format,
            n_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            data: DATA,
            data_size,
        }
    }

    pub fn get_duration_float(&self) -> f32 {
        (self.bits_per_sample as f32 / (32 * self.sample_rate) as f32) * self.data_size as f32
    }

    pub fn get_required_samples(&self) -> usize {
        (self.sample_rate as f32 * self.get_duration_float()) as usize * self.n_channels as usize
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { any_as_u8_slice(self) }
    }
}

pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}
