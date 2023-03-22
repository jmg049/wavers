use pyo3::prelude::pyclass;

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
