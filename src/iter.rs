use crate::{Sample, WavFile};

pub struct WavIterator {
    samples: Vec<Sample>,
    idx: usize,
    max_idx: usize,
}

impl WavIterator {
    pub fn new(wav_file: &WavFile, as_type: Option<Sample>) -> Self {
        let samples = wav_file.read(as_type);
        Self {
            samples,
            idx: 0,
            max_idx: wav_file.len(),
        }
    }
}

impl Iterator for WavIterator {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.max_idx {
            return None;
        }

        let sample = self.samples[self.idx];
        self.idx += 1;
        Some(sample)
    }
}

// TODO
#[allow(dead_code)]
pub struct OverlappingWavIterator {
    samples: Vec<Sample>,
    idx: usize,
    max_idx: usize,
    window_size: usize,
    hop_size: usize,
}
