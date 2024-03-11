pub(crate) const FACT: [u8; 4] = *b"fact";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactChunk {
    pub num_samples: u32,
}

impl FactChunk {
    pub(crate) fn new(num_samples: u32) -> Self {
        Self { num_samples }
    }

    pub(crate) fn as_bytes(&self) -> [u8; 12] {
        let mut bytes = [0; 12];
        bytes[0..4].copy_from_slice(&FACT);
        bytes[4..8].copy_from_slice(&4u32.to_ne_bytes());
        bytes[8..12].copy_from_slice(&self.num_samples.to_ne_bytes());
        bytes
    }
}

impl Default for FactChunk {
    fn default() -> Self {
        FactChunk { num_samples: 0 }
    }
}
