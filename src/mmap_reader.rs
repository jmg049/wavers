use std::io::Cursor;
use memmap2::MMap;

pub struct MMapReader {
    mmap: Cursor<MMap>
}

impl MMapReader {
    pub fn new(mmap: MMap) -> Self {
        Self {
            mmap: Cursor::new(mmap)
        }
    }
}

impl ReadSeek for MMapReader;