use std::io::{Read, Seek};

use crate::{chunks::RIFF, WaversResult};
#[derive(Debug)]
pub struct MMapReader {
    mmap: memmap2::Mmap,
    position: usize,
}

impl MMapReader {
    pub fn new(mmap: memmap2::Mmap) -> WaversResult<Self> {
        println!("First 4 bytes: {:?}", &mmap[0..4]);
        Ok(Self { mmap, position: 0 })
    }
}

impl Read for MMapReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.mmap.get(self.position..self.position + buf.len()) {
            Some(slice) => {
                println!("Position: {}", self.position);
                println!("Slice len: {}", slice.len());
                self.position += slice.len();
                println!("{:?}", slice);
                println!("{:?}", RIFF);
                panic!();
                Ok(slice.len())
            }
            None => Ok(0),
        }
    }
}

impl Seek for MMapReader {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match pos {
            std::io::SeekFrom::Start(offset) => {
                self.position = offset as usize;
                Ok(self.position as u64)
            }
            std::io::SeekFrom::Current(offset) => {
                let new_position = (self.position as i64 + offset) as usize;
                if new_position > self.mmap.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Seek out of bounds",
                    ));
                }
                self.position = new_position;
                Ok(self.position as u64)
            }
            std::io::SeekFrom::End(offset) => {
                let new_position = (self.mmap.len() as i64 + offset) as usize;
                if new_position > self.mmap.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Seek out of bounds",
                    ));
                }
                self.position = new_position;
                Ok(self.position as u64)
            }
        }
    }
}
