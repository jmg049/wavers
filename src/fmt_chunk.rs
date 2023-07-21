use std::{io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write}, fs::File, path::Path};

const RIFF: &[u8; 4] = b"RIFF";
const DATA: &[u8; 4] = b"data";
const WAVE: &[u8; 4] = b"WAVE";
const FMT: &[u8; 4] = b"fmt ";

///
/// A struct for storing the necessary format information about a wav file.
///
/// In total the struct is 20 bytes. 4 bytes storing the size of the chunk
/// and 16 bytes for the format information.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FmtChunk {
    /// Size of the chunk in bytes. Constant value of 16.
    pub size: i32,
    /// Format of the audio data. 1 for PCM, 3 for IEEE float.
    pub format: u16,
    /// Number of channels in the audio data.
    pub channels: u16,
    /// Sample rate of the audio data.
    pub sample_rate: i32,
    /// Byte rate of the audio data.
    pub byte_rate: i32,
    /// Block align of the audio data.
    pub block_align: u16,
    /// Bits per sample of the audio data.
    pub bits_per_sample: u16,
}

impl FmtChunk {
    ///
    /// Create a new ``FmtChunk`` from the given parameters.
    ///
    pub fn new(
        size: i32,            // 4
        format: u16,          // 2
        channels: u16,        // 2
        sample_rate: i32,     // 4
        byte_rate: i32,       // 4
        block_align: u16,     // 2
        bits_per_sample: u16, //2
    ) -> FmtChunk {
        FmtChunk {
            size,
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        }
    }

    ///
    /// Create a new ``FmtChunk`` from the given file path.
    ///
    /// Returns an error if the file cannot be opened or the ``FmtChunk`` cannot be read.
    ///

    pub fn from_path(signal_fp: &Path) -> Result<FmtChunk, std::io::Error> {
        let wav_file = File::open(signal_fp)?;
        let mut br = BufReader::new(wav_file);
        FmtChunk::from_buf_reader(&mut br)
    }

    ///
    /// Create a new ``FmtChunk`` from the given ``BufReader``.
    ///
    /// Returns an error if the ``FmtChunk`` cannot be read.
    ///

    pub fn from_buf_reader(br: &mut BufReader<File>) -> Result<FmtChunk, std::io::Error> {
        let mut buf: [u8; 4] = [0; 4];
        let mut buf_two: [u8; 2] = [0; 2];
        let (offset, _) = find_sub_chunk_id(br, b"fmt ")?;
        br.seek(SeekFrom::Start(offset as u64))?;
        br.read_exact(&mut buf)?;
        let size = i32::from_ne_bytes(buf);

        br.read_exact(&mut buf_two)?;
        let format = u16::from_ne_bytes(buf_two);

        br.read_exact(&mut buf_two)?;
        let channels = u16::from_ne_bytes(buf_two);

        br.read_exact(&mut buf)?;
        let sample_rate = i32::from_ne_bytes(buf);

        br.read_exact(&mut buf)?;
        let byte_rate = i32::from_ne_bytes(buf);

        br.read_exact(&mut buf_two)?;
        let block_align = u16::from_ne_bytes(buf_two);

        br.read_exact(&mut buf_two)?;
        let bits_per_sample = u16::from_ne_bytes(buf_two);
        br.seek(SeekFrom::Start(0))?;
        Ok(FmtChunk::new(
            size,
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        ))
    }

    ///
    /// Convert the ``FmtChunk`` to a byte array.
    ///

    pub fn as_bytes(&self) -> [u8; 24] {
        let mut buf: [u8; 24] = [0; 24];
        buf[0..4].copy_from_slice(FMT);
        buf[4..8].copy_from_slice(&self.size.to_ne_bytes());
        buf[8..10].copy_from_slice(&self.format.to_ne_bytes());
        buf[10..12].copy_from_slice(&self.channels.to_ne_bytes());
        buf[12..16].copy_from_slice(&self.sample_rate.to_ne_bytes());
        buf[16..20].copy_from_slice(&self.byte_rate.to_ne_bytes());
        buf[20..22].copy_from_slice(&self.block_align.to_ne_bytes());
        buf[22..24].copy_from_slice(&self.bits_per_sample.to_ne_bytes());
        buf
    }

    // Helper functions to avoid having to use the fmt struct field directly

    ///
    /// Get the number of bytes per sample.
    pub fn get_sample_size(&self) -> usize {
        self.bits_per_sample as usize / 8
    }

    /// Get the wave file data format.
    pub fn format(&self) -> u16 {
        self.format
    }

    /// Get the number of channels.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    /// Get the byte rate.
    pub fn byte_rate(&self) -> i32 {
        self.byte_rate
    }

    /// Get the block align.
    pub fn block_align(&self) -> u16 {
        self.block_align
    }

    /// Get the number of bits per sample.
    pub fn bits_per_sample(&self) -> u16 {
        self.bits_per_sample
    }
}

///
/// A function which searchs for a sub-chunk id in a wave file currently stored in a ``BufReader``.
///
/// Returns the offset of the sub-chunk id and the size of the sub-chunk.
///
/// Returns an error if the sub-chunk id cannot be found.
///

fn find_sub_chunk_id(
    file: &mut BufReader<File>,
    chunk_id: &[u8; 4],
) -> Result<(usize, usize), std::io::Error> {
    let mut buf: [u8; 4] = [0; 4];
    // Find the RIFF Tag

    file.read_exact(&mut buf)?;
    if !buf_eq(&buf, RIFF) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to find RIFF tag in {:?}", file.get_ref()),
        ));
    }

    file.seek(SeekFrom::Current(8))?;
    let mut tag_offset: usize = 0;
    let mut bytes_traversed: usize = 12;
    loop {
        // First sub-chunk is guaranteed to begin at byte 12 so seek forward by 8.
        // No other chunk is at a guaranteed offset.
        let bytes_read = file.read(&mut buf)?;
        if bytes_read == 0 {
            break;
        }

        bytes_traversed += bytes_read;

        if buf_eq(&buf, chunk_id) {
            tag_offset = bytes_traversed;
        }

        let bytes_read = file.read(&mut buf)?;
        if bytes_read == 0 {
            break;
        }
        bytes_traversed += bytes_read;

        let chunk_len =
            buf[0] as u32 | (buf[1] as u32) << 8 | (buf[2] as u32) << 16 | (buf[3] as u32) << 24;
        if tag_offset > 0 {
            let chunk_size = chunk_len as usize;
            file.seek(SeekFrom::Start(0))?; // Reset the file offset to the beginning
            return Ok((tag_offset, chunk_size));
        }
        file.seek(SeekFrom::Current(chunk_len as i64))?;

        bytes_traversed += chunk_len as usize;
    }
    file.seek(SeekFrom::Start(0))?;

    Err(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!(
            "Failed to find {:?} tag in {:?}",
            std::str::from_utf8(chunk_id).unwrap(),
            file.get_ref()
        ),
    ))
}