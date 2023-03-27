use std::path::Path;

use wavers::{read_wav_i16, write_pcm_i16};

fn main() -> Result<(),  Box<dyn std::error::Error>> {
    let (header, audio_data) = read_wav_i16(Path::new("./test.wav"))?;
    println!("{:#?}", header);
    write_pcm_i16(Path::new("./__test.wav"), audio_data.as_slice(), header.sample_rate, header.n_channels)?;
    let (header, audio_data) = read_wav_i16(Path::new("./__test.wav"))?;
    println!("{:#?}", header);
    Ok(())
}