pub mod wave;

use std::{fs::File, io::Write, path::Path};

use wave::{overlapping_chunks, WavFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A vec of the numbers 0-20
    let data: Vec<i16> = (0..101).collect();
    println!("{:?}", data);
    overlapping_chunks(data, 4, 3);

    // let test_data = WavFile::new(&Path::new("./test.wav"))?;
    // // let mut out_file = File::create("out.txt")?;
    // let data = test_data.read_pcm_i16();
    // println!("{}", data.len());
    // // let strings: Vec<String> = test_data.read_pcm_i16()?.iter().map(|n| n.to_string()).collect();
    // // writeln!(out_file, "{}", strings.join(" "))?;
    // println!("{:?}", data);
    Ok(())
}
