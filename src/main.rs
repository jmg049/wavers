#![feature(iter_collect_into)]
use std::{ error::Error,
};

mod wav;

fn main() -> Result<(), Box<dyn Error>> {
    let paths = std::fs::read_dir("./data").unwrap();
    let mut v = Vec::with_capacity(22);
    for path in paths {
        let name = path.unwrap().path();
        match wav::read_wav_i16_as_f32(&name) {
            Ok(data) => v.push(data),
            Err(_) => continue,
        }
    }

    Ok(())
}