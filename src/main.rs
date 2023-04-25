#![feature(slice_as_chunks)]

pub mod sample;
pub mod wave;

use std::path::Path;
use wave::WavFile;
use criterion::black_box;

const FILES: [&'static str; 20] = ["./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir1/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir1/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir1/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir1/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir2/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir2/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir2/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir2/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir3/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir3/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir3/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir3/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir4/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir4/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir4/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir4/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir5/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/mfall/dir5/lpcnq.wav",
"./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir5/ref.wav","./resources/quickstart_genspeech/LPCNet_listening_test/vec18/dir5/lpcnq.wav"];

fn main() {
    for _ in 0..100{
        for file in FILES {
            let wav = match WavFile::from_file(black_box(Path::new(file))) {
                Ok(wav) => wav,
                Err(e) => panic!("Error reading file: {}", e),
            };
            let _samples = wav.read(None);
        }
    }

}