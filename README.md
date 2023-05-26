<!-- [crates.io](https://crates.io/crates/wavers)

[Documentation](https://docs.rs/wavers/)

 -->
# Wavers
Wavers is a minimal wav file reader/writing written in Rust. It is minimal in the sense that it only cares about the essential chunks of a wave file when reading and writing, i.e. The ``RIFF``, ``WAVE``, ``fmt `` and ``data`` chunks. These are the only chunks read and written in the current version of Wavers. Wavers can read wav files with other chunks such as ``LIST``, but they get ignored and do not get written.
**<p style="text-align: center;">[![Crates.io version][crate-img]][crate] [![Documentation][docs-img]][docs]</p>**

[crate-pywavers]: https://crates.io/crates/wavers
[crate-img]:     https://img.shields.io/crates/v/wavers.svg
[crate]:         https://crates.io/crates/wavers

[docs-img]:      https://img.shields.io/badge/docs-online-blue.svg
[docs]:          https://docs.rs/wavers

---
## PyWavers
Wavers but in Python! An even bigger WIP than Wavers.

**<p style="text-align: center;">[![Crates.io version][pywavers-crate-img]][crate-pywavers] [![Documentation][pywavers-docs-img]][pywavers-docs]</p>**

[crate-pywavers]: https://crates.io/crates/pywavers
[pywavers-crate-img]:     https://img.shields.io/crates/v/pywavers.svg

[pywavers-docs]:          https://docs.rs/pywavers
[pywavers-docs-img]:      https://img.shields.io/badge/docs-online-blue.svg



# Motivation
There were several motivating factors when deciding to write Wavers. Firstly, my PhD involves quite a bit of audio processing and I have been working almost exclusively with wav files using Python. Python is a fantastic language but I have always had issues with aspects such as baseline memory usage. Secondly, after being interested in learning a more low-level language and not being bothered with the likes of C/C++ (again), Rust caught my attention. Thirdly, I had to do a Speech and Audio processing module, which involved a project. Mixing all of these together led me to start this project and gave me a deadline and goals for an MVP of Wavers.

Rust also has a limited number of audio-processing libraries and even fewer specific to the wav format. Currently, the most popular wav reader/writer crate is [hound](https://github.com/ruuda/hound), by ruuda. Hound is currently used as the **wav file reader/writer** for other Rust audio libraries such as [Rodio](https://github.com/RustAudio/rodio), **but Hound has last updated in September 2022**. The biggest **general-purpose audio-processing** library for Rust though is [CPAL](https://github.com/rustaudio/cpal). CPAL, the Cross-Platform Audio Library is a low-level library for audio input and output in pure Rust. Finally, there is also [Symphonia](https://github.com/pdeljanov/Symphonia), another general-purpose media encoder/decoder crate for Rust. Symphonia supports the reading and writing of wav files.

# Project Goals
The goals of the project are:
- Create a wav file reader/writer written in Rust.
    - It should support a variety of different wav formats, starting with PCM_16, PCM_32, Float_32 and Float_64.
    - It should support easy and efficient conversion between these formats.
- The library should maintain strong performance testing and benchmarking as a method of furthering the project in terms of performance.
    - Aspiration is to get close to the performance of [LibSndFile](https://github.com/libsndfile/libsndfile), an audio-processing library written in C. LibSndFile is the backend used for the [Soundfile](https://github.com/bastibe/python-soundfile) Python library.
- The library should be well-tested and testing should be a priority before pushing any feature/update/change to main.
- The library should provide an easy-to-use interface to the user.
- The library should be minimal in terms of dependencies.
- It should be possible to expose Wavers to Python to provide a Python package for reading and writing wav files backed by Rust.

# Anatomy of a wav file
The wave file format is a widely supported format for storing digital audio. A wave file uses the Resource Interchange File Format (RIFF) file structure and organizes the data in the file in chunks. Each chunk contains information about its type and size and can easily be skipped by software that does not understand the specific chunk type.

Excluding the RIFF chunk there is no guaranteed order to the remaining chunks and only the fmt and data chunk are guaranteed to exist. This means that rather than a specific structure for decoding chunks, chunks must be discovered by seeking through the wav file and reading the chunk ID aSeptember 2022nd chunk size fields. Then if the chunk is needed, read the chunk according to the chunk format, and if not, skip ahead by the size of the chunk. 

The chunk types supported and used by Wavers are described below. There are plans to expand the number of supported chunks as time goes on and the library matures.

## <u>RIFF</u>  
Total Bytes = 8 + 4 for the Chunk name + the size of the rest of the file less 8 bytes for the Chunk ID and size.

| Byte Sequence Description | Length(Bytes) | Offset(Bytes) | Description |
|:---| :---: |:---: |:---|
| Chunk ID                  | 4             | 0      | The character string "RIFF"|
| Size| 4 | 4 | The size, in bytes of the chunk|
| RIFF type ID | 4 | 8 | The character string "WAVE"|
| Chunks | $x$ |  12 | The other chunks |

## <u>fmt </u> 
Total bytes = 20 + 4 for the Chunk name
| Byte Sequence Description | Length(Bytes) | Offset(Bytes) | Description |
|:---| :---: |:---: |:---|
| Chunk ID                  | 4             | 0      | The character string "fmt " (note the space!)|
| Size| 4 | 4 | The size, in bytes of the chunk|
| Compression Code | 2 | 6 | Indicates the format of the wav file, e.g. PCM = 1 and IEEE Float = 3|
| Number of Channels | 2 |  8 | The number of channels in the wav file |
| Sampling Rate | 4 |  12 | The rate at which the audio is sampled |
| Byte Rate | 4 |  16 | Bytes per second |
| Block Align | 2 |  18 | Minimum atomic unit of data |
| Bits per Sample | 2 |  20 | The number of bits per sample, e.g. PCM_16 has 16 bits |

## <u>Data Chunk</u>
Total bytes = 4 for the chunk name + 4 for the size and then $x$ bytes for the data.
| Byte Sequence Description | Length(Bytes) | Offset(Bytes) | Description |
|:---| :---: |:---: |:---|
| Chunk ID                  | 4             | 0      | The character string "data" |
| size | 4 | 4 | The size of the data in bytes |
| data | $x$ | 8 | The encoded audio data |



# Using Wavers
A [key goal](#Project-Goals) of Wavers was to make an easy-to-use interface for reading and writing wav files. However, Rust is a typed programming language and is pretty strict when it comes to knowing what everything is. On top of this, there are a variety of different types that represent the data, for example, ``16-bit integers`` or ``32-bit floats``. Having to get the user to always specify/know the type doesn't further the goal of an easy-to-use interface, but luckily, Rust offers zero-cost abstraction through its  ``traits`` functionality and ``enum`` types. This allows for expressive and efficient code, that works for many different types, without incurring any additional runtime overhead. 

Just in case you just need to see how you can read and write Wavers here's everything in a nutshell

## Quick Guide - Reading
Channel data is interleaved in the returned vector.

```rust
use std::path::Path;
use wavers::{read, SampleType};

let input_fp = Path::new('./my_wav_file.wav')
//if we know that the wav file is encoded in the desired type, then we don't
// need to pass the type to function
let pcm_16_data: Vec<SampleType> = read(input_fp).unwrap();

// As PCM_16 (16-bit integers)
let pcm_16_data: Vec<SampleType> = read(input_fp, SampleType::I16(0)).unwrap();

// As PCM_32 (32-bit integers)
let pcm_32_data: Vec<SampleType> = read(input_fp, SampleType::I32(0)).unwrap();

// As FLOAT_32 (32-bit floats)
let float_32_data: Vec<SampleType> = read(input_fp, SampleType::F32(0.0)).unwrap();

// As FLOAT_64 (64-bit floats)
let pcm_32_data: Vec<SampleType> = read(input_fp, SampleType::F64(0.0)).unwrap();
``` 

## Quick Guide - Writing

```rust
use std::path::Path;
use wavers::{read, write_wav_as, SampleType};

let input_fp = Path::new("./my_test_wav.wav");
let wav_data_i16: Vec<SampleType> = read(input_fp, SampleType::I16(0));

// Re-encode it to IEEE 32-bit floats.
let output_fp = Path::new("./my_test_wav_f32.wav");
let sample_rate = 16000;
let n_channels = 1;
match write_wav_as(output_fp, &wav_data_i16, SampleType::F32(0.0), sample_rate, n_channels) {
    Ok(()) => println!("Succes"),
    Err(err) => println!("Failed with err {}", err),
}

```

## SampleType & Sample
To represent the different possible formats that can be read from a wav file, an enum was used. That enum is given in the code block below and is used to represent the type of a single sample in a wav file. Rust wav files allow values to be stored within the variants of the enum.

```rust
pub enum SampleType {
    I16(i16),
    I32(i32),
    F32(f32),
    F64(f64),
}
```
Then to allow for easy conversion between the different sample types a rust trait is used. The ``Sample`` trait is given in the code block below. A custom conversion function is necessary for an audio reader/writer as when converting between the likes of ``PCM_16`` and ``Float_32`` one cannot simply cast the 16-bit integer to a 32-bit float, as the range of values for ``PCM_16`` is $-32768$ to $+32768$, while ``Float_32`` is $-1.0$ to $1.0$.
In combination, the ``SampleType`` enum and ``Sample`` trait allows for each type/format of wav data to be converted easily between the available types. Each variant of ``SampleType`` can have the functions, ``to_i16()``, ``to_i32()``, ``to_f32()`` and ``to_f64()`` called on them to retrieve the value of variant as that type.  

The ``Sample`` trait must be implemented for the types available in ``SampleType``.

```rust
pub trait Sample {
    fn convert_to_i16(self) -> i16;
    fn convert_to_i32(self) -> i32;
    fn convert_to_f32(self) -> f32;
    fn convert_to_f64(self) -> f64;
}

```
Now that we can represent the individual samples of a wav file we need to represent the wav file itself. Introducing the ``wavers::wave::WavFile`` struct.

## WavFile
Wavers represents a wav file as the format chunk, the underlying byte data and the current seek position. The data is a heap-allocated array owned by the ``WavFile`` struct. The ``wavers::wave::FmtChunk`` struct just contains the fields [outlined above](#fmt-)
```rust 
pub struct WavFile {
    pub fmt_chunk: FmtChunk,
    pub data: Box<[u8]>,
    pub seek_pos: u64,
}
```
There are three ways of creating a ``WavFile``:
1. Manually, using ``WavFile::new(fmt_chunk: FmtChunk, data: Box<[u8]>, seek_pos: u64)``, or
2. From a file path, ``WavFile::from_file(fp: &Path)``, or
3. From existing wav data, ``WavFile::from_data(data: Array2<SampleType>, sample_rate: i32)``.

In the current implementation of Wavers, when using the ``from_file`` function the full wav data is read from the file and stored on the heap. This will likely change in future to a lazy approach where data is only loaded when needed. 

If you only need information relating to the format of the file, for example, the sample rate, number of channels or duration, then they can be determined without reading the entire file. See [Helper Functions](#helper-functions) for more details.

Now that the wav file has been loaded the data can be fully read or be written as the desired type.

## Reading
Wavers offers many ways to read the sample data contained within a wav file. These options include:
1. Reading the wav file in its native encoding. The type is inferred from the ``CompressionFormat`` and ``Bits per Sample`` fields of the wav header.
2. Reading the wav file as a particular, supported, type. The user passes the desired ``SampleType`` to the function.
3. Read the wav file, as a specified or unspecified type, in one step by using the ``wavers::read`` function.

```rust
use std::path::Path;
use wavers::{WavFile, SampleType, read};

fn main() {
    let input_fp = Path::new("./my_test_wav.wav");
    let wav_file = WavFile::from_path(input_fp).unwrap();

    // 1. Read using defaults
    let wav_data_default_format = wav_file.read(None);

    // 2. Read as a specific type
    let wav_data_as_f32 = wav_file.read(Some(SampleType::F32(0.0)));

    // 3. Read without creating a WavFile
    let wav_data_default_format = read(input_fp, None).unwrap();
    let wav_data_as_i16 = read(input_fp, Some(SampleType::I16(0)).unwrap();
}
```
By opting for methods 1 and 2, which explicitly create a ``WavFile`` variable, it is easy to re-decode the data as another ``SampleType``. However, achieving this with the last method is not much more work at all and can be achieved using the code below.

```rust
use std::path::Path;
use wavers::{WavFile, SampleType, read};

fn main() {
    let input_fp = Path::new("./my_test_wav.wav");
    let wav_data: Vec<SampleType> = read(input_fp, SampleType::I16(0)).unwrap();
    // do some other things ...
    
    let wav_data = wav_data.iter()
                           .map(|sample| sample.convert_to(Some(SampleType::F32(0.0))))
                           .collect::<Vec<_>>();

    // Planned syntax for in-place conversions
    wav_data.convert_to(SampleType::F32(0.0));
}
``` 

## Writing
There are two methods for writing a wav file with Wavers:

1. Writing a ``WavFile`` with its default encoding.
2. Writing wav data (a vector of samples) as a specific encoding without manually creating a ``WavFile``.

```rust
use std::path::Path;
use wavers::{WavFile, SampleType, write_wav_as};

fn main() {
    let input_fp = Path::new("./my_test_wav.wav");
    let wav_file = WavFile::from_file(input_fp).unwrap();

    // 1. Write the wav data based on the ``WavFile`` encoding
    match wav_file.write(Path::new("./my_output_wav.wav")) {
        Ok(()) => println!("Success"),
        Err(err) => eprintln!("Failed to write wav file due to {}", err),
    }

    let wav_data = wav_file.read(None);
    let sample_rate = 16000;
    // 2. Write the wav data as a specific type.

    match write_wav_as(Path::new("./my_output_wav.wav"), &wav_data_f32, Some(SampleType::F32(0.0), sample_rate) {
        Ok(()) => println!("Success"),
        Err(err) => eprintln!("Failed to write wav file due to {}", err),
    }

}
```

## Helper Functions
Wavers offers a collection of helper functions which relate to retrieving information about a particular wav file. None of the helper functions described below decode the underlying wav file data, but instead decode the relevant parts of the wav file header. These helper functions do not cover all/most of the fields of a wav file yet. There are plans to extend these functions as the project develops.

The ``SignalInfo`` struct best exemplifies the data which can be retrieved from a wav file by Wavers. The fields stored in the ``SignalInfo`` struct can be retrieved using the ``wavers::signal_info`` function. These fields can also be retrieved individually without using a ``SignalInfo`` struct.

```rust 
pub struct SignalInfo {
    pub sample_rate: i32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub duration: u64,
}
```

```rust
use std::path::Path;
use wavers::{signal_info, signal_duration, signal_channels, signal_sample_rate};

fn main() {
    let input_fp = Path::new("./my_test_wav.wav");

    let duration = signal_duration(input_fp).expect("Failed to read file");
    let channels = signal_channels(input_fp).expect("Failed to read file");
    let sample_rate = signal_sample_rate(input_fp).expect("Failed to read file");
    let signal_info = signal_info(input_fp).expect("Failed to read file");
}
```

# PyWavers - Wavers in Python
A key goal of Wavers is to expose its functionality to Python. This would allow for it to be used by a much wider audience and would provide a typed, mostly-memory safe backend. The goal of providing an easy-to-use interface originated from Python and audio processing packages such as SoundFile and ScipyIO. These packages offer ``read`` and ``write`` functions that allow the user to have to worry about the nitty-gritty types and other low-level aspects like reading/creating a header for a wav file. 

In a nutshell, the Python package works very similarly to the Rust version and like other well-established audio processing packages.

```python
import numpy as np
import pywavers as pw

if __name__ == '__main__':
    data_f32 = pywavers.read('./my_test_wav.wav' dtype=np.float32)
    pywavers.write('./my_output_test_wav.wav', data_f32, sample_rate=16000, dtype=np.int16)
```

For more information on the PyWavers side of the project please visit the [PyWavers Github](https://github.com/jmg049/PyWavers).