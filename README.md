
<div style="text-align: center;" align="center">
<h1>WaveRS - A Wav File Reader/Writer</h1>


**<p style="text-align: center;"> [![Crates.io][crate-img]][crate] [![Documentation][docs-img]][docs]  [![crate-pywavers-img]][crate-pywavers] [![crate-pywavers-docrs-img]][pywavers-docrs] [![crate-pywavers-readdocs-img]][pywavers-readdocs]</p>**

[crate-pywavers]: https://github.com/jmg049/pywavers

[crate-pywavers-img]: https://img.shields.io/badge/PyWavers-56B4E9?style=for-the-badge&labelColor=gray

[crate-pywavers-docrs-img]: https://img.shields.io/badge/Docs-56B4E9?style=for-the-badge&labelColor=gray


[crate-pywavers-readdocs-img]:  https://img.shields.io/badge/Read_The_Docs-56B4E9?style=for-the-badge&labelColor=gray

[crate]:         https://crates.io/crates/wavers

[crate-img]:     https://img.shields.io/badge/WaveRs-009E73?style=for-the-badge&labelColor=gray


[docs-img]:      https://img.shields.io/badge/docs-009E73.svg?style=for-the-badge&labelColor=gray

[docs]:          https://docs.rs/wavers

[pywavers-readdocs]: https://readthedocs.org/projects/pywavers

[pywavers-docrs]: https://docs.rs/pywavers

<p>
    <strong>
	WaveRs (pronounced wavers) is a Wav file reader/writer written in Rust and designed to fast and easy to use. WaveRs is also available in Python through the PyWaveRs package. 
    </strong>
</p>

<p>
    <h3>
        <a href="https://docs.rs/wavers">Getting Started</a>
        <span> · </span>
        <a href="">Benchmarks</a>
    </h3>
</p>
</div>

---

# Getting Started
This sections provides a quick overview of the functionality offered by WaveRs to help you get started quickly. WaveRs allows the user to read, write and perform conversions between different types of sampled audio, currently, ``i16``, ``i32``, ``f32`` and ``f64``. There is now **experimental** support for``i24`` now.

For more details on the project and wav files see the [WaveRs Project](#the-wavers-project) section below. For more detailed information on the functionality offered by WaveRs see the [the docs](https://docs.rs/wavers).

## Reading

```rust
use wavers::{Wav, read};
use std::path::Path;

fn main() {
	let fp = "path/to/wav.wav";
    // creates a Wav file struct, does not read the audio data. Just the header information.
    let wav: Wav<i16> = Wav::from_path(fp).unwrap();
	let samples: Samples<i16> = wav.read().unwrap();
    // or to read the audio data directly
    let (samples, sample_rate): (Samples<i16>, i32) = read::<i16, _>(fp).unwrap();
    // samples can be derefed to a slice of samples
    let samples: &[i16] = &samples;
}
```


## Conversion
```rust
use wavers::{Wav, read, ConvertTo};
use std::path::Path;

fn main() {
    // Two ways of converted a wav file
    let fp: "./path/to/i16_encoded_wav.wav";
    let wav: Wav<f32> = Wav::from_path(fp).unwrap();
    // conversion happens automatically when you read
    let samples: &[f32] = &wav.read().unwrap();

    // or read and then call the convert function on the samples.
    let (samples, sample_rate): (Samples<i16>, i32) = read::<i16, _>(fp).unwrap();
    let samples: &[f32] = &samples.convert();
}
```

## Writing
```rust
use wavers::Wav;
use std::path::Path;

fn main() {
	let fp: &Path = &Path::new("path/to/wav.wav");
	let out_fp: &Path = &Path::new("out/path/to/wav.wav");

	// two main ways, read and write as the type when reading
    let wav: Wav<i16> = Wav::from_path(fp).unwrap();
    wav.write(out_fp).unwrap();

	// or read, convert, and write
    let (samples, sample_rate): (Samples<i16>,i32) = read::<i16, _>(fp).unwrap();
    let sample_rate = wav.sample_rate();
    let n_channels = wav.n_channels();

    let samples: &[f32] = &samples.convert();
    write(out_fp, samples, sample_rate, n_channels).unwrap();
}
```

## Wav Utilities

```rust
use wavers::wav_spec;

fn main() {
 	let fp = "path/to/wav.wav";
    let wav: Wav<i16> = Wav::from_path(fp).unwrap();
    let sample_rate = wav.sample_rate();
    let n_channels = wav.n_channels();
    let duration = wav.duration();
    let encoding = wav.encoding();
    let (sample_rate, n_channels, duration, encoing) = wav_spec(fp).unwrap();
}
```

## Features
The following section describes the features available in the WaveRs crate. 

### Ndarray
The ``ndarray`` feature is used to provide functions that allow wav files to be read as ``ndarray`` 2-D arrays (samples x channels). There are two functions provided, ``into_ndarray`` and ``as_ndarray``. Both functions create an ``Array2`` from the samples and return it alongside the sample rate of the audio. ``into_ndarray`` consume the input ``Wav`` struct and ``as_ndarray`` does not.


```rust
use wavers::{read, Wav, AsNdarray, IntoNdarray};
use ndarray::{Array2, CowArray2};

fn main() {
	let fp = "path/to/wav.wav";
    let wav: Wav<i16> = Wav::from_path(fp).unwrap();

    // does not consume the wav file struct
	let (i16_array, sample_rate): (Array2<i16>, i32) = wav.as_ndarray().unwrap();
    
   	// consumes the wav file struct
	let (i16_array, sample_rate): (Array2<i16>, i32) = wav.into_ndarray().unwrap();
}
```

### PyO3
The ``pyo3`` feature is used to provide interoperabilty with the Python, specifically, the [PyWavers](https://github.com/jmg049/pywavers) project (Use Wavers in Python).

# The WaveRs Project
There were several motivating factors when deciding to write Wavers. Firstly, my PhD involves quite a bit of audio processing and I have been working almost exclusively with wav files using Python. Python is a fantastic language but I have always had issues with aspects such as baseline memory usage. Secondly, after being interested in learning a more low-level language and not being bothered with the likes of C/C++ (again), Rust caught my attention. Thirdly, I had to do a Speech and Audio processing module, which involved a project. Mixing all of these together led me to start this project and gave me a deadline and goals for an MVP of Wavers.

Rust also has a limited number of audio-processing libraries and even fewer specific to the wav format. Currently, the most popular wav reader/writer crate is [hound](https://github.com/ruuda/hound), by ruuda. Hound is currently used as the **wav file reader/writer** for other Rust audio libraries such as [Rodio](https://github.com/RustAudio/rodio), **but Hound was last updated in September 2022**. The biggest **general-purpose audio-processing** library for Rust though is [CPAL](https://github.com/rustaudio/cpal). CPAL, the Cross-Platform Audio Library is a low-level library for audio input and output in pure Rust. Finally, there is also [Symphonia](https://github.com/pdeljanov/Symphonia), another general-purpose media encoder/decoder crate for Rust. Symphonia supports the reading and writing of wav files.  Both ``CPAL`` and ``Symphonia`` are very low level and do not offer the ease of use that ``WaveRs`` strives for.

## Project Goals

## Anatomy of Wav File
The wave file format is a widely supported format for storing digital audio. A wave file uses the Resource Interchange File Format (RIFF) file structure and organizes the data in the file in chunks. Each chunk contains information about its type and size and can easily be skipped by software that does not understand the specific chunk type.

Excluding the RIFF chunk there is no guaranteed order to the remaining chunks and only the fmt and data chunk are guaranteed to exist. This means that rather than a specific structure for decoding chunks, chunks must be discovered by seeking through the wav file and reading the chunk ID and chunk size fields. Then if the chunk is needed, read the chunk according to the chunk format, and if not, skip ahead by the size of the chunk. 

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
Total bytes = 16 + 4 for the chunk size + 4 for the chunk name

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

# Benchmarks
The benchmarks below were recorded using ``Criterion`` and each benchmark was run on a small dataset of wav files taken from the GenSpeech data set. The durations vary between approx. 7s and 15s and each file is encoded as PCM_16. The results below are the time taken to load all the wav files in the data set. So the time per file is the total time divided by the number of files in the data set. The data set contains 10 files. There are some suspected anomalies in the benchmarks which warrant further investigation. The benchmarks were run on a desktop PC with the following (relevant) specs: 

- CPU: 13th Gen Intel® Core™ i7-13700KF
- RAM: 32Gb DDR4
- Storage: 1Tb SSD


## Hound vs Wavers - native i16
| benchmark                    | name       | min_time   | mean_time   | max_time   |
|:-----------------------------|:-----------|:-----------|:------------|:-----------|
| Hound vs Wavers - native i16 | Hound Read i16   | 7.4417 ms  | 7.4441 ms   | 7.4466 ms  |
| Hound vs Wavers - native i16 | Wavers Read i16   | 122.42 µs  | 122.56 µs   | 122.72 µs  |
| Hound vs Wavers - native i16 | Hound Write i16  | 2.1900 ms  | 2.2506 ms   | 2.3201ms   |
| Hound vs Wavers - native i16 | Wavers Write i16 | 5.9484 ms  | 6.2091 ms   | 6.5018 ms  |

## Reading
| benchmark   | name                       | min_time   | mean_time   | max_time   |
|:------------|:---------------------------|:-----------|:------------|:-----------|
| Reading     | Native i16 - Read          | 121.28 µs  | 121.36 µs   | 121.44 µs  |
| Reading     | Native i16 - Read Wav File | 121.56 µs  | 121.79 µs   | 122.08 µs  |
| Reading     | Native i16 As f32          | 287.63 µs  | 287.78 µs   | 287.97 µs  |


## Writing
| benchmark   | name                      | min_time   | mean_time   | max_time   |
|:------------|:--------------------------|:-----------|:------------|:-----------|
| Writing     | Slice - Native i16        | 5.9484 ms  | 6.2091 ms   | 6.5018 ms  |
| Writing     | Slice - Native i16 As f32 | 30.271 ms  | 33.773 ms   | 37.509 ms  |
| Writing     | Write native f32          | 11.286 ms  | 11.948 ms   | 12.648 ms  |

