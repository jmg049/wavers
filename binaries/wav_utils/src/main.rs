use std::{fs::File, io::BufReader, path::Path, time::Duration};

use audio_samples::{AudioEditing, AudioSample, AudioSamples, ConvertTo, I24};
use clap::{Parser, Subcommand};
use indicatif::ProgressBar;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use wavers::WaversResult;

#[derive(Parser)]
#[command(name = "wav_utils")]
#[command(about = "A utility for WAV file manipulation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Split {
        #[arg(help = "Input WAV file path")]
        input: String,
        #[arg(short, long, help = "Output directory (default: output)")]
        output: Option<String>,
        #[arg(short, long, help = "Window size in samples")]
        window_size: u32,
    },
    Resample {
        #[arg(help = "Input WAV file path")]
        input: String,
        #[arg(short, long, help = "Output file path")]
        output: Option<String>,
        #[arg(short, long, help = "Target sample rate")]
        sample_rate: u32,
    },
    Convert {
        #[arg(help = "Input WAV file path")]
        input: String,
        #[arg(short, long, help = "Output file path")]
        output: Option<String>,
        #[arg(short, long, help = "Target type (i16, i32, i64, f32, f64)")]
        target_type: String,
    },
}

fn main() -> WaversResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Split {
            input,
            output,
            window_size,
        } => {
            let params = SplitParameters::new(window_size);
            split_wav(input, output, params)?;
        }
        Commands::Resample {
            input,
            output,
            sample_rate,
        } => {
            println!("Resample functionality not yet implemented");
        }
        Commands::Convert {
            input,
            output,
            target_type,
        } => {
            println!("Convert functionality not yet implemented");
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum SupportedTypes {
    I16,
    // I24,
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy)]
pub struct SplitParameters {
    pub window_size: u32,
}

impl SplitParameters {
    pub fn new(window_size: u32) -> Self {
        Self { window_size }
    }

    pub fn window_duration_seconds(&self, sr: u32) -> u64 {
        self.window_size as u64 / sr as u64
    }
}

pub enum Tools<P: AsRef<Path>> {
    SplitWav(P, Option<P>, SplitParameters),
    ResampleWav(P, Option<P>, u32),
    ConvertWav(P, Option<P>, SupportedTypes),
}

pub fn split_wav<P: AsRef<Path>>(
    fp: P,
    output_path: Option<P>,
    params: SplitParameters,
) -> WaversResult<()>
where
    i16: AudioSample,
    I24: AudioSample,
    i32: AudioSample,
    f32: AudioSample,
    f64: AudioSample,
{
    let mut wav_data = wavers::Wav::<BufReader<File>>::from_path(fp)?;
    let split_duration = params.window_duration_seconds(wav_data.sample_rate());

    fn _write<T: AudioSample, P: AsRef<Path>>(
        wav_data: AudioSamples<T>,
        split_duration: f64,
        output_path: Option<P>,
    ) -> WaversResult<()>
    where
        T: FromPrimitive + ToPrimitive + num_traits::One,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        wav_data
            .split(split_duration as f64)
            .unwrap()
            .iter()
            .enumerate()
            .for_each(|(i, segment)| {
                let output_file = match &output_path {
                    Some(path) => path.as_ref().join(format!("segment_{}.wav", i)),
                    None => Path::new("output").join(format!("segment_{}.wav", i)),
                };
                if !output_file.exists() {
                    std::fs::create_dir_all(output_file.parent().unwrap())
                        .expect("Failed to create output directory");
                }
                wavers::write_audio(output_file, segment).expect("Failed to write audio segment")
            });
        Ok(())
    }

    match wav_data.wav_type {
        wavers::WavType::Pcm16 => _write::<i16, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::Pcm24 => _write::<I24, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::Pcm32 => _write::<i32, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::Float32 => _write::<f32, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::Float64 => _write::<f64, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::EPcm16 => _write::<i16, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::EPcm24 => _write::<I24, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::EPcm32 => _write::<i32, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::EFloat32 => _write::<f32, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        wavers::WavType::EFloat64 => _write::<f64, P>(
            wav_data.to_audio_samples()?,
            split_duration as f64,
            output_path,
        )?,
        _ => unreachable!(),
    };

    Ok(())
}
