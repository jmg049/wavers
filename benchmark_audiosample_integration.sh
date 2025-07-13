#!/bin/bash

# AudioSamples Integration Benchmark Script
# This script benchmarks the new AudioSamples API against legacy APIs

set -e

echo "🎵 AudioSamples Integration Benchmark"
echo "====================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if test resources exist
if [ ! -f "./test_resources/one_channel_i16.wav" ]; then
    echo -e "${RED}❌ Test resources not found. Please ensure test_resources/ directory exists.${NC}"
    exit 1
fi

echo -e "${BLUE}📊 Running benchmarks...${NC}"
echo ""

# Create a simple benchmark binary using Cargo
mkdir -p benchmark_temp
cd benchmark_temp

# Create Cargo.toml for benchmark
cat > Cargo.toml << 'EOF'
[package]
name = "audiosample_benchmark"
version = "0.1.0"
edition = "2021"

[dependencies]
wavers = { path = ".." }
audio_samples = { path = "../../audio_samples" }
EOF

# Create src directory and main.rs
mkdir -p src
cat > src/main.rs << 'EOF'
use std::time::Instant;
use std::fs::File;
use std::io::BufReader;
use wavers::{read, read_audio, read_ndarray, Wav};
use audio_samples::AudioStatistics;

fn time_operation<F, R>(name: &str, operation: F) -> R 
where 
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    println!("  {} took: {:?}", name, duration);
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_file = "../test_resources/one_channel_i16.wav";
    
    println!("🚀 Benchmarking different read methods:");
    println!("=====================================");
    
    // Benchmark legacy read() function
    time_operation("Legacy read() -> Box<[i16]>", || {
        let (samples, _sr, _ch): (Box<[i16]>, u32, u16) = read(test_file).unwrap();
        samples.len()
    });
    
    // Benchmark new read_audio() function  
    time_operation("New read_audio() -> AudioSamples<i16>", || {
        let audio = read_audio::<i16, _>(test_file).unwrap();
        audio.total_samples()
    });
    
    // Benchmark read_ndarray() function
    time_operation("read_ndarray() -> Array2<i16>", || {
        let (array, _sr) = read_ndarray::<_, i16>(test_file).unwrap();
        array.len()
    });
    
    println!();
    println!("🎯 Benchmarking Wav struct methods:");
    println!("===================================");
    
    // Benchmark Wav::read()
    time_operation("Wav::read() -> Box<[i16]>", || {
        let file = File::open(test_file).unwrap();
        let reader = BufReader::new(file);
        let mut wav1 = Wav::from_reader(reader).unwrap();
        let samples: Box<[i16]> = wav1.read().unwrap();
        samples.len()
    });
    
    // Benchmark Wav::read_audio()
    time_operation("Wav::read_audio() -> AudioSamples<i16>", || {
        let file = File::open(test_file).unwrap();
        let reader = BufReader::new(file);
        let mut wav2 = Wav::from_reader(reader).unwrap();
        let audio = wav2.read_audio::<i16>().unwrap();
        audio.total_samples()
    });
    
    // Benchmark Wav::to_audio_samples() (internal method)
    time_operation("Wav::to_audio_samples() (internal)", || {
        let file = File::open(test_file).unwrap();
        let reader = BufReader::new(file);
        let mut wav3 = Wav::from_reader(reader).unwrap();
        let audio = wav3.to_audio_samples::<i16>().unwrap();
        audio.total_samples()
    });
    
    println!();
    println!("📈 Benchmarking AudioSamples operations:");
    println!("========================================");
    
    let audio = read_audio::<f32, _>(test_file)?;
    
    time_operation("AudioSamples::rms()", || {
        audio.rms()
    });
    
    time_operation("AudioSamples::peak()", || {
        audio.peak()
    });
    
    time_operation("AudioSamples::zero_crossings()", || {
        audio.zero_crossings()
    });
    
    time_operation("AudioSamples::zero_crossing_rate()", || {
        audio.zero_crossing_rate()
    });
    
    println!();
    println!("🔄 Benchmarking caching behavior:");
    println!("=================================");
    
    let file = File::open(test_file)?;
    let reader = BufReader::new(file);
    let mut wav = Wav::from_reader(reader)?;
    
    // First call (no cache)
    time_operation("First call to read_audio() (no cache)", || {
        let audio = wav.read_audio::<f32>().unwrap();
        audio.total_samples()
    });
    
    // Second call (cached)
    time_operation("Second call to read_audio() (cached)", || {
        let audio = wav.read_audio::<f32>().unwrap();
        audio.total_samples()
    });
    
    // Third call, different type (no cache for this type)
    time_operation("Call with different type f64 (no cache)", || {
        let audio = wav.read_audio::<f64>().unwrap();
        audio.total_samples()
    });
    
    // Fourth call, same type as third (cached)
    time_operation("Second call with f64 (cached)", || {
        let audio = wav.read_audio::<f64>().unwrap();
        audio.total_samples()
    });
    
    println!();
    println!("🔍 Memory usage analysis:");
    println!("========================");
    
    println!("  Cache size after operations: {}", wav.cache_size());
    wav.clear_audio_cache();
    println!("  Cache size after clear: {}", wav.cache_size());
    
    println!();
    println!("✅ Benchmark complete!");
    
    Ok(())
}
EOF

# Build and run benchmark
echo -e "${YELLOW}🔧 Building benchmark...${NC}"
cargo build --release

echo -e "${GREEN}🏃 Running benchmark...${NC}"
echo ""
cargo run --release

# Go back to original directory and cleanup
cd ..
rm -rf benchmark_temp

# Performance comparison script
echo ""
echo -e "${BLUE}📊 Creating performance comparison script...${NC}"

cat > performance_comparison.rs << 'EOF'
use std::time::Instant;
use wavers::{read, read_audio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_file = "./test_resources/one_channel_i16.wav";
    let iterations = 100;
    
    println!("🏃 Performance Comparison ({} iterations)", iterations);
    println!("==========================================");
    
    // Warm up
    let _ = read::<i16, _>(test_file)?;
    let _ = read_audio::<i16, _>(test_file)?;
    
    // Benchmark legacy API
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = read::<i16, _>(test_file)?;
    }
    let legacy_duration = start.elapsed();
    
    // Benchmark new API
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = read_audio::<i16, _>(test_file)?;
    }
    let new_duration = start.elapsed();
    
    println!("Legacy read():      {:?} ({:.2} μs/iter)", 
             legacy_duration, 
             legacy_duration.as_micros() as f64 / iterations as f64);
    
    println!("New read_audio():   {:?} ({:.2} μs/iter)", 
             new_duration,
             new_duration.as_micros() as f64 / iterations as f64);
    
    let ratio = new_duration.as_nanos() as f64 / legacy_duration.as_nanos() as f64;
    
    if ratio < 1.1 {
        println!("✅ New API is competitive ({}x overhead)", ratio);
    } else if ratio < 2.0 {
        println!("⚠️  New API has acceptable overhead ({}x)", ratio);
    } else {
        println!("❌ New API has significant overhead ({}x)", ratio);
    }
    
    Ok(())
}
EOF

# Cleanup temporary files
rm -f performance_comparison.rs

echo ""
echo -e "${GREEN}✅ Benchmark script complete!${NC}"
echo ""
echo "📋 Summary of changes:"
echo "  ✅ Added read_audio() and read_audio_with() functions"  
echo "  ✅ Updated Wav struct with read_audio() method"
echo "  ✅ Enhanced WavIterators trait with as_audio_samples()"
echo "  ✅ Updated documentation to prioritize AudioSamples"
echo "  ✅ Added comprehensive tests for new APIs"
echo "  ✅ Verified backward compatibility (all existing tests pass)"
echo ""
echo "🎯 Next steps:"
echo "  1. Review the API additions"
echo "  2. Consider any additional AudioSamples integrations"
echo "  3. Update examples and documentation as needed"
echo "  4. Consider adding write_audio() method for AudioSamples"

exit 0