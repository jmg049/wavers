#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sndfile.h>
#include <sys/stat.h>
#include <errno.h>
#include <sys/time.h>

// Define sample formats
typedef enum {
    PCM_16,
    PCM_24,
    PCM_32,
    IEEE_F32,
    IEEE_F64
} SampleFormat;

// Define durations, sample rates, and channel counts
// Match the Rust benchmark settings exactly
const int DURATIONS[] = {1, 10, 60}; // in seconds
const int SAMPLE_RATES[] = {8000, 16000, 44100};
const int N_CHANNELS[] = {1, 2};
const int NUM_DURATIONS = 3;
const int NUM_SAMPLE_RATES = 3;
const int NUM_CHANNELS = 2;

// Statistics structure
typedef struct {
    double mean;
    double min;
    double max;
    double standard_deviation;
    double median;
    double throughput;
    double file_size;
    int sample_count;
} BenchmarkStats;

// High precision timing using gettimeofday
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Function to generate random noise between -1.0 and 1.0
// Similar to the Rust's random_range(-1.0..1.0)
void generate_noise(float* buffer, long num_samples) {
    for (long i = 0; i < num_samples; i++) {
        buffer[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// Get the appropriate format for libsndfile
int get_sndfile_format(SampleFormat format) {
    switch (format) {
        case PCM_16:
            return SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        case PCM_24:
            return SF_FORMAT_WAV | SF_FORMAT_PCM_24;
        case PCM_32:
            return SF_FORMAT_WAV | SF_FORMAT_PCM_32;
        case IEEE_F32:
            return SF_FORMAT_WAV | SF_FORMAT_FLOAT;
        case IEEE_F64:
            return SF_FORMAT_WAV | SF_FORMAT_DOUBLE;
        default:
            return SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    }
}

// Get format name as string
const char* get_format_name(SampleFormat format) {
    switch (format) {
        case PCM_16:
            return "PCM_16";
        case PCM_24:
            return "PCM_24";
        case PCM_32:
            return "PCM_32";
        case IEEE_F32:
            return "IEEE_F32";
        case IEEE_F64:
            return "IEEE_F64";
        default:
            return "UNKNOWN";
    }
}

// Calculate warm-up and measurement times based on duration
// Follows the exact same logic as in the Rust benchmark
void get_benchmark_times(int duration_sec, int* warm_up_time, int* measurement_time, int* sample_size) {
    *warm_up_time = duration_sec / 10;
    if (*warm_up_time < 3) *warm_up_time = 3;
    
    *measurement_time = duration_sec / 2;
    if (*measurement_time < 15) *measurement_time = 15;
    
    *sample_size = (duration_sec > 600) ? 50 : 100;
}

// Compare function for qsort
int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

// Calculate statistics
BenchmarkStats calculate_stats(double* measurements, int count, double file_size) {
    BenchmarkStats stats = {0};
    stats.sample_count = count;
    
    if (count == 0) return stats;
    
    // Sort measurements for median and percentiles
    qsort(measurements, count, sizeof(double), compare_doubles);
    
    // Calculate mean
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += measurements[i];
    }
    stats.mean = sum / count;
    
    // Min and max
    stats.min = measurements[0];
    stats.max = measurements[count - 1];
    
    // Standard deviation
    double variance = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = measurements[i] - stats.mean;
        variance += diff * diff;
    }
    variance /= count;
    stats.standard_deviation = sqrt(variance);
    
    // Median
    if (count % 2 == 0) {
        stats.median = (measurements[count/2 - 1] + measurements[count/2]) / 2.0;
    } else {
        stats.median = measurements[count/2];
    }
    
    // Throughput (bytes per second)
    stats.throughput = file_size / stats.mean;
    stats.file_size = file_size;
    
    return stats;
}

// Benchmark writing operation
BenchmarkStats benchmark_write(SampleFormat format, int duration_sec, int sample_rate, int n_channels) {
    char tmp_filename[256];
    sprintf(tmp_filename, "./tmp_bench_resources/Libsndfile_writing_%s_%ds_%dHz_%dch.wav", 
            get_format_name(format), duration_sec, sample_rate, n_channels);
    
    // Calculate number of samples
    long num_frames = duration_sec * sample_rate;
    long num_samples = num_frames * n_channels;
    
    // Generate noise samples in float format
    float* float_buffer = (float*)malloc(num_samples * sizeof(float));
    if (!float_buffer) {
        fprintf(stderr, "Failed to allocate memory for float buffer\n");
        BenchmarkStats error_stats = {0};
        return error_stats;
    }
    generate_noise(float_buffer, num_samples);
    
    // Calculate warm-up and measurement times
    int warm_up_time, measurement_time, sample_size;
    get_benchmark_times(duration_sec, &warm_up_time, &measurement_time, &sample_size);
    
    printf("Writing %s - %ds - %dHz - %dch (Warm-up: %ds, Measurement: %ds, Samples: %d)...\n", 
           get_format_name(format), duration_sec, sample_rate, n_channels, 
           warm_up_time, measurement_time, sample_size);
    
    // Perform warm-up
    double start_warmup = get_time();
    while ((get_time() - start_warmup) < warm_up_time) {
        SNDFILE* file;
        SF_INFO sfinfo;
        
        memset(&sfinfo, 0, sizeof(sfinfo));
        sfinfo.samplerate = sample_rate;
        sfinfo.channels = n_channels;
        sfinfo.format = get_sndfile_format(format);
        
        file = sf_open(tmp_filename, SFM_WRITE, &sfinfo);
        if (!file) {
            fprintf(stderr, "Failed to open %s for writing: %s\n", tmp_filename, sf_strerror(NULL));
            free(float_buffer);
            BenchmarkStats error_stats = {0};
            return error_stats;
        }
        
        // Write data based on format - always from float buffer to match Rust behavior
        sf_count_t frames_written = sf_writef_float(file, float_buffer, num_frames);
        sf_close(file);
    }
    
    // Perform measurement
    double* measurements = (double*)malloc(sample_size * sizeof(double));
    if (!measurements) {
        fprintf(stderr, "Failed to allocate memory for measurements\n");
        free(float_buffer);
        BenchmarkStats error_stats = {0};
        return error_stats;
    }
    
    int sample_count = 0;
    double start_measurement = get_time();
    
    while (sample_count < sample_size && (get_time() - start_measurement) < measurement_time) {
        SNDFILE* file;
        SF_INFO sfinfo;
        
        memset(&sfinfo, 0, sizeof(sfinfo));
        sfinfo.samplerate = sample_rate;
        sfinfo.channels = n_channels;
        sfinfo.format = get_sndfile_format(format);
        
        double start_sample = get_time();
        
        file = sf_open(tmp_filename, SFM_WRITE, &sfinfo);
        if (!file) {
            fprintf(stderr, "Failed to open %s for writing: %s\n", tmp_filename, sf_strerror(NULL));
            free(float_buffer);
            free(measurements);
            BenchmarkStats error_stats = {0};
            return error_stats;
        }
        
        // Write data
        sf_count_t frames_written = sf_writef_float(file, float_buffer, num_frames);
        
        sf_close(file);
        
        double end_sample = get_time();
        measurements[sample_count++] = end_sample - start_sample;
    }
    
    // Get file size
    struct stat st;
    if (stat(tmp_filename, &st) != 0) {
        fprintf(stderr, "Failed to get file size for %s: %s\n", tmp_filename, strerror(errno));
        free(float_buffer);
        free(measurements);
        BenchmarkStats error_stats = {0};
        return error_stats;
    }
    off_t file_size = st.st_size;
    
    // Calculate statistics
    BenchmarkStats stats = calculate_stats(measurements, sample_count, file_size);
    
    // Clean up
    free(float_buffer);
    free(measurements);
    
    return stats;
}

// Benchmark reading operation
BenchmarkStats benchmark_read(SampleFormat format, int duration_sec, int sample_rate, int n_channels) {
    char tmp_filename[256];
    sprintf(tmp_filename, "./tmp_bench_resources/Libsndfile_reading_%s_%ds_%dHz_%dch.wav", 
            get_format_name(format), duration_sec, sample_rate, n_channels);
    
    // Calculate number of samples
    long num_frames = duration_sec * sample_rate;
    long num_samples = num_frames * n_channels;
    
    // Generate and write a file for reading benchmark
    float* float_buffer = (float*)malloc(num_samples * sizeof(float));
    if (!float_buffer) {
        fprintf(stderr, "Failed to allocate memory for float buffer\n");
        BenchmarkStats error_stats = {0};
        return error_stats;
    }
    generate_noise(float_buffer, num_samples);
    
    SNDFILE* file;
    SF_INFO sfinfo;
    
    memset(&sfinfo, 0, sizeof(sfinfo));
    sfinfo.samplerate = sample_rate;
    sfinfo.channels = n_channels;
    sfinfo.format = get_sndfile_format(format);
    
    file = sf_open(tmp_filename, SFM_WRITE, &sfinfo);
    if (!file) {
        fprintf(stderr, "Failed to open %s for writing: %s\n", tmp_filename, sf_strerror(NULL));
        free(float_buffer);
        BenchmarkStats error_stats = {0};
        return error_stats;
    }
    
    sf_writef_float(file, float_buffer, num_frames);
    sf_close(file);
    
    // Allocate buffer for reading - use appropriate type for each format
    void* read_buffer = NULL;
    
    switch (format) {
        case PCM_16:
            read_buffer = malloc(num_samples * sizeof(short));
            break;
        case PCM_24:
        case PCM_32:
            read_buffer = malloc(num_samples * sizeof(int));
            break;
        case IEEE_F32:
            read_buffer = malloc(num_samples * sizeof(float));
            break;
        case IEEE_F64:
            read_buffer = malloc(num_samples * sizeof(double));
            break;
    }
    
    if (!read_buffer) {
        fprintf(stderr, "Failed to allocate memory for read buffer\n");
        free(float_buffer);
        BenchmarkStats error_stats = {0};
        return error_stats;
    }
    
    // Calculate warm-up and measurement times
    int warm_up_time, measurement_time, sample_size;
    get_benchmark_times(duration_sec, &warm_up_time, &measurement_time, &sample_size);
    
    printf("Reading %s - %ds - %dHz - %dch (Warm-up: %ds, Measurement: %ds, Samples: %d)...\n", 
           get_format_name(format), duration_sec, sample_rate, n_channels, 
           warm_up_time, measurement_time, sample_size);
    
    // Perform warm-up
    double start_warmup = get_time();
    while ((get_time() - start_warmup) < warm_up_time) {
        memset(&sfinfo, 0, sizeof(sfinfo));
        
        file = sf_open(tmp_filename, SFM_READ, &sfinfo);
        if (!file) {
            fprintf(stderr, "Failed to open %s for reading: %s\n", tmp_filename, sf_strerror(NULL));
            free(float_buffer);
            free(read_buffer);
            BenchmarkStats error_stats = {0};
            return error_stats;
        }
        
        // Read data based on format
        sf_count_t frames_read;
        switch (format) {
            case PCM_16:
                frames_read = sf_readf_short(file, (short*)read_buffer, num_frames);
                break;
            case PCM_24:
            case PCM_32:
                frames_read = sf_readf_int(file, (int*)read_buffer, num_frames);
                break;
            case IEEE_F32:
                frames_read = sf_readf_float(file, (float*)read_buffer, num_frames);
                break;
            case IEEE_F64:
                frames_read = sf_readf_double(file, (double*)read_buffer, num_frames);
                break;
        }
        
        sf_close(file);
    }
    
    // Perform measurement
    double* measurements = (double*)malloc(sample_size * sizeof(double));
    if (!measurements) {
        fprintf(stderr, "Failed to allocate memory for measurements\n");
        free(float_buffer);
        free(read_buffer);
        BenchmarkStats error_stats = {0};
        return error_stats;
    }
    
    int sample_count = 0;
    double start_measurement = get_time();
    
    while (sample_count < sample_size && (get_time() - start_measurement) < measurement_time) {
        memset(&sfinfo, 0, sizeof(sfinfo));
        
        double start_sample = get_time();
        
        file = sf_open(tmp_filename, SFM_READ, &sfinfo);
        if (!file) {
            fprintf(stderr, "Failed to open %s for reading: %s\n", tmp_filename, sf_strerror(NULL));
            free(float_buffer);
            free(read_buffer);
            free(measurements);
            BenchmarkStats error_stats = {0};
            return error_stats;
        }
        
        // Read data based on format
        sf_count_t frames_read;
        switch (format) {
            case PCM_16:
                frames_read = sf_readf_short(file, (short*)read_buffer, num_frames);
                break;
            case PCM_24:
            case PCM_32:
                frames_read = sf_readf_int(file, (int*)read_buffer, num_frames);
                break;
            case IEEE_F32:
                frames_read = sf_readf_float(file, (float*)read_buffer, num_frames);
                break;
            case IEEE_F64:
                frames_read = sf_readf_double(file, (double*)read_buffer, num_frames);
                break;
        }
        
        sf_close(file);
        
        double end_sample = get_time();
        measurements[sample_count++] = end_sample - start_sample;
    }
    
    // Get file size
    struct stat st;
    if (stat(tmp_filename, &st) != 0) {
        fprintf(stderr, "Failed to get file size for %s: %s\n", tmp_filename, strerror(errno));
        free(float_buffer);
        free(read_buffer);
        free(measurements);
        BenchmarkStats error_stats = {0};
        return error_stats;
    }
    off_t file_size = st.st_size;
    
    // Calculate statistics
    BenchmarkStats stats = calculate_stats(measurements, sample_count, file_size);
    
    // Clean up
    free(float_buffer);
    free(read_buffer);
    free(measurements);
    
    return stats;
}

// Write results to a JSON file for Python analysis
void write_json_results(BenchmarkStats* write_stats, BenchmarkStats* read_stats, 
                        SampleFormat format, int duration_sec, int sample_rate, int n_channels,
                        FILE* json_file) {
    
    const char* format_name = get_format_name(format);
    
    fprintf(json_file, "  {\n");
    fprintf(json_file, "    \"operation\": \"write\",\n");
    fprintf(json_file, "    \"format\": \"%s\",\n", format_name);
    fprintf(json_file, "    \"duration_sec\": %d,\n", duration_sec);
    fprintf(json_file, "    \"sample_rate\": %d,\n", sample_rate);
    fprintf(json_file, "    \"channels\": %d,\n", n_channels);
    fprintf(json_file, "    \"mean_time\": %.9f,\n", write_stats->mean);
    fprintf(json_file, "    \"min_time\": %.9f,\n", write_stats->min);
    fprintf(json_file, "    \"max_time\": %.9f,\n", write_stats->max);
    fprintf(json_file, "    \"median_time\": %.9f,\n", write_stats->median);
    fprintf(json_file, "    \"std_dev\": %.9f,\n", write_stats->standard_deviation);
    fprintf(json_file, "    \"throughput_bytes_per_sec\": %.2f,\n", write_stats->throughput);
    fprintf(json_file, "    \"file_size_bytes\": %.0f,\n", write_stats->file_size);
    fprintf(json_file, "    \"sample_count\": %d\n", write_stats->sample_count);
    fprintf(json_file, "  },\n");
    
    fprintf(json_file, "  {\n");
    fprintf(json_file, "    \"operation\": \"read\",\n");
    fprintf(json_file, "    \"format\": \"%s\",\n", format_name);
    fprintf(json_file, "    \"duration_sec\": %d,\n", duration_sec);
    fprintf(json_file, "    \"sample_rate\": %d,\n", sample_rate);
    fprintf(json_file, "    \"channels\": %d,\n", n_channels);
    fprintf(json_file, "    \"mean_time\": %.9f,\n", read_stats->mean);
    fprintf(json_file, "    \"min_time\": %.9f,\n", read_stats->min);
    fprintf(json_file, "    \"max_time\": %.9f,\n", read_stats->max);
    fprintf(json_file, "    \"median_time\": %.9f,\n", read_stats->median);
    fprintf(json_file, "    \"std_dev\": %.9f,\n", read_stats->standard_deviation);
    fprintf(json_file, "    \"throughput_bytes_per_sec\": %.2f,\n", read_stats->throughput);
    fprintf(json_file, "    \"file_size_bytes\": %.0f,\n", read_stats->file_size);
    fprintf(json_file, "    \"sample_count\": %d\n", read_stats->sample_count);
    fprintf(json_file, "  },\n");
}

// Main function that runs all the benchmarks
int main(int argc, char* argv[]) {
    // Seed the random number generator
    srand(time(NULL));
    
    // Create temp directory
    int result = mkdir("./tmp_bench_resources", 0777);
    if (result != 0 && errno != EEXIST) {
        fprintf(stderr, "Failed to create directory ./tmp_bench_resources: %s\n", strerror(errno));
        return 1;
    }
    
    // Open JSON file for results
    FILE* json_file = fopen("libsndfile_benchmark_results.json", "w");
    if (!json_file) {
        fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
        return 1;
    }
    
    fprintf(json_file, "[\n");
    
    // Set up all sample formats to test
    SampleFormat formats[] = {PCM_16, PCM_24, PCM_32, IEEE_F32, IEEE_F64};
    const int num_formats = 5;
    
    // Counter for tracking progress
    int total_benchmarks = num_formats * NUM_DURATIONS * NUM_SAMPLE_RATES * NUM_CHANNELS * 2; // *2 for read and write
    int completed_benchmarks = 0;
    
    for (int format_idx = 0; format_idx < num_formats; format_idx++) {
        SampleFormat format = formats[format_idx];
        
        for (int duration_idx = 0; duration_idx < NUM_DURATIONS; duration_idx++) {
            int duration_sec = DURATIONS[duration_idx];
            
            for (int sr_idx = 0; sr_idx < NUM_SAMPLE_RATES; sr_idx++) {
                int sample_rate = SAMPLE_RATES[sr_idx];
                
                for (int ch_idx = 0; ch_idx < NUM_CHANNELS; ch_idx++) {
                    int n_channels = N_CHANNELS[ch_idx];
                    
                    printf("\n[%d/%d] Benchmarking %s - %ds - %dHz - %dch\n", 
                           completed_benchmarks + 1, total_benchmarks,
                           get_format_name(format), duration_sec, sample_rate, n_channels);
                    
                    // Run write benchmark
                    BenchmarkStats write_stats = benchmark_write(format, duration_sec, sample_rate, n_channels);
                    completed_benchmarks++;
                    
                    printf("[%d/%d] ", completed_benchmarks + 1, total_benchmarks);
                    
                    // Run read benchmark
                    BenchmarkStats read_stats = benchmark_read(format, duration_sec, sample_rate, n_channels);
                    completed_benchmarks++;
                    
                    // Write results to JSON
                    write_json_results(&write_stats, &read_stats, format, duration_sec, sample_rate, n_channels, json_file);
                    
                    // Print progress information
                    printf("Progress: %.1f%%\n", (float)completed_benchmarks / total_benchmarks * 100.0);
                }
            }
        }
    }
    
    // Finalize JSON (remove trailing comma from last entry and close array)
    fseek(json_file, -2, SEEK_CUR); // Move back 2 characters to remove the last comma and newline
    fprintf(json_file, "\n]\n");
    fclose(json_file);
    
    // Clean up temp directory
    system("rm -rf ./tmp_bench_resources");
    
    printf("\nBenchmark complete. Results saved to libsndfile_benchmark_results.json\n");
    
    return 0;
}