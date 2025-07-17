#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sndfile.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/sysinfo.h>
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

// Configuration for benchmark
typedef struct {
    size_t n_iter;
    size_t n_warmup;
} BenchConfig;

// Benchmark condition descriptor
typedef struct {
    const char *func;      // "read" or "write"
    const char *sample_type; // "i16", "i32", "f32", "f64"
    int sample_rate;
    unsigned int duration; // seconds
    int channels;
} BenchCondition;

// Timing utility: returns elapsed microseconds
static long time_us(void (*fn)(void *), void *arg) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    fn(arg);
    clock_gettime(CLOCK_MONOTONIC, &end);
    long sec = end.tv_sec - start.tv_sec;
    long nsec = end.tv_nsec - start.tv_nsec;
    return sec * 1000000 + nsec / 1000;
}

// Synthetic audio generation:
void generate_sine(void *buffer, size_t frames, int channels, const char *type) {
    for (size_t i = 0; i < frames * channels; ++i) {
        double v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        if (strcmp(type, "i16") == 0) {
            ((int16_t*)buffer)[i] = (int16_t)(v * INT16_MAX);
        } else if (strcmp(type, "i32") == 0) {
            ((int32_t*)buffer)[i] = (int32_t)(v * INT32_MAX);
        } else if (strcmp(type, "f32") == 0) {
            ((float*)buffer)[i] = (float)v;
        } else if (strcmp(type, "f64") == 0) {
            ((double*)buffer)[i] = v;
        }
    }
}

// Read benchmark function wrapper arg
typedef struct {
    BenchCondition *cond;
    const char *tmpfile;
} ReadArg;

void bench_read(void *arg) {
    ReadArg *ra = (ReadArg*)arg;
    SF_INFO sfinfo;
    SNDFILE *infile = sf_open(ra->tmpfile, SFM_READ, &sfinfo);
    if (!infile) {
        perror("sf_open for read"); exit(1);
    }
    size_t frames = sfinfo.frames;
    int channels = sfinfo.channels;
    const char *type = ra->cond->sample_type;
    void *buf = malloc(frames * channels * 
        (strcmp(type, "i16")==0 ? sizeof(int16_t) :
         strcmp(type, "i32")==0 ? sizeof(int32_t) :
         strcmp(type, "f32")==0 ? sizeof(float) : sizeof(double)));
    sf_readf_short(infile, buf, frames); // always reading into i16 for simplicity
    sf_close(infile);
    free(buf);
}

// Write benchmark wrapper arg
typedef struct {
    BenchCondition *cond;
    void *buffer;
    size_t frames;
    const char *type;
} WriteArg;

void bench_write(void *arg) {
    WriteArg *wa = (WriteArg*)arg;
    SF_INFO sfinfo = {0};
    sfinfo.samplerate = wa->cond->sample_rate;
    sfinfo.channels = wa->cond->channels;
    sfinfo.format = SF_FORMAT_WAV |
        (strcmp(wa->type, "i16")==0 ? SF_FORMAT_PCM_16 :
         strcmp(wa->type, "i32")==0 ? SF_FORMAT_PCM_32 :
         strcmp(wa->type, "f32")==0 ? SF_FORMAT_FLOAT : SF_FORMAT_DOUBLE);
    SNDFILE *outfile = sf_open("temp_test.wav", SFM_WRITE, &sfinfo);
    if (!outfile) {
        perror("sf_open for write"); exit(1);
    }
    sf_writef_short(outfile, wa->buffer, wa->frames); // writing i16 for simplicity
    sf_close(outfile);
}

int main(void) {
    srand((unsigned)time(NULL));
    BenchConfig config = { .n_iter = 10000, .n_warmup = 10 };
    int sample_rates[] = {8000, 16000, 44100};
    unsigned int durations[] = {1, 5, 10, 30, 60};
    int channels_arr[] = {1, 2};
    const char *sample_types[] = {"i16", "i32", "f32", "f64"};
    const char *functions[] = {"read", "write"};

    // Open JSON output
    FILE *out = fopen("C_benchmark_results.json", "w");
    fprintf(out, "[");
    int first = 1;

    for (size_t f = 0; f < ARRAY_LEN(functions); ++f) {
        for (size_t t = 0; t < ARRAY_LEN(sample_types); ++t) {
            for (size_t sr = 0; sr < ARRAY_LEN(sample_rates); ++sr) {
                for (size_t d = 0; d < ARRAY_LEN(durations); ++d) {
                    printf("\n  // %s %s %dHz %us %dch\n", functions[f], sample_types[t], sample_rates[sr], durations[d], channels_arr[0]);
                    for (size_t ch = 0; ch < ARRAY_LEN(channels_arr); ++ch) {
                        BenchCondition cond = {
                            .func = functions[f],
                            .sample_type = sample_types[t],
                            .sample_rate = sample_rates[sr],
                            .duration = durations[d],
                            .channels = channels_arr[ch]
                        };
                        size_t frames = cond.sample_rate * cond.duration;
                        size_t total_samples = frames * cond.channels;
                        // Allocate buffer for synthetic audio
                        void *buffer = malloc(total_samples *
                            (strcmp(cond.sample_type, "i16")==0 ? sizeof(int16_t) :
                             strcmp(cond.sample_type, "i32")==0 ? sizeof(int32_t) :
                             strcmp(cond.sample_type, "f32")==0 ? sizeof(float) : sizeof(double)));
                        generate_sine(buffer, frames, cond.channels, cond.sample_type);
                        
                        // Write temp file if benchmarking read
                        if (strcmp(cond.func, "read")==0) {
                            SF_INFO sfinfo = {0};
                            sfinfo.samplerate = cond.sample_rate;
                            sfinfo.channels = cond.channels;
                            sfinfo.format = SF_FORMAT_WAV |
                                (strcmp(cond.sample_type, "i16")==0 ? SF_FORMAT_PCM_16 :
                                 strcmp(cond.sample_type, "i32")==0 ? SF_FORMAT_PCM_32 :
                                 strcmp(cond.sample_type, "f32")==0 ? SF_FORMAT_FLOAT : SF_FORMAT_DOUBLE);
                            SNDFILE *tmp = sf_open("temp_test.wav", SFM_WRITE, &sfinfo);
                            sf_writef_short(tmp, buffer, frames);
                            sf_close(tmp);
                        }
                        
                        // Warmup
                        for (size_t w = 0; w < config.n_warmup; ++w) {
                            if (strcmp(cond.func, "read")==0) {
                                ReadArg ra = { .cond = &cond, .tmpfile = "temp_test.wav" };
                                bench_read(&ra);
                            } else {
                                WriteArg wa = { .cond = &cond, .buffer = buffer, .frames = frames, .type = cond.sample_type };
                                bench_write(&wa);
                                remove("temp_test.wav");
                            }
                        }
                        // Benchmark iterations
                        long total_time = 0;
                        for (size_t it = 0; it < config.n_iter; ++it) {
                            if (!first && it == 0) fprintf(out, ",\n");
                            long t_us;
                            if (strcmp(cond.func, "read")==0) {
                                ReadArg ra = { .cond = &cond, .tmpfile = "temp_test.wav" };
                                t_us = time_us(&bench_read, &ra);
                            } else {
                                WriteArg wa = { .cond = &cond, .buffer = buffer, .frames = frames, .type = cond.sample_type };
                                t_us = time_us(&bench_write, &wa);
                                remove("temp_test.wav");
                            }
                            total_time += t_us;
                        }
                        double avg_us = (double)total_time / config.n_iter;
                        
                        // JSON record
                        if (!first) fprintf(out, ",\n");
                        first = 0;
                        fprintf(out,
                            "  { \"func\": \"%s\", \"type\": \"%s\", \"sr\": %d, \"duration\": %u, \"channels\": %d, \"avg_us\": %.2f }",
                            cond.func, cond.sample_type, cond.sample_rate,
                            cond.duration, cond.channels, avg_us);

                        free(buffer);
                    }
                }
            }
        }
    }
    fprintf(out, "\n]\n");
    fclose(out);

    // Simple report to stdout
    printf("Benchmark completed. Results written to benchmark_results.json\n");
    return 0;
}
