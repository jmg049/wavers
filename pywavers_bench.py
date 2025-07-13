import pywavers as wavers
import soundfile as sf
import numpy as np
import time
import tempfile
import os
import pandas as pd
from itertools import product
from tqdm import tqdm
from statistics import mean, median
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
DURATIONS = [1, 5, 10]
SAMPLE_RATES = [16000, 44100]
DTYPES = [np.int16, np.int32, np.float32, np.float64]
CHANNELS = [1, 2]
WARMUP_RUNS = 5
MEASURE_RUNS = 100

# --- Helper: WAV generation ---
def generate_wav(path, sr, duration, channels, dtype):
    samples = int(sr * duration)
    data = np.random.rand(samples, channels).astype(np.float32) * 2 - 1
    data = (data * np.iinfo(dtype).max).astype(dtype) if np.issubdtype(dtype, np.integer) else data.astype(dtype)
    sf.write(path, data, sr)

# --- Generic benchmark runner ---
def benchmark(func, warmup_runs, measure_runs):
    for _ in range(warmup_runs):
        try:
            func()
        except Exception:
            return None, "Warmup failed"

    times = []
    for _ in range(measure_runs):
        try:
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        except Exception as e:
            return None, str(e)

    return {
        "min_ms": min(times),
        "mean_ms": mean(times),
        "median_ms": median(times),
        "max_ms": max(times),
        "runs": times,
    }, None

# --- Main benchmark ---
def run_benchmarks():
    results = []
    cases = list(product(DURATIONS, SAMPLE_RATES, DTYPES, CHANNELS))

    with tempfile.TemporaryDirectory() as tmpdir, tqdm(total=len(cases), desc="Benchmarking reads") as pbar:
        for duration, sr, dtype, channels in cases:
            fname = f"{duration}s_{sr}Hz_{dtype.__name__}_{channels}ch.wav"
            path = os.path.join(tmpdir, fname)
            generate_wav(path, sr, duration, channels, dtype)

            # --- PyWavers ---
            def read_wavers():
                wavers.read(path, dtype=dtype)

            res_wavers, err_wavers = benchmark(read_wavers, WARMUP_RUNS, MEASURE_RUNS)
            info = None
            try:
                info = wavers.info(path)
            except Exception:
                pass

            results.append({
                "tool": "pywavers",
                "file": fname,
                "duration_s": duration,
                "sample_rate": sr,
                "dtype": dtype.__name__,
                "channels": channels,
                "min_ms": res_wavers["min_ms"] if res_wavers else None,
                "mean_ms": res_wavers["mean_ms"] if res_wavers else None,
                "median_ms": res_wavers["median_ms"] if res_wavers else None,
                "max_ms": res_wavers["max_ms"] if res_wavers else None,
                "runs": res_wavers["runs"] if res_wavers else None,
                "error": err_wavers,
                "format": getattr(info, "format", None),
                "n_samples": getattr(info, "n_samples", None),
            })

            # --- SoundFile ---
            def read_sf():
                sf.read(path, dtype='float32' if dtype == np.float32 else 'float64' if dtype == np.float64 else 'int32' if dtype == np.int32 else 'int16')

            res_sf, err_sf = benchmark(read_sf, WARMUP_RUNS, MEASURE_RUNS)

            results.append({
                "tool": "soundfile",
                "file": fname,
                "duration_s": duration,
                "sample_rate": sr,
                "dtype": dtype.__name__,
                "channels": channels,
                "min_ms": res_sf["min_ms"] if res_sf else None,
                "mean_ms": res_sf["mean_ms"] if res_sf else None,
                "median_ms": res_sf["median_ms"] if res_sf else None,
                "max_ms": res_sf["max_ms"] if res_sf else None,
                "runs": res_sf["runs"] if res_sf else None,
                "error": err_sf,
                "format": getattr(info, "format", None),
                "n_samples": getattr(info, "n_samples", None),
            })

            pbar.set_postfix_str(fname)
            pbar.update(1)

    return pd.DataFrame(results)

# --- Run & save ---
df = run_benchmarks()
df.to_csv("wavers_vs_soundfile_benchmark.csv", index=False)
df.to_json("wavers_vs_soundfile_benchmark.json", orient="records", indent=2)

# --- Summary Table ---
summary = df[df["error"].isnull()].groupby(["tool", "dtype"])[["mean_ms", "min_ms", "median_ms", "max_ms"]].mean().round(2)
print("\n📊 Average Timing Summary (ms):")
print(summary)

# --- Boxplot ---
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df[df["error"].isnull()].explode("runs"),
    x="dtype", y="runs", hue="tool"
)
plt.ylabel("Read Time (ms)")
plt.title("Read Performance: PyWavers vs SoundFile")
plt.savefig("read_benchmark_comparison_boxplot.png")
plt.tight_layout()
plt.show()
