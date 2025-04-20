
# WAV Library Read and Write Benchmarks - libsndfile vs. Wavers vs. hound

## 1 Methodology in Brief

The benchmark exercises 180 parameter combinations (5 sample formats × 3 sample‑rates × 3 durations × 2 channel counts, each for read + write).  

* A warm‑up phase removes cold‑start artefacts.  
* Up to 100 timed iterations per configuration supply robust mean/median figures.  
* Open → process → close is timed as a single operation, matching typical “whole‑file” workflows.  

---

## 2 Reading Performance  

| Scenario | Fastest library | Relative observations |
|----------|-----------------|-----------------------|
|Mono, 1 s, 8 kHz, PCM‑16|`wavers`|2.79 µs (Wavers) vs 5.02 µs (libsndfile), 52.3 µs (hound)|
|Mono, 1 s, 44.1 kHz, PCM‑16|`wavers`|5.19 µs (Wavers) vs 6.45 µs (libsndfile) vs 294 µs (hound) |
|Mono, 10 s, 8 kHz, PCM‑16|`libsndfile`|7.95 µs (libsndfile) vs 13.3 µs (wavers) vs 514 µs (hound) |

* `wavers` records the lowest latency in every short‑file test and maintains that lead for most longer files.  
* `libsndfile` narrows—or reverses—the gap as file duration and size increase, suggesting its internal buffering becomes advantageous.  
* `hound` incurs higher per‑sample cost across the board, most evident at higher sample rates.

---

## 3 Writing Performance  

| Scenario | Fastest library | Relative observations |
|----------|-----------------|-----------------------|
|Mono, 1 s, 44.1 kHz, PCM‑16|`hound`|302 µs (hound) vs 960 µs (Wavers) vs 5599 µs (libsndfile)|
|Mono, 10 s, 8 kHz, PCM‑16|`libsndfile`|714 µs (libsndfile) vs 982 µs (hound) vs 1121 µs (wavers) |
|Stereo, 1 s, 8 kHz, PCM‑32|`wavers`|181 µs (Wavers) vs 192 µs (libsndfile) vs 235 µs (hound)|

* For very small files at high sample rates, `hound` shows the shortest write time; its minimal header‑generation path dominates the timing.  
* On longer writes (≥ 10 s) and lower rates, `libsndfile` generally delivers the best throughput.  
* `wavers`’ write times remain close to libsndfile in most integer‑format tests and lead when 24‑ or 32‑bit PCM is involved.

---

## 4 Effect of Individual Parameters  

| Parameter | Observed scaling |
|-----------|------------------|
|**Sample format**|Integer → float adds ≤ 10 % in `wavers` and libsndfile; ~40 – 50 % in `hound`.|
|**Sample rate**|Per‑frame time increases almost linearly for `wavers` and libsndfile; non‑linear growth (×5–10) in `hound`.|
|**Duration**|Time grows sub‑linearly with duration for all libraries; fixed overheads are amortised quickly.|
|**Channels (1 → 2)**|Timing doubles in libsndfile and wavers (ideal scaling); increase is >2× in `hound`, implying extra per‑channel overhead.|

---

## 5 Practical Guidance  

| Requirement | Candidate libraries |
|-------------|--------------------|
|Fastest reads, broad format set (incl. 24‑/64‑bit float)|`wavers` or `libsndfile` (both close; `wavers` leads in most short‑file cases)|
|Sustained high‑throughput writes (minutes‑long exports)|`libsndfile`|
|Tiny telemetry logs, limited to PCM‑16/32/F32|`hound` (small binary, quickest small‑file writes)|
|Archival in IEEE F64|`wavers` or `libsndfile` (not supported by `hound`)|

---

## 6 Limitations and Next Steps

* **Format and channel scope.** Only 8–44.1 kHz, mono/stereo cases were profiled. Extending to 96 kHz+ and >2 channels would provide a fuller picture.  
* **Platform features.** The benchmark is single‑threaded; libraries offering asynchronous or multi‑threaded paths were not exercised.  
* **External resources.** Memory consumption, CPU events and low‑level cache behaviour were not recorded; a complementary study is planned.  

Within its defined scope, the data indicate that no single library dominates every workload:  

* `wavers` tends to minimise read latency and maintains competitive write times across all supported formats.  
* `libsndfile` achieves the highest sustained write throughput for larger files and offers the widest feature set.  
* `hound` is efficient for short PCM‑16/32/F32 writes and has the lightest dependency footprint.

The results show no single library is universally “best”; each one excels along different axes that matter in practice:

| Library        | Read Profile                                                                                                                         | Write Profile                                                                                                    | Format coverage                                    | Ecosystem & integration                                                           |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------------------------------------|
| wavers (Rust)  | Lowest mean latency across every short‑file test and many longer runs; very consistent std‑dev figures (sub‑µs jitter)               | Within 10 – 15 % of the fastest implementation in most integer‑PCM cases and top performer for 24‑/32‑bit writes | PCM‑16/24/32, IEEE‑F32/64                          | Memory‑safe, pure‑Rust, feature rich, doc-rich, benchmarked and a Python API;                      |
| libsndfile (C) | Second‑fastest reader overall, occasionally leading once file size is large enough for its internal buffer to dominate transfer cost | Highest sustained throughput on ≥ 10 s files at all tested rates; scales linearly with channels                  | Same set as wavers plus RF64 & non‑standard chunks | Mature C ABI, bindings for Python, Rust, Go, etc.; widely deployed in DAWs & CLIs |
| hound (Rust)   | Higher per‑sample overhead; suitable when read speed is not critical                                                                 | Fastest for very small (< 1 s) PCM‑16/32/IEEE‑F32 writes thanks to lightweight header emission                   | PCM‑16/32, IEEE‑F32 only                           | Single‑crate, zero‑FFI dependency; ideal for embedded or audit‑friendly builds and small storage footprint (38.5Kib vs. 199Kib (Wavers) and approx. 535Kib (on my Kubuntu machine))   |

Key take‑aways for users:

* Latency‑sensitive readers (DSP front‑ends, packetised streaming) benefit most from wavers thanks to its minimal per‑frame cost and broad sample‑format support.
* Bulk exporters and long offline renders are still best served by libsndfile, whose mature buffering yields the greatest bytes‑per‑second once header overhead is amortised.
* Resource‑constrained or audit‑heavy projects that touch only PCM‑16/32/F32 can pick hound for its tiny footprint and excellent small‑file write speeds.
* Feature completeness matters. If 24‑bit integer, 64‑bit float, or RF64 is required, wavers or libsndfile are the only candidates in this set. Conversely, if eliminating native dependencies is a priority, hound is the only pure‑Rust option.

Future benchmarks should extend the matrix (96 kHz+, multichannel, threaded I/O) to validate whether these performance hierarchies persist under higher data rates and parallel workloads, but within the present scope the data provide a clear decision table rather than a single winner.
