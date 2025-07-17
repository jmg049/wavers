# Wavers Basic Benchmarks

This benchmark is designed to test the basic functionality of the Wavers library.
The benchmarks will cover the core reading and writing operations of Wavers.
The exact functions tested are:

* read_audio
* write_audio
* read (legacy)
* write (legacy)

---

## How the Benchmarks Work

Each function is exhaustively tested on the follow parameter sets:

1. Samples rates (8000Hz, 16,000Hz, 44,100Hz)
2. Duration (1s, 5s, 10s, 30s, 60s)
3. Sample Type (``i16``, ``i24``, ``i32``, ``f32``, ``f64``)
4. Channels (``mono``, ``stereo``)

The data will be random noise based on the above parameters.

Each functions is run 10 times as a warmup.
Then they are executed and measured 10,000 times.

---
