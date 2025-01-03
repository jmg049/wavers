use criterion::criterion_main;
mod common;
mod comparison_benches;
mod core_benches;
mod memory_benches;
pub use common::generate_bench_configs;
criterion_main! {
    core_benches::core_benches,
    // comparison_benches::comparison_benches,
    // memory_benches::memory_benches,
}
