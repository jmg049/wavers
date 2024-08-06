pub mod interleave;
pub mod resample;

pub use interleave::{deinterleave_in_place, interleave_in_place};

#[cfg(feature = "simd")]
pub use interleave::deinterleave_simd;
