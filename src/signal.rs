// pub mod mono {
//     use std::ops::{Add, Div};

//     use itertools::Itertools;
//     use ndarray::{
//         Array1, ArrayBase, Axis, CowArray, CowRepr, Dim, Dimension, Ix1, OwnedRepr, RemoveAxis, Ix2, Array2,
//     };
//     use num_traits::{FromPrimitive, Zero};

//     pub trait Mono<'a> {
//         type T: Clone
//             + FromPrimitive
//             + Add<Self::T, Output = Self::T>
//             + Div<Self::T, Output = Self::T>
//             + Zero;
//         fn to_mono(&self) -> CowArray<'a, Self::T, Ix2>;
//     }

//     impl<'a, X> Mono<'a> for ArrayBase<CowRepr<'a, X>, Ix2>
//     where
//         X: 'a + Clone + FromPrimitive + Add<X, Output = X> + Div<X, Output = X> + Zero,
//     {
//         type T = X;
//         fn to_mono(&self) -> CowArray<'a, Self::T, Ix2>{
//             let mean_arr = match self.mean_axis(Axis(0)) {
//                 Some(v) => match v.into_shape((1, self.shape()[1])) {
//                     Ok(v) => v,
//                     Err(err) => panic!("{:?}", err)},
//                 None => panic!("{:?}, {}", "Cannot get mean axis {axis}, {}", self.len()),
//             };
//             CowArray::from(mean_arr)

//         }
//     }

//     pub fn pad_mono_to_len<'a, T>(
//         array: CowArray<T, Ix2>,
//         desired_length: usize,
//     ) -> CowArray<'a, T, Ix2>
//     where
//         T: 'a
//             + Clone
//             + Copy
//             + FromPrimitive
//             + Add<T, Output = T>
//             + Div<T, Output = T>
//             + Zero
//             + std::fmt::Debug,
//     {
//         let mut new_array: Array2<T> = Array2::zeros((1, desired_length));

//         for i in 0..array.shape()[1] {

//             new_array[[0, i]] = array[[0, i]];
//         }

//         CowArray::from(new_array)
//     }
// }

// #[cfg(test)]
// pub mod MonoTest {
//     use std::path::Path;

//     use super::mono::Mono;
//     use crate::wave::WavFile;
//     use ndarray::{arr2, Array2, CowArray, CowRepr, Ix1, Ix2};

//     #[test]
//     pub fn stereo_to_mono() {
//         let stereo: CowArray<f32, Ix2> = CowArray::from(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
//         let mono: CowArray<f32, Ix2> = stereo.to_mono();
//         assert_eq!(mono, CowArray::from(Array2::from_shape_vec((1,3),vec![2.5, 3.5, 4.5]).unwrap()));
//     }

//     #[test]
//     pub fn pad_mono_to_len() {
//         let mono: CowArray<f32, Ix2> = CowArray::from(Array2::from_shape_vec((1,3), vec![1.0, 2.0, 3.0]).unwrap());
//         let padded_mono = super::mono::pad_mono_to_len(mono, 5);
//         assert_eq!(padded_mono, Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 0.0, 0.0]).unwrap());
//     }

//     // read in test.wav using wavfile and read_pcm_i16
//     // then test to make sure the pad_to_len function works with the read in file
//     #[test]
//     pub fn pad_mono_to_len_from_file() {
//         let wav_file = WavFile::from_file(Path::new("./test.wav")).unwrap();
//         let data: CowArray<i16, Ix2> = wav_file.read_pcm_i16();
//         let padded_mono = super::mono::pad_mono_to_len(data, 60 * 16000 + 16000);
//         assert_eq!(padded_mono.len(), 60 * 16000 + 16000);
//     }
// }

// pub mod stereo {
//     use std::ops::{Add, Div};

//     use ndarray::{Array2, ArrayBase, Axis, Ix2, OwnedRepr, ShapeBuilder};
//     use num_traits::{FromPrimitive, Zero};

//     pub fn pad_stereo_to_len<'a, T, S>(
//         array: Array2<T>,
//         desired_shape: S,
//     ) -> ArrayBase<OwnedRepr<T>, Ix2>
//     where
//         T: 'a
//             + Clone
//             + Copy
//             + FromPrimitive
//             + Add<T, Output = T>
//             + Div<T, Output = T>
//             + Zero
//             + std::fmt::Debug,
//         S: ShapeBuilder<Dim = Ix2>,
//     {
//         let mut new_array: Array2<T> = Array2::zeros(desired_shape);
//         for i in 0..array.len_of(Axis(0)) {
//             new_array[[0, i]] = array[[0, i]];
//             new_array[[1, i]] = array[[1, i]];
//         }
//         new_array
//     }
// }
