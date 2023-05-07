pub enum Sample {
    I16(i16),
    I32(i32),
    F32(f32),
    F64(f64),
}

pub trait AudioConversion {
    fn as_i16(self) -> i16;
    fn as_i32(self) -> i32;
    fn as_f32(self) -> f32;
    fn as_f64(self) -> f64;
    fn as_type(self, as_type: Sample) -> Sample;
}

impl AudioConversion for Sample {
    fn as_i16(self) -> i16 {
        todo!()
    }

    fn as_i32(self) -> i32 {
        todo!()
    }

    fn as_f32(self) -> f32 {
        todo!()
    }

    fn as_f64(self) -> f64 {
        todo!()
    }

    fn as_type(self, as_type: Sample) -> Sample {
        todo!()
    }
}

impl AudioConversion for Vec<Sample> {
    fn as_i16(self) -> i16 {
        todo!()
    }

    fn as_i32(self) -> i32 {
        todo!()
    }

    fn as_f32(self) -> f32 {
        todo!()
    }

    fn as_f64(self) -> f64 {
        todo!()
    }

    fn as_type(self, as_type: Sample) -> Sample {
        todo!()
    }
}