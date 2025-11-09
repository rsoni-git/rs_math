use crate::tensor::*;
pub trait AddAArch64<U: TensorTypeNumeric> {
    fn add_l4(tensor_a: &TensorView<'_, U>, tensor_b: &TensorView<'_, U>) -> Tensor<'static, U>;
}

pub trait TensorArithmetic<U: TensorTypeNumeric>: AddAArch64<U> {}

impl TensorArithmetic<u8> for u8 {}
impl TensorArithmetic<i32> for i32 {}
impl TensorArithmetic<i64> for i64 {}
impl TensorArithmetic<f32> for f32 {}
impl TensorArithmetic<f64> for f64 {}

pub mod add_aarch64;
pub mod add_generic;
pub mod arithmetic;
