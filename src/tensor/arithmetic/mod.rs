use crate::tensor::*;
pub trait AddAArch64<U> {
    fn add_l4(tensor_a: &TensorView<'_, U>, tensor_b: &TensorView<'_, U>) -> Tensor<'static, U>;
}

pub mod add_aarch64;
pub mod add_generic;
pub mod arithmetic;
