use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Matrix<'a, U> {
    tensor: Tensor<'a, U>,
}

pub mod matrix_core;
pub mod matrix_linalg;
