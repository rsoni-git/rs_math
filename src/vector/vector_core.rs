use crate::tensor::{Error, Tensor, TensorEq, TensorTypeNumeric};
use std::fmt::Debug;

#[derive(Debug)]
pub struct Vector<'a, U> {
    tensor: Tensor<'a, U>,
}

impl<'a, U> Vector<'a, U>
where
    U: TensorTypeNumeric + Default + Debug,
{
    pub fn from_zeros(nelems: usize) -> Result<Vector<'a, U>, Error> {
        let vector = Tensor::from_zeros(&[nelems, 1])?;
        Ok(Vector { tensor: vector })
    }

    pub fn from_vec(data: Vec<U>) -> Result<Vector<'a, U>, Error> {
        let vector = Tensor::from_shape(&[data.len(), 1], &data)?;
        Ok(Vector { tensor: vector })
    }

    pub fn from_arr<const N: usize>(data: &[U; N]) -> Result<Vector<'a, U>, Error> {
        let vector = Tensor::from_shape(&[N, 1], &data.to_vec())?;
        Ok(Vector { tensor: vector })
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Result<U, Error> {
        self.tensor.getval(&[index, 0])
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, val: U) -> Result<bool, Error> {
        self.tensor.setval(&[index, 0], val)
    }
}

impl<'a, U, T> PartialEq<Vec<T>> for Vector<'a, U>
where
    U: TensorTypeNumeric + PartialEq + Debug,
    T: TensorEq<U> + Debug,
{
    fn eq(&self, other: &Vec<T>) -> bool {
        self.tensor == other
    }
}

impl<'a, U> PartialEq<Vector<'a, U>> for Vector<'a, U>
where
    U: TensorTypeNumeric + PartialEq + Debug,
{
    fn eq(&self, other: &Vector<U>) -> bool {
        self.tensor == &other.tensor
    }
}
