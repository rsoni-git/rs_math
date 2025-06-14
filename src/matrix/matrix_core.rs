use super::*;
use crate::tensor::{Error, Tensor, TensorEq, TensorTypeNumeric, TensorView, TensorViewMut};
use std::fmt::Debug;

impl<'a, U> Matrix<'a, U>
where
    U: TensorTypeNumeric + Default + Debug,
{
    pub fn from_zeros(rows: usize, cols: usize) -> Result<Matrix<'a, U>, Error> {
        let matrix = Tensor::from_zeros(&[rows, cols])?;
        Ok(Matrix { tensor: matrix })
    }

    pub fn from_vec(data: Vec<Vec<U>>) -> Result<Matrix<'a, U>, Error> {
        let matrix = Tensor::from_vec(data)?;
        Ok(Matrix { tensor: matrix })
    }

    pub fn from_arr<const R: usize, const C: usize>(
        data: [[U; C]; R],
    ) -> Result<Matrix<'a, U>, Error> {
        let data_vec: Vec<Vec<U>> = data.iter().map(|row| row.to_vec()).collect();
        let matrix = Tensor::from_vec(data_vec)?;
        Ok(Matrix { tensor: matrix })
    }

    pub fn from_shape(shape: &[usize], data: &Vec<U>) -> Result<Matrix<'static, U>, Error> {
        let matrix = Tensor::from_shape(shape, data)?;
        Ok(Matrix { tensor: matrix })
    }

    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape()
    }

    #[inline(always)]
    pub fn row(&self, index: usize) -> Result<TensorView<'_, U>, Error> {
        self.tensor.slice(&[index])
    }

    #[inline(always)]
    pub fn col(&self, index: usize) -> Result<TensorView<'_, U>, Error> {
        self.tensor.axis_slice(1, &[index])
    }

    #[inline(always)]
    pub fn update(&'a mut self, data: Vec<Vec<U>>) -> Result<TensorViewMut<'a, U>, Error> {
        self.tensor.update(&[], data)
    }

    #[inline(always)]
    pub fn update_row(
        &'a mut self,
        index: usize,
        data: Vec<U>,
    ) -> Result<TensorViewMut<'a, U>, Error> {
        self.tensor.update(&[index], data)
    }

    pub fn add(&self, matrix_b: &Matrix<'a, U>) -> Result<Matrix<'static, U>, Error> {
        let matrix_c = self.tensor.add(&matrix_b.tensor.view())?;

        Ok(Matrix { tensor: matrix_c })
    }

    pub fn sub(&self, matrix_b: &Matrix<'a, U>) -> Result<Matrix<'static, U>, Error> {
        let matrix_c = self.tensor.sub(&matrix_b.tensor.view())?;

        Ok(Matrix { tensor: matrix_c })
    }

    pub fn mul(&self, matrix_b: &Matrix<'a, U>) -> Result<Matrix<'static, U>, Error> {
        let matrix_c = self.tensor.mul(&matrix_b.tensor.view())?;

        Ok(Matrix { tensor: matrix_c })
    }

    pub fn add_scalar(&mut self, scalar: U) {
        self.tensor.add_scalar(scalar);
    }

    pub fn sub_scalar(&mut self, scalar: U) {
        self.tensor.sub_scalar(scalar);
    }

    pub fn mul_scalar(&mut self, scalar: U) {
        self.tensor.mul_scalar(scalar);
    }

    #[inline(always)]
    pub fn t(&self) -> Result<TensorView<'_, U>, Error> {
        self.tensor.t()
    }

    #[inline(always)]
    pub fn t_mut(&'a mut self) -> Result<TensorViewMut<'a, U>, Error> {
        self.tensor.t_mut()
    }

    #[inline(always)]
    pub fn print(&self) -> String {
        self.tensor.print()
    }
}

impl<'a, U, T> PartialEq<Vec<T>> for Matrix<'a, U>
where
    U: TensorTypeNumeric + PartialEq + Debug,
    T: TensorEq<U> + Debug,
{
    fn eq(&self, other: &Vec<T>) -> bool {
        self.tensor == other
    }
}

impl<'a, U> PartialEq<Matrix<'a, U>> for Matrix<'a, U>
where
    U: TensorTypeNumeric + PartialEq + Debug,
{
    fn eq(&self, other: &Matrix<U>) -> bool {
        self.tensor == &other.tensor
    }
}
