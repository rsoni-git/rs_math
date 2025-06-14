use super::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

impl<'a, U, S> TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U>,
{
    fn shape_indexes(&self, shape: &'a [usize]) -> impl Iterator<Item = Vec<usize>> + 'a {
        let ndim = shape.len();
        let total = shape.iter().product();

        (0..total).map(move |mut idx| {
            let mut index = vec![0; ndim];
            for dim in (0..ndim).rev() {
                index[dim] = idx % shape[dim];
                idx /= shape[dim];
            }
            index
        })
    }

    fn shape_bc(
        shape_a: &[usize],
        shape_b: &[usize],
        batch_mul: bool,
    ) -> Result<Vec<usize>, Error> {
        let ndim_a = shape_a.len();
        let ndim_b = shape_b.len();
        let ndim_c = ndim_a.max(ndim_b);
        let mut shape_c;

        let elem_wise_bc = |batch_a: &[usize],
                            batch_b: &[usize],
                            ndim_a: usize,
                            ndim_b: usize,
                            ndim_c: usize|
         -> Result<Vec<usize>, Error> {
            // println!("batch_a: {:?}, batch_b: {:?}, ndim_c: {}", batch_a, batch_b, ndim_c);
            let mut shape_c = Vec::with_capacity(ndim_c);
            for dim in 0..ndim_c {
                let dim_a = if dim >= ndim_c - ndim_a {
                    batch_a[dim - (ndim_c - ndim_a)]
                } else {
                    1
                };
                let dim_b = if dim >= ndim_c - ndim_b {
                    batch_b[dim - (ndim_c - ndim_b)]
                } else {
                    1
                };

                if dim_a == dim_b || dim_a == 1 || dim_b == 1 {
                    shape_c.push(dim_a.max(dim_b));
                } else {
                    return Err(Error::ShapeMismatchBroadcast {
                        shape_a: shape_a.to_vec(),
                        shape_b: shape_b.to_vec(),
                    });
                }
            }

            Ok(shape_c)
        };

        if batch_mul {
            let (batch_a, m, k_a) = match ndim_a {
                2 => (vec![], shape_a[0], shape_a[1]),
                _ => (
                    shape_a[..ndim_a - 2].to_vec(),
                    shape_a[ndim_a - 2],
                    shape_a[ndim_a - 1],
                ),
            };

            let (batch_b, k_b, n) = match ndim_b {
                2 => (vec![], shape_b[0], shape_b[1]),
                _ => (
                    shape_b[..ndim_b - 2].to_vec(),
                    shape_b[ndim_b - 2],
                    shape_b[ndim_b - 1],
                ),
            };

            if k_a != k_b {
                return Err(Error::ShapeMismatchBroadcast {
                    shape_a: shape_a.to_vec(),
                    shape_b: shape_b.to_vec(),
                });
            }

            shape_c = elem_wise_bc(&batch_a, &batch_b, ndim_a - 2, ndim_b - 2, ndim_c - 2)?;
            shape_c.push(m);
            shape_c.push(n);
        } else {
            shape_c = elem_wise_bc(shape_a, shape_b, ndim_a, ndim_b, ndim_c)?;
        }

        Ok(shape_c)
    }

    #[inline(always)]
    fn offset(index: &[usize], shape: &[usize], strides: &[usize]) -> usize {
        index
            .iter()
            .zip(shape.iter().zip(strides))
            .map(|(&idx, (&shape, &stride))| {
                let i = if shape == 1 { 0 } else { idx };
                i * stride
            })
            .sum()
    }

    #[inline(always)]
    pub fn add_alias(&self, tensor_b: &TensorView<'_, U>) -> Result<Tensor<'static, U>, Error> {
        self.add(tensor_b)
    }

    pub fn add(&self, tensor_b: &TensorView<'_, U>) -> Result<Tensor<'static, U>, Error> {
        let shape_c = Self::shape_bc(&self.shape, &tensor_b.shape, false)?;
        let strides_c = self.compute_strides(&shape_c);
        let mut data_c = vec![U::default(); shape_c.iter().product()];

        for index in self.shape_indexes(&shape_c) {
            let offset_a = Self::offset(&index, &self.shape, &self.strides);
            let offset_b = Self::offset(&index, &tensor_b.shape, &tensor_b.strides);
            let offset_c = Self::offset(&index, &shape_c, &strides_c);
            data_c[offset_c] = self.data[offset_a] + tensor_b.data[offset_b];
        }

        Ok(Tensor {
            data: data_c,
            shape: shape_c,
            strides: strides_c,
            offset: 0,
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    #[inline(always)]
    pub fn sub_alias(&self, tensor_b: &TensorView<'_, U>) -> Result<Tensor<'static, U>, Error> {
        self.sub(tensor_b)
    }

    pub fn sub(&self, tensor_b: &TensorView<'_, U>) -> Result<Tensor<'static, U>, Error> {
        let shape_c = Self::shape_bc(&self.shape, &tensor_b.shape, false)?;
        let strides_c = self.compute_strides(&shape_c);
        let mut data_c = vec![U::default(); shape_c.iter().product()];

        for index in self.shape_indexes(&shape_c) {
            let offset_a = Self::offset(&index, &self.shape, &self.strides);
            let offset_b = Self::offset(&index, &tensor_b.shape, &tensor_b.strides);
            let offset_c = Self::offset(&index, &shape_c, &strides_c);
            data_c[offset_c] = self.data[offset_a] - tensor_b.data[offset_b];
        }

        Ok(Tensor {
            data: data_c,
            shape: shape_c,
            strides: strides_c,
            offset: 0,
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    #[inline(always)]
    pub fn mul_alias(&self, tensor_b: &TensorView<'_, U>) -> Result<Tensor<'static, U>, Error> {
        self.mul(tensor_b)
    }

    pub fn mul(&self, tensor_b: &TensorView<U>) -> Result<Tensor<'static, U>, Error> {
        let shape_a = self.shape();
        let shape_b = tensor_b.shape();
        // let shape_c = self.bc_shape(&tensor_b, true)?;
        let shape_c = Self::shape_bc(&shape_a, &shape_b, true)?;

        let strides_a = self.strides();
        let strides_b = tensor_b.strides();
        let strides_c = self.compute_strides(&shape_c);

        let ndim_a = self.ndim();
        let ndim_b = tensor_b.ndim();
        let ndim_c = shape_c.len();

        let m = shape_c[ndim_c - 2];
        let n = shape_c[ndim_c - 1];
        let k = shape_a[ndim_a - 1];

        let mut data_c = vec![U::default(); shape_c.iter().product()];

        for index in self.shape_indexes(&shape_c[..ndim_c - 2]) {
            let base_a = Self::offset(&index, &shape_a[..ndim_a - 2], &strides_a[..ndim_a - 2]);
            let base_b = Self::offset(&index, &shape_b[..ndim_b - 2], &strides_b[..ndim_b - 2]);
            let base_c = Self::offset(&index, &shape_c[..ndim_c - 2], &strides_c[..ndim_c - 2]);

            // MxN - NxK
            for mi in 0..m {
                for ji in 0..n {
                    let mut sum = U::default();
                    for ki in 0..k {
                        let offset_a =
                            base_a + (mi * strides_a[ndim_a - 2]) + (ki * strides_a[ndim_a - 1]);
                        let offset_b =
                            base_b + (ki * strides_b[ndim_b - 2]) + (ji * strides_b[ndim_b - 1]);
                        sum = sum + self.data[offset_a] * tensor_b.data[offset_b];
                    }
                    let offset_c =
                        base_c + (mi * strides_c[ndim_c - 2]) + (ji * strides_c[ndim_c - 1]);
                    data_c[offset_c] = sum;
                }
            }
        }

        Ok(Tensor {
            data: data_c,
            shape: shape_c,
            strides: strides_c,
            offset: 0,
            _u: PhantomData,
            _s: PhantomData,
        })
    }
}

impl<'a, U, S> TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    pub fn add_scalar(&mut self, scaler: U) {
        for idx in 0..self.nelems() {
            self.data[idx] = self.data[idx] + scaler;
        }
    }

    pub fn sub_scalar(&mut self, scaler: U) {
        for idx in 0..self.nelems() {
            self.data[idx] = self.data[idx] - scaler;
        }
    }

    pub fn mul_scalar(&mut self, scaler: U) {
        for idx in 0..self.nelems() {
            self.data[idx] = self.data[idx] * scaler;
        }
    }
}

impl<U, S1, S2> Add<&TensorBase<'_, U, S1>> for &TensorBase<'_, U, S2>
where
    U: TensorTypeNumeric + 'static,
    S1: TensorStorage<U>,
    S2: TensorStorage<U>,
{
    type Output = TensorBase<'static, U, Vec<U>>;

    fn add(self, other: &TensorBase<'_, U, S1>) -> Self::Output {
        // TODO: Use TensorView::from()
        let tensor = self.add_alias(&other.view());
        match tensor {
            Ok(val) => return val,
            Err(_) => Tensor {
                shape: vec![],
                strides: vec![],
                offset: 0,
                data: vec![],
                _u: PhantomData,
                _s: PhantomData,
            },
        }
    }
}

impl<U, S1, S2> Sub<&TensorBase<'_, U, S1>> for &TensorBase<'_, U, S2>
where
    U: TensorTypeNumeric + 'static,
    S1: TensorStorage<U>,
    S2: TensorStorage<U>,
{
    type Output = TensorBase<'static, U, Vec<U>>;

    fn sub(self, other: &TensorBase<'_, U, S1>) -> Self::Output {
        // TODO: Use TensorView::from()
        let tensor = self.sub_alias(&other.view());
        match tensor {
            Ok(val) => return val,
            Err(_) => Tensor {
                shape: vec![],
                strides: vec![],
                offset: 0,
                data: vec![],
                _u: PhantomData,
                _s: PhantomData,
            },
        }
    }
}

impl<U, S1, S2> Mul<&TensorBase<'_, U, S1>> for &TensorBase<'_, U, S2>
where
    U: TensorTypeNumeric + 'static,
    S1: TensorStorage<U>,
    S2: TensorStorage<U>,
{
    type Output = TensorBase<'static, U, Vec<U>>;

    fn mul(self, other: &TensorBase<'_, U, S1>) -> Self::Output {
        // TODO: Use TensorView::from()
        let tensor = self.mul_alias(&other.view());
        match tensor {
            Ok(val) => return val,
            Err(_) => Tensor {
                shape: vec![],
                strides: vec![],
                offset: 0,
                data: vec![],
                _u: PhantomData,
                _s: PhantomData,
            },
        }
    }
}

impl<'a, U, S> AddAssign<U> for TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    fn add_assign(&mut self, other: U) {
        self.add_scalar(other);
    }
}

impl<'a, U, S> SubAssign<U> for TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    fn sub_assign(&mut self, other: U) {
        self.sub_scalar(other);
    }
}

impl<'a, U, S> MulAssign<U> for TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    fn mul_assign(&mut self, other: U) {
        self.mul_scalar(other);
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, TensorBase};

    #[test]
    fn shape_bc() {
        /* BatchMultiplication: False */

        // Positive: 1x3 & 1x3: complete match
        let shape_a = vec![3];
        let shape_b = vec![3];
        let shape_c = TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, false).unwrap();
        assert_eq!(shape_c, vec![3]);

        // Positive: 1x3 & 1x1: Partial match
        let shape_a = vec![3];
        let shape_b = vec![1];
        let shape_c = TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, false).unwrap();
        assert_eq!(shape_c, vec![3]);

        // Positive: 2x3x5 & 2x1x1: Partial match
        let shape_a = vec![2, 3, 5];
        let shape_b = vec![2, 1, 1];
        let shape_c = TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, false).unwrap();
        assert_eq!(shape_c, vec![2, 3, 5]);

        // Positive: 2x1x5 & 2x4x1: Mix partial match
        let shape_a = vec![2, 1, 5];
        let shape_b = vec![2, 4, 1];
        let shape_c = TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, false).unwrap();
        assert_eq!(shape_c, vec![2, 4, 5]);

        // Negative: 1x3 & 1x4: Mismatch on second dim
        let shape_a = vec![1, 3];
        let shape_b = vec![1, 4];
        assert!(matches!(
            TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, false),
            Err(Error::ShapeMismatchBroadcast { .. })
        ));

        // Negative: 2x1x4 + 2x3x5: Mismatch on third dim, partial match on second dim
        let shape_a = vec![2, 1, 4];
        let shape_b = vec![2, 3, 5];
        assert!(matches!(
            TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, false),
            Err(Error::ShapeMismatchBroadcast { .. })
        ));

        /* BatchMultiplication: True */

        // Positive: (2x3) * (3x2) = 2x2
        let shape_a = vec![2, 3];
        let shape_b = vec![3, 2];
        let shape_c = TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, true).unwrap();
        assert_eq!(shape_c, vec![2, 2]);

        // Positive: (1x3x4) * (2x4x3): Partial match on 1st dim
        //                              matrix-match on the remaining two
        let shape_a = vec![1, 3, 4];
        let shape_b = vec![2, 4, 3];
        let shape_c = TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, true).unwrap();
        assert_eq!(shape_c, vec![2, 3, 3]);

        // Negative: (2x3) * (2x3)
        let shape_a = vec![2, 3];
        let shape_b = vec![2, 3];
        assert!(matches!(
            TensorBase::<i32, Vec<i32>>::shape_bc(&shape_a, &shape_b, true),
            Err(Error::ShapeMismatchBroadcast { .. })
        ));
    }
}
