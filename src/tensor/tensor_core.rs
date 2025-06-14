use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;

use super::{
    Error, Tensor, TensorBase, TensorStorage, TensorStorageMut, TensorTypeNumeric, TensorView,
    TensorViewMut,
};

pub trait TensorFromNDim<T, U> {
    fn compute_shape(data: &T) -> Vec<usize>;
    fn compute_strides(shape: &[usize]) -> Vec<usize>;
    fn flatten_data(data: &T, data_flat: &mut Vec<U>);
}

impl<U: TensorTypeNumeric> TensorFromNDim<U, U> for U {
    fn compute_shape(_: &U) -> Vec<usize> {
        vec![]
    }

    fn compute_strides(_: &[usize]) -> Vec<usize> {
        vec![]
    }

    fn flatten_data(data: &U, data_flat: &mut Vec<U>) {
        data_flat.push(data.clone());
    }
}

impl<T, U> TensorFromNDim<Vec<T>, U> for Vec<T>
where
    T: TensorFromNDim<T, U>,
    U: TensorTypeNumeric,
{
    fn compute_shape(data: &Vec<T>) -> Vec<usize> {
        if data.is_empty() {
            return vec![0];
        }
        let mut shape = vec![data.len()];
        shape.extend(T::compute_shape(&data[0]));
        shape
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() || shape.contains(&0) {
            return vec![];
        }
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn flatten_data(data: &Vec<T>, data_flat: &mut Vec<U>) {
        for item in data {
            T::flatten_data(item, data_flat);
        }
    }
}

impl<'a, U> Tensor<'a, U>
where
    U: TensorTypeNumeric,
{
    pub fn from_vec<T>(data_ndim: T) -> Result<Self, Error>
    where
        T: TensorFromNDim<T, U>,
    {
        // TODO: Sanity check for data_ndim
        let mut data = Vec::new();
        let shape = T::compute_shape(&data_ndim);
        let strides = T::compute_strides(&shape);
        let offset = 0;

        T::flatten_data(&data_ndim, &mut data);

        Ok(Tensor {
            data,
            shape,
            strides,
            offset,
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn from_vec_ref<T>(data_ndim: &T) -> Result<Self, Error>
    where
        T: TensorFromNDim<T, U>,
    {
        // TODO: Sanity check for data_ndim
        let mut data = Vec::new();
        let shape = T::compute_shape(&data_ndim);
        let strides = T::compute_strides(&shape);
        let offset = 0;

        T::flatten_data(&data_ndim, &mut data);

        Ok(Tensor {
            data,
            shape,
            strides,
            offset,
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn from_zeros(shape: &[usize]) -> Result<Self, Error> {
        // TODO: Sanity check for data_ndim
        let data: Vec<U> = vec![U::default(); shape.iter().product()];
        let shape = shape.to_vec();
        let strides = <Vec<U> as TensorFromNDim<Vec<U>, U>>::compute_strides(&shape);
        let offset = 0;

        Ok(Tensor {
            shape,
            strides,
            offset,
            data,
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn from_shape(shape: &[usize], data: &Vec<U>) -> Result<Self, Error> {
        // TODO: Sanity check for data_ndim
        let shape = shape.to_vec();
        let strides = <Vec<U> as TensorFromNDim<Vec<U>, U>>::compute_strides(&shape);
        let offset = 0;

        Ok(Tensor {
            shape,
            strides,
            offset,
            data: data.clone(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }
}

impl<U: TensorTypeNumeric> TensorStorage<U> for Vec<U> {
    #[inline(always)]
    fn get(&self, index: usize) -> U {
        self[index]
    }
}

impl<'a, U: TensorTypeNumeric> TensorStorage<U> for &'a [U] {
    #[inline(always)]
    fn get(&self, index: usize) -> U {
        self[index]
    }
}

impl<'a, U: TensorTypeNumeric> TensorStorage<U> for &'a mut [U] {
    #[inline(always)]
    fn get(&self, index: usize) -> U {
        self[index]
    }
}

impl<U: TensorTypeNumeric> TensorStorageMut<U> for Vec<U> {
    #[inline(always)]
    fn get(&self, index: usize) -> U {
        self[index]
    }

    #[inline(always)]
    fn set(&mut self, index: usize, val: U) {
        self[index] = val;
    }
}

impl<'a, U: TensorTypeNumeric> TensorStorageMut<U> for &'a mut [U] {
    #[inline(always)]
    fn get(&self, index: usize) -> U {
        self[index]
    }

    #[inline(always)]
    fn set(&mut self, index: usize, val: U) {
        self[index] = val;
    }
}

impl<'a, U, S> TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U>,
{
    #[inline(always)]
    pub fn data(&self) -> Vec<U> {
        self.data.as_ref().to_vec()
    }

    #[inline(always)]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[inline(always)]
    pub fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }

    #[inline(always)]
    pub fn ndim(&self) -> usize {
        self.strides.len()
    }

    #[inline(always)]
    pub fn nelems(&self) -> usize {
        self.data.as_ref().len()
    }

    pub fn compute_strides(&self, shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() || shape.contains(&0) {
            return vec![];
        }

        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn getval(&self, index: &[usize]) -> Result<U, Error> {
        self.check_shape(&index, &self.shape)?;

        let mut flat_index = self.offset;

        for (i, &idx) in index.iter().enumerate() {
            flat_index += idx * self.strides[i];
        }

        if flat_index >= self.nelems() {
            return Err(Error::IndexOutOfRange {
                index: flat_index,
                nelems: self.ndim(),
            });
        }

        Ok(self.data.get(flat_index))
    }

    pub fn view(&'a self) -> TensorView<'a, U> {
        TensorView {
            shape: self.shape(),
            strides: self.strides(),
            offset: self.offset,
            data: self.data.as_ref(),
            _u: PhantomData,
            _s: PhantomData,
        }
    }

    pub fn axis_impl(&self, axis: usize) -> Result<(Vec<usize>, Vec<usize>, usize), Error> {
        if axis >= self.ndim() {
            return Err(Error::InvalidAxis {
                axis: axis,
                ndim: self.ndim(),
            });
        }

        let mut sl_shape = Vec::with_capacity(self.ndim());
        let mut sl_strides = Vec::with_capacity(self.ndim());
        let sl_offset = 0;

        sl_shape.push(self.shape[axis]);
        sl_strides.push(self.strides[axis]);

        for dim in 0..self.ndim() {
            if dim != axis {
                sl_shape.push(self.shape[dim]);
                sl_strides.push(self.strides[dim]);
            }
        }

        Ok((sl_shape, sl_strides, sl_offset))
    }

    pub fn axis(&'a self, axis: usize) -> Result<TensorView<'a, U>, Error> {
        let (sl_shape, sl_strides, sl_offset) = self.axis_impl(axis)?;

        Ok(TensorView {
            shape: sl_shape,
            strides: sl_strides,
            offset: sl_offset,
            data: &self.data.as_ref(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    fn slice_impl(&self, index: &[usize]) -> Result<(Vec<usize>, Vec<usize>, usize), Error> {
        if index.len() > self.ndim() {
            return Err(Error::ShapeMismatch {
                shape_a: index.to_vec(),
                shape_b: self.shape(),
            });
        }

        let sl_shape = self.shape[index.len()..].to_vec();
        let sl_strides = self.strides[index.len()..].to_vec();
        let mut sl_offset = self.offset;

        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(Error::InvalidSlicing {
                    slice: index.to_vec(),
                    shape: self.shape(),
                });
            }
            sl_offset += idx * self.strides[i];
        }

        Ok((sl_shape, sl_strides, sl_offset))
    }

    pub fn slice(&'a self, index: &[usize]) -> Result<TensorView<'a, U>, Error> {
        let (sl_shape, sl_strides, sl_offset) = self.slice_impl(index)?;

        Ok(TensorView {
            shape: sl_shape,
            strides: sl_strides,
            offset: sl_offset,
            data: &self.data.as_ref(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn axis_slice(&'a self, axis: usize, index: &[usize]) -> Result<TensorView<'a, U>, Error> {
        let view = self.axis(axis)?;
        let (sl_shape, sl_strides, sl_offset) = view.slice_impl(index)?;

        Ok(TensorView {
            shape: sl_shape,
            strides: sl_strides,
            offset: sl_offset,
            data: &self.data.as_ref(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    fn batch_impl(&'a self, range: Range<usize>) -> Result<(Vec<usize>, Vec<usize>, usize), Error> {
        let shape = std::iter::once(range.end - range.start)
            .chain(self.shape[1..].iter().copied())
            .collect();
        let strides = self.strides();
        let offset = range.start * strides[0];
        Ok((shape, strides, offset))
    }

    pub fn batch(&'a self, range: Range<usize>) -> Result<TensorView<'a, U>, Error> {
        let (shape, strides, offset) = self.batch_impl(range)?;

        Ok(TensorView {
            shape,
            strides,
            offset,
            data: &self.data.as_ref(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    fn permute_impl(&self, axes: &[usize]) -> Result<(Vec<usize>, Vec<usize>), Error> {
        if axes.len() != self.ndim() {
            return Err(Error::DimensionMismatch {
                tensor_dim: self.ndim(),
                dim: axes.len(),
            });
        }

        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());

        for axis in axes {
            shape.push(self.shape[*axis]);
            strides.push(self.strides[*axis]);
        }
        Ok((shape, strides))
    }

    pub fn permute(&'a self, axes: &[usize]) -> Result<TensorView<'a, U>, Error> {
        let (shape, strides) = self.permute_impl(axes)?;

        Ok(TensorView {
            shape,
            strides,
            offset: self.offset,
            data: &self.data.as_ref(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn transpose(&'a self) -> Result<TensorView<'a, U>, Error> {
        let t_axes: Vec<usize> = (0..self.ndim()).rev().collect();
        let (shape, strides) = self.permute_impl(&t_axes)?;

        Ok(TensorView {
            shape,
            strides,
            offset: self.offset,
            data: &self.data.as_ref(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn t(&'a self) -> Result<TensorView<'a, U>, Error> {
        self.transpose()
    }

    fn flatten_impl(&'a self) -> Result<(Vec<usize>, Vec<usize>), Error> {
        let mut shape_fl = vec![self.shape[0], 0];
        shape_fl[1] = self.shape.iter().skip(1).product();
        let strides_fl = vec![shape_fl[1], 1];

        Ok((shape_fl, strides_fl))
    }

    pub fn flatten(&'a self) -> Result<TensorView<'a, U>, Error> {
        let (shape, strides) = self.flatten_impl()?;

        Ok(TensorView {
            shape,
            strides,
            offset: 0,
            data: &self.data.as_ref(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn max(&self) -> U {
        let mut max = U::min_value();

        for val in self.iter() {
            if val > max {
                max = val;
            }
        }

        max
    }

    pub fn min(&self) -> U {
        let mut min = U::max_value();

        for val in self.iter() {
            if val < min {
                min = val;
            }
        }

        min
    }

    pub fn print(&self) -> String {
        fn print_recursive<U, S>(
            view: &TensorBase<U, S>,
            depth: usize,
            indices: &mut Vec<usize>,
        ) -> String
        where
            U: TensorTypeNumeric,
            S: TensorStorage<U> + AsRef<[U]>,
        {
            if depth == view.ndim() {
                format!("{}", view.getval(indices).unwrap())
            } else {
                let mut nested = Vec::with_capacity(view.shape()[depth]);
                for index in 0..view.shape()[depth] {
                    indices.push(index);
                    nested.push(print_recursive(view, depth + 1, indices));
                    indices.pop();
                }

                format!("[{}]", nested.join(", "))
            }
        }

        let mut indices: Vec<usize> = Vec::with_capacity(self.shape.len());
        print_recursive(self, 0, &mut indices)
    }

    #[inline(always)]
    fn check_shape(&self, shape_a: &[usize], shape_b: &[usize]) -> Result<bool, Error> {
        if shape_a.len() != shape_b.len() {
            return Err(Error::ShapeMismatch {
                shape_a: shape_a.to_vec(),
                shape_b: shape_b.to_vec(),
            });
        }

        Ok(true)
    }
}

impl<'a, U, S> TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    pub fn setval(&mut self, index: &[usize], val: U) -> Result<bool, Error> {
        self.check_shape(&index, &self.shape)?;

        let mut flat_index = self.offset;

        for (i, &idx) in index.iter().enumerate() {
            flat_index += idx * self.strides[i];
        }

        if flat_index >= self.nelems() {
            return Err(Error::IndexOutOfRange {
                index: flat_index,
                nelems: self.ndim(),
            });
        }

        self.data.set(flat_index, val);

        Ok(true)
    }

    pub fn view_mut(&'a mut self) -> TensorViewMut<'a, U> {
        TensorViewMut {
            shape: self.shape(),
            strides: self.strides(),
            offset: self.offset,
            data: self.data.as_mut(),
            _u: PhantomData,
            _s: PhantomData,
        }
    }

    pub fn axis_mut(&'a mut self, axis: usize) -> Result<TensorViewMut<'a, U>, Error> {
        let (sl_shape, sl_strides, sl_offset) = self.axis_impl(axis)?;

        Ok(TensorViewMut {
            shape: sl_shape,
            strides: sl_strides,
            offset: sl_offset,
            data: self.data.as_mut(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn slice_mut(&'a mut self, index: &[usize]) -> Result<TensorViewMut<'a, U>, Error> {
        let (sl_shape, sl_strides, sl_offset) = self.slice_impl(index)?;

        Ok(TensorViewMut {
            shape: sl_shape,
            strides: sl_strides,
            offset: sl_offset,
            data: self.data.as_mut(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn permute_mut(&'a mut self, axes: &[usize]) -> Result<TensorViewMut<'a, U>, Error> {
        let (shape, strides) = self.permute_impl(axes)?;

        Ok(TensorViewMut {
            shape,
            strides,
            offset: self.offset,
            data: self.data.as_mut(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn transpose_mut(&'a mut self) -> Result<TensorViewMut<'a, U>, Error> {
        let t_axes: Vec<usize> = (0..self.ndim()).rev().collect();
        let (shape, strides) = self.permute_impl(&t_axes)?;

        Ok(TensorViewMut {
            shape,
            strides,
            offset: self.offset,
            data: self.data.as_mut(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }

    pub fn t_mut(&'a mut self) -> Result<TensorViewMut<'a, U>, Error> {
        self.transpose_mut()
    }

    pub fn flatten_mut(&'a mut self) -> Result<TensorViewMut<'a, U>, Error> {
        let (shape, strides) = self.flatten_impl()?;

        Ok(TensorViewMut {
            shape,
            strides,
            offset: 0,
            data: self.data.as_mut(),
            _u: PhantomData,
            _s: PhantomData,
        })
    }
}

impl<'a, U> TensorViewMut<'a, U>
where
    U: TensorTypeNumeric,
{
    pub fn update<T>(
        &'a mut self,
        slice: &[usize],
        data: Vec<T>,
    ) -> Result<TensorViewMut<'a, U>, Error>
    where
        Vec<T>: TensorUpdate<Vec<T>, U>,
        T: Debug,
    {
        let mut index = slice.to_vec();
        let mut sl_view = self.slice_mut(&[])?;
        <Vec<T> as TensorUpdate<Vec<T>, U>>::_update(&mut sl_view, &mut index, &data);
        Ok(sl_view)
    }
}

impl<'a, U> Tensor<'a, U>
where
    U: TensorTypeNumeric,
{
    pub fn update<T>(
        &'a mut self,
        slice: &[usize],
        data: Vec<T>,
    ) -> Result<TensorViewMut<'a, U>, Error>
    where
        Vec<T>: TensorUpdate<Vec<T>, U>,
        T: Debug,
    {
        let mut sl_view = self.slice_mut(&[])?;
        let mut index = slice.to_vec();
        <Vec<T> as TensorUpdate<Vec<T>, U>>::_update(&mut sl_view, &mut index, &data);
        Ok(sl_view)
    }
}

pub trait TensorUpdate<T, U> {
    fn _update(view: &mut TensorViewMut<U>, index: &mut Vec<usize>, data: &T);
}

impl<U: TensorTypeNumeric> TensorUpdate<U, U> for U {
    fn _update(view: &mut TensorViewMut<U>, index: &mut Vec<usize>, data: &U) {
        view.setval(index, *data).unwrap();
    }
}

impl<T, U> TensorUpdate<Vec<T>, U> for Vec<T>
where
    T: TensorUpdate<T, U> + Debug,
    U: TensorTypeNumeric,
{
    fn _update(view: &mut TensorViewMut<U>, index: &mut Vec<usize>, data: &Vec<T>) {
        // TODO: Verify the length of data is equal to the slice length
        if index.len() == view.ndim() {
            T::_update(view, index, &data[0]);
        } else {
            for (i, elem) in data.iter().enumerate() {
                index.push(i);
                T::_update(view, index, elem);
                index.pop();
            }
        }
    }
}

impl<'a, U: TensorTypeNumeric> From<&'a Tensor<'a, U>> for TensorView<'a, U> {
    fn from(item: &'a Tensor<U>) -> Self {
        TensorView {
            shape: item.shape(),
            strides: item.strides(),
            offset: item.offset,
            data: &item.data,
            _u: PhantomData,
            _s: PhantomData,
        }
    }
}

impl<'a, U: TensorTypeNumeric> From<&'a mut Tensor<'a, U>> for TensorViewMut<'a, U> {
    fn from(item: &'a mut Tensor<U>) -> Self {
        TensorViewMut {
            shape: item.shape(),
            strides: item.strides(),
            offset: item.offset,
            data: item.data.as_mut(),
            _u: PhantomData,
            _s: PhantomData,
        }
    }
}

impl<U> Clone for Tensor<'_, U>
where
    U: TensorTypeNumeric + 'static,
{
    fn clone(&self) -> Self {
        Tensor {
            shape: self.shape(),
            strides: self.strides(),
            offset: self.offset,
            data: self.data(),
            _u: PhantomData,
            _s: PhantomData,
        }
    }
}
