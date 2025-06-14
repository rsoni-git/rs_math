use super::{
    TensorBase, TensorIter, TensorIterMut, TensorStorage, TensorStorageMut, TensorTypeNumeric,
};
use std::marker::PhantomData;

impl<'a, U> Iterator for TensorIter<'a, U>
where
    U: TensorTypeNumeric,
{
    type Item = U;

    fn next(&mut self) -> Option<Self::Item> {
        let mut flat_index = *self.offset;

        if self.index[0] >= self.shape[0] {
            return None;
        }

        for (i, &idx) in self.index.iter().enumerate() {
            flat_index += idx * self.strides[i];
        }

        for idx in (0..self.index.len()).rev() {
            self.index[idx] += 1;

            if (self.index[idx] < self.shape[idx])
                || (idx == 0 && self.index[idx] == self.shape[idx])
            {
                break;
            } else {
                self.index[idx] = 0;
            }
        }

        Some(self.data[flat_index])
    }
}

impl<'a, U> Iterator for TensorIterMut<'a, U>
where
    U: TensorTypeNumeric,
{
    type Item = &'a mut U;

    fn next(&mut self) -> Option<Self::Item> {
        let mut flat_index = *self.offset;

        if self.index[0] >= self.shape[0] {
            return None;
        }

        for (i, &idx) in self.index.iter().enumerate() {
            flat_index += idx * self.strides[i];
        }

        for idx in (0..self.index.len()).rev() {
            self.index[idx] += 1;

            if (self.index[idx] < self.shape[idx])
                || (idx == 0 && self.index[idx] == self.shape[idx])
            {
                break;
            } else {
                self.index[idx] = 0;
            }
        }

        // Converting raw pointer to mutable reference
        Some(unsafe { &mut *self.data.as_mut_ptr().add(flat_index) })
    }
}

impl<'a, U, S> TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U>,
{
    pub fn iter(&'a self) -> TensorIter<'a, U> {
        TensorIter {
            shape: &self.shape,
            strides: &self.strides,
            offset: &self.offset,
            data: &self.data.as_ref(),
            index: vec![0; self.ndim()],
            _u: PhantomData,
        }
    }
}

impl<'a, U, S> TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    pub fn iter_mut(&'_ mut self) -> TensorIterMut<'_, U> {
        let ndim = self.ndim();

        TensorIterMut {
            shape: &self.shape,
            strides: &self.strides,
            offset: &self.offset,
            data: self.data.as_mut(),
            index: vec![0; ndim],
            _u: PhantomData,
        }
    }
}
