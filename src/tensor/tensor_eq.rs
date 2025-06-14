use super::*;

impl<U> TensorEq<U> for U
where
    U: TensorTypeNumeric,
{
    fn tensor_eq(&self, view: &TensorView<U>, indices: &mut Vec<usize>, _depth: usize) -> bool {
        if view.getval(indices).unwrap() == *self {
            true
        } else {
            false
        }
    }
}

impl<U, T> TensorEq<U> for Vec<T>
where
    U: TensorTypeNumeric,
    T: TensorEq<U> + Debug,
{
    fn tensor_eq(&self, view: &TensorView<U>, indices: &mut Vec<usize>, depth: usize) -> bool {
        for (index, item) in self.iter().enumerate() {
            indices.push(index);
            if !item.tensor_eq(view, indices, depth + 1) {
                return false;
            }
            indices.pop();
        }

        true
    }
}

impl<'a, U, T> TensorEq<U> for &'a Vec<T>
where
    U: TensorTypeNumeric,
    T: TensorEq<U> + Debug,
{
    fn tensor_eq(&self, view: &TensorView<U>, indices: &mut Vec<usize>, depth: usize) -> bool {
        for (index, item) in self.iter().enumerate() {
            indices.push(index);
            if !item.tensor_eq(view, indices, depth + 1) {
                return false;
            }
            indices.pop();
        }

        true
    }
}

impl<'a, U> TensorEq<U> for Tensor<'a, U>
where
    U: TensorTypeNumeric,
{
    fn tensor_eq(&self, view: &TensorView<U>, _indices: &mut Vec<usize>, _depth: usize) -> bool {
        self.data == view.data && self.shape == view.shape && self.strides == view.strides
    }
}

impl<'a, U> TensorEq<U> for &Tensor<'a, U>
where
    U: TensorTypeNumeric,
{
    fn tensor_eq(&self, view: &TensorView<U>, _indices: &mut Vec<usize>, _depth: usize) -> bool {
        self.data == view.data && self.shape == view.shape && self.strides == view.strides
    }
}

impl<'a, U> TensorEq<U> for TensorView<'a, U>
where
    U: TensorTypeNumeric,
{
    fn tensor_eq(&self, view: &TensorView<U>, _indices: &mut Vec<usize>, _depth: usize) -> bool {
        self.data == view.data && self.shape == view.shape && self.strides == view.strides
    }
}

impl<'a, U> TensorEq<U> for &TensorView<'a, U>
where
    U: TensorTypeNumeric,
{
    fn tensor_eq(&self, view: &TensorView<U>, _indices: &mut Vec<usize>, _depth: usize) -> bool {
        self.data == view.data && self.shape == view.shape && self.strides == view.strides
    }
}

impl<'a, U> TensorEq<U> for TensorViewMut<'a, U>
where
    U: TensorTypeNumeric,
{
    fn tensor_eq(&self, view: &TensorView<U>, _indices: &mut Vec<usize>, _depth: usize) -> bool {
        self.data == view.data && self.shape == view.shape && self.strides == view.strides
    }
}

impl<'a, U> TensorEq<U> for &TensorViewMut<'a, U>
where
    U: TensorTypeNumeric,
{
    fn tensor_eq(&self, view: &TensorView<U>, _indices: &mut Vec<usize>, _depth: usize) -> bool {
        self.data == view.data && self.shape == view.shape && self.strides == view.strides
    }
}

impl<'a, U, T> PartialEq<T> for Tensor<'a, U>
where
    U: TensorTypeNumeric,
    T: TensorEq<U> + Debug,
{
    fn eq(&self, other: &T) -> bool {
        let mut indices = Vec::with_capacity(self.ndim());
        other.tensor_eq(&self.axis(0).unwrap(), &mut indices, 0)
    }
}

impl<'a, U, T> PartialEq<T> for TensorView<'a, U>
where
    U: TensorTypeNumeric,
    T: TensorEq<U> + Debug,
{
    fn eq(&self, other: &T) -> bool {
        let mut indices = Vec::with_capacity(self.ndim());
        other.tensor_eq(self, &mut indices, 0)
    }
}

impl<'a, U, T> PartialEq<T> for TensorViewMut<'a, U>
where
    U: TensorTypeNumeric,
    T: TensorEq<U> + Debug,
{
    fn eq(&self, other: &T) -> bool {
        let mut indices = Vec::with_capacity(self.ndim());
        other.tensor_eq(&self.axis(0).unwrap(), &mut indices, 0)
    }
}
