use super::*;
use indexmap::IndexMap;
use num_traits::Float;
use std::cmp::Eq;
use std::hash::Hash;

impl<'a> Tensor<'a, u8> {
    pub fn from_one_hot_enc<L: Eq + Hash>(labels: &Vec<L>) -> Result<Self, Error> {
        let nlabels = labels.len();
        if nlabels == 0 {
            return Err(Error::InvalidParam {
                err_msg: "Label data shouldn't be an empty vector".to_string(),
            });
        }
        let mut classes = IndexMap::new();

        for label in labels {
            let nclasses = classes.len();
            classes.entry(label).or_insert(nclasses);
        }

        let nclasses = classes.len();
        let mut data: Vec<u8> = Vec::with_capacity(nlabels * nclasses);

        for label in labels {
            let mut label_enc = vec![0u8; nclasses];
            if let Some(&index) = classes.get(label) {
                label_enc[index] = 1;
            }
            data.extend(label_enc);
        }

        Ok(Tensor {
            shape: vec![nlabels, nclasses],
            strides: vec![nclasses, 1],
            offset: 0,
            data: data,
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
    pub fn relu(&mut self) -> Result<bool, Error> {
        for val in self.iter_mut() {
            if *val < U::default() {
                *val = U::default();
            }
        }
        Ok(true)
    }
}

impl<'a, F, S> TensorBase<'a, F, S>
where
    F: TensorTypeFloat,
    S: TensorStorage<F> + TensorStorageMut<F>,
{
    pub fn softmax(&'a mut self, axis: usize) -> Result<TensorViewMut<'a, F>, Error> {
        let mut view = self.axis_mut(axis)?;
        let max = view.max();
        let mut sum_exps = F::default();

        for val in view.iter_mut() {
            *val = Float::exp(*val - max);
            sum_exps += *val;
        }

        for val in view.iter_mut() {
            *val = *val / sum_exps
        }

        Ok(view)
    }
}
