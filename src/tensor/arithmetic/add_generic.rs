use crate::tensor::*;

impl<'a, U, S> TensorBase<'a, U, S>
where
    U: TensorTypeNumeric,
    S: TensorStorage<U>,
{
    pub fn add_generic(&self, tensor_b: &TensorView<'_, U>) -> Result<Tensor<'static, U>, Error> {
        let shape_c = Self::shape_bc(&self.shape, &tensor_b.shape, false)?;
        let strides_c = self.compute_strides(&shape_c);
        let nelems_c = shape_c.iter().product();
        let mut data_c: Vec<U, TensorAllocator> = Vec::with_capacity_in(nelems_c, TensorAllocator);
        data_c.resize(nelems_c, U::default());

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
}
