use super::*;
use core::arch::aarch64::*;

impl AddAArch64<u8> for u8 {
    fn add_l4(tensor_a: &TensorView<'_, u8>, tensor_b: &TensorView<'_, u8>) -> Tensor<'static, u8> {
        let nelems = tensor_a.nelems();
        let mut data_c: Vec<u8, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
        unsafe {
            data_c.set_len(nelems);
            let data_a_ptr = tensor_a.data.as_ptr();
            let data_b_ptr = tensor_b.data.as_ptr();
            let data_c_ptr = data_c.as_mut_ptr();

            for idx in (0..(nelems / 8) * 8).step_by(8) {
                let va = vld1q_u8(data_a_ptr.add(idx));
                let vb = vld1q_u8(data_b_ptr.add(idx));
                let vc = vaddq_u8(va, vb);
                vst1q_u8(data_c_ptr.add(idx), vc);
            }
        }

        for idx in (nelems - (nelems % 8))..nelems {
            data_c[idx] = tensor_a.data[idx] + tensor_b.data[idx];
        }

        Tensor {
            data: data_c,
            shape: tensor_a.shape(),
            strides: tensor_a.strides(),
            offset: 0,
            _u: PhantomData,
            _s: PhantomData,
        }
    }
}

impl AddAArch64<i32> for i32 {
    fn add_l4(
        tensor_a: &TensorView<'_, i32>,
        tensor_b: &TensorView<'_, i32>,
    ) -> Tensor<'static, i32> {
        let nelems = tensor_a.nelems();
        let mut data_c: Vec<i32, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
        unsafe {
            data_c.set_len(nelems);

            let data_a_ptr = tensor_a.data.as_ptr();
            let data_b_ptr = tensor_b.data.as_ptr();
            let data_c_ptr = data_c.as_mut_ptr();

            for idx in (0..(nelems / 4) * 4).step_by(4) {
                let va = vld1q_s32(data_a_ptr.add(idx));
                let vb = vld1q_s32(data_b_ptr.add(idx));
                let vc = vaddq_s32(va, vb);
                vst1q_s32(data_c_ptr.add(idx), vc);
            }
        }

        for idx in (nelems - (nelems % 4))..nelems {
            data_c[idx] = tensor_a.data[idx] + tensor_b.data[idx];
        }

        Tensor {
            data: data_c,
            shape: tensor_a.shape(),
            strides: tensor_a.strides(),
            offset: 0,
            _u: PhantomData,
            _s: PhantomData,
        }
    }
}

impl AddAArch64<i64> for i64 {
    fn add_l4(
        tensor_a: &TensorView<'_, i64>,
        tensor_b: &TensorView<'_, i64>,
    ) -> Tensor<'static, i64> {
        let nelems = tensor_a.nelems();
        let mut data_c: Vec<i64, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
        unsafe {
            data_c.set_len(nelems);

            let data_a_ptr = tensor_a.data.as_ptr();
            let data_b_ptr = tensor_b.data.as_ptr();
            let data_c_ptr = data_c.as_mut_ptr();

            for idx in (0..(nelems / 2) * 2).step_by(2) {
                let va = vld1q_s64(data_a_ptr.add(idx));
                let vb = vld1q_s64(data_b_ptr.add(idx));
                let vc = vaddq_s64(va, vb);
                vst1q_s64(data_c_ptr.add(idx), vc);
            }
        }

        for idx in (nelems - (nelems % 2))..nelems {
            data_c[idx] = tensor_a.data[idx] + tensor_b.data[idx];
        }

        Tensor {
            data: data_c,
            shape: tensor_a.shape(),
            strides: tensor_a.strides(),
            offset: 0,
            _u: PhantomData,
            _s: PhantomData,
        }
    }
}

impl AddAArch64<f32> for f32 {
    fn add_l4(
        tensor_a: &TensorView<'_, f32>,
        tensor_b: &TensorView<'_, f32>,
    ) -> Tensor<'static, f32> {
        let nelems = tensor_a.nelems();
        let mut data_c: Vec<f32, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
        unsafe {
            data_c.set_len(nelems);

            let data_a_ptr = tensor_a.data.as_ptr();
            let data_b_ptr = tensor_b.data.as_ptr();
            let data_c_ptr = data_c.as_mut_ptr();

            for idx in (0..(nelems / 4) * 4).step_by(4) {
                let va = vld1q_f32(data_a_ptr.add(idx));
                let vb = vld1q_f32(data_b_ptr.add(idx));
                let vc = vaddq_f32(va, vb);
                vst1q_f32(data_c_ptr.add(idx), vc);
            }
        }

        for idx in (nelems - (nelems % 4))..nelems {
            data_c[idx] = tensor_a.data[idx] + tensor_b.data[idx];
        }

        Tensor {
            data: data_c,
            shape: tensor_a.shape(),
            strides: tensor_a.strides(),
            offset: 0,
            _u: PhantomData,
            _s: PhantomData,
        }
    }
}

impl AddAArch64<f64> for f64 {
    fn add_l4(
        tensor_a: &TensorView<'_, f64>,
        tensor_b: &TensorView<'_, f64>,
    ) -> Tensor<'static, f64> {
        let nelems = tensor_a.nelems();
        let mut data_c: Vec<f64, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
        unsafe {
            data_c.set_len(nelems);

            let data_a_ptr = tensor_a.data.as_ptr();
            let data_b_ptr = tensor_b.data.as_ptr();
            let data_c_ptr = data_c.as_mut_ptr();

            for idx in (0..(nelems / 2) * 2).step_by(2) {
                let va = vld1q_f64(data_a_ptr.add(idx));
                let vb = vld1q_f64(data_b_ptr.add(idx));
                let vc = vaddq_f64(va, vb);
                vst1q_f64(data_c_ptr.add(idx), vc);
            }
        }

        for idx in (nelems - (nelems % 2))..nelems {
            data_c[idx] = tensor_a.data[idx] + tensor_b.data[idx];
        }

        Tensor {
            data: data_c,
            shape: tensor_a.shape(),
            strides: tensor_a.strides(),
            offset: 0,
            _u: PhantomData,
            _s: PhantomData,
        }
    }
}

impl<'a, U, S> TensorBase<'a, U, S>
where
    U: TensorTypeNumeric + AddAArch64<U>,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    pub fn add_aarch64(&self, tensor_b: &TensorView<'_, U>) -> Result<Tensor<'static, U>, Error> {
        if self.shape == tensor_b.shape {
            Ok(U::add_l4(&self.view(), tensor_b))
        } else {
            // Temporary until I implement aarch64 add() with broadcasting
            self.add_generic_bc(tensor_b)
        }
    }
}
