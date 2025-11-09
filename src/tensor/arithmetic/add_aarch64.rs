use super::*;
use core::arch::aarch64::*;

impl AddAArch64<u8> for u8 {
    // fn add_l4(tensor_a: &TensorView<'_, u8>, tensor_b: &TensorView<'_, u8>) -> Tensor<'static, u8> {
    //     let nelems = tensor_a.nelems();
    //     let mut data_c: Vec<u8, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
    //     unsafe {
    //         data_c.set_len(nelems);
    //         let data_a_ptr = tensor_a.data.as_ptr();
    //         let data_b_ptr = tensor_b.data.as_ptr();
    //         let data_c_ptr = data_c.as_mut_ptr();

    //         for idx in (0..(nelems / 8) * 8).step_by(8) {
    //             let va = vld1q_u8(data_a_ptr.add(idx));
    //             let vb = vld1q_u8(data_b_ptr.add(idx));
    //             let vc = vaddq_u8(va, vb);
    //             vst1q_u8(data_c_ptr.add(idx), vc);
    //         }
    //     }

    //     for idx in (nelems - (nelems % 8))..nelems {
    //         data_c[idx] = tensor_a.data[idx] + tensor_b.data[idx];
    //     }

    //     Tensor {
    //         data: data_c,
    //         shape: tensor_a.shape(),
    //         strides: tensor_a.strides(),
    //         offset: 0,
    //         _u: PhantomData,
    //         _s: PhantomData,
    //     }
    // }

    fn add_l4(tensor_a: &TensorView<'_, u8>, tensor_b: &TensorView<'_, u8>) -> Tensor<'static, u8> {
        let nelems = tensor_a.nelems();
        let mut data_c: Vec<u8, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
        unsafe {
            data_c.set_len(nelems);
            let mut data_a = tensor_a.data.as_ptr();
            let mut data_b = tensor_b.data.as_ptr();
            let mut data_c = data_c.as_mut_ptr();

            #[cfg(target_arch = "aarch64")]
            {
                let mut n256 = tensor_a.nelems() & !255usize;
                core::arch::asm!(
                    "cbz {n256}, 2f",
                    "1:",
                    // "prfm pldl1keep, [{data_a}, #512]",
                    // "prfm pldl1keep, [{data_b}, #512]",
                    // "prfm pldl1keep, [{data_c}, #512]",

                    "ld1 {{v0.16b, v1.16b, v2.16b, v3.16b}},     [{data_a}], #64",
                    "ld1 {{v4.16b, v5.16b, v6.16b, v7.16b}},     [{data_a}], #64",
                    "ld1 {{v8.16b, v9.16b, v10.16b, v11.16b}},   [{data_a}], #64",
                    "ld1 {{v12.16b, v13.16b, v14.16b, v15.16b}}, [{data_a}], #64",
                    "ld1 {{v16.16b, v17.16b, v18.16b, v19.16b}}, [{data_b}], #64",
                    "ld1 {{v20.16b, v21.16b, v22.16b, v23.16b}}, [{data_b}], #64",
                    "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{data_b}], #64",
                    "ld1 {{v28.16b, v29.16b, v30.16b, v31.16b}}, [{data_b}], #64",

                    "add v0.16b,  v0.16b,  v16.16b",
                    "add v1.16b,  v1.16b,  v17.16b",
                    "add v2.16b,  v2.16b,  v18.16b",
                    "add v3.16b,  v3.16b,  v19.16b",
                    "add v4.16b,  v4.16b,  v20.16b",
                    "add v5.16b,  v5.16b,  v21.16b",
                    "add v6.16b,  v6.16b,  v22.16b",
                    "add v7.16b,  v7.16b,  v23.16b",
                    "add v8.16b,  v8.16b,  v24.16b",
                    "add v9.16b,  v9.16b,  v25.16b",
                    "add v10.16b, v10.16b, v26.16b",
                    "add v11.16b, v11.16b, v27.16b",
                    "add v12.16b, v12.16b, v28.16b",
                    "add v13.16b, v13.16b, v29.16b",
                    "add v14.16b, v14.16b, v30.16b",
                    "add v15.16b, v15.16b, v31.16b",

                    "st1 {{v0.16b, v1.16b, v2.16b, v3.16b}},     [{data_c}], #64",
                    "st1 {{v4.16b, v5.16b, v6.16b, v7.16b}},     [{data_c}], #64",
                    "st1 {{v8.16b, v9.16b, v10.16b, v11.16b}},   [{data_c}], #64",
                    "st1 {{v12.16b, v13.16b, v14.16b, v15.16b}}, [{data_c}], #64",

                    "subs {n256}, {n256}, #256",
                    "b.hi 1b",
                    "2:",

                    data_a = inout(reg) data_a,
                    data_b = inout(reg) data_b,
                    data_c = inout(reg) data_c,

                    n256 = inout(reg) n256,
                    options(nostack, preserves_flags)
                );
            }

            let mut n16 = (nelems & 255) & !15usize;
            core::arch::asm!(
                "cbz {n16}, 2f",
                "1:",

                "ld1 {{v0.16b}}, [{data_a}], #16",
                "ld1 {{v1.16b}}, [{data_b}], #16",

                "add v0.16b, v0.16b, v1.16b",
                "st1 {{v0.16b}}, [{data_c}], #16",

                "subs {n16}, {n16}, #16",
                "b.hi 1b",
                "2:",

                data_a = inout(reg) data_a,
                data_b = inout(reg) data_b,
                data_c = inout(reg) data_c,

                n16 = inout(reg) n16,

                options(nostack, preserves_flags)
            );

            for offset in 0..(nelems & 15) {
                *data_c.add(offset) = *data_a.add(offset) + *data_b.add(offset);
            }
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

// impl AddAArch64<f32> for f32 {
//     fn add_l4(
//         tensor_a: &TensorView<'_, f32>,
//         tensor_b: &TensorView<'_, f32>,
//     ) -> Tensor<'static, f32> {
//         let nelems = tensor_a.nelems();
//         let mut data_c: Vec<f32, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
//         unsafe {
//             data_c.set_len(nelems);

//             let data_a_ptr = tensor_a.data.as_ptr();
//             let data_b_ptr = tensor_b.data.as_ptr();
//             let data_c_ptr = data_c.as_mut_ptr();

//             for idx in (0..(nelems / 4) * 4).step_by(4) {
//                 let va = vld1q_f32(data_a_ptr.add(idx));
//                 let vb = vld1q_f32(data_b_ptr.add(idx));
//                 let vc = vaddq_f32(va, vb);
//                 vst1q_f32(data_c_ptr.add(idx), vc);
//             }
//         }

//         for idx in (nelems - (nelems % 4))..nelems {
//             data_c[idx] = tensor_a.data[idx] + tensor_b.data[idx];
//         }

//         Tensor {
//             data: data_c,
//             shape: tensor_a.shape(),
//             strides: tensor_a.strides(),
//             offset: 0,
//             _u: PhantomData,
//             _s: PhantomData,
//         }
//     }
// }

impl AddAArch64<f32> for f32 {
    fn add_l4(
        tensor_a: &TensorView<'_, f32>,
        tensor_b: &TensorView<'_, f32>,
    ) -> Tensor<'static, f32> {
        let nelems = tensor_a.nelems();
        let mut data_c: Vec<f32, TensorAllocator> = Vec::with_capacity_in(nelems, TensorAllocator);
        unsafe {
            data_c.set_len(nelems);

            let mut data_a_ptr = tensor_a.data.as_ptr();
            let mut data_b_ptr = tensor_b.data.as_ptr();
            let mut data_c_ptr = data_c.as_mut_ptr();

            #[cfg(target_arch = "aarch64")]
            {
                let mut n16 = tensor_a.nelems() & !15usize;
                core::arch::asm!(
                    "cbz {n16}, 2f",
                    "1:",
                    // "prfm pldl1keep, [{data_a}, #512]",
                    // "prfm pldl1keep, [{data_b}, #512]",
                    // "prfm pldl1keep, [{data_c}, #512]",

                    "ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}},       [{data_a_ptr}], #64",
                    "ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}},       [{data_a_ptr}], #64",
                    "ld1 {{v8.4s, v9.4s, v10.4s, v11.4s}},     [{data_a_ptr}], #64",
                    "ld1 {{v12.4s, v13.4s, v14.4s, v15.4s}},   [{data_a_ptr}], #64",

                    "ld1 {{v16.4s, v17.4s, v18.4s, v19.4s}},   [{data_b_ptr}], #64",
                    "ld1 {{v20.4s, v21.4s, v22.4s, v23.4s}},   [{data_b_ptr}], #64",
                    "ld1 {{v24.4s, v25.4s, v26.4s, v27.4s}},   [{data_b_ptr}], #64",
                    "ld1 {{v28.4s, v29.4s, v30.4s, v31.4s}},   [{data_b_ptr}], #64",

                    "fadd v0.4s,   v0.4s,   v16.4s",
                    "fadd v1.4s,   v1.4s,   v17.4s",
                    "fadd v2.4s,   v2.4s,   v18.4s",
                    "fadd v3.4s,   v3.4s,   v19.4s",
                    "fadd v4.4s,   v4.4s,   v20.4s",
                    "fadd v5.4s,   v5.4s,   v21.4s",
                    "fadd v6.4s,   v6.4s,   v22.4s",
                    "fadd v7.4s,   v7.4s,   v23.4s",
                    "fadd v8.4s,   v8.4s,   v24.4s",
                    "fadd v9.4s,   v9.4s,   v25.4s",
                    "fadd v10.4s,  v10.4s,  v26.4s",
                    "fadd v11.4s,  v11.4s,  v27.4s",
                    "fadd v12.4s,  v12.4s,  v28.4s",
                    "fadd v13.4s,  v13.4s,  v29.4s",
                    "fadd v14.4s,  v14.4s,  v30.4s",
                    "fadd v15.4s,  v15.4s,  v31.4s",

                    "st1 {{v0.4s, v1.4s, v2.4s, v3.4s}},     [{data_c_ptr}], #64",
                    "st1 {{v4.4s, v5.4s, v6.4s, v7.4s}},     [{data_c_ptr}], #64",
                    "st1 {{v8.4s, v9.4s, v10.4s, v11.4s}},   [{data_c_ptr}], #64",
                    "st1 {{v12.4s, v13.4s, v14.4s, v15.4s}}, [{data_c_ptr}], #64",

                    "subs {n16}, {n16}, #16",
                    "b.hi 1b",
                    "2:",

                    data_a_ptr = inout(reg) data_a_ptr,
                    data_b_ptr = inout(reg) data_b_ptr,
                    data_c_ptr = inout(reg) data_c_ptr,
                    n16 = inout(reg) n16,
                    options(nostack, preserves_flags)
                );
            }
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
    U: TensorTypeNumeric + TensorArithmetic<U>,
    S: TensorStorage<U>,
{
    pub fn add_aarch64(
        &self,
        tensor_b: &'a impl TensorAsView<'a, U>,
    ) -> Result<Tensor<'static, U>, Error> {
        let tensor_b = tensor_b.as_view();
        if self.shape == tensor_b.shape {
            if (self.nelems() >= 64) {
                Ok(U::add_l4(&self.view(), &tensor_b))
            } else {
                self.add_generic(&tensor_b)
            }
        } else {
            // TODO: Implement aarch64 specific broadcast addition
            self.add_generic_bc(&tensor_b.as_view())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AddAArch64, Tensor};

    #[test]
    fn add_l4() {
        // /* Testing u8 */
        // /* 1x256 + 1x256 */
        // let tensor_1x256_a = Tensor::<u8>::from_vec(vec![11; 256]).unwrap();
        // let tensor_1x256_b = Tensor::<u8>::from_vec(vec![11; 256]).unwrap();
        // let tensor_1x256_c =
        //     <u8 as AddAArch64<u8>>::add_l4(&tensor_1x256_a.view(), &tensor_1x256_b.view());
        // assert_eq!(tensor_1x256_c, vec![22; 256]);

        // /* 1x511 + 1x511 (256 * 2 = 512 - 1 = 511) */
        // let tensor_1x511_a = Tensor::<u8>::from_vec(vec![11; 511]).unwrap();
        // let tensor_1x511_b = Tensor::<u8>::from_vec(vec![11; 511]).unwrap();
        // let tensor_1x511_c =
        //     <u8 as AddAArch64<u8>>::add_l4(&tensor_1x511_a.view(), &tensor_1x511_b.view());
        // assert_eq!(tensor_1x511_c, vec![22; 511]);

        /* Testing f32 */

        /* 1x256 + 1x256 */
        let tensor_1x256_a = Tensor::<f32>::from_vec(vec![11.1; 32]).unwrap();
        let tensor_1x256_b = Tensor::<f32>::from_vec(vec![11.1; 32]).unwrap();
        let tensor_1x256_c =
            <f32 as AddAArch64<f32>>::add_l4(&tensor_1x256_a.view(), &tensor_1x256_b.view());
        assert_eq!(tensor_1x256_c, vec![22.2; 32]);
    }
}
