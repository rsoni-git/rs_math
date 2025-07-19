use rs_math::tensor::Tensor;

#[path = "utils/ndim_vec.rs"]
mod ndim_vec;

#[path = "utils/py_ndarray.rs"]
mod py_ndarray;

#[test]
fn add_aarch64() {
    let tensor_1x4_a = Tensor::from_vec(ndim_vec::ndim_vec_1d::<i32>(4, false)).unwrap();
    let tensor_1x4_b = Tensor::from_vec(ndim_vec::ndim_vec_1d::<i32>(4, false)).unwrap();
    let tensor_1x4_c = tensor_1x4_a.add_aarch64(&tensor_1x4_b.view()).unwrap();
    assert_eq!(tensor_1x4_c, vec![2, 4, 6, 8]);

    let tensor_1x5_a = Tensor::from_vec(ndim_vec::ndim_vec_1d::<i32>(5, false)).unwrap();
    let tensor_1x5_b = Tensor::from_vec(ndim_vec::ndim_vec_1d::<i32>(5, false)).unwrap();
    let tensor_1x5_c = tensor_1x5_a.add_aarch64(&tensor_1x5_b.view()).unwrap();
    assert_eq!(tensor_1x5_c, vec![2, 4, 6, 8, 10]);

    let tensor_1x6_a = Tensor::from_vec(ndim_vec::ndim_vec_1d::<i32>(6, false)).unwrap();
    let tensor_1x6_b = Tensor::from_vec(ndim_vec::ndim_vec_1d::<i32>(6, false)).unwrap();
    let tensor_1x6_c = tensor_1x6_a.add_aarch64(&tensor_1x6_b.view()).unwrap();
    assert_eq!(tensor_1x6_c, vec![2, 4, 6, 8, 10, 12]);

    let tensor_1x7_a = Tensor::from_vec(ndim_vec::ndim_vec_1d::<i32>(7, false)).unwrap();
    let tensor_1x7_b = Tensor::from_vec(ndim_vec::ndim_vec_1d::<i32>(7, false)).unwrap();
    let tensor_1x7_c = tensor_1x7_a.add_aarch64(&tensor_1x7_b.view()).unwrap();
    assert_eq!(tensor_1x7_c, vec![2, 4, 6, 8, 10, 12, 14]);

    // Tensor 3D: Adding two u8 types
    let tensor_7x5x3_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<u8>(&[7, 5, 3], true)).unwrap();
    let tensor_7x5x3_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<u8>(&[7, 5, 3], true)).unwrap();
    py_ndarray::add_and_verify(&tensor_7x5x3_a, &tensor_7x5x3_b);

    // Tensor 3D: Adding two i32 types
    let tensor_7x5x3_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[7, 5, 3], true)).unwrap();
    let tensor_7x5x3_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[7, 5, 3], true)).unwrap();
    py_ndarray::add_and_verify(&tensor_7x5x3_a, &tensor_7x5x3_b);

    // Tensor 3D: Adding two i64 types
    let tensor_7x5x3_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i64>(&[7, 5, 3], true)).unwrap();
    let tensor_7x5x3_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i64>(&[7, 5, 3], true)).unwrap();
    py_ndarray::add_and_verify(&tensor_7x5x3_a, &tensor_7x5x3_b);

    // Tensor 3D: Adding two f32 types
    let tensor_7x5x3_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[7, 5, 3], true)).unwrap();
    let tensor_7x5x3_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[7, 5, 3], true)).unwrap();
    py_ndarray::add_and_verify(&tensor_7x5x3_a, &tensor_7x5x3_b);

    // Tensor 3D: Adding two f64 types
    let tensor_7x5x3_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<f64>(&[7, 5, 3], true)).unwrap();
    let tensor_7x5x3_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<f64>(&[7, 5, 3], true)).unwrap();
    py_ndarray::add_and_verify(&tensor_7x5x3_a, &tensor_7x5x3_b);
}
