use rs_math::tensor::Tensor;

#[path = "utils/ndim_vec.rs"]
mod ndim_vec;

#[path = "utils/py_ndarray.rs"]
mod py_ndarray;

#[test]
fn add_aarch64() {
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
