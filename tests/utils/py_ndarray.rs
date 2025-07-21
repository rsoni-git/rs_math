use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, Bound, PyResult, Python};
use rand::distr::uniform::SampleUniform;
use rs_math::tensor::{
    arithmetic::AddAArch64, Error, Tensor, TensorBase, TensorStorage, TensorStorageMut,
    TensorTypeNumeric,
};
#[path = "ndim_vec.rs"]
mod ndim_vec;

pub fn tensor_to_pyarray<'a, 'py, U, S>(
    py: Python<'py>,
    tensor: &TensorBase<'a, U, S>,
) -> PyResult<Bound<'py, PyArrayDyn<U>>>
where
    U: TensorTypeNumeric + numpy::Element,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    match ArrayD::from_shape_vec(IxDyn(&tensor.shape()), tensor.data().to_vec()) {
        Ok(array) => Ok(array.into_pyarray(py)),
        Err(msg) => Err(PyValueError::new_err(format!(
            "Failed to create array: {} (expected product of shape = {:?})",
            msg,
            tensor.shape()
        ))),
    }
}

pub fn pyarray_to_tensor<'py, U>(
    py_array: &Bound<'py, PyArrayDyn<U>>,
) -> Result<Tensor<'static, U>, Error>
where
    U: TensorTypeNumeric + numpy::Element,
{
    let shape = py_array.shape().to_vec();
    let data = py_array.readonly().as_slice().unwrap().to_vec();
    Tensor::from_shape(&shape, &data)
}

pub fn assert_eq<'a, 'py, U, S>(tensor: &TensorBase<'a, U, S>, py_array: &Bound<'py, PyArrayDyn<U>>)
where
    U: TensorTypeNumeric + numpy::Element + 'static,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    // println!("tensor----> {:?}", tensor.view());
    // println!("pyarray --> {:?}", pyarray_to_tensor(py_array).unwrap());
    assert_eq!(tensor.view(), &pyarray_to_tensor(py_array).unwrap());
}

pub fn add_and_verify<U>(tensor_a: &Tensor<U>, tensor_b: &Tensor<U>)
where
    U: TensorTypeNumeric + AddAArch64<U> + From<u8> + numpy::Element + SampleUniform + 'static,
{
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let tensor_c = tensor_a.add_aarch64(&tensor_b.view()).unwrap();

        let pyarray_a = tensor_to_pyarray(py, &tensor_a).unwrap();
        let pyarray_b = tensor_to_pyarray(py, &tensor_b).unwrap();

        unsafe {
            let pyarray_c = pyarray_a.as_array().to_owned() + pyarray_b.as_array().to_owned();
            assert_eq(&tensor_c, &pyarray_c.into_pyarray(py));
        }
    });
}
