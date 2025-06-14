use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, Bound, PyResult, Python};
use rs_math::tensor::{
    Error, Tensor, TensorBase, TensorStorage, TensorStorageMut, TensorTypeNumeric,
};

pub fn tensor_to_pyarray<'a, 'py, U, S>(
    py: Python<'py>,
    tensor: &TensorBase<'a, U, S>,
) -> PyResult<Bound<'py, PyArrayDyn<U>>>
where
    U: TensorTypeNumeric + numpy::Element,
    S: TensorStorage<U> + TensorStorageMut<U>,
{
    match ArrayD::from_shape_vec(IxDyn(&tensor.shape()), tensor.data()) {
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
    assert_eq!(
        tensor.slice(&[0]).unwrap(),
        &pyarray_to_tensor(py_array).unwrap()
    );
}
