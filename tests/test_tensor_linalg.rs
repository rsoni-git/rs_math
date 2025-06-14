use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::Python;
use rs_math::tensor::{Error, Tensor};

#[path = "utils/py_ndarray.rs"]
mod py_ndarray;

#[test]
fn from_one_hot_enc() {
    let labels = vec![
        "cat", "dog", "cat", "cat", "elephant", "tiger", "cat", "tiger", "lion", "zebra", "zebra",
    ];
    let uniq_labels = labels
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();

    let tensor = Tensor::from_one_hot_enc(&labels).unwrap();
    assert_eq!(tensor.shape(), vec![labels.len(), uniq_labels]);
    assert_eq!(tensor.strides(), vec![uniq_labels, 1]);
    assert_eq!(
        tensor,
        vec![
            vec![1, 0, 0, 0, 0, 0],
            vec![0, 1, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
            vec![1, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 1],
        ]
    );

    // Negative: Empty label data
    let labels: Vec<&str> = vec![];
    let tensor = Tensor::from_one_hot_enc(&labels);
    assert!(matches!(tensor, Err(Error::InvalidParam { .. })));
}

#[test]
fn relu() {
    let mut tensor_2x2 = Tensor::from_vec(vec![vec![1, -3, 0], vec![-4, 5, 0]]).unwrap();
    {
        tensor_2x2.relu().unwrap();
    }
    assert_eq!(tensor_2x2, vec![vec![1, 0, 0], vec![0, 5, 0]]);
}

#[test]
fn softmax() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut ts_2x2 =
            Tensor::<f32>::from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();

        let ts_2x2 = ts_2x2.softmax(0).unwrap();

        let py_array = py_ndarray::tensor_to_pyarray(py, &ts_2x2).unwrap();
        let py_scipy = py.import("scipy.special").unwrap();
        let py_softmax = py_scipy.getattr("softmax").unwrap();
        let py_result = py_softmax.call1((py_array,)).unwrap();
        let py_result = py_result.downcast::<PyArrayDyn<f32>>().unwrap();
        py_ndarray::assert_eq(&ts_2x2, py_result);
    });
}
