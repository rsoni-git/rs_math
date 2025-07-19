use criterion::{criterion_group, criterion_main, Criterion};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Python;
use rs_math::tensor::Tensor;

#[path = "../tests/utils/ndim_vec.rs"]
mod ndim_vec;

#[path = "../tests/utils/py_ndarray.rs"]
mod py_ndarray;

fn compare_numpy(c: &mut Criterion) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        /* Compare dataset performance: 100x10x1 */

        let mut group = c.benchmark_group("compare_numpy");
        let tensor_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[100, 10, 1], true)).unwrap();
        let tensor_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[100, 10, 1], true)).unwrap();
        let pyarray_a = py_ndarray::tensor_to_pyarray(py, &tensor_a).unwrap();
        let pyarray_b = py_ndarray::tensor_to_pyarray(py, &tensor_b).unwrap();
        let tensor_b = tensor_b.view();

        group.bench_function("tensor::add_aarch64::[100x10x1]", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_aarch64(&tensor_b);
            })
        });

        let np = PyModule::import(py, "numpy").unwrap();
        let np_add = np.getattr("add").unwrap();
        group.bench_function("pyarray::add::[100x10x1]", |bench| {
            bench.iter(|| {
                let _ = np_add.call1((&pyarray_a, &pyarray_b));
            })
        });

        /* Compare dataset performance: 1000x100x10 */

        let tensor_a =
            Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[1000, 100, 10], true)).unwrap();
        let tensor_b =
            Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[1000, 100, 10], true)).unwrap();
        let pyarray_a = py_ndarray::tensor_to_pyarray(py, &tensor_a).unwrap();
        let pyarray_b = py_ndarray::tensor_to_pyarray(py, &tensor_b).unwrap();
        let tensor_b = tensor_b.view();

        group.bench_function("tensor::add_aarch64::[1000x100x10]", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_aarch64(&tensor_b);
            })
        });

        let np = PyModule::import(py, "numpy").unwrap();
        let np_add = np.getattr("add").unwrap();
        group.bench_function("pyarray::add::[1000x100x10]", |bench| {
            bench.iter(|| {
                let _ = np_add.call1((&pyarray_a, &pyarray_b));
            })
        });

        /* Compare dataset performance: 1000x100x100 */

        let tensor_a =
            Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[1000, 100, 100], true)).unwrap();
        let tensor_b =
            Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[1000, 100, 100], true)).unwrap();
        let pyarray_a = py_ndarray::tensor_to_pyarray(py, &tensor_a).unwrap();
        let pyarray_b = py_ndarray::tensor_to_pyarray(py, &tensor_b).unwrap();
        let tensor_b = tensor_b.view();

        group.bench_function("tensor::add_aarch64::[1000x100x100]", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_aarch64(&tensor_b);
            })
        });

        let np = PyModule::import(py, "numpy").unwrap();
        let np_add = np.getattr("add").unwrap();
        group.bench_function("pyarray::add::[1000x100x100]", |bench| {
            bench.iter(|| {
                let _ = np_add.call1((&pyarray_a, &pyarray_b));
            })
        });
        group.finish();
    });
}

criterion_group!(benches, compare_numpy);
criterion_main!(benches);
