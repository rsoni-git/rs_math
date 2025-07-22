use criterion::{criterion_group, criterion_main, Criterion};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Python;
use rs_math::tensor::Tensor;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use walkdir::WalkDir;

#[path = "../tests/utils/ndim_vec.rs"]
mod ndim_vec;

#[path = "../tests/utils/py_ndarray.rs"]
mod py_ndarray;

fn bench_add_aarch64(c: &mut Criterion) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("NUMEXPR_NUM_THREADS", "1");
        std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
        let mut group = c.benchmark_group("bench_add_aarch64");

        /* Compare dataset performance: 100x10x1 */
        let tensor_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[100, 10, 1], true)).unwrap();
        let tensor_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<f32>(&[100, 10, 1], true)).unwrap();
        let pyarray_a = py_ndarray::tensor_to_pyarray(py, &tensor_a).unwrap();
        let pyarray_b = py_ndarray::tensor_to_pyarray(py, &tensor_b).unwrap();
        let tensor_b = tensor_b.view();

        group.bench_function("[100x10x1]:add_aarch64", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_aarch64(&tensor_b);
            })
        });

        group.bench_function("[100x10x1]:add_generic", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_generic(&tensor_b);
            })
        });

        let np = PyModule::import(py, "numpy").unwrap();
        let np_add = np.getattr("add").unwrap();
        group.bench_function("[100x10x1]:numpy", |bench| {
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

        group.bench_function("[1000x100x10]:add_aarch64", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_aarch64(&tensor_b);
            })
        });

        group.bench_function("[1000x100x10]:add_generic", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_generic(&tensor_b);
            })
        });

        let np = PyModule::import(py, "numpy").unwrap();
        let np_add = np.getattr("add").unwrap();
        group.bench_function("[1000x100x10]:numpy", |bench| {
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

        group.bench_function("[1000x100x100]:add_aarch64", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_aarch64(&tensor_b);
            })
        });

        group.bench_function("[1000x100x100]:add_generic", |bench| {
            bench.iter(|| {
                let _ = tensor_a.add_generic(&tensor_b);
            })
        });

        let np = PyModule::import(py, "numpy").unwrap();
        let np_add = np.getattr("add").unwrap();
        group.bench_function("[1000x100x100]:numpy", |bench| {
            bench.iter(|| {
                let _ = np_add.call1((&pyarray_a, &pyarray_b));
            })
        });
        group.finish();
    });

    display_results("bench_add_aarch64");
}

#[derive(Debug, Deserialize)]
struct Estimate {
    point_estimate: f64,
}

#[derive(Debug, Deserialize)]
struct Estimates {
    mean: Estimate,
    std_dev: Estimate,
}

fn display_nanosecs(nano_secs: f64) -> String {
    if nano_secs < 1e3 {
        format!("{:.2} ns", nano_secs)
    } else if nano_secs >= 1e3 && nano_secs < 1e6 {
        format!("{:.2} µs", nano_secs / 1e3)
    } else if nano_secs > 1e6 {
        format!("{:.2} ms", nano_secs / 1e6)
    } else {
        format!("{:.2} s", nano_secs / 1e9)
    }
}

fn display_results(bench_name: &str) {
    let path_base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/criterion/")
        .join(bench_name);

    use prettytable::{cell, row, Table};
    let mut table = Table::new();
    table.add_row(row!["Bench", "Mean", "Std"]);

    println!("{}", path_base.display());

    let mut directory: Vec<_> = WalkDir::new(&path_base)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name() == "estimates.json"
                && e.path()
                    .parent()
                    .and_then(|p| p.file_name())
                    .map(|name| name == "new")
                    .unwrap_or(false)
        })
        .collect();

    directory.sort_by_key(|e| e.path().to_path_buf());

    for file in directory {
        let estimate_file = File::open(file.path()).unwrap();
        let estimate_reader = BufReader::new(estimate_file);
        let estimates: Estimates = serde_json::from_reader(estimate_reader).unwrap();
        let bench_tag = file
            .path()
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.file_name())
            .unwrap()
            .to_string_lossy();

        table.add_row(row![
            format!("{}", bench_tag),
            format!("{}", display_nanosecs(estimates.mean.point_estimate)),
            format!("{}", display_nanosecs(estimates.std_dev.point_estimate))
        ]);
    }

    println!("Benchmarking comparison table: {}", bench_name);
    table.printstd();
}

criterion_group!(benches, bench_add_aarch64);
criterion_main!(benches);
