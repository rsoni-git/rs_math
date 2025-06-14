use criterion::{criterion_group, criterion_main, Criterion};
use image::ImageReader;
use rs_math::tensor::Tensor;
use std::mem::drop;

fn img_to_vec(path: &str) -> Vec<u8> {
    let img = ImageReader::open(path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8();

    let (width, height) = img.dimensions();
    println!("Image: {}, Width: {}, Height: {}", path, width, height);
    let mut img_vec = vec![0u8; (width * height * 3) as usize];

    for x in 0..width {
        for y in 0..height {
            let pixel = img.get_pixel(x, y);
            img_vec[(x + y + 0) as usize] = pixel[0]; // R
            img_vec[(x + y + 1) as usize] = pixel[1]; // G
            img_vec[(x + y + 2) as usize] = pixel[2]; // B
        }
    }
    img_vec
}

fn bench_mul(c: &mut Criterion) {
    println!("#################### Bench: tensor::mul ####################");

    /* Bench: (100x100x3) * (100x3x3) = (100x100x3) */
    println!("##### Bench: (100x100x3) * (100x3x3) = (100x100x3) #####");
    let img = img_to_vec("/usr/local/dev/rs_math/rs_math/benches/input/matrix_imgs/100x100x3.png");
    let tensor_a = Tensor::from_shape(&[100, 100, 3], &img).unwrap();
    println!("Shape (A): {:?}", tensor_a.shape());

    let img = img_to_vec("/usr/local/dev/rs_math/rs_math/benches/input/matrix_imgs/100x3x3.png");
    let tensor_b = Tensor::from_shape(&[100, 3, 3], &img).unwrap();
    println!("Shape (B): {:?}", tensor_b.shape());
    drop(img);

    c.bench_function("tensor::mul: (100x100x3) * (100x3x3) = (100x100x3)", |b| {
        b.iter(|| {
            tensor_a.mul(&tensor_b.view()).unwrap();
        })
    });

    /* Bench: (1000x1000x3) * (1000x3x3) = (1000x1000x3) */
    println!("##### Bench: (1000x1000x3) * (1000x3x3) = (1000x1000x3) #####");
    let img =
        img_to_vec("/usr/local/dev/rs_math/rs_math/benches/input/matrix_imgs/1000x1000x3.png");
    let tensor_a = Tensor::from_shape(&[1000, 1000, 3], &img).unwrap();
    println!("Shape (A): {:?}", tensor_a.shape());

    let img = img_to_vec("/usr/local/dev/rs_math/rs_math/benches/input/matrix_imgs/1000x3x3.png");
    let tensor_b = Tensor::from_shape(&[1000, 3, 3], &img).unwrap();
    println!("Shape (B): {:?}", tensor_b.shape());
    drop(img);

    c.bench_function(
        "tensor::mul: (1000x1000x3) * (1000x3x3) = (1000x1000x3)",
        |b| {
            b.iter(|| {
                tensor_a.mul(&tensor_b.view()).unwrap();
            })
        },
    );

    /* Bench: (10000x10000x3) * (10000x3x3) = (10000x10000x3) */
    println!("##### Bench: (10000x10000x3) * (10000x3x3) = (10000x10000x3) #####");
    let img =
        img_to_vec("/usr/local/dev/rs_math/rs_math/benches/input/matrix_imgs/10000x10000x3.png");
    let tensor_a = Tensor::from_shape(&[10000, 10000, 3], &img).unwrap();
    println!("Shape (A): {:?}", tensor_a.shape());

    let img = img_to_vec("/usr/local/dev/rs_math/rs_math/benches/input/matrix_imgs/10000x3x3.png");
    let tensor_b = Tensor::from_shape(&[10000, 3, 3], &img).unwrap();
    println!("Shape (B): {:?}", tensor_b.shape());
    drop(img);

    let mut group = c.benchmark_group("sample_size:10");
    group.sample_size(10);
    group.bench_function(
        "tensor::mul: (10000x10000x3) * (10000x3x3) = (10000x10000x3)",
        |b| {
            b.iter(|| {
                tensor_a.mul(&tensor_b.view()).unwrap();
            })
        },
    );
}

criterion_group!(benches, bench_mul);
criterion_main!(benches);
