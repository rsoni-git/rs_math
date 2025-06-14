use rand::distr::uniform::SampleUniform;
use rand::rng;
use rand_distr::{Distribution, Uniform};
use std::ops::Add;

pub fn ndim_vec_2d<T>(shape: &[usize; 2], rand: bool) -> Vec<Vec<T>>
where
    T: Copy + Add<Output = T> + From<u8> + SampleUniform,
    Uniform<T>: Distribution<T>,
{
    let mut counter: T = T::from(1u8);
    let mut rng = rng();
    let dist = Uniform::new(T::from(1u8), T::from(100u8)).unwrap();

    (0..shape[0])
        .map(|_| {
            (0..shape[1])
                .map(|_| {
                    if rand {
                        dist.sample(&mut rng)
                    } else {
                        let val = counter;
                        counter = counter + T::from(1u8);
                        val
                    }
                })
                .collect()
        })
        .collect()
}

pub fn ndim_vec_3d<T>(shape: &[usize; 3], rand: bool) -> Vec<Vec<Vec<T>>>
where
    T: Copy + Add<Output = T> + From<u8> + SampleUniform,
    Uniform<T>: Distribution<T>,
{
    let mut counter: T = T::from(1u8);
    let mut rng = rng();
    let dist = Uniform::new(T::from(1u8), T::from(100u8)).unwrap();

    (0..shape[0])
        .map(|_| {
            (0..shape[1])
                .map(|_| {
                    (0..shape[2])
                        .map(|_| {
                            if rand {
                                dist.sample(&mut rng)
                            } else {
                                let val = counter;
                                counter = counter + T::from(1u8);
                                val
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn ndim_vec_4d<T>(shape: &[usize; 4], rand: bool) -> Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy + Add<Output = T> + From<u8> + SampleUniform,
    Uniform<T>: Distribution<T>,
{
    let mut counter: T = T::from(1u8);
    let mut rng = rng();
    let dist = Uniform::new(T::from(1u8), T::from(100u8)).unwrap();

    (0..shape[0])
        .map(|_| {
            (0..shape[1])
                .map(|_| {
                    (0..shape[2])
                        .map(|_| {
                            (0..shape[3])
                                .map(|_| {
                                    if rand {
                                        dist.sample(&mut rng)
                                    } else {
                                        let val = counter;
                                        counter = counter + T::from(1u8);
                                        val
                                    }
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn ndim_vec_5d<T>(shape: &[usize; 5], rand: bool) -> Vec<Vec<Vec<Vec<Vec<T>>>>>
where
    T: Copy + Add<Output = T> + From<u8> + SampleUniform,
    Uniform<T>: Distribution<T>,
{
    let mut counter: T = T::from(1u8);
    let mut rng = rng();
    let dist = Uniform::new(T::from(1u8), T::from(100u8)).unwrap();

    (0..shape[0])
        .map(|_| {
            (0..shape[1])
                .map(|_| {
                    (0..shape[2])
                        .map(|_| {
                            (0..shape[3])
                                .map(|_| {
                                    (0..shape[4])
                                        .map(|_| {
                                            if rand {
                                                dist.sample(&mut rng)
                                            } else {
                                                let val = counter;
                                                counter = counter + T::from(1u8);
                                                val
                                            }
                                        })
                                        .collect()
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}
