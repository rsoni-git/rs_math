use crate::tensor::Error;
// use rand::distr::{Distribution, Uniform};
use rand::rng;
use rand_distr::{Distribution, Normal, Uniform};

pub fn uniform(len: usize, start: f32, end: f32) -> Result<Vec<f32>, Error> {
    let gen = match Uniform::new(start, end) {
        Ok(gen) => gen,
        Err(err) => {
            return Err(Error::Error {
                err_msg: err.to_string(),
            })
        }
    };

    let mut rng = rng();
    let mut samples: Vec<f32> = Vec::with_capacity(len);
    for _ in 0..len {
        samples.push(gen.sample(&mut rng));
    }
    Ok(samples)
}

pub fn uniform_he(len: usize, ninputs: usize) -> Result<Vec<f32>, Error> {
    let bound = (6.0 / ninputs as f32).sqrt();

    let gen = match Uniform::new(-bound, bound) {
        Ok(gen) => gen,
        Err(err) => {
            return Err(Error::Error {
                err_msg: err.to_string(),
            })
        }
    };

    let mut rng = rng();
    let mut samples: Vec<f32> = Vec::with_capacity(len);
    for _ in 0..len {
        samples.push(gen.sample(&mut rng));
    }
    Ok(samples)
}

pub fn normal(len: usize, mean: f32, sd: f32) -> Result<Vec<f32>, Error> {
    let gen = match Normal::new(mean, sd) {
        Ok(gen) => gen,
        Err(err) => {
            return Err(Error::Error {
                err_msg: err.to_string(),
            })
        }
    };

    let mut rng = rng();
    let mut samples: Vec<f32> = Vec::with_capacity(len);
    for _ in 0..len {
        samples.push(gen.sample(&mut rng));
    }
    Ok(samples)
}

pub fn normal_he(len: usize, ninputs: usize) -> Result<Vec<f32>, Error> {
    let sd = (2.0 / ninputs as f32).sqrt();
    let gen = match Normal::new(0.0, sd) {
        Ok(gen) => gen,
        Err(err) => {
            return Err(Error::Error {
                err_msg: err.to_string(),
            })
        }
    };

    let mut rng = rng();
    let mut samples: Vec<f32> = Vec::with_capacity(len);
    for _ in 0..len {
        samples.push(gen.sample(&mut rng));
    }
    Ok(samples)
}

#[inline(always)]
pub fn standard(len: usize) -> Result<Vec<f32>, Error> {
    normal(len, 0f32, 1f32)
}
