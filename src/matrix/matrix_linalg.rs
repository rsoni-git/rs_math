use super::*;
use crate::tensor::{Error, Tensor};
use std::cmp::Eq;
use std::hash::Hash;

impl<'a> Matrix<'a, u8> {
    pub fn from_one_hot_enc<L: Eq + Hash>(labels: &Vec<L>) -> Result<Self, Error> {
        let tensor = Tensor::from_one_hot_enc(labels)?;
        Ok(Matrix { tensor })
    }
}
