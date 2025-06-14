use num_traits::Bounded;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::ops::{Deref, DerefMut};

pub trait TensorTypeNumeric:
    Default
    + Copy
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + Bounded
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
}

impl TensorTypeNumeric for i8 {}
impl TensorTypeNumeric for u8 {}
impl TensorTypeNumeric for i32 {}
impl TensorTypeNumeric for u32 {}
impl TensorTypeNumeric for i64 {}
impl TensorTypeNumeric for u64 {}
impl TensorTypeNumeric for f32 {}
impl TensorTypeNumeric for f64 {}

pub trait TensorTypeFloat: TensorTypeNumeric + num_traits::Float {}

impl TensorTypeFloat for f32 {}
impl TensorTypeFloat for f64 {}

#[derive(Debug)]
pub struct TensorBase<'a, U, S> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    data: S,
    _u: PhantomData<U>,
    _s: PhantomData<&'a S>,
}

pub type Tensor<'a, U> = TensorBase<'a, U, Vec<U>>;
pub type TensorView<'a, U> = TensorBase<'a, U, &'a [U]>;
pub type TensorViewMut<'a, U> = TensorBase<'a, U, &'a mut [U]>;

#[derive(Debug)]
pub enum Error {
    InvalidAxis {
        axis: usize,
        ndim: usize,
    },
    InvalidSlicing {
        slice: Vec<usize>,
        shape: Vec<usize>,
    },
    ShapeMismatch {
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    },
    DimensionMismatch {
        tensor_dim: usize,
        dim: usize,
    },
    ShapeMismatchBroadcast {
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    },
    IndexOutOfRange {
        index: usize,
        nelems: usize,
    },
    InvalidParam {
        err_msg: String,
    },
    InvalidFileContents {
        err_msg: String,
    },
    Error {
        err_msg: String,
    },
}

pub trait TensorStorage<U: TensorTypeNumeric>: AsRef<[U]> + Deref<Target = [U]> {
    fn get(&self, index: usize) -> U;
}

pub trait TensorStorageMut<U: TensorTypeNumeric>: AsMut<[U]> + DerefMut<Target = [U]> {
    fn get(&self, index: usize) -> U;
    fn set(&mut self, index: usize, value: U);
}

pub trait TensorEq<U> {
    fn tensor_eq(&self, view: &TensorView<U>, indices: &mut Vec<usize>, depth: usize) -> bool;
}

pub struct TensorIter<'a, U> {
    shape: &'a [usize],
    strides: &'a [usize],
    offset: &'a usize,
    data: &'a [U],
    index: Vec<usize>,
    _u: PhantomData<U>,
}

pub struct TensorIterMut<'a, U> {
    shape: &'a [usize],
    strides: &'a [usize],
    offset: &'a usize,
    data: &'a mut [U],
    index: Vec<usize>,
    _u: PhantomData<U>,
}

pub mod tensor_arithmetic;
pub mod tensor_core;
pub mod tensor_eq;
pub mod tensor_error;
pub mod tensor_iter;
pub mod tensor_linalg;
