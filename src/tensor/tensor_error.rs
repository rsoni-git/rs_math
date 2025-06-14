use super::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Error::InvalidAxis { axis, ndim } => {
                write!(f, "Invalid axis [ AXIS: {} | NDIM: {} ]", axis, ndim)
            }

            Error::ShapeMismatch { shape_a, shape_b } => {
                write!(
                    f,
                    "Shape mismatch [ SHAPE(A): {:?} | SHAPE(B): {:?} ]",
                    shape_a, shape_b
                )
            }

            Error::DimensionMismatch { tensor_dim, dim } => {
                write!(
                    f,
                    "Dimension mismatch [ TENSOR_DIMENSION: {:?} | DIMENSION: {:?} ]",
                    tensor_dim, dim
                )
            }

            Error::ShapeMismatchBroadcast { shape_a, shape_b } => {
                write!(
                    f,
                    "The two shapes are not compatible for broadcasting [ SHAPE(A): {:?} | SHAPE(B): {:?} ]",
                    shape_a, shape_b
                )
            }

            Error::InvalidSlicing { slice, shape } => {
                write! {
                    f,
                    "Invalid slicing [ SLICE: {:?} | SHAPE: {:?} ]",
                    slice, shape
                }
            }

            Error::IndexOutOfRange { index, nelems } => {
                write! {
                    f,
                    "Index out of range [ INDEX: {:?} | NUM_ELEMENTS: {:?} ]",
                    index, nelems
                }
            }
            Error::InvalidParam { err_msg } => {
                write! {
                    f,
                    "Invalid parameter [ DESC: {} ]",
                    err_msg
                }
            }
            Error::InvalidFileContents { err_msg } => {
                write! {
                    f,
                    "Error [ ERR_MSG: {} ]",
                    err_msg,
                }
            }
            Error::Error { err_msg } => {
                write! {
                    f,
                    "Error [ ERR_MSG: {} ]",
                    err_msg,
                }
            }
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Error {
            err_msg: err.to_string(),
        }
    }
}
