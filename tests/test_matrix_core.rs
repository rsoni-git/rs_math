use rs_math::matrix::Matrix;

#[test]
fn from_zeros() {
    let matrix = Matrix::<i32>::from_zeros(2, 3).unwrap();
    assert_eq!(matrix, vec![vec![0, 0, 0], vec![0, 0, 0]]);
}

#[test]
fn from_vec() {
    let matrix = Matrix::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(matrix, vec![vec![1, 2, 3], vec![4, 5, 6]]);
}

#[test]
fn row() {
    let matrix = Matrix::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(matrix.row(0).unwrap(), vec![1, 2, 3]);
    assert_eq!(matrix.row(1).unwrap(), vec![4, 5, 6]);
}

#[test]
fn col() {
    let matrix = Matrix::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(matrix.col(0).unwrap(), vec![1, 4]);
    assert_eq!(matrix.col(1).unwrap(), vec![2, 5]);
    assert_eq!(matrix.col(2).unwrap(), vec![3, 6]);
}

#[test]
fn update() {
    let mut matrix = Matrix::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(
        matrix
            .update(vec![vec![11, 12, 13], vec![14, 15, 16]])
            .unwrap(),
        vec![vec![11, 12, 13], vec![14, 15, 16]]
    );
}

#[test]
fn update_row() {
    let mut matrix = Matrix::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(
        matrix.update_row(0, vec![11, 12, 13]).unwrap(),
        vec![vec![11, 12, 13], vec![4, 5, 6]]
    );

    // TODO: Immutable re-borrow: Implement "Eq" for references
    // assert_eq!(
    //     matrix.update_row(1, vec![14, 15, 16]).unwrap(),
    //     vec![vec![11, 12, 13], vec![14, 15, 16]]
    // );
}

#[test]
fn eq() {
    let matrix1 = Matrix::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    let matrix2 = Matrix::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();

    // Compare two matrix objects
    assert_eq!(matrix1, matrix2);

    // Compare a matrix object with a vector object
    assert_eq!(matrix1, vec![vec![1, 2, 3], vec![4, 5, 6]]);
}
