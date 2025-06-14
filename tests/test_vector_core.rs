use rs_math::vector::Vector;

#[test]
fn from_zeros() {
    let vector = Vector::from_zeros(3).unwrap();
    assert_eq!(vector, vec![vec![0], vec![0], vec![0]]);
}

#[test]
fn from_vec() {
    let vector = Vector::from_vec(vec![1, 2, 3]).unwrap();
    assert_eq!(vector, vec![vec![1], vec![2], vec![3]]);
}

#[test]
fn from_arr() {
    let vector = Vector::from_arr(&[1, 2, 3]).unwrap();
    assert_eq!(vector, vec![vec![1], vec![2], vec![3]]);
}

#[test]
fn get() {
    let vector = Vector::from_arr(&[1, 2, 3]).unwrap();
    assert_eq!(vector.get(0).unwrap(), 1);
    assert_eq!(vector.get(1).unwrap(), 2);
    assert_eq!(vector.get(2).unwrap(), 3);
}

#[test]
fn set() {
    let mut vector = Vector::from_arr(&[1, 2, 3]).unwrap();
    vector.set(0, 11).unwrap();
    vector.set(1, 12).unwrap();
    vector.set(2, 13).unwrap();

    assert_eq!(vector, vec![vec![11], vec![12], vec![13]]);
}

#[test]
fn eq() {
    // Immutable matrices
    let vector1 = Vector::from_vec(vec![1, 2, 3]).unwrap();
    let vector2 = Vector::from_vec(vec![1, 2, 3]).unwrap();

    // Compare two immutable matrices
    assert_eq!(vector1, vector2);

    // Compare immutable matrix and vector
    assert_eq!(vector1, vec![vec![1], vec![2], vec![3]]);

    // Mutable matrices
    let vector1_mut = Vector::from_vec(vec![1, 2, 3]).unwrap();
    let vector2_mut = Vector::from_vec(vec![1, 2, 3]).unwrap();

    // Compare two mutable matrices
    assert_eq!(vector1_mut, vector2_mut);

    // Compare mutable matrix and vector
    assert_eq!(vector1_mut, vec![vec![1], vec![2], vec![3]]);
}
