use rs_math::tensor::{Error, Tensor};

#[path = "utils/ndim_vec.rs"]
mod ndim_vec;

#[path = "utils/py_ndarray.rs"]
mod py_ndarray;

#[test]
fn from_zeros() {
    // 1D tensor
    let tensor_1d = Tensor::<i32>::from_zeros(&[2]).unwrap();
    assert_eq!(tensor_1d.strides(), [1]);

    // 2D tensor
    let tensor_2d = Tensor::<i32>::from_zeros(&[2, 3]).unwrap();
    assert_eq!(tensor_2d.strides(), [3, 1]);

    // 3D tensor
    let tensor_3d = Tensor::<i32>::from_zeros(&[2, 3, 4]).unwrap();
    assert_eq!(tensor_3d.strides(), [12, 4, 1]);

    // 4D tensor
    let tensor_4d = Tensor::<i32>::from_zeros(&[2, 3, 4, 5]).unwrap();
    assert_eq!(tensor_4d.strides(), [60, 20, 5, 1]);

    // 5D tensor
    let tensor_5d = Tensor::<i32>::from_zeros(&[2, 3, 4, 5, 6]).unwrap();
    assert_eq!(tensor_5d.strides(), [360, 120, 30, 6, 1]);

    // TODO: Negative test related to input data sanity
}

#[test]
fn data() {
    // 1D tensor
    let tensor_1d = Tensor::from_vec(vec![1, 2, 3, 4]).unwrap();
    assert_eq!(tensor_1d.data(), vec![1, 2, 3, 4]);

    // 2D tensor
    let tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(tensor_2d.data(), vec![1, 2, 3, 4, 5, 6]);

    // 3D tensor
    let tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();
    assert_eq!(tensor_3d.data(), vec![1, 2, 3, 4, 5, 6, 7, 8]);

    // 4D tensor
    let tensor_4d = Tensor::from_vec(vec![
        vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]],
        vec![
            vec![vec![9, 10], vec![11, 12]],
            vec![vec![13, 14], vec![15, 16]],
        ],
    ])
    .unwrap();

    assert_eq!(
        tensor_4d.data(),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );

    // TODO: 5D tensor
}

#[test]
fn shape() {
    // 1D tensor
    let tensor_1d = Tensor::from_vec(vec![1, 2, 3, 4]).unwrap();
    assert_eq!(tensor_1d.shape(), vec![4]);

    // 2D tensor
    let tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(tensor_2d.shape(), vec![2, 3]);

    // 3D tensor
    let tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();
    assert_eq!(tensor_3d.shape(), vec![2, 2, 2]);

    // 4D tensor
    let tensor_4d = Tensor::from_vec(vec![
        vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]],
        vec![
            vec![vec![9, 10], vec![11, 12]],
            vec![vec![13, 14], vec![15, 16]],
        ],
    ])
    .unwrap();
    assert_eq!(tensor_4d.shape(), vec![2, 2, 2, 2]);

    // TODO: 5D tensor
}

#[test]
fn strides() {
    // 1D tensor
    let tensor_1d = Tensor::from_vec(vec![1, 2, 3, 4]).unwrap();
    assert_eq!(tensor_1d.strides(), vec![1]);

    // 2D tensor
    let tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(tensor_2d.strides(), vec![3, 1]);

    // 3D tensor
    let tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();
    assert_eq!(tensor_3d.strides(), vec![4, 2, 1]);

    // 4D tensor
    let tensor_4d = Tensor::from_vec(vec![
        vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]],
        vec![
            vec![vec![9, 10], vec![11, 12]],
            vec![vec![13, 14], vec![15, 16]],
        ],
    ])
    .unwrap();

    assert_eq!(tensor_4d.strides(), vec![8, 4, 2, 1]);

    // TODO: 5D tensor
}

#[test]
fn dimension() {
    // 1D tensor
    let tensor_1d = Tensor::<i32>::from_zeros(&[2]).unwrap();
    assert_eq!(tensor_1d.ndim(), 1);

    // 2D tensor
    let tensor_2d = Tensor::<i32>::from_zeros(&[2, 3]).unwrap();
    assert_eq!(tensor_2d.ndim(), 2);

    // 3D tensor
    let tensor_3d = Tensor::<i32>::from_zeros(&[2, 3, 4]).unwrap();
    assert_eq!(tensor_3d.ndim(), 3);

    // 4D tensor
    let tensor_4d = Tensor::<i32>::from_zeros(&[2, 3, 4, 5]).unwrap();
    assert_eq!(tensor_4d.ndim(), 4);

    // 5D tensor
    let tensor_5d = Tensor::<i32>::from_zeros(&[2, 3, 4, 5, 6]).unwrap();
    assert_eq!(tensor_5d.ndim(), 5);
}

#[test]
fn getval() {
    // 1D tensor
    let tensor_1d = Tensor::from_vec(vec![1, 2, 3, 4]).unwrap();
    assert_eq!(tensor_1d.getval(&[0]).unwrap(), 1);
    assert_eq!(tensor_1d.getval(&[1]).unwrap(), 2);
    assert_eq!(tensor_1d.getval(&[2]).unwrap(), 3);
    assert_eq!(tensor_1d.getval(&[3]).unwrap(), 4);

    // 2D tensor
    let tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(tensor_2d.getval(&[0, 0]).unwrap(), 1);
    assert_eq!(tensor_2d.getval(&[0, 1]).unwrap(), 2);
    assert_eq!(tensor_2d.getval(&[0, 2]).unwrap(), 3);
    assert_eq!(tensor_2d.getval(&[1, 0]).unwrap(), 4);
    assert_eq!(tensor_2d.getval(&[1, 1]).unwrap(), 5);
    assert_eq!(tensor_2d.getval(&[1, 2]).unwrap(), 6);

    // 3D tensor
    let tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();
    assert_eq!(tensor_3d.getval(&[0, 0, 0]).unwrap(), 1);
    assert_eq!(tensor_3d.getval(&[0, 0, 1]).unwrap(), 2);
    assert_eq!(tensor_3d.getval(&[0, 1, 0]).unwrap(), 3);
    assert_eq!(tensor_3d.getval(&[0, 1, 1]).unwrap(), 4);
    assert_eq!(tensor_3d.getval(&[1, 0, 0]).unwrap(), 5);
    assert_eq!(tensor_3d.getval(&[1, 0, 1]).unwrap(), 6);
    assert_eq!(tensor_3d.getval(&[1, 1, 0]).unwrap(), 7);
    assert_eq!(tensor_3d.getval(&[1, 1, 1]).unwrap(), 8);

    // Negative: Empty index
    assert!(matches!(
        tensor_1d.getval(&[]),
        Err(Error::ShapeMismatch { .. })
    ));

    // Negative: Out of bound
    assert!(matches!(
        tensor_1d.getval(&[4]),
        Err(Error::IndexOutOfRange { .. })
    ));
    assert!(matches!(
        tensor_3d.getval(&[1, 1, 8]),
        Err(Error::IndexOutOfRange { .. })
    ));

    // Negative: Shape mismatch
    assert!(matches!(
        tensor_3d.getval(&[1, 1]),
        Err(Error::ShapeMismatch { .. })
    ));

    assert!(matches!(
        tensor_3d.getval(&[1, 1, 1, 1]),
        Err(Error::ShapeMismatch { .. })
    ));

    // TODO: 4D tensor
    // TODO: 5D tensor
}

#[test]
fn axis() {
    // 2D tensor
    let tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(
        tensor_2d.axis(0).unwrap(),
        vec![vec![1, 2, 3], vec![4, 5, 6]]
    );
    assert_eq!(
        tensor_2d.axis(1).unwrap(),
        vec![vec![1, 4], vec![2, 5], vec![3, 6]]
    );

    // 3D tensor
    let tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();

    assert_eq!(
        tensor_3d.axis(0).unwrap(),
        vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]]
    );
    assert_eq!(
        tensor_3d.axis(1).unwrap(),
        vec![vec![vec![1, 2], vec![5, 6]], vec![vec![3, 4], vec![7, 8]]]
    );
    assert_eq!(
        tensor_3d.axis(2).unwrap(),
        vec![vec![vec![1, 3], vec![5, 7]], vec![vec![2, 4], vec![6, 8]]]
    );

    // Negative: Invalid axis
    assert!(matches!(tensor_3d.axis(3), Err(Error::InvalidAxis { .. })));

    // TODO: 1D tensor
    // TODO: 4D tensor
    // TODO: 5D tensor
}

#[test]
fn slice() {
    // 2D tensor
    let tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(tensor_2d.slice(&[0]).unwrap(), vec![1, 2, 3]);
    assert_eq!(tensor_2d.slice(&[0, 0]).unwrap(), 1);
    assert_eq!(tensor_2d.slice(&[0, 1]).unwrap(), 2);
    assert_eq!(tensor_2d.slice(&[0, 2]).unwrap(), 3);

    assert_eq!(tensor_2d.slice(&[1]).unwrap(), vec![4, 5, 6]);
    assert_eq!(tensor_2d.slice(&[1, 0]).unwrap(), 4);
    assert_eq!(tensor_2d.slice(&[1, 1]).unwrap(), 5);
    assert_eq!(tensor_2d.slice(&[1, 2]).unwrap(), 6);

    // 3D tensor
    let tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();
    assert_eq!(tensor_3d.slice(&[0]).unwrap(), vec![vec![1, 2], vec![3, 4]]);
    assert_eq!(tensor_3d.slice(&[0, 0]).unwrap(), vec![1, 2]);
    assert_eq!(tensor_3d.slice(&[0, 1]).unwrap(), vec![3, 4]);
    assert_eq!(tensor_3d.slice(&[0, 0, 0]).unwrap(), 1);
    assert_eq!(tensor_3d.slice(&[0, 0, 1]).unwrap(), 2);
    assert_eq!(tensor_3d.slice(&[0, 1, 0]).unwrap(), 3);
    assert_eq!(tensor_3d.slice(&[0, 1, 1]).unwrap(), 4);
    assert_eq!(tensor_3d.slice(&[1]).unwrap(), vec![vec![5, 6], vec![7, 8]]);
    assert_eq!(tensor_3d.slice(&[1, 0]).unwrap(), vec![5, 6]);
    assert_eq!(tensor_3d.slice(&[1, 1]).unwrap(), vec![7, 8]);
    assert_eq!(tensor_3d.slice(&[1, 0, 0]).unwrap(), 5);
    assert_eq!(tensor_3d.slice(&[1, 0, 1]).unwrap(), 6);
    assert_eq!(tensor_3d.slice(&[1, 1, 0]).unwrap(), 7);
    assert_eq!(tensor_3d.slice(&[1, 1, 1]).unwrap(), 8);

    // Negative: Slice length is greater than the tensor shape
    assert!(matches!(
        tensor_3d.slice(&[1, 1, 1, 1]),
        Err(Error::ShapeMismatch { .. })
    ));

    // Negative: Invalid slice index - 1
    assert!(matches!(
        tensor_3d.slice(&[2]),
        Err(Error::InvalidSlicing { .. })
    ));

    // Negative: Invalid slice index - 2
    assert!(matches!(
        tensor_3d.slice(&[2, 2]),
        Err(Error::InvalidSlicing { .. })
    ));

    // Negative: Invalid slice index - 3
    assert!(matches!(
        tensor_3d.slice(&[2, 2, 2]),
        Err(Error::InvalidSlicing { .. })
    ));

    // TODO: 1D tensor
    // TODO: 4D tensor
    // TODO: 5D tensor
}

#[test]
fn setval() {
    // 1D tensor
    let mut tensor_1d = Tensor::from_vec(vec![1, 2, 3]).unwrap();
    tensor_1d.setval(&[0], 11).unwrap();
    tensor_1d.setval(&[1], 12).unwrap();
    tensor_1d.setval(&[2], 13).unwrap();
    assert_eq!(tensor_1d, vec![11, 12, 13]);

    // 2D tensor
    let mut tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    tensor_2d.setval(&[0, 0], 11).unwrap();
    tensor_2d.setval(&[0, 1], 12).unwrap();
    tensor_2d.setval(&[0, 2], 13).unwrap();
    tensor_2d.setval(&[1, 0], 14).unwrap();
    tensor_2d.setval(&[1, 1], 15).unwrap();
    tensor_2d.setval(&[1, 2], 16).unwrap();
    assert_eq!(tensor_2d, vec![vec![11, 12, 13], vec![14, 15, 16]]);

    // 3D tensor
    let mut tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();

    tensor_3d.setval(&[0, 0, 0], 11).unwrap();
    tensor_3d.setval(&[0, 0, 1], 12).unwrap();
    tensor_3d.setval(&[0, 1, 0], 13).unwrap();
    tensor_3d.setval(&[0, 1, 1], 14).unwrap();
    tensor_3d.setval(&[1, 0, 0], 15).unwrap();
    tensor_3d.setval(&[1, 0, 1], 16).unwrap();
    tensor_3d.setval(&[1, 1, 0], 17).unwrap();
    tensor_3d.setval(&[1, 1, 1], 18).unwrap();

    assert_eq!(
        tensor_3d,
        vec![
            vec![vec![11, 12], vec![13, 14]],
            vec![vec![15, 16], vec![17, 18]],
        ]
    );

    // TODO: 4D tensor
    // TODO: 5D tensor
}

#[test]
fn update() {
    // 2D tensor
    let mut tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    let mut tensor_2d = tensor_2d
        .update(&[], vec![vec![11, 12, 13], vec![14, 15, 16]])
        .unwrap();
    assert_eq!(tensor_2d, vec![vec![11, 12, 13], vec![14, 15, 16]]);
    let mut tensor_2d = tensor_2d.update(&[0], vec![21, 22, 23]).unwrap();
    let tensor_2d = tensor_2d.update(&[1], vec![24, 25, 26]).unwrap();
    assert_eq!(tensor_2d, vec![vec![21, 22, 23], vec![24, 25, 26]]);

    // 3D tensor
    let mut tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();

    let mut tensor_3d = tensor_3d
        .update(
            &[],
            vec![
                vec![vec![11, 12], vec![13, 14]],
                vec![vec![15, 16], vec![17, 18]],
            ],
        )
        .unwrap();
    assert_eq!(
        tensor_3d,
        vec![
            vec![vec![11, 12], vec![13, 14]],
            vec![vec![15, 16], vec![17, 18]]
        ]
    );
    let mut tensor_3d = tensor_3d
        .update(&[0], vec![vec![21, 22], vec![23, 24]])
        .unwrap();
    let mut tensor_3d = tensor_3d
        .update(&[1], vec![vec![25, 26], vec![27, 28]])
        .unwrap();
    assert_eq!(
        tensor_3d,
        vec![
            vec![vec![21, 22], vec![23, 24]],
            vec![vec![25, 26], vec![27, 28]]
        ]
    );

    let mut tensor_3d = tensor_3d.update(&[0, 0], vec![31, 32]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[0, 1], vec![33, 34]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[1, 0], vec![35, 36]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[1, 1], vec![37, 38]).unwrap();
    assert_eq!(
        tensor_3d,
        vec![
            vec![vec![31, 32], vec![33, 34]],
            vec![vec![35, 36], vec![37, 38]]
        ]
    );

    let mut tensor_3d = tensor_3d.update(&[0, 0, 0], vec![41]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[0, 0, 1], vec![42]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[0, 1, 0], vec![43]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[0, 1, 1], vec![44]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[1, 0, 0], vec![45]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[1, 0, 1], vec![46]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[1, 1, 0], vec![47]).unwrap();
    let mut tensor_3d = tensor_3d.update(&[1, 1, 1], vec![48]).unwrap();

    assert_eq!(
        tensor_3d,
        vec![
            vec![vec![41, 42], vec![43, 44]],
            vec![vec![45, 46], vec![47, 48]]
        ]
    );

    // Updating a mutable slice
    let mut tensor_3d_sl = tensor_3d.slice_mut(&[0]).unwrap();

    let tensor_3d_sl = tensor_3d_sl
        .update(&[], vec![vec![51, 52], vec![53, 54]])
        .unwrap();

    assert_eq!(tensor_3d_sl.data(), vec![51, 52, 53, 54, 45, 46, 47, 48]);

    // TODO: 1D tensor
    // TODO: 4D tensor
    // TODO: 5D tensor
}

#[test]
fn print() {
    let tensor_2d = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    // println!("Tensor(2D): {}", tensor_2d.print());
    assert_eq!(tensor_2d.print(), "[[1, 2, 3], [4, 5, 6]]");

    let tensor_3d = Tensor::from_vec(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ])
    .unwrap();

    // println!("Tensor(3D): {}", tensor_3d.print());
    assert_eq!(tensor_3d.print(), "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]");
}

#[test]
fn transpose() {
    /* 2D tensor */
    let tensor_2x3 = Tensor::from_vec(ndim_vec::ndim_vec_2d::<i32>(&[2, 3], false)).unwrap();
    let tensor_t_3x2 = vec![vec![1, 4], vec![2, 5], vec![3, 6]];

    let tensor_t = tensor_2x3.transpose().unwrap();
    assert_eq!(tensor_t.shape(), vec!(3, 2));
    assert_eq!(tensor_t, tensor_t_3x2);

    /* 3D tensor */
    let tensor_4x3x2 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();

    #[rustfmt::skip]
    let tensor_t_2x3x4= vec![
        vec![
            vec![1, 7, 13, 19],
            vec![3, 9, 15, 21],
            vec![5, 11, 17, 23]
        ],
        vec![
            vec![2, 8, 14, 20],
            vec![4, 10, 16, 22],
            vec![6, 12, 18, 24]
        ]
    ];
    let tensor_t = tensor_4x3x2.transpose().unwrap();
    assert_eq!(tensor_t.shape(), vec!(2, 3, 4));
    assert_eq!(tensor_t, tensor_t_2x3x4)
}

#[test]
fn flatten() {
    #[rustfmt::skip]
    let tensor_4d = Tensor::from_vec(vec![
        vec![
            vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6],
            ]
        ],
        vec![
            vec![
                vec![7, 8],
                vec![9, 10],
                vec![11, 12],
            ]
        ]
    ])
    .unwrap();

    #[rustfmt::skip]
    assert_eq!(
        tensor_4d.flatten().unwrap(),
        vec![
            vec![1, 2, 3, 4, 5, 6],
            vec![7, 8, 9, 10, 11, 12]
        ]
    );
}

#[test]
fn batch() {
    let vec_10x3 = ndim_vec::ndim_vec_2d::<u8>(&[10, 3], false);
    let tensor_10x3 = Tensor::from_vec_ref(&vec_10x3).unwrap();
    let tensor_3x3 = tensor_10x3.batch(3..6).unwrap();
    assert_eq!(tensor_3x3, vec_10x3[3..6].to_vec());
}

#[test]
fn iter() {
    let tensor_5x4x3x2 =
        Tensor::from_vec(ndim_vec::ndim_vec_4d::<u8>(&[5, 4, 3, 2], false)).unwrap();

    let mut data = Vec::with_capacity(tensor_5x4x3x2.nelems());
    for val in tensor_5x4x3x2.iter() {
        data.push(val)
    }
    assert_eq!(
        Tensor::from_shape(&tensor_5x4x3x2.shape(), &data).unwrap(),
        tensor_5x4x3x2
    );

    let tensor_5x1x3x1x2 =
        Tensor::from_vec(ndim_vec::ndim_vec_5d::<u8>(&[5, 1, 3, 1, 2], false)).unwrap();

    let mut data = Vec::with_capacity(tensor_5x1x3x1x2.nelems());
    for val in tensor_5x1x3x1x2.iter() {
        data.push(val)
    }
    assert_eq!(
        Tensor::from_shape(&tensor_5x1x3x1x2.shape(), &data).unwrap(),
        tensor_5x1x3x1x2
    );
}

#[test]
fn iter_mut() {
    let mut tensor_5x4x3x2 =
        Tensor::from_vec(ndim_vec::ndim_vec_4d::<u8>(&[5, 4, 3, 2], false)).unwrap();

    let mut data = Vec::with_capacity(tensor_5x4x3x2.nelems());

    for val in tensor_5x4x3x2.iter_mut() {
        data.push(*val)
    }

    assert_eq!(
        Tensor::from_shape(&tensor_5x4x3x2.shape(), &data).unwrap(),
        tensor_5x4x3x2
    );

    let mut tensor_5x1x3x1x2 =
        Tensor::from_vec(ndim_vec::ndim_vec_5d::<u8>(&[5, 1, 3, 1, 2], false)).unwrap();

    let mut data = Vec::with_capacity(tensor_5x1x3x1x2.nelems());

    for val in tensor_5x1x3x1x2.iter_mut() {
        data.push(*val)
    }

    assert_eq!(
        Tensor::from_shape(&tensor_5x1x3x1x2.shape(), &data).unwrap(),
        tensor_5x1x3x1x2
    );

    // Increment all values by 1
    for val in tensor_5x1x3x1x2.iter_mut() {
        *val += 1;
    }

    let data: Vec<u8> = data.iter().map(|x| x + 1).collect();
        assert_eq!(
        Tensor::from_shape(&tensor_5x1x3x1x2.shape(), &data).unwrap(),
        tensor_5x1x3x1x2
    );

}
