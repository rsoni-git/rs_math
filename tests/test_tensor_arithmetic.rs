use rs_math::tensor::Tensor;

#[path = "utils/ndim_vec.rs"]
mod ndim_vec;

#[test]
fn add() {
    /* 3D tensor: Adding two similar tensors */
    let tensor_4x3x2_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor_4x3x2_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor = tensor_4x3x2_a.add(&tensor_4x3x2_b.view()).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), vec![4, 3, 2]);

    // Verify data
    assert_eq!(
        tensor,
        vec![
            vec![vec![2, 4], vec![6, 8], vec![10, 12]],
            vec![vec![14, 16], vec![18, 20], vec![22, 24]],
            vec![vec![26, 28], vec![30, 32], vec![34, 36]],
            vec![vec![38, 40], vec![42, 44], vec![46, 48]]
        ]
    );

    /* 3D tensor: Partial match on third dimension */

    let tensor_4x3x2 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor_4x3x1 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 1], false)).unwrap();
    let tensor = tensor_4x3x2.add(&tensor_4x3x1.view()).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), vec![4, 3, 2]);

    // Verify data
    assert_eq!(
        tensor,
        vec![
            vec![vec![2, 3], vec![5, 6], vec![8, 9]],
            vec![vec![11, 12], vec![14, 15], vec![17, 18]],
            vec![vec![20, 21], vec![23, 24], vec![26, 27]],
            vec![vec![29, 30], vec![32, 33], vec![35, 36]]
        ]
    );

    // Check that the addition is commutative
    assert_eq!(
        tensor_4x3x2.add(&tensor_4x3x1.view()).unwrap(),
        tensor_4x3x1.add(&tensor_4x3x2.view()).unwrap()
    );

    /* 3D tensor: Partial match on second and third dimensions */

    let tensor_4x3x2 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor_4x1x1 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 1, 1], false)).unwrap();
    let tensor = tensor_4x3x2.add(&tensor_4x1x1.view()).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), vec![4, 3, 2]);

    // Verify data
    assert_eq!(
        tensor,
        vec![
            vec![vec![2, 3], vec![4, 5], vec![6, 7]],
            vec![vec![9, 10], vec![11, 12], vec![13, 14]],
            vec![vec![16, 17], vec![18, 19], vec![20, 21]],
            vec![vec![23, 24], vec![25, 26], vec![27, 28]]
        ]
    );

    // Check that the addition is commutative
    assert_eq!(
        tensor_4x3x2.add(&tensor_4x1x1.view()).unwrap(),
        tensor_4x1x1.add(&tensor_4x3x2.view()).unwrap()
    );

    /* 3D tensor: Partial match on all dimensions */

    let tensor_4x3x2 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor_1x1x1 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[1, 1, 1], false)).unwrap();
    let tensor = tensor_4x3x2.add(&tensor_1x1x1.view()).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), vec![4, 3, 2]);

    // Verify data
    assert_eq!(
        tensor,
        vec![
            vec![vec![2, 3], vec![4, 5], vec![6, 7]],
            vec![vec![8, 9], vec![10, 11], vec![12, 13]],
            vec![vec![14, 15], vec![16, 17], vec![18, 19]],
            vec![vec![20, 21], vec![22, 23], vec![24, 25]]
        ]
    );

    // Check that the addition is commutative
    assert_eq!(
        tensor_4x3x2.add(&tensor_4x1x1.view()).unwrap(),
        tensor_4x1x1.add(&tensor_4x3x2.view()).unwrap()
    );

    /* Operator overloading (+) */

    // TensorBase + TensorBase
    assert_eq!(
        (&tensor_4x3x2 + &tensor_4x1x1),
        tensor_4x3x2.add(&tensor_4x1x1.view()).unwrap()
    );

    // TensorView + TensorView
    assert_eq!(
        (&tensor_4x3x2.view() + &tensor_4x1x1.view()),
        tensor_4x3x2.add(&tensor_4x1x1.view()).unwrap()
    );
}

#[test]
fn sub() {
    /* 3D tensors: Adding two similar tensors */
    let tensor_4x3x2_a = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor_4x3x2_b = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor = tensor_4x3x2_a.sub(&tensor_4x3x2_b.view()).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), vec![4, 3, 2]);

    // Verify data
    assert_eq!(
        tensor,
        vec![
            vec![vec![0, 0], vec![0, 0], vec![0, 0]],
            vec![vec![0, 0], vec![0, 0], vec![0, 0]],
            vec![vec![0, 0], vec![0, 0], vec![0, 0]],
            vec![vec![0, 0], vec![0, 0], vec![0, 0]]
        ]
    );

    /* 3D tensors: Partial match on third dimension */

    let tensor_4x3x2 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor_4x3x1 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 1], false)).unwrap();
    let tensor = tensor_4x3x2.sub(&tensor_4x3x1.view()).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), vec![4, 3, 2]);

    // Verify data
    assert_eq!(
        tensor,
        vec![
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![vec![3, 4], vec![4, 5], vec![5, 6]],
            vec![vec![6, 7], vec![7, 8], vec![8, 9]],
            vec![vec![9, 10], vec![10, 11], vec![11, 12]]
        ]
    );

    /* 3D tensors: Partial match on second and third dimensions */

    let tensor_4x3x2 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor_4x1x1 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 1, 1], false)).unwrap();
    let tensor = tensor_4x3x2.sub(&tensor_4x1x1.view()).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), vec![4, 3, 2]);

    // Verify data
    assert_eq!(
        tensor,
        vec![
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            vec![vec![5, 6], vec![7, 8], vec![9, 10]],
            vec![vec![10, 11], vec![12, 13], vec![14, 15]],
            vec![vec![15, 16], vec![17, 18], vec![19, 20]]
        ]
    );

    /* 3D tensors: Partial match on all dimensions */

    let tensor_4x3x2 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[4, 3, 2], false)).unwrap();
    let tensor_1x1x1 = Tensor::from_vec(ndim_vec::ndim_vec_3d::<i32>(&[1, 1, 1], false)).unwrap();
    let tensor = tensor_4x3x2.sub(&tensor_1x1x1.view()).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), vec![4, 3, 2]);

    // Verify data
    assert_eq!(
        tensor,
        vec![
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            vec![vec![6, 7], vec![8, 9], vec![10, 11]],
            vec![vec![12, 13], vec![14, 15], vec![16, 17]],
            vec![vec![18, 19], vec![20, 21], vec![22, 23]]
        ]
    );

    /* Operator overloading (-) */

    // TensorBase - TensorBase
    assert_eq!(
        (&tensor_4x3x2 - &tensor_4x1x1),
        tensor_4x3x2.sub(&tensor_4x1x1.view()).unwrap()
    );

    // TensorView - TensorView
    assert_eq!(
        (&tensor_4x3x2.view() - &tensor_4x1x1.view()),
        tensor_4x3x2.sub(&tensor_4x1x1.view()).unwrap()
    );
}

#[test]
fn mul() {
    /* 2D tensors: 2x3 * 3x2 = 2x2 */

    let tensor_2x3 = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    let tensor_3x2 = Tensor::from_vec(vec![vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();

    let tensor_2x2 = tensor_2x3.mul(&tensor_3x2.view()).unwrap();
    assert_eq!(tensor_2x2, vec![vec![22, 28], vec![49, 64]]);

    /* 2D tensors: 3x2 * 2x3 = 3x3 */

    let tensor_3x3 = tensor_3x2.mul(&tensor_2x3.view()).unwrap();
    assert_eq!(
        tensor_3x3,
        vec![vec![9, 12, 15], vec![19, 26, 33], vec![29, 40, 51]]
    );

    /* 3D tensors: 2x4x3 * 2x3x4 = 2x4x4 */

    let tensor_2x4x3 = Tensor::from_vec(vec![
        vec![
            vec![11, 12, 13],
            vec![14, 15, 16],
            vec![17, 18, 19],
            vec![20, 21, 22],
        ],
        vec![
            vec![23, 24, 25],
            vec![26, 27, 28],
            vec![29, 30, 31],
            vec![32, 33, 34],
        ],
    ])
    .unwrap();

    let tensor_2x3x4 = Tensor::from_vec(vec![
        vec![
            vec![11, 12, 13, 14],
            vec![15, 16, 17, 18],
            vec![19, 20, 21, 22],
        ],
        vec![
            vec![23, 24, 25, 26],
            vec![27, 28, 29, 30],
            vec![31, 32, 33, 34],
        ],
    ])
    .unwrap();

    let tensor_2x4x4 = tensor_2x4x3.mul(&tensor_2x3x4.view()).unwrap();

    assert_eq!(
        tensor_2x4x4,
        vec![
            vec![
                vec![548, 584, 620, 656],
                vec![683, 728, 773, 818],
                vec![818, 872, 926, 980],
                vec![953, 1016, 1079, 1142],
            ],
            vec![
                vec![1952, 2024, 2096, 2168],
                vec![2195, 2276, 2357, 2438],
                vec![2438, 2528, 2618, 2708],
                vec![2681, 2780, 2879, 2978],
            ],
        ]
    );

    assert_eq!(tensor_2x4x4.shape(), vec![2, 4, 4]);

    /* 3D tensors: 2x3x4 * 2x4x3 = 2x3x3 */

    let tensor_2x3x3 = tensor_2x3x4.mul(&tensor_2x4x3.view()).unwrap();

    assert_eq!(
        tensor_2x3x3,
        vec![
            vec![
                vec![790, 840, 890],
                vec![1038, 1104, 1170],
                vec![1286, 1368, 1450],
            ],
            vec![
                vec![2710, 2808, 2906],
                vec![3150, 3264, 3378],
                vec![3590, 3720, 3850],
            ],
        ]
    );

    /* 4D tensors: 2x2x2x3 * 2x2x3x2 = 2x2x2x2 */

    #[rustfmt::skip]
    let tensor_2x2x2x3 = Tensor::from_vec(vec![
        vec![
            vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
            ],
            vec![
                vec![7, 8, 9],
                vec![10, 11, 12],
            ],
        ],
        vec![
            vec![
                vec![13, 14, 15],
                vec![16, 17, 18],
            ],
            vec![
                vec![19, 20, 21],
                vec![22, 23, 24],
            ],
        ],
    ])
    .unwrap();

    #[rustfmt::skip]
    let tensor_2x2x3x2 = Tensor::from_vec(vec![
        vec![
            vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6],
            ],
            vec![
                vec![7, 8],
                vec![9, 10],
                vec![11, 12],
            ],
        ],
        vec![
            vec![
                vec![13, 14],
                vec![15, 16],
                vec![17, 18],
            ],
            vec![
                vec![19, 20],
                vec![21, 22],
                vec![23, 24],
            ],
        ],
    ])
    .unwrap();

    let tensor_2x2x2x2 = tensor_2x2x2x3.mul(&tensor_2x2x3x2.view()).unwrap();

    #[rustfmt::skip]
    assert_eq!(
        tensor_2x2x2x2,
        vec![
            vec![
                vec![
                    vec![22, 28],
                    vec![49, 64]
                ],
                vec![
                    vec![220, 244],
                    vec![301, 334]
                ]
            ],
            vec![
                vec![
                    vec![634, 676],
                    vec![769, 820]
                ],
                vec![
                    vec![1264, 1324],
                    vec![1453, 1522]
                ]
            ]
        ]
    );

    /* 4D tensors: 2x3x2x3 * 2x1x3x2 = 2x3x2x2 (Partial match on second dimension) */

    #[rustfmt::skip]
    let tensor_2x3x2x3 = Tensor::from_vec(vec![
        vec![
            vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
            ],
            vec![
                vec![7, 8, 9],
                vec![10, 11, 12],
            ],
            vec![
                vec![13, 14, 15],
                vec![16, 17, 18],
            ]
        ],
        vec![
            vec![
                vec![19, 20, 21],
                vec![22, 23, 24],
            ],
            vec![
                vec![25, 26, 27],
                vec![28, 29, 30],
            ],
            vec![
                vec![31, 32, 33],
                vec![34, 35, 36],
            ]
        ]
    ])
    .unwrap();

    #[rustfmt::skip]
    let tensor_2x1x3x2 = Tensor::from_vec(vec![
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

    let tensor_2x3x2x2 = tensor_2x3x2x3.mul(&tensor_2x1x3x2.view()).unwrap();

    #[rustfmt::skip]
    assert_eq!(
        tensor_2x3x2x2,
        vec![
            vec![
                vec![
                    vec![22, 28],
                    vec![49, 64]
                ],
                vec![
                    vec![76, 100],
                    vec![103, 136]
                ],
                vec![
                    vec![130, 172],
                    vec![157, 208]
                ]
            ],
            vec![
                vec![
                    vec![544, 604],
                    vec![625, 694]
                ],
                vec![
                    vec![706, 784],
                    vec![787, 874]
                ],
                vec![
                    vec![868, 964],
                    vec![949, 1054]
                ]
            ]
        ]
    );

    /* Operator overloading (-) */

    // TensorBase - TensorBase
    assert_eq!(
        (&tensor_2x1x3x2 * &tensor_2x3x2x2),
        tensor_2x1x3x2.mul(&tensor_2x3x2x2.view()).unwrap()
    );

    // TensorView - TensorView
    assert_eq!(
        (&tensor_2x1x3x2.view() * &tensor_2x3x2x2.view()),
        tensor_2x1x3x2.mul(&tensor_2x3x2x2.view()).unwrap()
    );

    // TODO: TensoBase - TensorView
    // TODO: TensoBase - TensorViewMut
    // TODO: TensorView - TensorViewMut
}

#[test]
fn add_scalar() {
    /* 2D tensor: (2x3) + 5 */
    let mut tensor_2x3 = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    tensor_2x3.add_scalar(5);
    assert_eq!(tensor_2x3, vec![vec![6, 7, 8], vec![9, 10, 11]]);

    /* Operator overloading (+=) */
    let mut tensor_2x3_a = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    let mut tensor_2x3_b = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(tensor_2x3_a.add_scalar(5), (tensor_2x3_b += 5));
}

#[test]
fn sub_scalar() {
    /* 2D tensor: (2x3) + 5 */
    let mut tensor_2x3 = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    tensor_2x3.sub_scalar(5);
    assert_eq!(tensor_2x3, vec![vec![-4, -3, -2], vec![-1, 0, 1]]);

    /* Operator overloading (+=) */
    let mut tensor_2x3_a = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    let mut tensor_2x3_b = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(tensor_2x3_a.sub_scalar(5), (tensor_2x3_b -= 5));
}

#[test]
fn mul_scalar() {
    /* 2D tensor: (2x3) + 5 */
    let mut tensor_2x3 = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    tensor_2x3.mul_scalar(5);
    assert_eq!(tensor_2x3, vec![vec![5, 10, 15], vec![20, 25, 30]]);

    /* Operator overloading (+=) */
    let mut tensor_2x3_a = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    let mut tensor_2x3_b = Tensor::from_vec(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    assert_eq!(tensor_2x3_a.mul_scalar(5), (tensor_2x3_b *= 5));
}
