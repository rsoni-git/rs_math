import numpy as np

A = np.array([
    [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [7, 8, 9],
            [10, 11, 12]
        ],
        [
            [13, 14, 15],
            [16, 17, 18]
        ]
    ],
    [
        [
            [19, 20, 21],
            [22, 23, 24]
        ],
        [
            [25, 26, 27],
            [28, 29, 30]
        ],
        [
            [31, 32, 33],
            [34, 35, 36]
        ]
    ]
])

B = arr = np.array([
    [
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    ],
    [
        [
            [7, 8],
            [9, 10],
            [11, 12],
        ]
    ]
])

C = np.matmul(A, B)  # shape: (2, 4, 4)

print(C)
print(f"Shape_a: {A.shape}")
print(f"Shape_b: {B.shape}")
print(f"Shape_c: {C.shape}")
