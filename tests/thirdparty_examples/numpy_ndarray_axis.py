#!/usr/bin/env python3

import numpy as np

arrays = {
  "1D": np.array([10, 20, 30]),

  "2D": np.array(
    [
      [1, 2, 3],
      [4, 5, 6]
    ]
  ),

  "3D": np.array(
    [
      [
        [1, 2],
        [3, 4]
      ],
      [
        [5, 6],
        [7, 8]
      ]
    ]
  )
}

for name, arr in arrays.items():
  print(f"\n{'='*10} {name} array, shape={arr.shape} {'='*10}")
  print(arr)

  for axis in range(arr.ndim):
    print(f"\n---- Axis {axis} slices ----")
    for i in range(arr.shape[axis]):
      print(f"\nIndex {i}:\n{np.take(arr, i, axis=axis)}")
