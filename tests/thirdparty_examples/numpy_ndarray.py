#!/usr/bin/env python3

import numpy as np

# 1D array
a1 = np.zeros((2,), dtype=np.float32)
print(f"\n1D array:\n    Dimensions: {a1.ndim}, Shape: {a1.shape}, Strides: {a1.strides}")

# 2D array
a2 = np.zeros((2, 3), dtype=np.float32)
print(f"\n2D array:\n    Dimensions: {a2.ndim}, Shape: {a2.shape}, Strides: {a2.strides}")

# 3D array
a3 = np.zeros((2, 3, 4), dtype=np.float32)
print(f"\n3D array:\n    Dimensions: {a3.ndim}, Shape: {a3.shape}, Strides: {a3.strides}")

# 4D array
a4 = np.zeros((2, 3, 4, 5), dtype=np.float32)
print(f"\n4D array:\n    Dimensions: {a4.ndim}, Shape: {a4.shape}, Strides: {a4.strides}")

# 5D array
a5 = np.zeros((2, 3, 4, 5, 6), dtype=np.float32)
print(f"\n5D array:\n    Dimensions: {a5.ndim}, Shape: {a5.shape}, Strides: {a5.strides}")
