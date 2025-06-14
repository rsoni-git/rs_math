import numpy as np

mat_a = np.array([
  [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
  [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]
])

mat_b = np.array([
  [[[25, 26], [27, 28]], [[29, 30], [31, 32]]],  # Shape: (2,2,2,2)
  [[[33, 34], [35, 36]], [[37, 38], [39, 40]]]
])

print(f"Shape of matrix A: {mat_a.shape}")
print(f"Shape of matrix B: {mat_b.shape}")

mat_c = np.tensordot(mat_a, mat_b, axes = 0)
print(f"Shape of matrix C (axis 0): {mat_c.shape}")

mat_c = np.tensordot(mat_a, mat_b, axes = 1)
print(f"Shape of matrix C (axis 1): {mat_c.shape}")

mat_c = np.tensordot(mat_a, mat_b, axes = 2)
print(f"Shape of matrix C (axis 2): {mat_c.shape}")
