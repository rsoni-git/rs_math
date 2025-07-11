# Changelog

[v1.1.1]
- Added core n-dimensional tensor framework.
  - Tensor from the n-dimensional vector.
  - Tensor from 1D vector and shape.
  - Zero-initialized tensor from shape.
  - Member get-set functionality.
- Added TensorView framework.
  - Creating TensorView from TensorBase.
  - Tensor axis view.
  - Tensor batch view.
- Added Tensor slicing framework.
   - Indices slicing.
   - Axis-Indices slicing.
- Added tensor iteration functionality.
  - Iteration by value.
  - Iteration by reference.
- Added TensorError framework.
- Added Tensor comparison framework.
  - Tensor - Tensor comparison.
  - Tensor - TensorView comparison.
- Added tensor arithmetic framework.
  - Addition with broadcasting.
  - Subtraction with broadcasting.
  - Multiplication with broadcasting.
  - Scaler addition.
  - Scaler subtraction.
  - Scaler multiplication.
  - Get max/min value.
- Added Tensor permute/transpose functionality.
- Added Tensor cloning functionality.
- Added activation functions.
  - ReLU
  - Softmax
