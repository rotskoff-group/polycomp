import cupy as cp

# Kernel to multiply by exp(-K2 * h) on the GPU
# Used in the MDE
exp_mult_comp = cp.ElementwiseKernel(
    "complex128 q_k, float64 K2, float64 h",
    "complex128 out",
    "out = q_k * exp(-K2 * h)",
    "exp_mult_comp",
)


# Kernel to multiply by exp(-w_P * h) on the GPU
# Note that externalizing the division significantly speeds up this funciton
# Used in the MDE
exp_mult = cp.ElementwiseKernel(
    "complex128 q_r, complex128 w_P, float64 h_2",
    "complex128 out",
    "out = q_r * exp(-w_P  * h_2)",
    "exp_mult",
)


# Kernel for simple multiplication on the GPU
# Used in convolutions
kernel_mult_float = cp.ElementwiseKernel(
    "complex128 q_k, float64 kernel",
    "complex128 out",
    "out = q_k * kernel",
    "kernel_mult",
)


# Kernel for simple multiplication on the GPU
# Used in convolutions
kernel_mult_complex = cp.ElementwiseKernel(
    "complex128 q_k, complex128 kernel",
    "complex128 out",
    "out = q_k * kernel",
    "kernel_mult",
)
