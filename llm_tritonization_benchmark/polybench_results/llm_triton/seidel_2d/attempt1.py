import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # Calculate linear indices for the 9 neighbors
                idx_center = i * N + j
                idx_up_left = (i - 1) * N + (j - 1)
                idx_up = (i - 1) * N + j
                idx_up_right = (i - 1) * N + (j + 1)
                idx_left = i * N + (j - 1)
                idx_right = i * N + (j + 1)
                idx_down_left = (i + 1) * N + (j - 1)
                idx_down = (i + 1) * N + j
                idx_down_right = (i + 1) * N + (j + 1)
                
                # Load the 9 values
                val_up_left = tl.load(A_ptr + idx_up_left)
                val_up = tl.load(A_ptr + idx_up)
                val_up_right = tl.load(A_ptr + idx_up_right)
                val_left = tl.load(A_ptr + idx_left)
                val_center = tl.load(A_ptr + idx_center)
                val_right = tl.load(A_ptr + idx_right)
                val_down_left = tl.load(A_ptr + idx_down_left)
                val_down = tl.load(A_ptr + idx_down)
                val_down_right = tl.load(A_ptr + idx_down_right)
                
                # Compute average and store
                result = (val_up_left + val_up + val_up_right +
                         val_left + val_center + val_right +
                         val_down_left + val_down + val_down_right) / 9.0
                
                tl.store(A_ptr + idx_center, result)

def seidel_2d_triton(A, N, TSTEPS):
    grid = (1,)
    seidel_2d_kernel[grid](A, N, TSTEPS)