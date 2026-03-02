import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program IDs for 2D parallelization
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Check if this thread should process this timestep
    if pid_t >= TSTEPS:
        return
    
    # Calculate spatial indices for this block
    block_start = pid_i * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + 1  # +1 because we start from i=1
    
    # Mask for valid spatial indices
    mask = (i_indices >= 1) & (i_indices < N - 1)
    
    # Each thread block processes one timestep independently
    t = pid_t
    
    # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
    A_prev = tl.load(A_ptr + i_indices - 1, mask=mask)
    A_curr = tl.load(A_ptr + i_indices, mask=mask)
    A_next = tl.load(A_ptr + i_indices + 1, mask=mask)
    B_vals = 0.33333 * (A_prev + A_curr + A_next)
    tl.store(B_ptr + i_indices, B_vals, mask=mask)
    
    # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
    B_prev = tl.load(B_ptr + i_indices - 1, mask=mask)
    B_curr = tl.load(B_ptr + i_indices, mask=mask)
    B_next = tl.load(B_ptr + i_indices + 1, mask=mask)
    A_vals = 0.33333 * (B_prev + B_curr + B_next)
    tl.store(A_ptr + i_indices, A_vals, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 64
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE)  # N-2 because we process indices 1 to N-2
    grid_t = TSTEPS
    
    # Launch 2D grid to parallelize both time and space
    grid = (grid_t, grid_i)
    
    jacobi_1d_kernel[grid](A, B, N, TSTEPS, BLOCK_SIZE)