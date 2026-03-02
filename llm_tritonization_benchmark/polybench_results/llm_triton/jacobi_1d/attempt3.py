import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for spatial parallelization only
    pid_i = tl.program_id(0)
    
    # Calculate spatial indices for this block
    block_start = pid_i * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + 1  # +1 because we start from i=1
    
    # Mask for valid spatial indices
    mask = (i_indices >= 1) & (i_indices < N - 1)
    
    # Sequential loop over timesteps
    for t in range(TSTEPS):
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
    
    # Calculate grid dimensions - only parallelize spatial dimension
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE)  # N-2 because we process indices 1 to N-2
    
    # Launch 1D grid to parallelize space, sequential time
    grid = (grid_i,)
    
    jacobi_1d_kernel[grid](A, B, N, TSTEPS, BLOCK_SIZE)