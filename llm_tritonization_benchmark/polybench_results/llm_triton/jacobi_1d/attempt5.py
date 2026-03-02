import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for spatial dimension
    pid = tl.program_id(0)
    
    # Calculate spatial indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + 1  # +1 because we start from i=1
    
    # Mask for valid spatial indices
    mask = (i_indices >= 1) & (i_indices < N - 1)
    
    # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
    A_prev = tl.load(A_ptr + i_indices - 1, mask=mask)
    A_curr = tl.load(A_ptr + i_indices, mask=mask)
    A_next = tl.load(A_ptr + i_indices + 1, mask=mask)
    B_vals = 0.33333 * (A_prev + A_curr + A_next)
    tl.store(B_ptr + i_indices, B_vals, mask=mask)

@triton.jit
def jacobi_1d_kernel_second(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for spatial dimension
    pid = tl.program_id(0)
    
    # Calculate spatial indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + 1  # +1 because we start from i=1
    
    # Mask for valid spatial indices
    mask = (i_indices >= 1) & (i_indices < N - 1)
    
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
    
    # Sequential execution over timesteps
    for t in range(TSTEPS):
        # First kernel: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        jacobi_1d_kernel[(grid_i,)](A, B, N, BLOCK_SIZE)
        
        # Second kernel: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        jacobi_1d_kernel_second[(grid_i,)](A, B, N, BLOCK_SIZE)