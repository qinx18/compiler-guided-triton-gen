import torch
import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset array once
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices (1 <= i < N-1)
    mask = (indices >= 1) & (indices < N - 1)
    
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        # Load A values with offset patterns
        a_prev = tl.load(A_ptr + indices - 1, mask=mask)
        a_curr = tl.load(A_ptr + indices, mask=mask)
        a_next = tl.load(A_ptr + indices + 1, mask=mask)
        
        # Compute B values
        b_vals = 0.33333 * (a_prev + a_curr + a_next)
        
        # Store B values
        tl.store(B_ptr + indices, b_vals, mask=mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        # Load B values with offset patterns
        b_prev = tl.load(B_ptr + indices - 1, mask=mask)
        b_curr = tl.load(B_ptr + indices, mask=mask)
        b_next = tl.load(B_ptr + indices + 1, mask=mask)
        
        # Compute A values
        a_vals = 0.33333 * (b_prev + b_curr + b_next)
        
        # Store A values
        tl.store(A_ptr + indices, a_vals, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    # Block size for processing
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    jacobi_1d_kernel[(num_blocks,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )