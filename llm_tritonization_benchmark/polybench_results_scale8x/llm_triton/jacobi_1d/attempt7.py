import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get thread block info
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices (1 to N-2 inclusive)
    mask = (indices >= 1) & (indices < N - 1)
    
    for t in range(TSTEPS):
        # Synchronization barrier to ensure all threads complete before next phase
        tl.debug_barrier()
        
        # First phase: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        a_left = tl.load(A_ptr + indices - 1, mask=mask, other=0.0)
        a_center = tl.load(A_ptr + indices, mask=mask, other=0.0)
        a_right = tl.load(A_ptr + indices + 1, mask=mask, other=0.0)
        
        b_new = 0.33333 * (a_left + a_center + a_right)
        tl.store(B_ptr + indices, b_new, mask=mask)
        
        # Synchronization barrier to ensure all threads complete before next phase
        tl.debug_barrier()
        
        # Second phase: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        b_left = tl.load(B_ptr + indices - 1, mask=mask, other=0.0)
        b_center = tl.load(B_ptr + indices, mask=mask, other=0.0)
        b_right = tl.load(B_ptr + indices + 1, mask=mask, other=0.0)
        
        a_new = 0.33333 * (b_left + b_center + b_right)
        tl.store(A_ptr + indices, a_new, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel with single grid to ensure proper synchronization
    jacobi_1d_kernel[(1,)](
        A, B, N, TSTEPS, N
    )