import torch
import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the starting position for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (1 <= idx < N-1)
    mask = (offsets >= 1) & (offsets < N - 1)
    
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        a_prev = tl.load(A_ptr + offsets - 1, mask=mask)
        a_curr = tl.load(A_ptr + offsets, mask=mask)
        a_next = tl.load(A_ptr + offsets + 1, mask=mask)
        b_val = 0.33333 * (a_prev + a_curr + a_next)
        tl.store(B_ptr + offsets, b_val, mask=mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        b_prev = tl.load(B_ptr + offsets - 1, mask=mask)
        b_curr = tl.load(B_ptr + offsets, mask=mask)
        b_next = tl.load(B_ptr + offsets + 1, mask=mask)
        a_val = 0.33333 * (b_prev + b_curr + b_next)
        tl.store(A_ptr + offsets, a_val, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    jacobi_1d_kernel[(grid_size,)](A, B, N, TSTEPS, BLOCK_SIZE)