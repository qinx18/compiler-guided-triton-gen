import torch
import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (1 to N-2)
    mask = (offsets >= 1) & (offsets < N - 1)
    
    for t in range(TSTEPS):
        # First phase: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        # Load A values
        A_left = tl.load(A_ptr + offsets - 1, mask=mask)
        A_center = tl.load(A_ptr + offsets, mask=mask)
        A_right = tl.load(A_ptr + offsets + 1, mask=mask)
        
        # Compute B values
        B_new = 0.33333 * (A_left + A_center + A_right)
        
        # Store B values
        tl.store(B_ptr + offsets, B_new, mask=mask)
        
        # Synchronize to ensure all threads complete phase 1 before phase 2
        tl.debug_barrier()
        
        # Second phase: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        # Load B values
        B_left = tl.load(B_ptr + offsets - 1, mask=mask)
        B_center = tl.load(B_ptr + offsets, mask=mask)
        B_right = tl.load(B_ptr + offsets + 1, mask=mask)
        
        # Compute A values
        A_new = 0.33333 * (B_left + B_center + B_right)
        
        # Store A values
        tl.store(A_ptr + offsets, A_new, mask=mask)
        
        # Synchronize to ensure all threads complete phase 2 before next iteration
        tl.debug_barrier()

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    jacobi_1d_kernel[grid](A, B, N, TSTEPS, BLOCK_SIZE)