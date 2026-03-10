import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < N - 1
    
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        A_left = tl.load(A_ptr + current_offsets - 1, mask=mask)
        A_center = tl.load(A_ptr + current_offsets, mask=mask)
        A_right = tl.load(A_ptr + current_offsets + 1, mask=mask)
        
        B_vals = 0.33333 * (A_left + A_center + A_right)
        tl.store(B_ptr + current_offsets, B_vals, mask=mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        B_left = tl.load(B_ptr + current_offsets - 1, mask=mask)
        B_center = tl.load(B_ptr + current_offsets, mask=mask)
        B_right = tl.load(B_ptr + current_offsets + 1, mask=mask)
        
        A_vals = 0.33333 * (B_left + B_center + B_right)
        tl.store(A_ptr + current_offsets, A_vals, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N - 2, BLOCK_SIZE),)
    
    jacobi_1d_kernel[grid](A, B, N, TSTEPS, BLOCK_SIZE)