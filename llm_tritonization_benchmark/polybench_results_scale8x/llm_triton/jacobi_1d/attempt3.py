import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + 1
    mask = indices < N - 1
    
    for t in range(TSTEPS):
        tl.debug_barrier()
        
        # First phase: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        A_left = tl.load(A_ptr + indices - 1, mask=mask)
        A_center = tl.load(A_ptr + indices, mask=mask)
        A_right = tl.load(A_ptr + indices + 1, mask=mask)
        B_new = 0.33333 * (A_left + A_center + A_right)
        tl.store(B_ptr + indices, B_new, mask=mask)
        
        tl.debug_barrier()
        
        # Second phase: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        B_left = tl.load(B_ptr + indices - 1, mask=mask)
        B_center = tl.load(B_ptr + indices, mask=mask)
        B_right = tl.load(B_ptr + indices + 1, mask=mask)
        A_new = 0.33333 * (B_left + B_center + B_right)
        tl.store(A_ptr + indices, A_new, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 64
    num_elements = N - 2
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    jacobi_1d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )