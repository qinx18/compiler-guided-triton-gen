import torch
import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A, B, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < N - 1)
    
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        a_prev = tl.load(A + offsets - 1, mask=mask)
        a_curr = tl.load(A + offsets, mask=mask)
        a_next = tl.load(A + offsets + 1, mask=mask)
        
        b_val = 0.33333 * (a_prev + a_curr + a_next)
        tl.store(B + offsets, b_val, mask=mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        b_prev = tl.load(B + offsets - 1, mask=mask)
        b_curr = tl.load(B + offsets, mask=mask)
        b_next = tl.load(B + offsets + 1, mask=mask)
        
        a_val = 0.33333 * (b_prev + b_curr + b_next)
        tl.store(A + offsets, a_val, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N - 2, BLOCK_SIZE),)
    
    jacobi_1d_kernel[grid](
        A, B, 
        N=N, 
        TSTEPS=TSTEPS,
        BLOCK_SIZE=BLOCK_SIZE
    )