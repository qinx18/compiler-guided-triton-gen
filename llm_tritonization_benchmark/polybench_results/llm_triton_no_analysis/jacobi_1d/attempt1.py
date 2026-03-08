import torch
import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A, B, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        for block_start in range(1, N - 1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N - 1)
            
            a_prev = tl.load(A + current_offsets - 1, mask=mask)
            a_curr = tl.load(A + current_offsets, mask=mask)
            a_next = tl.load(A + current_offsets + 1, mask=mask)
            
            b_val = 0.33333 * (a_prev + a_curr + a_next)
            tl.store(B + current_offsets, b_val, mask=mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        for block_start in range(1, N - 1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N - 1)
            
            b_prev = tl.load(B + current_offsets - 1, mask=mask)
            b_curr = tl.load(B + current_offsets, mask=mask)
            b_next = tl.load(B + current_offsets + 1, mask=mask)
            
            a_val = 0.33333 * (b_prev + b_curr + b_next)
            tl.store(A + current_offsets, a_val, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    
    jacobi_1d_kernel[(1,)](
        A, B, 
        N=N, 
        TSTEPS=TSTEPS,
        BLOCK_SIZE=BLOCK_SIZE
    )