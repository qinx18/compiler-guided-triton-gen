import torch
import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A, B, N: tl.constexpr, TSTEPS: tl.constexpr):
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        for i in range(1, N - 1):
            a_prev = tl.load(A + i - 1)
            a_curr = tl.load(A + i)
            a_next = tl.load(A + i + 1)
            b_val = 0.33333 * (a_prev + a_curr + a_next)
            tl.store(B + i, b_val)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        for i in range(1, N - 1):
            b_prev = tl.load(B + i - 1)
            b_curr = tl.load(B + i)
            b_next = tl.load(B + i + 1)
            a_val = 0.33333 * (b_prev + b_curr + b_next)
            tl.store(A + i, a_val)

def jacobi_1d_triton(A, B, N, TSTEPS):
    grid = (1,)
    
    jacobi_1d_kernel[grid](
        A, B, 
        N=N, 
        TSTEPS=TSTEPS
    )