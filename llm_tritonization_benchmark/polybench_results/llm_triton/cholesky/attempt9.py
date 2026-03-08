import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr):
    for i in range(N):
        # Process j < i
        for j in range(i):
            # Compute sum for k < j
            for k in range(j):
                a_ik = tl.load(A_ptr + i * N + k)
                a_jk = tl.load(A_ptr + j * N + k)
                a_ij = tl.load(A_ptr + i * N + j)
                a_ij = a_ij - a_ik * a_jk
                tl.store(A_ptr + i * N + j, a_ij)
            
            # Divide by diagonal element
            a_ij = tl.load(A_ptr + i * N + j)
            a_jj = tl.load(A_ptr + j * N + j)
            a_ij = a_ij / a_jj
            tl.store(A_ptr + i * N + j, a_ij)
        
        # Diagonal case: compute A[i][i]
        for k in range(i):
            a_ik = tl.load(A_ptr + i * N + k)
            a_ii = tl.load(A_ptr + i * N + i)
            a_ii = a_ii - a_ik * a_ik
            tl.store(A_ptr + i * N + i, a_ii)
        
        # Take square root of diagonal element
        a_ii = tl.load(A_ptr + i * N + i)
        a_ii = tl.sqrt(a_ii)
        tl.store(A_ptr + i * N + i, a_ii)

def cholesky_triton(A, N):
    grid = (1,)
    cholesky_kernel[grid](A, N)