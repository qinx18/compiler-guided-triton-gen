import triton
import triton.language as tl
import torch

@triton.jit
def lu_kernel(A, N: tl.constexpr):
    for i in range(N):
        # First nested loop: j < i
        for j in range(i):
            # Inner k loop: k < j
            for k in range(j):
                A[i * N + j] -= A[i * N + k] * A[k * N + j]
            # Division step
            A[i * N + j] /= A[j * N + j]
        
        # Second nested loop: j >= i
        for j in range(i, N):
            # Inner k loop: k < i
            for k in range(i):
                A[i * N + j] -= A[i * N + k] * A[k * N + j]

def lu_triton(A, N):
    grid = (1,)
    lu_kernel[grid](A, N)