import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N, i, BLOCK_SIZE: tl.constexpr):
    # Process j < i
    for j in range(i):
        # Compute A[i][j] -= A[i][k] * A[j][k] for k < j
        for k in range(j):
            i_k_idx = i * N + k
            j_k_idx = j * N + k
            a_i_k = tl.load(A_ptr + i_k_idx)
            a_j_k = tl.load(A_ptr + j_k_idx)
            
            i_j_idx = i * N + j
            a_i_j = tl.load(A_ptr + i_j_idx)
            a_i_j -= a_i_k * a_j_k
            tl.store(A_ptr + i_j_idx, a_i_j)
        
        # A[i][j] /= A[j][j]
        i_j_idx = i * N + j
        j_j_idx = j * N + j
        a_i_j = tl.load(A_ptr + i_j_idx)
        a_j_j = tl.load(A_ptr + j_j_idx)
        a_i_j /= a_j_j
        tl.store(A_ptr + i_j_idx, a_i_j)
    
    # Process diagonal case (i == j)
    i_i_idx = i * N + i
    a_i_i = tl.load(A_ptr + i_i_idx)
    
    for k in range(i):
        i_k_idx = i * N + k
        a_i_k = tl.load(A_ptr + i_k_idx)
        a_i_i -= a_i_k * a_i_k
    
    a_i_i = tl.sqrt(a_i_i)
    tl.store(A_ptr + i_i_idx, a_i_i)

def cholesky_triton(A, N):
    # Launch kernel for each i value sequentially
    for i in range(N):
        grid = (1,)
        cholesky_kernel[grid](
            A, N, i, BLOCK_SIZE=32
        )