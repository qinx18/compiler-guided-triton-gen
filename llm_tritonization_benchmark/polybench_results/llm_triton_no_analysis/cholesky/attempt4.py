import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel_j(A, N, i, BLOCK_SIZE_J: tl.constexpr):
    j = tl.program_id(0)
    
    if j >= i:
        return
    
    # Compute A[i][j] -= sum(A[i][k] * A[j][k]) for k < j
    sum_val = 0.0
    for k in range(j):
        a_i_k = tl.load(A + i * N + k)
        a_j_k = tl.load(A + j * N + k)
        sum_val += a_i_k * a_j_k
    
    # Update A[i][j]
    a_ij = tl.load(A + i * N + j)
    a_ij = a_ij - sum_val
    
    # A[i][j] /= A[j][j]
    a_jj = tl.load(A + j * N + j)
    a_ij = a_ij / a_jj
    tl.store(A + i * N + j, a_ij)

@triton.jit
def cholesky_kernel_diag(A, N, i):
    # Process i == i case: A[i][i] -= sum(A[i][k] * A[i][k]) for k < i
    sum_val = 0.0
    for k in range(i):
        a_i_k = tl.load(A + i * N + k)
        sum_val += a_i_k * a_i_k
    
    # Update A[i][i]
    a_ii = tl.load(A + i * N + i)
    a_ii = a_ii - sum_val
    
    # A[i][i] = sqrt(A[i][i])
    a_ii = tl.sqrt(a_ii)
    tl.store(A + i * N + i, a_ii)

def cholesky_triton(A, N):
    BLOCK_SIZE_J = 64
    
    for i in range(N):
        if i > 0:
            grid = (i,)
            cholesky_kernel_j[grid](A, N, i, BLOCK_SIZE_J)
        
        cholesky_kernel_diag[(1,)](A, N, i)