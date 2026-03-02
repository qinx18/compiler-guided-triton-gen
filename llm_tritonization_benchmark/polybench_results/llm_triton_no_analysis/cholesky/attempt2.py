import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A, N, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Process j < i case
    for j in range(i):
        # Compute A[i][j] -= sum(A[i][k] * A[j][k]) for k < j
        sum_val = 0.0
        k_offsets = tl.arange(0, BLOCK_SIZE)
        
        for k_start in range(0, j, BLOCK_SIZE):
            current_k = k_start + k_offsets
            k_mask = current_k < j
            
            if tl.sum(k_mask.to(tl.int32)) > 0:
                a_i_k = tl.load(A + i * N + current_k, mask=k_mask, other=0.0)
                a_j_k = tl.load(A + j * N + current_k, mask=k_mask, other=0.0)
                prod = a_i_k * a_j_k
                sum_val += tl.sum(tl.where(k_mask, prod, 0.0))
        
        # Update A[i][j]
        a_ij = tl.load(A + i * N + j)
        a_ij = a_ij - sum_val
        
        # A[i][j] /= A[j][j]
        a_jj = tl.load(A + j * N + j)
        a_ij = a_ij / a_jj
        tl.store(A + i * N + j, a_ij)
    
    # Process i == j case: A[i][i] -= sum(A[i][k] * A[i][k]) for k < i
    sum_val = 0.0
    k_offsets = tl.arange(0, BLOCK_SIZE)
    
    for k_start in range(0, i, BLOCK_SIZE):
        current_k = k_start + k_offsets
        k_mask = current_k < i
        
        if tl.sum(k_mask.to(tl.int32)) > 0:
            a_i_k = tl.load(A + i * N + current_k, mask=k_mask, other=0.0)
            sq = a_i_k * a_i_k
            sum_val += tl.sum(tl.where(k_mask, sq, 0.0))
    
    # Update A[i][i]
    a_ii = tl.load(A + i * N + i)
    a_ii = a_ii - sum_val
    
    # A[i][i] = sqrt(A[i][i])
    a_ii = tl.sqrt(a_ii)
    tl.store(A + i * N + i, a_ii)

def cholesky_triton(A, N):
    BLOCK_SIZE = 64
    grid = (N,)
    cholesky_kernel[grid](A, N, BLOCK_SIZE)