import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles one row
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Process j < i case
    for j in range(i):
        # Inner k loop: A[i][j] -= A[i][k] * A[j][k] for k < j
        k_offsets = tl.arange(0, BLOCK_SIZE)
        for k_start in range(0, j, BLOCK_SIZE):
            current_k = k_start + k_offsets
            k_mask = current_k < j
            
            # Load A[i][k] and A[j][k]
            a_i_k = tl.load(A + i * N + current_k, mask=k_mask, other=0.0)
            a_j_k = tl.load(A + j * N + current_k, mask=k_mask, other=0.0)
            
            # Compute reduction
            prod = a_i_k * a_j_k
            reduction = tl.sum(tl.where(k_mask, prod, 0.0))
            
            # Update A[i][j]
            old_val = tl.load(A + i * N + j)
            tl.store(A + i * N + j, old_val - reduction)
        
        # A[i][j] /= A[j][j]
        a_ij = tl.load(A + i * N + j)
        a_jj = tl.load(A + j * N + j)
        tl.store(A + i * N + j, a_ij / a_jj)
    
    # Process i == j case: A[i][i] -= A[i][k] * A[i][k] for k < i
    k_offsets = tl.arange(0, BLOCK_SIZE)
    for k_start in range(0, i, BLOCK_SIZE):
        current_k = k_start + k_offsets
        k_mask = current_k < i
        
        # Load A[i][k]
        a_i_k = tl.load(A + i * N + current_k, mask=k_mask, other=0.0)
        
        # Compute reduction
        sq = a_i_k * a_i_k
        reduction = tl.sum(tl.where(k_mask, sq, 0.0))
        
        # Update A[i][i]
        old_val = tl.load(A + i * N + i)
        tl.store(A + i * N + i, old_val - reduction)
    
    # A[i][i] = sqrt(A[i][i])
    a_ii = tl.load(A + i * N + i)
    tl.store(A + i * N + i, tl.sqrt(a_ii))

def cholesky_triton(A, N):
    BLOCK_SIZE = 64
    grid = (N,)
    cholesky_kernel[grid](A, N, BLOCK_SIZE)