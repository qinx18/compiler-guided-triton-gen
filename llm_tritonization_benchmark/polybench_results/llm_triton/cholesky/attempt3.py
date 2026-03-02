import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel_j(A_ptr, N, i, BLOCK_SIZE: tl.constexpr):
    j = tl.program_id(0)
    
    if j >= i:
        return
    
    # Compute A[i][j] -= A[i][k] * A[j][k] for k < j
    k_offsets = tl.arange(0, BLOCK_SIZE)
    
    for k_start in range(0, j, BLOCK_SIZE):
        current_k_offsets = k_start + k_offsets
        k_mask = current_k_offsets < j
        
        i_k_indices = i * N + current_k_offsets
        j_k_indices = j * N + current_k_offsets
        
        a_i_k = tl.load(A_ptr + i_k_indices, mask=k_mask, other=0.0)
        a_j_k = tl.load(A_ptr + j_k_indices, mask=k_mask, other=0.0)
        
        products = a_i_k * a_j_k
        reduction = tl.sum(products)
        
        i_j_idx = i * N + j
        a_i_j = tl.load(A_ptr + i_j_idx)
        a_i_j -= reduction
        tl.store(A_ptr + i_j_idx, a_i_j)
    
    # A[i][j] /= A[j][j]
    i_j_idx = i * N + j
    j_j_idx = j * N + j
    a_i_j = tl.load(A_ptr + i_j_idx)
    a_j_j = tl.load(A_ptr + j_j_idx)
    a_i_j /= a_j_j
    tl.store(A_ptr + i_j_idx, a_i_j)

@triton.jit
def cholesky_kernel_diag(A_ptr, N, i, BLOCK_SIZE: tl.constexpr):
    # Process diagonal case (i == i)
    i_i_idx = i * N + i
    a_i_i = tl.load(A_ptr + i_i_idx)
    
    # Compute A[i][i] -= A[i][k] * A[i][k] for k < i
    k_offsets = tl.arange(0, BLOCK_SIZE)
    
    for k_start in range(0, i, BLOCK_SIZE):
        current_k_offsets = k_start + k_offsets
        k_mask = current_k_offsets < i
        
        i_k_indices = i * N + current_k_offsets
        a_i_k = tl.load(A_ptr + i_k_indices, mask=k_mask, other=0.0)
        
        squares = a_i_k * a_i_k
        reduction = tl.sum(squares)
        
        a_i_i -= reduction
    
    a_i_i = tl.sqrt(a_i_i)
    tl.store(A_ptr + i_i_idx, a_i_i)

def cholesky_triton(A, N):
    BLOCK_SIZE = 32
    
    for i in range(N):
        # Launch kernel for j < i with parallel j
        if i > 0:
            grid = (i,)
            cholesky_kernel_j[grid](
                A, N, i, BLOCK_SIZE=BLOCK_SIZE
            )
        
        # Launch diagonal kernel
        grid = (1,)
        cholesky_kernel_diag[grid](
            A, N, i, BLOCK_SIZE=BLOCK_SIZE
        )