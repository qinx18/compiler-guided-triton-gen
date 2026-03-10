import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles one row i
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Process j < i
    for j in range(i):
        # Compute A[i][j] -= sum(A[i][k] * A[j][k]) for k < j
        sum_val = 0.0
        
        # Process k values in blocks
        k_offsets = tl.arange(0, BLOCK_SIZE)
        for k_block in range(0, j, BLOCK_SIZE):
            k_indices = k_block + k_offsets
            k_mask = (k_indices < j)
            
            # Load A[i][k_indices] and A[j][k_indices]
            i_k_ptrs = A_ptr + i * N + k_indices
            j_k_ptrs = A_ptr + j * N + k_indices
            
            a_i_k = tl.load(i_k_ptrs, mask=k_mask, other=0.0)
            a_j_k = tl.load(j_k_ptrs, mask=k_mask, other=0.0)
            
            # Compute products and sum
            products = a_i_k * a_j_k
            sum_val += tl.sum(products)
        
        # Update A[i][j]
        a_ij_ptr = A_ptr + i * N + j
        a_ij = tl.load(a_ij_ptr)
        a_ij = a_ij - sum_val
        
        # Divide by A[j][j]
        a_jj_ptr = A_ptr + j * N + j
        a_jj = tl.load(a_jj_ptr)
        a_ij = a_ij / a_jj
        
        tl.store(a_ij_ptr, a_ij)
    
    # Process i == j case
    sum_val = 0.0
    
    # Process k values in blocks for diagonal element
    k_offsets = tl.arange(0, BLOCK_SIZE)
    for k_block in range(0, i, BLOCK_SIZE):
        k_indices = k_block + k_offsets
        k_mask = (k_indices < i)
        
        # Load A[i][k_indices]
        i_k_ptrs = A_ptr + i * N + k_indices
        a_i_k = tl.load(i_k_ptrs, mask=k_mask, other=0.0)
        
        # Compute squares and sum
        squares = a_i_k * a_i_k
        sum_val += tl.sum(squares)
    
    # Update A[i][i]
    a_ii_ptr = A_ptr + i * N + i
    a_ii = tl.load(a_ii_ptr)
    a_ii = a_ii - sum_val
    a_ii = tl.sqrt(a_ii)
    
    tl.store(a_ii_ptr, a_ii)

def cholesky_triton(A, N):
    assert A.is_contiguous()
    assert A.dtype == torch.float32
    
    BLOCK_SIZE = 64
    grid = (N,)
    
    cholesky_kernel[grid](A, N, BLOCK_SIZE)