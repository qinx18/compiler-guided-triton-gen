import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each block processes a different row i
    i = pid
    if i >= N:
        return
    
    # Process j < i case
    for j in range(i):
        # Vectorized k loop: A[i][j] -= A[i][k] * A[j][k]
        k_offsets = tl.arange(0, BLOCK_SIZE)
        
        for k_start in range(0, j, BLOCK_SIZE):
            current_k_offsets = k_start + k_offsets
            k_mask = (current_k_offsets < j)
            
            i_k_offsets = i * N + current_k_offsets
            j_k_offsets = j * N + current_k_offsets
            
            a_i_k = tl.load(A_ptr + i_k_offsets, mask=k_mask, other=0.0)
            a_j_k = tl.load(A_ptr + j_k_offsets, mask=k_mask, other=0.0)
            
            products = a_i_k * a_j_k
            products = tl.where(k_mask, products, 0.0)
            
            # Reduce sum for this block
            reduction = tl.sum(products)
            
            # Update A[i][j] atomically
            if k_start == 0:
                i_j_offset = i * N + j
                a_i_j = tl.load(A_ptr + i_j_offset)
                tl.store(A_ptr + i_j_offset, a_i_j - reduction)
            else:
                i_j_offset = i * N + j
                a_i_j = tl.load(A_ptr + i_j_offset)
                tl.store(A_ptr + i_j_offset, a_i_j - reduction)
        
        # A[i][j] /= A[j][j]
        i_j_offset = i * N + j
        j_j_offset = j * N + j
        
        a_i_j = tl.load(A_ptr + i_j_offset)
        a_j_j = tl.load(A_ptr + j_j_offset)
        
        a_i_j = a_i_j / a_j_j
        tl.store(A_ptr + i_j_offset, a_i_j)
    
    # i == j case: A[i][i] -= A[i][k] * A[i][k]
    k_offsets = tl.arange(0, BLOCK_SIZE)
    diagonal_sum = 0.0
    
    for k_start in range(0, i, BLOCK_SIZE):
        current_k_offsets = k_start + k_offsets
        k_mask = (current_k_offsets < i)
        
        i_k_offsets = i * N + current_k_offsets
        
        a_i_k = tl.load(A_ptr + i_k_offsets, mask=k_mask, other=0.0)
        
        squares = a_i_k * a_i_k
        squares = tl.where(k_mask, squares, 0.0)
        
        diagonal_sum += tl.sum(squares)
    
    # A[i][i] = sqrt(A[i][i] - sum)
    i_i_offset = i * N + i
    a_i_i = tl.load(A_ptr + i_i_offset)
    a_i_i = tl.sqrt(a_i_i - diagonal_sum)
    tl.store(A_ptr + i_i_offset, a_i_i)

def cholesky_triton(A, N):
    BLOCK_SIZE = 64
    grid = (N,)
    cholesky_kernel[grid](A, N, BLOCK_SIZE)