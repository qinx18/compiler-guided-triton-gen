import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
    
    # Process column j < row
    for j in range(row):
        # Vectorized reduction for k < j
        if j > 0:
            k_offsets = tl.arange(0, BLOCK_SIZE)
            sum_val = 0.0
            
            for k_start in range(0, j, BLOCK_SIZE):
                current_k = k_start + k_offsets
                k_mask = (current_k < j)
                
                a_ik_ptr = A_ptr + row * N + current_k
                a_jk_ptr = A_ptr + j * N + current_k
                
                a_ik = tl.load(a_ik_ptr, mask=k_mask, other=0.0)
                a_jk = tl.load(a_jk_ptr, mask=k_mask, other=0.0)
                
                products = a_ik * a_jk
                sum_val += tl.sum(tl.where(k_mask, products, 0.0))
            
            a_ij = tl.load(A_ptr + row * N + j)
            a_ij -= sum_val
            tl.store(A_ptr + row * N + j, a_ij)
        
        # A[row][j] /= A[j][j]
        a_ij = tl.load(A_ptr + row * N + j)
        a_jj = tl.load(A_ptr + j * N + j)
        a_ij /= a_jj
        tl.store(A_ptr + row * N + j, a_ij)
    
    # Diagonal case: A[row][row]
    if row > 0:
        k_offsets = tl.arange(0, BLOCK_SIZE)
        sum_val = 0.0
        
        for k_start in range(0, row, BLOCK_SIZE):
            current_k = k_start + k_offsets
            k_mask = (current_k < row)
            
            a_ik_ptr = A_ptr + row * N + current_k
            a_ik = tl.load(a_ik_ptr, mask=k_mask, other=0.0)
            
            squares = a_ik * a_ik
            sum_val += tl.sum(tl.where(k_mask, squares, 0.0))
        
        a_ii = tl.load(A_ptr + row * N + row)
        a_ii -= sum_val
        tl.store(A_ptr + row * N + row, a_ii)
    
    # A[row][row] = sqrt(A[row][row])
    a_ii = tl.load(A_ptr + row * N + row)
    a_ii = tl.sqrt(a_ii)
    tl.store(A_ptr + row * N + row, a_ii)

def cholesky_triton(A, N):
    BLOCK_SIZE = 32
    grid = (N,)
    cholesky_kernel[grid](A, N, BLOCK_SIZE)