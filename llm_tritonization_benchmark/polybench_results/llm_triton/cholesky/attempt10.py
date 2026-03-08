import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        for i in range(N):
            # j < i case
            for j in range(i):
                # Vectorized k < j operations
                if j > 0:
                    k_offsets = tl.arange(0, BLOCK_SIZE)
                    for k_start in range(0, j, BLOCK_SIZE):
                        k_indices = k_start + k_offsets
                        k_mask = k_indices < j
                        
                        a_ik_vals = tl.load(A_ptr + i * N + k_indices, mask=k_mask, other=0.0)
                        a_jk_vals = tl.load(A_ptr + j * N + k_indices, mask=k_mask, other=0.0)
                        
                        products = a_ik_vals * a_jk_vals
                        sum_products = tl.sum(products, axis=0)
                        
                        current_val = tl.load(A_ptr + i * N + j)
                        new_val = current_val - sum_products
                        tl.store(A_ptr + i * N + j, new_val)
                
                # Divide by diagonal
                a_ij = tl.load(A_ptr + i * N + j)
                a_jj = tl.load(A_ptr + j * N + j)
                result = a_ij / a_jj
                tl.store(A_ptr + i * N + j, result)
            
            # Diagonal case i == j
            if i > 0:
                k_offsets = tl.arange(0, BLOCK_SIZE)
                diagonal_val = tl.load(A_ptr + i * N + i)
                
                for k_start in range(0, i, BLOCK_SIZE):
                    k_indices = k_start + k_offsets
                    k_mask = k_indices < i
                    
                    a_ik_vals = tl.load(A_ptr + i * N + k_indices, mask=k_mask, other=0.0)
                    squares = a_ik_vals * a_ik_vals
                    sum_squares = tl.sum(squares, axis=0)
                    
                    diagonal_val = diagonal_val - sum_squares
                
                tl.store(A_ptr + i * N + i, diagonal_val)
            
            # Square root of diagonal
            diagonal_val = tl.load(A_ptr + i * N + i)
            sqrt_val = tl.sqrt(diagonal_val)
            tl.store(A_ptr + i * N + i, sqrt_val)

def cholesky_triton(A, N):
    BLOCK_SIZE = 32
    grid = (1,)
    cholesky_kernel[grid](A, N, BLOCK_SIZE)