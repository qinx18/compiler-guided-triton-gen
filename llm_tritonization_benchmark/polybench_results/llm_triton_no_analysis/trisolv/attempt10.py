import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # x[i] = b[i]
    x_i = tl.load(b_ptr + i)
    
    # Vectorized computation of sum(L[i][j] * x[j]) for j < i
    if i > 0:
        j_offsets = tl.arange(0, BLOCK_SIZE)
        sum_val = 0.0
        
        for j_start in range(0, i, BLOCK_SIZE):
            current_j_offsets = j_start + j_offsets
            mask = current_j_offsets < i
            
            # Load L[i][j] values
            L_indices = i * N + current_j_offsets
            L_vals = tl.load(L_ptr + L_indices, mask=mask, other=0.0)
            
            # Load x[j] values  
            x_vals = tl.load(x_ptr + current_j_offsets, mask=mask, other=0.0)
            
            # Compute products and accumulate
            products = L_vals * x_vals
            sum_val = sum_val + tl.sum(products)
        
        x_i = x_i - sum_val
    
    # x[i] = x[i] / L[i][i]
    L_ii = tl.load(L_ptr + i * N + i)
    x_i = x_i / L_ii
    
    tl.store(x_ptr + i, x_i)

def trisolv_triton(L, b, x, N):
    x.copy_(b)
    
    BLOCK_SIZE = 64
    grid = (N,)
    
    trisolv_kernel[grid](
        L, b, x, N, BLOCK_SIZE,
        num_warps=4
    )