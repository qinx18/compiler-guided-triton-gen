import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= N:
        return
    
    # x[i] = b[i]
    x_i = tl.load(b_ptr + i)
    
    # Vectorized processing of j from 0 to i-1 in blocks
    for j_start in range(0, i, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < i
        
        # Load L[i][j] values
        L_indices = i * N + j_offsets
        L_vals = tl.load(L_ptr + L_indices, mask=mask, other=0.0)
        
        # Load x[j] values
        x_vals = tl.load(x_ptr + j_offsets, mask=mask, other=0.0)
        
        # Compute products and sum
        products = L_vals * x_vals
        x_i = x_i - tl.sum(products)
    
    # x[i] = x[i] / L[i][i]
    L_ii = tl.load(L_ptr + i * N + i)
    x_i = x_i / L_ii
    
    tl.store(x_ptr + i, x_i)

def trisolv_triton(L, b, x, N):
    x.copy_(b)
    
    BLOCK_SIZE = 32
    grid = (N,)
    
    for i in range(N):
        trisolv_kernel[grid](
            L, b, x, N, BLOCK_SIZE,
            num_warps=1
        )