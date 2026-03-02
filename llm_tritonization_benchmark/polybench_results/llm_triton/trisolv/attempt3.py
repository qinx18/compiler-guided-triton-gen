import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N, L_stride, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
    
    # x[row] = b[row]
    b_val = tl.load(b_ptr + row)
    x_val = b_val
    
    # Process columns in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, row, BLOCK_SIZE):
        current_offsets = block_start + col_offsets
        mask = current_offsets < row
        
        # Load L[row][j] for current block
        L_ptrs = L_ptr + row * L_stride + current_offsets
        L_vals = tl.load(L_ptrs, mask=mask, other=0.0)
        
        # Load x[j] for current block
        x_ptrs = x_ptr + current_offsets
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute partial sum: L[row][j] * x[j]
        products = L_vals * x_vals
        x_val -= tl.sum(products)
    
    # x[row] = x[row] / L[row][row]
    L_diag = tl.load(L_ptr + row * L_stride + row)
    x_val = x_val / L_diag
    
    tl.store(x_ptr + row, x_val)

def trisolv_triton(L, b, x, N):
    BLOCK_SIZE = 64
    grid = (N,)
    
    for i in range(N):
        trisolv_kernel[grid](L, b, x, N, L.stride(0), BLOCK_SIZE)