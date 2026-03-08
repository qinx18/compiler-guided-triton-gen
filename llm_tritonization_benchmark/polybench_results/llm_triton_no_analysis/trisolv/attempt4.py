import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, row_to_process: tl.constexpr, N: tl.constexpr, stride_L: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Initialize x[row] = b[row]
    b_val = tl.load(b_ptr + row_to_process)
    x_val = b_val
    
    # Compute sum of L[row][j] * x[j] for j < row using vectorized loads
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, row_to_process, BLOCK_SIZE):
        current_cols = block_start + col_offsets
        mask = current_cols < row_to_process
        
        # Load L[row][j] values
        L_ptrs = L_ptr + row_to_process * stride_L + current_cols
        L_vals = tl.load(L_ptrs, mask=mask, other=0.0)
        
        # Load x[j] values
        x_ptrs = x_ptr + current_cols
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute partial sum
        products = L_vals * x_vals
        partial_sum = tl.sum(products)
        x_val = x_val - partial_sum
    
    # Divide by diagonal element L[row][row]
    L_diag = tl.load(L_ptr + row_to_process * stride_L + row_to_process)
    x_val = x_val / L_diag
    
    # Store result
    tl.store(x_ptr + row_to_process, x_val)

def trisolv_triton(L, b, x, N):
    BLOCK_SIZE = 64
    
    # Process each row sequentially to maintain dependency
    for i in range(N):
        # Launch kernel for current row
        grid = (1,)
        trisolv_kernel[grid](
            L, b, x,
            row_to_process=i,
            N=N,
            stride_L=L.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
        )