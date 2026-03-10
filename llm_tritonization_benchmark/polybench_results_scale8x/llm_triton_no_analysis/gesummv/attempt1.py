import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(A_ptr, B_ptr, tmp_ptr, x_ptr, y_ptr, alpha, beta, 
                   N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = block_start + offsets
    row_mask = row_offsets < N
    
    # Initialize tmp and y to 0
    tmp_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    y_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Inner loop over j
    col_offsets = tl.arange(0, N)
    for j in range(N):
        # Load x[j] (scalar broadcast)
        x_j = tl.load(x_ptr + j)
        
        # Load A[i][j] for all rows in block
        A_offsets = row_offsets[:, None] * N + j
        A_vals = tl.load(A_ptr + A_offsets, mask=row_mask[:, None], other=0.0)
        A_vals = tl.reshape(A_vals, (BLOCK_SIZE,))
        
        # Load B[i][j] for all rows in block  
        B_offsets = row_offsets[:, None] * N + j
        B_vals = tl.load(B_ptr + B_offsets, mask=row_mask[:, None], other=0.0)
        B_vals = tl.reshape(B_vals, (BLOCK_SIZE,))
        
        # Accumulate
        tmp_vals = tmp_vals + A_vals * x_j
        y_vals = y_vals + B_vals * x_j
    
    # Final computation: y[i] = alpha * tmp[i] + beta * y[i]
    final_y = alpha * tmp_vals + beta * y_vals
    
    # Store results
    tl.store(tmp_ptr + row_offsets, tmp_vals, mask=row_mask)
    tl.store(y_ptr + row_offsets, final_y, mask=row_mask)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gesummv_kernel[grid](
        A, B, tmp, x, y, alpha, beta, N, BLOCK_SIZE
    )