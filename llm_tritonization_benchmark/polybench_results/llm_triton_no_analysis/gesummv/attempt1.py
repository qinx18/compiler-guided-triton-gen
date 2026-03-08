import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(A_ptr, B_ptr, tmp_ptr, x_ptr, y_ptr, alpha, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = block_start + offsets
    row_mask = row_offsets < N
    
    # Initialize tmp and y arrays
    tl.store(tmp_ptr + row_offsets, 0.0, mask=row_mask)
    tl.store(y_ptr + row_offsets, 0.0, mask=row_mask)
    
    # Inner loop over j
    for j in range(N):
        x_val = tl.load(x_ptr + j)
        
        # Load A[i][j] and B[i][j] for current rows
        a_offsets = row_offsets * N + j
        b_offsets = row_offsets * N + j
        
        a_vals = tl.load(A_ptr + a_offsets, mask=row_mask)
        b_vals = tl.load(B_ptr + b_offsets, mask=row_mask)
        
        # Load current tmp[i] and y[i] values
        tmp_vals = tl.load(tmp_ptr + row_offsets, mask=row_mask)
        y_vals = tl.load(y_ptr + row_offsets, mask=row_mask)
        
        # Update tmp[i] and y[i]
        new_tmp = a_vals * x_val + tmp_vals
        new_y = b_vals * x_val + y_vals
        
        tl.store(tmp_ptr + row_offsets, new_tmp, mask=row_mask)
        tl.store(y_ptr + row_offsets, new_y, mask=row_mask)
    
    # Final computation: y[i] = alpha * tmp[i] + beta * y[i]
    final_tmp = tl.load(tmp_ptr + row_offsets, mask=row_mask)
    final_y = tl.load(y_ptr + row_offsets, mask=row_mask)
    result_y = alpha * final_tmp + beta * final_y
    tl.store(y_ptr + row_offsets, result_y, mask=row_mask)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gesummv_kernel[grid](
        A, B, tmp, x, y, alpha, beta, N, BLOCK_SIZE
    )