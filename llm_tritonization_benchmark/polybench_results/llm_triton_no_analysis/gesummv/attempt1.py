import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(A_ptr, B_ptr, tmp_ptr, x_ptr, y_ptr, alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    i_mask = i_offsets < N
    
    # Initialize tmp and y to 0
    tmp_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    y_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Inner loop over j
    for j in range(N):
        # Load x[j]
        x_j = tl.load(x_ptr + j)
        
        # Load A[i][j] for all i in block
        a_offsets = i_offsets * N + j
        a_vals = tl.load(A_ptr + a_offsets, mask=i_mask, other=0.0)
        
        # Load B[i][j] for all i in block
        b_offsets = i_offsets * N + j
        b_vals = tl.load(B_ptr + b_offsets, mask=i_mask, other=0.0)
        
        # Update tmp and y
        tmp_vals = tmp_vals + a_vals * x_j
        y_vals = y_vals + b_vals * x_j
    
    # Final computation: y[i] = alpha * tmp[i] + beta * y[i]
    y_vals = alpha * tmp_vals + beta * y_vals
    
    # Store results
    tl.store(tmp_ptr + i_offsets, tmp_vals, mask=i_mask)
    tl.store(y_ptr + i_offsets, y_vals, mask=i_mask)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gesummv_kernel[grid](
        A, B, tmp, x, y, alpha, beta, N, BLOCK_SIZE
    )