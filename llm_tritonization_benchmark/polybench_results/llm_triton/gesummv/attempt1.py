import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(A, B, tmp, x, y, alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    tmp_val = 0.0
    y_val = 0.0
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, N, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < N
        
        # Load x values
        x_vals = tl.load(x + j_indices, mask=j_mask, other=0.0)
        
        # Load A[i, j] values
        A_indices = i * N + j_indices
        A_vals = tl.load(A + A_indices, mask=j_mask, other=0.0)
        
        # Load B[i, j] values
        B_indices = i * N + j_indices
        B_vals = tl.load(B + B_indices, mask=j_mask, other=0.0)
        
        # Accumulate tmp and y
        tmp_val += tl.sum(A_vals * x_vals)
        y_val += tl.sum(B_vals * x_vals)
    
    # Store tmp[i]
    tl.store(tmp + i, tmp_val)
    
    # Compute final y[i] = alpha * tmp[i] + beta * y[i]
    final_y = alpha * tmp_val + beta * y_val
    tl.store(y + i, final_y)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    BLOCK_SIZE = 32
    grid = (N,)
    
    gesummv_kernel[grid](A, B, tmp, x, y, alpha, beta, N, BLOCK_SIZE)