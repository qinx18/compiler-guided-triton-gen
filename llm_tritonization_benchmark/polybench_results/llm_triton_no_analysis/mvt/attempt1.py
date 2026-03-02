import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(A, x1, x2, y_1, y_2, N, BLOCK_SIZE: tl.constexpr):
    # Get row index for this program
    row_idx = tl.program_id(0)
    
    if row_idx >= N:
        return
    
    # Define column offsets once
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulators
    acc1 = 0.0
    acc2 = 0.0
    
    # Process columns in blocks
    for col_start in range(0, N, BLOCK_SIZE):
        current_col_offsets = col_start + col_offsets
        col_mask = current_col_offsets < N
        
        # First loop: x1[i] += A[i][j] * y_1[j]
        a_offsets = row_idx * N + current_col_offsets
        a_vals = tl.load(A + a_offsets, mask=col_mask, other=0.0)
        y1_vals = tl.load(y_1 + current_col_offsets, mask=col_mask, other=0.0)
        acc1 += tl.sum(a_vals * y1_vals)
        
        # Second loop: x2[i] += A[j][i] * y_2[j]
        # A[j][i] means we need A at positions [j*N + i] for j in current_col_offsets
        a_transpose_offsets = current_col_offsets * N + row_idx
        a_transpose_vals = tl.load(A + a_transpose_offsets, mask=col_mask, other=0.0)
        y2_vals = tl.load(y_2 + current_col_offsets, mask=col_mask, other=0.0)
        acc2 += tl.sum(a_transpose_vals * y2_vals)
    
    # Load current values and add accumulation
    x1_current = tl.load(x1 + row_idx)
    x2_current = tl.load(x2 + row_idx)
    
    # Store results
    tl.store(x1 + row_idx, x1_current + acc1)
    tl.store(x2 + row_idx, x2_current + acc2)

def mvt_triton(A, x1, x2, y_1, y_2, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, 1),)
    
    mvt_kernel[grid](A, x1, x2, y_1, y_2, N, BLOCK_SIZE)