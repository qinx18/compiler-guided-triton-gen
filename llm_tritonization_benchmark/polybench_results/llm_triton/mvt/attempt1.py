import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(A_ptr, x1_ptr, x2_ptr, y_1_ptr, y_2_ptr, N, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_id * BLOCK_SIZE + offsets
    i_mask = i_offsets < N
    
    # First loop: x1[i] = x1[i] + A[i][j] * y_1[j]
    x1_vals = tl.load(x1_ptr + i_offsets, mask=i_mask, other=0.0)
    
    for j_start in range(0, N, BLOCK_SIZE):
        j_offsets = j_start + offsets
        j_mask = j_offsets < N
        
        y_1_vals = tl.load(y_1_ptr + j_offsets, mask=j_mask, other=0.0)
        
        for j_idx in range(BLOCK_SIZE):
            if j_start + j_idx < N:
                j = j_start + j_idx
                y_1_val = tl.load(y_1_ptr + j)
                
                A_offsets = i_offsets * N + j
                A_vals = tl.load(A_ptr + A_offsets, mask=i_mask, other=0.0)
                x1_vals = x1_vals + A_vals * y_1_val
    
    tl.store(x1_ptr + i_offsets, x1_vals, mask=i_mask)
    
    # Second loop: x2[i] = x2[i] + A[j][i] * y_2[j]
    x2_vals = tl.load(x2_ptr + i_offsets, mask=i_mask, other=0.0)
    
    for j_start in range(0, N, BLOCK_SIZE):
        j_offsets = j_start + offsets
        j_mask = j_offsets < N
        
        y_2_vals = tl.load(y_2_ptr + j_offsets, mask=j_mask, other=0.0)
        
        for j_idx in range(BLOCK_SIZE):
            if j_start + j_idx < N:
                j = j_start + j_idx
                y_2_val = tl.load(y_2_ptr + j)
                
                A_offsets = j * N + i_offsets
                A_vals = tl.load(A_ptr + A_offsets, mask=i_mask, other=0.0)
                x2_vals = x2_vals + A_vals * y_2_val
    
    tl.store(x2_ptr + i_offsets, x2_vals, mask=i_mask)

def mvt_triton(A, x1, x2, y_1, y_2, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    mvt_kernel[grid](A, x1, x2, y_1, y_2, N, BLOCK_SIZE)