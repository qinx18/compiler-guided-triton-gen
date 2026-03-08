import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(A_ptr, x1_ptr, x2_ptr, y_1_ptr, y_2_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= N:
        return
    
    # First loop: x1[i] = x1[i] + A[i][j] * y_1[j]
    j_offsets = tl.arange(0, BLOCK_SIZE)
    x1_acc = tl.load(x1_ptr + i)
    
    for j_block in range(0, N, BLOCK_SIZE):
        j_idx = j_block + j_offsets
        mask = j_idx < N
        
        A_offsets = i * N + j_idx
        A_vals = tl.load(A_ptr + A_offsets, mask=mask, other=0.0)
        y_1_vals = tl.load(y_1_ptr + j_idx, mask=mask, other=0.0)
        
        x1_acc += tl.sum(A_vals * y_1_vals)
    
    tl.store(x1_ptr + i, x1_acc)
    
    # Second loop: x2[i] = x2[i] + A[j][i] * y_2[j]
    x2_acc = tl.load(x2_ptr + i)
    
    for j_block in range(0, N, BLOCK_SIZE):
        j_idx = j_block + j_offsets
        mask = j_idx < N
        
        A_offsets = j_idx * N + i
        A_vals = tl.load(A_ptr + A_offsets, mask=mask, other=0.0)
        y_2_vals = tl.load(y_2_ptr + j_idx, mask=mask, other=0.0)
        
        x2_acc += tl.sum(A_vals * y_2_vals)
    
    tl.store(x2_ptr + i, x2_acc)

def mvt_triton(A, x1, x2, y_1, y_2, N):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, 1),)
    
    mvt_kernel[grid](
        A, x1, x2, y_1, y_2,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )