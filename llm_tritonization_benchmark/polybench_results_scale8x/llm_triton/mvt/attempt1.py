import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(A_ptr, x1_ptr, x2_ptr, y_1_ptr, y_2_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First loop: x1[i] = x1[i] + A[i][j] * y_1[j] for j in range(N)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    x1_acc = 0.0
    
    for j_start in range(0, N, BLOCK_SIZE):
        current_j = j_start + j_offsets
        mask = current_j < N
        
        # Load A[i][j]
        A_idx = i * N + current_j
        A_vals = tl.load(A_ptr + A_idx, mask=mask, other=0.0)
        
        # Load y_1[j]
        y_1_vals = tl.load(y_1_ptr + current_j, mask=mask, other=0.0)
        
        # Accumulate A[i][j] * y_1[j]
        x1_acc += tl.sum(A_vals * y_1_vals)
    
    # Load current x1[i] and add accumulated result
    x1_current = tl.load(x1_ptr + i)
    tl.store(x1_ptr + i, x1_current + x1_acc)
    
    # Second loop: x2[i] = x2[i] + A[j][i] * y_2[j] for j in range(N)
    x2_acc = 0.0
    
    for j_start in range(0, N, BLOCK_SIZE):
        current_j = j_start + j_offsets
        mask = current_j < N
        
        # Load A[j][i] (transpose access)
        A_idx = current_j * N + i
        A_vals = tl.load(A_ptr + A_idx, mask=mask, other=0.0)
        
        # Load y_2[j]
        y_2_vals = tl.load(y_2_ptr + current_j, mask=mask, other=0.0)
        
        # Accumulate A[j][i] * y_2[j]
        x2_acc += tl.sum(A_vals * y_2_vals)
    
    # Load current x2[i] and add accumulated result
    x2_current = tl.load(x2_ptr + i)
    tl.store(x2_ptr + i, x2_current + x2_acc)

def mvt_triton(A, x1, x2, y_1, y_2, N):
    # Launch kernel with one thread per i
    grid = (N,)
    BLOCK_SIZE = 128
    
    mvt_kernel[grid](
        A, x1, x2, y_1, y_2,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )