import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(A_ptr, B_ptr, tmp_ptr, x_ptr, y_ptr, alpha, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Initialize accumulators
    tmp_acc = 0.0
    y_acc = 0.0
    
    # Vectorized reduction over j dimension
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, N, BLOCK_SIZE):
        current_j = j_start + j_offsets
        mask = current_j < N
        
        # Load x values
        x_vals = tl.load(x_ptr + current_j, mask=mask, other=0.0)
        
        # Load A[i][j] and B[i][j] values
        a_idx = i * N + current_j
        b_idx = i * N + current_j
        a_vals = tl.load(A_ptr + a_idx, mask=mask, other=0.0)
        b_vals = tl.load(B_ptr + b_idx, mask=mask, other=0.0)
        
        # Compute contributions
        tmp_contrib = a_vals * x_vals
        y_contrib = b_vals * x_vals
        
        # Reduce
        tmp_acc += tl.sum(tmp_contrib)
        y_acc += tl.sum(y_contrib)
    
    # Store tmp[i]
    tl.store(tmp_ptr + i, tmp_acc)
    
    # Compute final y[i] = alpha * tmp[i] + beta * y[i]
    final_y = alpha * tmp_acc + beta * y_acc
    tl.store(y_ptr + i, final_y)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, 1),)
    
    gesummv_kernel[grid](
        A, B, tmp, x, y,
        alpha, beta,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )