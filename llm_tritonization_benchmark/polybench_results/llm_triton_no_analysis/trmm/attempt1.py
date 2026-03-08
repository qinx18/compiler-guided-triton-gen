import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which (i, j) pair this program handles
    i = pid // N
    j = pid % N
    
    # Check bounds
    if i >= M:
        return
    if j >= N:
        return
    
    # Load initial B[i][j]
    b_idx = i * N + j
    b_val = tl.load(B_ptr + b_idx)
    
    # Compute sum over k from i+1 to M-1
    sum_val = 0.0
    k_start = i + 1
    
    # Process k values in blocks
    k_offsets = tl.arange(0, BLOCK_SIZE)
    for k_block_start in range(k_start, M, BLOCK_SIZE):
        k_indices = k_block_start + k_offsets
        k_mask = (k_indices < M) & (k_indices >= k_start)
        
        # Load A[k][i] values
        a_indices = k_indices * M + i
        a_vals = tl.load(A_ptr + a_indices, mask=k_mask, other=0.0)
        
        # Load B[k][j] values
        b_indices = k_indices * N + j
        b_vals = tl.load(B_ptr + b_indices, mask=k_mask, other=0.0)
        
        # Compute partial sum
        partial_sum = tl.sum(a_vals * b_vals)
        sum_val += partial_sum
    
    # Update B[i][j] = alpha * (B[i][j] + sum)
    final_val = alpha * (b_val + sum_val)
    tl.store(B_ptr + b_idx, final_val)

def trmm_triton(A, B, alpha, M, N):
    # Launch one thread per (i, j) pair
    grid = (M * N,)
    
    # Choose block size for k-loop
    BLOCK_SIZE = 32
    
    trmm_kernel[grid](
        A, B, alpha,
        M, N, BLOCK_SIZE
    )