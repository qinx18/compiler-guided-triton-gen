import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(
    A_ptr, B_ptr, alpha, M, N,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate i, j from program ID
    i = pid // N
    j = pid % N
    
    # Check bounds
    if i >= M or j >= N:
        return
    
    # Load initial B[i][j]
    b_idx = i * N + j
    b_val = tl.load(B_ptr + b_idx)
    
    # Accumulate sum for k from i+1 to M-1
    sum_val = 0.0
    k_start = i + 1
    
    # Process k values in blocks
    for k_block_start in range(k_start, M, BLOCK_SIZE):
        k_offsets = tl.arange(0, BLOCK_SIZE)
        k_vals = k_block_start + k_offsets
        k_mask = (k_vals < M) & (k_vals >= k_start)
        
        # Load A[k][i] values
        a_indices = k_vals * M + i
        a_vals = tl.load(A_ptr + a_indices, mask=k_mask, other=0.0)
        
        # Load B[k][j] values
        b_indices = k_vals * N + j
        b_vals = tl.load(B_ptr + b_indices, mask=k_mask, other=0.0)
        
        # Compute products and sum
        products = a_vals * b_vals
        sum_val += tl.sum(tl.where(k_mask, products, 0.0))
    
    # Update B[i][j] = alpha * (B[i][j] + sum)
    final_val = alpha * (b_val + sum_val)
    tl.store(B_ptr + b_idx, final_val)

def trmm_triton(A, B, alpha, M, N):
    # Total number of (i, j) pairs
    grid_size = M * N
    
    # Block size for k-dimension processing
    BLOCK_SIZE = 32
    
    # Launch kernel with one thread per (i, j) pair
    trmm_kernel[(grid_size,)](
        A, B, alpha, M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return B