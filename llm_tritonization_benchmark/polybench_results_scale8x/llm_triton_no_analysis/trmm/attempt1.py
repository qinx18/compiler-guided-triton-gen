import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which (i, j) this program handles
    i = pid // N
    j = pid % N
    
    if i >= M or j >= N:
        return
    
    # Load B[i][j] once
    b_idx = i * N + j
    b_val = tl.load(B_ptr + b_idx)
    
    # Accumulate over k = i+1 to M-1
    start_k = i + 1
    if start_k < M:
        # Process k values in blocks
        k_offsets = tl.arange(0, BLOCK_SIZE)
        
        for k_block in range(start_k, M, BLOCK_SIZE):
            k_indices = k_block + k_offsets
            k_mask = (k_indices < M) & (k_indices >= start_k)
            
            # Load A[k][i] values
            a_indices = k_indices * M + i
            a_vals = tl.load(A_ptr + a_indices, mask=k_mask, other=0.0)
            
            # Load B[k][j] values  
            b_indices = k_indices * N + j
            b_k_vals = tl.load(B_ptr + b_indices, mask=k_mask, other=0.0)
            
            # Compute products and accumulate
            products = a_vals * b_k_vals
            b_val += tl.sum(products)
    
    # Apply alpha scaling and store result
    b_val = alpha * b_val
    tl.store(B_ptr + b_idx, b_val)

def trmm_triton(A, B, alpha, M, N):
    # Launch one thread per (i, j) pair
    grid = (M * N,)
    
    # Choose block size for k-dimension processing
    BLOCK_SIZE = 64
    
    trmm_kernel[grid](
        A, B, alpha, 
        M, N, BLOCK_SIZE
    )
    
    return B