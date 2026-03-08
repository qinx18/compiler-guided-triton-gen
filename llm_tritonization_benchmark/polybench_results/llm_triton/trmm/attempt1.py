import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M: tl.constexpr, N: tl.constexpr, BLOCK_J: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Calculate j indices for this block
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    j_mask = j_offsets < N
    
    # Sequential loop over i
    for i in range(M):
        # Load B[i, j] values for this row
        b_ptrs = B_ptr + i * N + j_offsets
        b_vals = tl.load(b_ptrs, mask=j_mask, other=0.0)
        
        # Inner loop over k from i+1 to M-1
        for k in range(i + 1, M):
            # Load A[k, i] (scalar)
            a_val = tl.load(A_ptr + k * M + i)
            
            # Load B[k, j] values
            bk_ptrs = B_ptr + k * N + j_offsets
            bk_vals = tl.load(bk_ptrs, mask=j_mask, other=0.0)
            
            # Accumulate: B[i][j] += A[k][i] * B[k][j]
            b_vals += a_val * bk_vals
        
        # Scale by alpha: B[i][j] = alpha * B[i][j]
        b_vals = alpha * b_vals
        
        # Store back to B[i, j]
        tl.store(b_ptrs, b_vals, mask=j_mask)

def trmm_triton(A, B, alpha, M, N):
    # Choose block size for j dimension
    BLOCK_J = 32
    
    # Calculate grid size
    grid_j = triton.cdiv(N, BLOCK_J)
    
    # Launch kernel
    trmm_kernel[(grid_j,)](
        A, B, alpha, M, N, BLOCK_J
    )