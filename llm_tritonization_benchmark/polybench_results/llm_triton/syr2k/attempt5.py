import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Compute block boundaries
    block_i_start = pid_i * BLOCK_I
    block_j_start = pid_j * BLOCK_J
    
    # Create offset ranges
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    # Compute actual indices
    i_indices = block_i_start + i_offsets
    j_indices = block_j_start + j_offsets
    
    # Create 2D indices for this block
    i_idx = i_indices[:, None]
    j_idx = j_indices[None, :]
    
    # Mask for valid indices and triangular condition (j <= i)
    valid_mask = (i_idx < N) & (j_idx < N) & (j_idx <= i_idx)
    
    # Scale C[i][j] by beta
    c_indices = i_idx * N + j_idx
    c_vals = tl.load(C_ptr + c_indices, mask=valid_mask, other=0.0)
    c_vals = c_vals * beta
    
    # Main computation loop over k
    for k in range(M):
        # Masks for accessing A and B arrays
        a_j_mask = (j_idx < N) & (i_idx < N) & (j_idx <= i_idx)
        b_i_mask = (i_idx < N) & (j_idx < N) & (j_idx <= i_idx)
        b_j_mask = (j_idx < N) & (i_idx < N) & (j_idx <= i_idx) 
        a_i_mask = (i_idx < N) & (j_idx < N) & (j_idx <= i_idx)
        
        # Compute indices for current k
        a_j_indices = j_idx * M + k
        b_i_indices = i_idx * M + k
        b_j_indices = j_idx * M + k
        a_i_indices = i_idx * M + k
        
        # Load A[j][k] and B[i][k]
        a_vals_j = tl.load(A_ptr + a_j_indices, mask=a_j_mask, other=0.0)
        b_vals_i = tl.load(B_ptr + b_i_indices, mask=b_i_mask, other=0.0)
        b_vals_j = tl.load(B_ptr + b_j_indices, mask=b_j_mask, other=0.0)
        a_vals_i = tl.load(A_ptr + a_i_indices, mask=a_i_mask, other=0.0)
        
        # Compute the update: A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
        update = a_vals_j * alpha * b_vals_i + b_vals_j * alpha * a_vals_i
        
        # Update C[i][j]
        c_vals = c_vals + update
    
    # Store final result
    tl.store(C_ptr + c_indices, c_vals, mask=valid_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    # Block sizes
    BLOCK_I = 32
    BLOCK_J = 32
    
    # Grid dimensions
    grid_i = triton.cdiv(N, BLOCK_I)
    grid_j = triton.cdiv(N, BLOCK_J)
    
    # Launch kernel
    syr2k_kernel[(grid_i, grid_j)](
        A, B, C, alpha, beta, M, N, BLOCK_I, BLOCK_J
    )