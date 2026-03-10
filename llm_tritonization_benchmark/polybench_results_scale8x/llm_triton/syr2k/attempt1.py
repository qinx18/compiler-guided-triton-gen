import torch
import triton
import triton.language as tl

@triton.jit
def syr2k_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, 
                 BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < N
    j_mask = j_indices < N
    
    # Create 2D masks for valid indices and triangular constraint
    i_indices_2d = i_indices[:, None]
    j_indices_2d = j_indices[None, :]
    
    valid_mask = (i_indices_2d < N) & (j_indices_2d < N) & (j_indices_2d <= i_indices_2d)
    
    # Load and scale C matrix elements
    c_offsets = i_indices_2d * N + j_indices_2d
    c_vals = tl.load(C_ptr + c_offsets, mask=valid_mask, other=0.0)
    c_vals = c_vals * beta
    
    # Accumulate over k dimension
    for k in range(M):
        # Load A[j, k] and B[i, k]
        a_offsets_j = j_indices_2d * M + k
        b_offsets_i = i_indices_2d * M + k
        
        a_vals_j = tl.load(A_ptr + a_offsets_j, mask=(j_indices_2d < N), other=0.0)
        b_vals_i = tl.load(B_ptr + b_offsets_i, mask=(i_indices_2d < N), other=0.0)
        
        # Load A[i, k] and B[j, k] 
        a_offsets_i = i_indices_2d * M + k
        b_offsets_j = j_indices_2d * M + k
        
        a_vals_i = tl.load(A_ptr + a_offsets_i, mask=(i_indices_2d < N), other=0.0)
        b_vals_j = tl.load(B_ptr + b_offsets_j, mask=(j_indices_2d < N), other=0.0)
        
        # Compute contribution: A[j,k]*alpha*B[i,k] + B[j,k]*alpha*A[i,k]
        contrib = a_vals_j * alpha * b_vals_i + b_vals_j * alpha * a_vals_i
        c_vals = c_vals + contrib * valid_mask.to(tl.float32)
    
    # Store results back to C
    tl.store(C_ptr + c_offsets, c_vals, mask=valid_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid_i = triton.cdiv(N, BLOCK_I)
    grid_j = triton.cdiv(N, BLOCK_J)
    
    syr2k_kernel[(grid_i, grid_j)](
        A, B, C, alpha, beta, M, N, BLOCK_I, BLOCK_J
    )