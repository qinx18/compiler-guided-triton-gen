import torch
import triton
import triton.language as tl

@triton.jit
def syrk_kernel(A_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offset = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offset = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    i_mask = i_offset < N
    j_mask = j_offset < N
    
    # Create masks for the triangular condition j <= i
    i_expanded = i_offset[:, None]
    j_expanded = j_offset[None, :]
    triangular_mask = j_expanded <= i_expanded
    
    # Combined mask
    valid_mask = i_mask[:, None] & j_mask[None, :] & triangular_mask
    
    # Scale C[i][j] by beta
    c_offsets = i_expanded * N + j_expanded
    c_vals = tl.load(C_ptr + c_offsets, mask=valid_mask, other=0.0)
    c_vals = c_vals * beta
    
    # Compute the matrix multiplication part
    for k in range(M):
        a_i_offsets = i_offset * M + k
        a_j_offsets = j_offset * M + k
        
        a_i_vals = tl.load(A_ptr + a_i_offsets, mask=i_mask, other=0.0)
        a_j_vals = tl.load(A_ptr + a_j_offsets, mask=j_mask, other=0.0)
        
        # Broadcast for matrix multiplication
        a_i_expanded = a_i_vals[:, None]
        a_j_expanded = a_j_vals[None, :]
        
        # Update C values
        update_vals = alpha * a_i_expanded * a_j_expanded
        c_vals = c_vals + tl.where(valid_mask, update_vals, 0.0)
    
    # Store results back
    tl.store(C_ptr + c_offsets, c_vals, mask=valid_mask)

def syrk_triton(A, C, alpha, beta, M, N):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid = (triton.cdiv(N, BLOCK_I), triton.cdiv(N, BLOCK_J))
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N,
        BLOCK_I, BLOCK_J
    )