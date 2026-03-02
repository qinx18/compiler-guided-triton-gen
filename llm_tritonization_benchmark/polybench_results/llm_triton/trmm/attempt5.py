import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M, N, stride_A_0, stride_A_1, stride_B_0, stride_B_1, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < M
    j_mask = j_indices < N
    
    # Load initial B[i, j] values
    B_ptrs = B_ptr + i_indices[:, None] * stride_B_0 + j_indices[None, :] * stride_B_1
    B_vals = tl.load(B_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)
    
    # For each i in the block
    for i_block_idx in range(BLOCK_SIZE_I):
        i_actual = i_start + i_block_idx
        i_valid = i_actual < M
        
        # Inner k loop: k from i+1 to M-1
        for k in range(M):
            k_valid = (k > i_actual) & (k < M) & i_valid
            
            # Load A[k, i_actual]
            A_ptr_k = A_ptr + k * stride_A_0 + i_actual * stride_A_1
            A_val = tl.load(A_ptr_k)
            
            # Load B[k, j] for all j in block
            B_k_ptrs = B_ptr + k * stride_B_0 + j_indices * stride_B_1
            B_k_vals = tl.load(B_k_ptrs, mask=j_mask, other=0.0)
            
            # Update B[i_actual, j] for all j in block
            i_match_mask = (i_indices == i_actual)
            update_mask = i_match_mask[:, None] & j_mask[None, :] & k_valid
            B_vals = B_vals + tl.where(update_mask, A_val * B_k_vals[None, :], 0.0)
    
    # Scale by alpha
    B_vals = alpha * B_vals
    
    # Store back to B[i, j]
    tl.store(B_ptrs, B_vals, mask=i_mask[:, None] & j_mask[None, :])

def trmm_triton(A, B, alpha, M, N):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 32
    
    grid_i = triton.cdiv(M, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N, BLOCK_SIZE_J)
    grid = (grid_i, grid_j)
    
    trmm_kernel[grid](
        A, B, alpha, M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )