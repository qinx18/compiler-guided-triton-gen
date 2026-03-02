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
    
    # Load B[i, j] values
    B_ptrs = B_ptr + i_indices[:, None] * stride_B_0 + j_indices[None, :] * stride_B_1
    B_vals = tl.load(B_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)
    
    # Inner k loop
    for k in range(M):
        k_greater_mask = k > i_indices
        
        # Load A[k, i] values
        A_ptrs = A_ptr + k * stride_A_0 + i_indices * stride_A_1
        A_vals = tl.load(A_ptrs, mask=i_mask & k_greater_mask, other=0.0)
        
        # Load B[k, j] values
        B_k_ptrs = B_ptr + k * stride_B_0 + j_indices * stride_B_1
        B_k_vals = tl.load(B_k_ptrs, mask=j_mask, other=0.0)
        
        # Accumulate where k > i
        update_mask = i_mask[:, None] & j_mask[None, :] & k_greater_mask[:, None]
        B_vals = B_vals + tl.where(update_mask, A_vals[:, None] * B_k_vals[None, :], 0.0)
    
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