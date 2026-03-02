import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, NI, NJ, NK, 
                stride_A_i, stride_A_k, stride_B_k, stride_B_j, stride_C_i, stride_C_j,
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < NI
    j_mask = j_indices < NJ
    
    # First multiply C by beta
    c_ptrs = C_ptr + i_indices[:, None] * stride_C_i + j_indices[None, :] * stride_C_j
    c_mask = i_mask[:, None] & j_mask[None, :]
    c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
    c_vals = c_vals * beta
    
    # Accumulate A * B over k dimension
    for k in range(NK):
        a_ptrs = A_ptr + i_indices * stride_A_i + k * stride_A_k
        a_vals = tl.load(a_ptrs, mask=i_mask, other=0.0)
        
        b_ptrs = B_ptr + k * stride_B_k + j_indices * stride_B_j
        b_vals = tl.load(b_ptrs, mask=j_mask, other=0.0)
        
        ab_vals = alpha * a_vals[:, None] * b_vals[None, :]
        c_vals = c_vals + ab_vals
    
    tl.store(c_ptrs, c_vals, mask=c_mask)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    
    gemm_kernel[grid](
        A, B, C, alpha, beta, NI, NJ, NK,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_I, BLOCK_J
    )