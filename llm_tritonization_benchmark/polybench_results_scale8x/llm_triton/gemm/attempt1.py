import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A, B, C, alpha, beta, NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    i_mask = (i_start + i_offsets) < NI
    j_mask = (j_start + j_offsets) < NJ
    
    # Initialize C block
    c_offsets = (i_start + i_offsets[:, None]) * NJ + (j_start + j_offsets[None, :])
    c_mask = i_mask[:, None] & j_mask[None, :]
    c_vals = tl.load(C + c_offsets, mask=c_mask, other=0.0)
    c_vals = c_vals * beta
    
    # Compute GEMM
    for k in range(NK):
        a_offsets = (i_start + i_offsets) * NK + k
        a_mask = i_mask
        a_vals = tl.load(A + a_offsets, mask=a_mask, other=0.0)
        
        b_offsets = k * NJ + (j_start + j_offsets)
        b_mask = j_mask
        b_vals = tl.load(B + b_offsets, mask=b_mask, other=0.0)
        
        c_vals += alpha * a_vals[:, None] * b_vals[None, :]
    
    tl.store(C + c_offsets, c_vals, mask=c_mask)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    
    gemm_kernel[grid](A, B, C, alpha, beta, NI, NJ, NK, BLOCK_I, BLOCK_J)