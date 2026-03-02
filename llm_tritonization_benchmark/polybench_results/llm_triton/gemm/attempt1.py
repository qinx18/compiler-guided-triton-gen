import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A, B, C, alpha, beta, NI, NJ, NK, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    i_range = i_start + i_offsets
    j_range = j_start + j_offsets
    
    i_mask = i_range < NI
    j_mask = j_range < NJ
    
    # Load C block and multiply by beta
    c_ptrs = C + i_range[:, None] * NJ + j_range[None, :]
    mask_2d = i_mask[:, None] & j_mask[None, :]
    c_vals = tl.load(c_ptrs, mask=mask_2d, other=0.0)
    c_vals = c_vals * beta
    
    # Accumulate A[i,k] * B[k,j] over k
    for k in range(NK):
        a_ptrs = A + i_range * NK + k
        b_ptrs = B + k * NJ + j_range
        
        a_vals = tl.load(a_ptrs, mask=i_mask, other=0.0)
        b_vals = tl.load(b_ptrs, mask=j_mask, other=0.0)
        
        c_vals += alpha * a_vals[:, None] * b_vals[None, :]
    
    # Store result
    tl.store(c_ptrs, c_vals, mask=mask_2d)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid_i = triton.cdiv(NI, BLOCK_I)
    grid_j = triton.cdiv(NJ, BLOCK_J)
    
    gemm_kernel[(grid_i, grid_j)](
        A, B, C, alpha, beta, NI, NJ, NK, BLOCK_I, BLOCK_J
    )