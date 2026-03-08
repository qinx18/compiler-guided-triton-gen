import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, 
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    i = pid
    
    if i >= NI:
        return
    
    # First loop: C[i][j] *= beta for j in range(NJ)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, NJ, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < NJ
        
        c_offsets = i * NJ + j_indices
        c_vals = tl.load(C_ptr + c_offsets, mask=j_mask)
        c_vals = c_vals * beta
        tl.store(C_ptr + c_offsets, c_vals, mask=j_mask)
    
    # Second loop: C[i][j] += alpha * A[i][k] * B[k][j]
    for k in range(NK):
        a_val = tl.load(A_ptr + i * NK + k)
        
        for j_start in range(0, NJ, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < NJ
            
            b_offsets = k * NJ + j_indices
            b_vals = tl.load(B_ptr + b_offsets, mask=j_mask)
            
            c_offsets = i * NJ + j_indices
            c_vals = tl.load(C_ptr + c_offsets, mask=j_mask)
            
            c_vals = c_vals + alpha * a_val * b_vals
            tl.store(C_ptr + c_offsets, c_vals, mask=j_mask)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(NI, 1),)
    
    gemm_kernel[grid](
        A, B, C, alpha, beta,
        NI, NJ, NK, BLOCK_SIZE
    )