import triton
import triton.language as tl

@triton.jit
def k2mm_kernel_tmp(A_ptr, B_ptr, tmp_ptr, alpha,
                    NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr):
    
    row = tl.program_id(0)
    
    if row < NI:
        col_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, NJ, BLOCK_SIZE):
            j_offsets = j_start + col_offsets
            j_mask = j_offsets < NJ
            
            tmp_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            for k in range(NK):
                a_val = tl.load(A_ptr + row * NK + k)
                b_ptrs = B_ptr + k * NJ + j_offsets
                b_vals = tl.load(b_ptrs, mask=j_mask, other=0.0)
                tmp_vals += alpha * a_val * b_vals
            
            tmp_ptrs = tmp_ptr + row * NJ + j_offsets
            tl.store(tmp_ptrs, tmp_vals, mask=j_mask)

@triton.jit
def k2mm_kernel_d(tmp_ptr, C_ptr, D_ptr, beta,
                  NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr):
    
    row = tl.program_id(0)
    
    if row < NI:
        col_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, NL, BLOCK_SIZE):
            j_offsets = j_start + col_offsets
            j_mask = j_offsets < NL
            
            d_ptrs = D_ptr + row * NL + j_offsets
            d_vals = tl.load(d_ptrs, mask=j_mask, other=0.0)
            d_vals *= beta
            
            for k in range(NJ):
                tmp_val = tl.load(tmp_ptr + row * NJ + k)
                c_ptrs = C_ptr + k * NL + j_offsets
                c_vals = tl.load(c_ptrs, mask=j_mask, other=0.0)
                d_vals += tmp_val * c_vals
            
            tl.store(d_ptrs, d_vals, mask=j_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_SIZE = 32
    grid = (NI,)
    
    k2mm_kernel_tmp[grid](A, B, tmp, alpha, NI, NJ, NK, BLOCK_SIZE)
    k2mm_kernel_d[grid](tmp, C, D, beta, NI, NJ, NL, BLOCK_SIZE)