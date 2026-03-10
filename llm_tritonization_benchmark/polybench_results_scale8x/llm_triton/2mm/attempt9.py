import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel_phase1(A, B, tmp, alpha,
                       NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    i_start = pid_m * BLOCK_M
    j_start = pid_n * BLOCK_N
    
    i_offsets = tl.arange(0, BLOCK_M)
    j_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, NK, BLOCK_K):
        i_indices = i_start + i_offsets[:, None]
        k_indices = k_start + k_offsets[None, :]
        a_mask = (i_indices < NI) & (k_indices < NK)
        a_ptrs = A + i_indices * NK + k_indices
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        k_indices_b = k_start + k_offsets[:, None]
        j_indices = j_start + j_offsets[None, :]
        b_mask = (k_indices_b < NK) & (j_indices < NJ)
        b_ptrs = B + k_indices_b * NJ + j_indices
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a_tile, b_tile)
    
    acc = alpha * acc
    
    i_indices = i_start + i_offsets[:, None]
    j_indices = j_start + j_offsets[None, :]
    tmp_mask = (i_indices < NI) & (j_indices < NJ)
    tmp_ptrs = tmp + i_indices * NJ + j_indices
    tl.store(tmp_ptrs, acc, mask=tmp_mask)

@triton.jit
def k2mm_kernel_phase2(tmp, C, D, beta,
                       NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    i_start = pid_m * BLOCK_M
    l_start = pid_n * BLOCK_N
    
    i_offsets = tl.arange(0, BLOCK_M)
    l_offsets = tl.arange(0, BLOCK_N)
    j_offsets = tl.arange(0, BLOCK_K)
    
    i_indices = i_start + i_offsets[:, None]
    l_indices = l_start + l_offsets[None, :]
    d_mask = (i_indices < NI) & (l_indices < NL)
    d_ptrs = D + i_indices * NL + l_indices
    
    d_vals = tl.load(d_ptrs, mask=d_mask, other=0.0)
    acc = beta * d_vals
    
    for j_start_inner in range(0, NJ, BLOCK_K):
        i_indices_tmp = i_start + i_offsets[:, None]
        j_indices_tmp = j_start_inner + j_offsets[None, :]
        tmp_mask = (i_indices_tmp < NI) & (j_indices_tmp < NJ)
        tmp_ptrs = tmp + i_indices_tmp * NJ + j_indices_tmp
        tmp_tile = tl.load(tmp_ptrs, mask=tmp_mask, other=0.0)
        
        j_indices_c = j_start_inner + j_offsets[:, None]
        l_indices_c = l_start + l_offsets[None, :]
        c_mask = (j_indices_c < NJ) & (l_indices_c < NL)
        c_ptrs = C + j_indices_c * NL + l_indices_c
        c_tile = tl.load(c_ptrs, mask=c_mask, other=0.0)
        
        acc += tl.dot(tmp_tile, c_tile)
    
    tl.store(d_ptrs, acc, mask=d_mask)

@triton.jit
def k2mm_kernel(A, B, C, D, tmp, alpha, beta,
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, NL: tl.constexpr):
    pid = tl.program_id(0)
    
    for i in range(NI):
        for j in range(NJ):
            tmp_val = 0.0
            for k in range(NK):
                a_ptr = A + i * NK + k
                b_ptr = B + k * NJ + j
                a_val = tl.load(a_ptr)
                b_val = tl.load(b_ptr)
                tmp_val += alpha * a_val * b_val
            
            tmp_ptr = tmp + i * NJ + j
            tl.store(tmp_ptr, tmp_val)
    
    for i in range(NI):
        for j in range(NL):
            d_ptr = D + i * NL + j
            d_val = tl.load(d_ptr)
            acc = beta * d_val
            
            for k in range(NJ):
                tmp_ptr = tmp + i * NJ + k
                c_ptr = C + k * NL + j
                tmp_val = tl.load(tmp_ptr)
                c_val = tl.load(c_ptr)
                acc += tmp_val * c_val
            
            tl.store(d_ptr, acc)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    grid = (1,)
    k2mm_kernel[grid](
        A, B, C, D, tmp, alpha, beta,
        NI, NJ, NK, NL
    )