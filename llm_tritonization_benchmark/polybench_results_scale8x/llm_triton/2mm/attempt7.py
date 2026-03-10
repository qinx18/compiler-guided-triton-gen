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
    
    acc = tl.load(d_ptrs, mask=d_mask, other=0.0) * beta
    
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

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16
    
    grid1 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NJ, BLOCK_N))
    k2mm_kernel_phase1[grid1](
        A, B, tmp, alpha,
        NI, NJ, NK,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    grid2 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NL, BLOCK_N))
    k2mm_kernel_phase2[grid2](
        tmp, C, D, beta,
        NI, NJ, NL,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

def k2mm_kernel():
    pass